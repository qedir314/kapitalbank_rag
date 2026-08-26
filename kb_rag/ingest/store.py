"""Chroma vector store wrapper (embedded persistent client)."""

from __future__ import annotations

from dataclasses import dataclass

import chromadb
import numpy as np

from kb_rag.config import Settings
from kb_rag.ingest.chunking import Chunk


@dataclass(frozen=True)
class RetrievedChunk:
    text: str
    url: str           # final URL after redirects (birbank.az)
    source_url: str    # original sitemap URL (kapitalbank.az slug)
    title: str
    lang: str
    section: str
    section_path: str
    score: float  # cosine similarity in [0, 1]


class VectorStore:
    """Thin wrapper around a persistent Chroma collection."""

    def __init__(self, settings: Settings):
        self.settings = settings
        settings.chroma_dir.mkdir(parents=True, exist_ok=True)
        self.client = chromadb.PersistentClient(path=str(settings.chroma_dir))
        self.collection = self.client.get_or_create_collection(
            name=settings.vector_store.collection,
            metadata={"hnsw:space": settings.vector_store.distance},
        )

    # ---------------------------------------------------------------- write
    def reset(self) -> None:
        name = self.settings.vector_store.collection
        self.client.delete_collection(name)
        self.collection = self.client.get_or_create_collection(
            name=name,
            metadata={"hnsw:space": self.settings.vector_store.distance},
        )

    def add_chunks(self, chunks: list[Chunk], embeddings) -> int:
        if not chunks:
            return 0
        self.collection.upsert(
            ids=[c.chunk_id for c in chunks],
            documents=[c.text for c in chunks],
            metadatas=[c.metadata for c in chunks],
            embeddings=embeddings.tolist() if hasattr(embeddings, "tolist") else list(embeddings),
        )
        return len(chunks)

    # ----------------------------------------------------------------- read
    def count(self) -> int:
        return self.collection.count()

    def get_all_chunks(self) -> list[RetrievedChunk]:
        """Every indexed chunk (score unused) — feeds the BM25 side of hybrid search."""
        result = self.collection.get(include=["documents", "metadatas"])
        docs = result.get("documents") or []
        metas = result.get("metadatas") or []
        out: list[RetrievedChunk] = []
        for doc, meta in zip(docs, metas):
            meta = meta or {}
            out.append(
                RetrievedChunk(
                    text=doc,
                    url=meta.get("url", ""),
                    source_url=meta.get("source_url", ""),
                    title=meta.get("title", ""),
                    lang=meta.get("lang", ""),
                    section=meta.get("section", ""),
                    section_path=meta.get("section_path", ""),
                    score=0.0,
                )
            )
        return out

    def query(
        self,
        query_embedding,
        top_k: int,
        where: dict | None = None,
    ) -> list[RetrievedChunk]:
        emb = np.asarray(query_embedding, dtype=np.float32)
        if emb.ndim == 1:  # accept a single vector or a batch of queries
            emb = emb[None, :]
        n = min(top_k, max(self.count(), 1))
        result = self.collection.query(
            query_embeddings=emb.tolist(),
            n_results=n,
            where=where,
            include=["documents", "metadatas", "distances"],
        )
        out: list[RetrievedChunk] = []
        docs = result.get("documents") or [[]]
        metas = result.get("metadatas") or [[]]
        dists = result.get("distances") or [[]]
        for doc, meta, dist in zip(docs[0], metas[0], dists[0]):
            meta = meta or {}
            out.append(
                RetrievedChunk(
                    text=doc,
                    url=meta.get("url", ""),
                    source_url=meta.get("source_url", ""),
                    title=meta.get("title", ""),
                    lang=meta.get("lang", ""),
                    section=meta.get("section", ""),
                    section_path=meta.get("section_path", ""),
                    score=1.0 - float(dist),
                )
            )
        return out
