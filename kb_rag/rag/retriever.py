"""Semantic retrieval over the Chroma index with optional metadata filters.

Pipeline per query: dense ANN search + BM25 lexical search -> reciprocal rank
fusion -> cross-encoder re-scoring of the fused pool -> per-URL dedupe -> top-k.
Every stage is toggleable from config, degrading gracefully to plain dense
search when disabled.
"""

from __future__ import annotations

from kb_rag.config import Settings
from kb_rag.ingest.embeddings import E5Embedder
from kb_rag.ingest.store import RetrievedChunk, VectorStore
from kb_rag.rag.hybrid import BM25Index, CrossEncoderReranker, reciprocal_rank_fusion


def dedupe_by_url(chunks: list[RetrievedChunk], max_per_url: int = 1) -> list[RetrievedChunk]:
    """Keep at most ``max_per_url`` chunks per page, best score first.

    Without this, a single long page can fill the whole context window (its
    sections rank similarly), starving the prompt of page diversity.
    """
    counts: dict[str, int] = {}
    kept: list[RetrievedChunk] = []
    for chunk in chunks:  # inputs are score-ordered by every upstream stage
        n = counts.get(chunk.url, 0)
        if n >= max_per_url:
            continue
        counts[chunk.url] = n + 1
        kept.append(chunk)
    return kept


class Retriever:
    def __init__(self, settings: Settings, embedder: E5Embedder, store: VectorStore):
        self.settings = settings
        self.embedder = embedder
        self.store = store
        self._bm25: BM25Index | None = None   # lazy: scans the whole collection once
        self._reranker: CrossEncoderReranker | None = None

    @property
    def bm25(self) -> BM25Index:
        if self._bm25 is None:
            self._bm25 = BM25Index.from_store(self.store)
        return self._bm25

    @property
    def reranker(self) -> CrossEncoderReranker:
        if self._reranker is None:
            cfg = self.settings.retrieval
            self._reranker = CrossEncoderReranker(cfg.rerank_model, cfg.rerank_max_length)
        return self._reranker

    @staticmethod
    def _build_where(
        lang: str | None,
        section: str | list[str] | None,
        exclude_sections: list[str],
    ) -> dict | None:
        conditions = []
        if lang:
            conditions.append({"lang": {"$eq": lang}})
        if section:
            values = [section] if isinstance(section, str) else list(section)
            conditions.append({"section": {"$in": values}})
        elif exclude_sections:
            # only applied when no explicit section filter — an explicit
            # selection means the caller knows what they want
            conditions.append({"section": {"$nin": exclude_sections}})
        if not conditions:
            return None
        return conditions[0] if len(conditions) == 1 else {"$and": conditions}

    @staticmethod
    def _passes_filter(
        chunk: RetrievedChunk,
        lang: str | None,
        section: str | list[str] | None,
        exclude_sections: list[str],
    ) -> bool:
        """Same semantics as ``_build_where``, as a Python predicate for BM25 results."""
        if lang and chunk.lang != lang:
            return False
        if section:
            allowed = [section] if isinstance(section, str) else list(section)
            if chunk.section not in allowed:
                return False
        elif exclude_sections and chunk.section in exclude_sections:
            return False
        return True

    def retrieve(
        self,
        query: str,
        top_k: int | None = None,
        lang: str | None = None,
        section: str | list[str] | None = None,
    ) -> list[RetrievedChunk]:
        cfg = self.settings.retrieval
        k = top_k or cfg.top_k
        pool = max(k * 3, cfg.candidate_pool)
        where = self._build_where(lang, section, cfg.exclude_sections)

        dense = self.store.query(self.embedder.embed_query(query), top_k=pool, where=where)

        candidates = dense
        if cfg.enable_bm25 and self.bm25.chunks:
            filter_fn = lambda c: self._passes_filter(c, lang, section, cfg.exclude_sections)
            sparse = self.bm25.search(query, limit=pool, filter_fn=filter_fn)
            candidates = reciprocal_rank_fusion([dense, sparse])

        # dedupe BEFORE re-ranking: two sections of one page would burn two of
        # the few expensive cross-encoder slots on near-identical text
        candidates = dedupe_by_url(candidates)[:cfg.rerank_candidates]

        if cfg.rerank_model:
            candidates = self.reranker.rerank(query, candidates)

        return candidates[:k]
