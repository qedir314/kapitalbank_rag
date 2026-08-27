"""Embedding wrapper for multilingual models (E5, BGE-M3, etc.).

E5 requires task prefixes — queries must be embedded as ``"query: ..."`` and
documents as ``"passage: ..."``. BGE-M3 and other modern models don't need
prefixes. Vectors are L2-normalized so similarity is a plain dot product / cosine.
"""

from __future__ import annotations

import numpy as np

from kb_rag.config import EmbeddingConfig

QUERY_PREFIX = "query: "
PASSAGE_PREFIX = "passage: "


class E5Embedder:
    def __init__(self, config: EmbeddingConfig):
        self.config = config
        self._model = None  # lazy: avoid loading torch on import (keeps tests fast)
        # E5 requires task prefixes; BGE-M3 and others don't
        self._use_prefixes = "e5" in config.model_name.lower()

    @property
    def model(self):
        if self._model is None:
            from sentence_transformers import SentenceTransformer

            import torch
            device = "cuda" if torch.cuda.is_available() else "cpu"
            self._model = SentenceTransformer(self.config.model_name, device=device)
            # sentence-transformers >= 6 removed encode(max_length=...);
            # sequence length is configured on the model itself
            self._model.max_seq_length = self.config.max_seq_tokens
        return self._model

    def embed_passages(self, texts: list[str]) -> np.ndarray:
        if self._use_prefixes:
            texts = [PASSAGE_PREFIX + t for t in texts]
        return self.model.encode(
            texts,
            batch_size=self.config.batch_size,
            normalize_embeddings=True,
            show_progress_bar=True,
        )

    def embed_query(self, query: str | list[str]) -> np.ndarray:
        queries = [query] if isinstance(query, str) else query
        if self._use_prefixes:
            queries = [QUERY_PREFIX + q for q in queries]
        return self.model.encode(
            queries,
            batch_size=self.config.batch_size,
            normalize_embeddings=True,
        )
