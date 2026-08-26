"""Embedding wrapper for multilingual E5 models.

E5 requires task prefixes — queries must be embedded as ``"query: ..."`` and
documents as ``"passage: ..."``. Vectors are L2-normalized so similarity is a
plain dot product / cosine.
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

    @property
    def model(self):
        if self._model is None:
            from sentence_transformers import SentenceTransformer

            self._model = SentenceTransformer(self.config.model_name, device="cpu")
            # sentence-transformers >= 6 removed encode(max_length=...);
            # sequence length is configured on the model itself
            self._model.max_seq_length = self.config.max_seq_tokens
        return self._model

    def embed_passages(self, texts: list[str]) -> np.ndarray:
        return self.model.encode(
            [PASSAGE_PREFIX + t for t in texts],
            batch_size=self.config.batch_size,
            normalize_embeddings=True,
            show_progress_bar=True,
        )

    def embed_query(self, query: str | list[str]) -> np.ndarray:
        queries = [query] if isinstance(query, str) else query
        return self.model.encode(
            [QUERY_PREFIX + q for q in queries],
            batch_size=self.config.batch_size,
            normalize_embeddings=True,
        )
