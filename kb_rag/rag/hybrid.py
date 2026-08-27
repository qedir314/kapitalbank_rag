"""Hybrid retrieval helpers: BM25 sparse search, RRF fusion, cross-encoder re-ranking.

Why hybrid: dense embeddings paraphrase well but under-match exact tokens — product
names ("BirKart"), numbers ("10.9%"), transliterated terms — which is precisely what
bank queries hinge on. BM25 covers lexical precision, reciprocal rank fusion merges
the two rankings without score-scale calibration, and a multilingual cross-encoder
re-scores the fused pool so the final top-k is ordered by actual query-passage fit.
"""

from __future__ import annotations

import math
import re
from dataclasses import replace

from rank_bm25 import BM25Okapi

from kb_rag.ingest.store import RetrievedChunk

_TOKEN_RE = re.compile(r"\w+", re.UNICODE)
_RRF_K = 60  # standard damping constant

# --- morphology-aware augmentation (plan 2.3, behind a config flag) -----------
# Azerbaijani is agglutinative: ``kreditlərin`` (genitive plural of "kredit")
# shares no substring with the query form ``kredit`` that bare \\w+ tokens
# would match. We don't replace the surface token — we ADD normalized forms,
# so exact matches keep their full BM25 weight and stems only widen recall.
# Order of augmentation: transliterate Cyrillic → Latin first (bank queries
# arrive in ru while product nouns live in az), then strip one suffix.
_CYR_TO_LAT = str.maketrans({
    "а": "a", "б": "b", "в": "v", "г": "g", "д": "d", "е": "e", "ё": "yo",
    "ж": "j", "з": "z", "и": "i", "й": "y", "к": "k", "л": "l", "м": "m",
    "н": "n", "о": "o", "п": "p", "р": "r", "с": "s", "т": "t", "у": "u",
    "ф": "f", "х": "h", "ц": "ts", "ч": "ç", "ш": "ş", "щ": "şç", "ъ": "",
    "ы": "ı", "ь": "", "э": "e", "ю": "yu", "я": "ya",
})
_HAS_CYRILLIC = re.compile(r"[а-яё]")
# longest-first so compound forms (lərin = lər+in) win over their tails
_AZ_SUFFIXES = sorted(
    [
        "lar", "lər", "ların", "lərin", "larda", "lərdə", "lardan", "lərdən",
        "da", "də", "ndan", "ndən", "dan", "dən", "tan", "tən",
        "a", "ə", "nın", "nin", "nun", "nün", "nı", "ni", "nu", "nü",
        "ın", "in", "un", "ün", "ı", "i", "u", "ü", "sı", "si", "su", "sü",
        "yla", "lə",
    ],
    key=len,
    reverse=True,
)
_MIN_STEM = 4  # don't chop tokens down to fragments


def _has_cyrillic(token: str) -> bool:
    return bool(_HAS_CYRILLIC.search(token))


def _strip_az_suffix(token: str) -> str | None:
    """One-step conservative suffix strip; None when nothing safely strips."""
    for suffix in _AZ_SUFFIXES:
        if token.endswith(suffix) and len(token) - len(suffix) >= _MIN_STEM:
            return token[: -len(suffix)]
    return None


def tokenize(text: str, morph: bool = False) -> list[str]:
    """Lowercase word tokens; ``\\w`` keeps az/ru letters intact.

    With ``morph=True`` each token also contributes a Latin transliteration
    (Cyrillic input) and a one-step Azerbaijani stem — additive, never a
    replacement, so plain surface matches keep scoring exactly as before.
    """
    tokens = _TOKEN_RE.findall(text.lower())
    if not morph:
        return tokens
    out: list[str] = []
    seen = set()

    def emit(tok: str) -> None:
        if tok and tok not in seen:
            seen.add(tok)
            out.append(tok)

    for token in tokens:
        emit(token)
        if _has_cyrillic(token):
            latin = token.translate(_CYR_TO_LAT)
            emit(latin)
            stem = _strip_az_suffix(latin)
        else:
            stem = _strip_az_suffix(token)
        emit(stem)
    return out


class BM25Index:
    """In-memory BM25 over the indexed chunks (corpus fits comfortably in RAM).

    ``exclude_sections`` lets queries drop chunks from the BM25 corpus so IDF
    statistics aren't skewed by content the queries would never return (e.g.
    news pages excluded at query time by config). ``morph_tokens`` enables
    the additive transliteration + suffix-stripping augmentation (plan 2.3).
    """

    def __init__(
        self,
        chunks: list[RetrievedChunk],
        exclude_sections: list[str] | None = None,
        morph_tokens: bool = False,
    ):
        exclude = set(exclude_sections or [])
        self.chunks = [c for c in chunks if c.section not in exclude] if exclude else list(chunks)
        self._all_chunks = list(chunks)  # keep reference for introspection / tests
        self._morph = morph_tokens
        self._bm25 = (
            BM25Okapi([tokenize(c.text, morph=morph_tokens) for c in self.chunks])
            if self.chunks else None
        )

    @classmethod
    def from_store(
        cls,
        store,
        exclude_sections: list[str] | None = None,
        morph_tokens: bool = False,
    ) -> "BM25Index":
        return cls(store.get_all_chunks(), exclude_sections=exclude_sections,
                   morph_tokens=morph_tokens)

    def search(
        self,
        query: str,
        limit: int,
        filter_fn=None,
    ) -> list[RetrievedChunk]:
        """Top-``limit`` chunks by BM25, optionally filtered; [] when no term matches."""
        if self._bm25 is None:
            return []
        tokens = tokenize(query, morph=self._morph)
        if not tokens:
            return []
        token_set = set(tokens)
        # candidate = any doc containing at least one query token; raw BM25
        # scores can go negative on skewed corpora (tiny N / ubiquitous terms),
        # so overlap — not score sign — decides who competes
        overlapping = [
            i for i, freqs in enumerate(self._bm25.doc_freqs)
            if token_set & freqs.keys()
        ]
        scores = self._bm25.get_scores(tokens)
        order = sorted(overlapping, key=lambda i: scores[i], reverse=True)

        out: list[RetrievedChunk] = []
        for i in order:
            chunk = self.chunks[i]
            if filter_fn and not filter_fn(chunk):
                continue
            out.append(replace(chunk, score=float(scores[i])))
            if len(out) >= limit:
                break
        return out


def reciprocal_rank_fusion(rankings: list[list[RetrievedChunk]]) -> list[RetrievedChunk]:
    """Fuse ranked lists by RRF: score(d) = sum(1 / (k + rank_i(d))).

    Rank-based, so cosine similarities and BM25 scores never need calibrating
    against each other.
    """
    fused: dict[str, float] = {}
    first_seen: dict[str, RetrievedChunk] = {}
    for ranking in rankings:
        for pos, chunk in enumerate(ranking):
            key = chunk.url + "\x00" + chunk.text[:128]
            fused[key] = fused.get(key, 0.0) + 1.0 / (_RRF_K + pos + 1)
            first_seen.setdefault(key, chunk)
    ordered = sorted(fused.items(), key=lambda kv: kv[1], reverse=True)
    return [replace(first_seen[key], score=score) for key, score in ordered]


class CrossEncoderReranker:
    """Multilingual cross-encoder scoring of (query, passage) pairs."""

    def __init__(self, model_name: str, max_length: int = 448):
        self.model_name = model_name
        self.max_length = max_length
        self._model = None

    @property
    def model(self):
        if self._model is None:  # lazy: heavy download/load on first query only
            import torch
            from sentence_transformers import CrossEncoder

            device = "cuda" if torch.cuda.is_available() else "cpu"
            self._model = CrossEncoder(self.model_name, max_length=self.max_length, device=device)
        return self._model

    def rerank(self, query: str, chunks: list[RetrievedChunk]) -> list[RetrievedChunk]:
        if not chunks:
            return []
        pairs = [(query, c.text) for c in chunks]
        raw = self.model.predict(pairs)
        scores = [_to_probability(float(s)) for s in raw]
        rescored = [replace(c, score=s) for c, s in zip(chunks, scores)]
        return sorted(rescored, key=lambda c: c.score, reverse=True)


def _to_probability(score: float) -> float:
    """Normalize model output to [0, 1]; some CEs emit logits, others probabilities."""
    if 0.0 <= score <= 1.0:
        return score
    return 1.0 / (1.0 + math.exp(-score))
