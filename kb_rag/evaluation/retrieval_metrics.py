"""Retrieval quality metrics — pure functions over ranks (no network).

A retrieval "hit" means: at least one returned URL contains one of the
question's expected source fragments. Matching is substring-based so trailing
slashes / scheme / www variations don't matter.
"""

from __future__ import annotations


def first_relevant_rank(retrieved_urls: list[str], expected_fragments: list[str]) -> int | None:
    """0-based rank of the first retrieved URL matching an expected fragment."""
    for rank, url in enumerate(retrieved_urls):
        if any(fragment.lower() in url.lower() for fragment in expected_fragments):
            return rank
    return None


def hit_at_k(rank: int | None, k: int) -> bool:
    return rank is not None and rank < k


def reciprocal_rank(rank: int | None, k: int) -> float:
    """1/(rank+1) when the first hit is inside top-k, else 0."""
    if rank is None or rank >= k:
        return 0.0
    return 1.0 / (rank + 1)


def mean(values: list[float]) -> float:
    return sum(values) / len(values) if values else 0.0
