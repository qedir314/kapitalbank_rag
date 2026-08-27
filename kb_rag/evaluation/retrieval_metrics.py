"""Retrieval quality metrics — pure functions over ranks (no network).

A retrieval "hit" means: at least one returned URL contains one of the
question's expected source fragments. Matching is substring-based so trailing
slashes / scheme / www variations don't matter.
"""

from __future__ import annotations


def first_relevant_rank(retrieved_urls: list[str], expected_fragments: list[str]) -> int | None:
    """0-based rank of the first retrieved URL matching an expected fragment.

    ⚠ ``retrieved_urls`` must have exactly ONE entry per retrieved document —
    callers that carry two URL identities per page (final + source slug) must
    use ``first_relevant_source_rank`` instead. Feeding flattened
    (url, source_url) pairs here doubles every position, so pages at chunk
    ranks 4–6 read as rank 6–11 and score as misses of a top-6 cut
    (the original runner did exactly that until the Phase 4 A/B caught it —
    every pre-fix hit@6 in this repo's history was quietly a hit@3).
    """
    for rank, url in enumerate(retrieved_urls):
        if any(fragment.lower() in url.lower() for fragment in expected_fragments):
            return rank
    return None


def first_relevant_source_rank(sources, expected_fragments: list[str]) -> int | None:
    """0-based rank over SOURCES, matching either URL identity of each page.

    ``sources`` is any sequence of objects with ``.url`` and ``.source_url``
    (pipeline.Source). One entry per retrieved document keeps k aligned with
    the actual context size, while still matching a fragment against either
    the final URL (birbank.az) or the original sitemap slug (kapitalbank.az).
    """
    for rank, src in enumerate(sources):
        identities = f"{getattr(src, 'url', '') or ''} {getattr(src, 'source_url', '') or ''}".lower()
        if any(fragment.lower() in identities for fragment in expected_fragments):
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
