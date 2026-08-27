from types import SimpleNamespace

from kb_rag.evaluation.retrieval_metrics import (
    first_relevant_rank,
    first_relevant_source_rank,
    hit_at_k,
    mean,
    reciprocal_rank,
)


def _src(url, source_url):
    return SimpleNamespace(url=url, source_url=source_url)


def test_first_relevant_rank_case_insensitive():
    urls = ["https://www.kapitalbank.az/en/loans", "https://KAPITALBANK.AZ/en/cards/miles-debet"]
    rank = first_relevant_rank(urls, ["miles-debet"])
    assert rank == 1


def test_first_relevant_rank_returns_none_when_absent():
    assert first_relevant_rank(["a.com", "b.com"], ["nope"]) is None


# ----------------------------------------------------- source-granularity (Phase 4)
def test_source_rank_matches_either_url_identity_one_entry_per_page():
    # a page whose FINAL URL is the expected slug, but whose source slug is
    # unrelated — must still count, and occupy exactly one rank slot
    sources = [_src("https://birbank.az/overdraft", "https://kapitalbank.az/x"),
               _src("https://birbank.az/deposits", "https://kapitalbank.az/depozitler")]
    assert first_relevant_source_rank(sources, ["depozitler"]) == 1
    assert first_relevant_source_rank(sources, ["overdraft"]) == 0


def test_source_rank_fourth_page_is_inside_top6_not_seventh():
    """The regression the whole project's hit@6 understated: ranking over a
    flattened (url, source_url) list pushed the 4th page to rank 6/7, outside
    a top-6 cut, so only 3 pages could ever register a hit. Ranking over
    sources puts page 4 at 0-based rank 3 → a genuine top-6 hit."""
    pages = [_src(f"https://birbank.az/p{i}", f"https://kapitalbank.az/s{i}") for i in range(6)]
    rank = first_relevant_source_rank(pages, ["p3"])   # 4th page (0-based index 3)
    assert rank == 3
    assert hit_at_k(rank, 6) is True                    # inside top-6
    # ...while the same page under the old flattened scheme read as rank 6:
    flat = [u for s in pages for u in (s.url, s.source_url)]
    old = first_relevant_rank(flat, ["p3"])
    assert old == 6
    assert hit_at_k(old, 6) is False                    # the phantom-miss bug


def test_hit_at_k_boundaries():
    assert hit_at_k(5, 6) is True     # last slot inside top-k
    assert hit_at_k(6, 6) is False    # one past top-k
    assert hit_at_k(None, 6) is False


def test_reciprocal_rank_values():
    assert reciprocal_rank(0, 10) == 1.0
    assert reciprocal_rank(1, 10) == 0.5
    assert reciprocal_rank(9, 10) == 0.1
    assert reciprocal_rank(10, 10) == 0.0
    assert reciprocal_rank(None, 10) == 0.0


def test_mean_empty_is_zero():
    assert mean([]) == 0.0
    assert mean([1.0, 2.0]) == 1.5
