from kb_rag.evaluation.retrieval_metrics import (
    first_relevant_rank,
    hit_at_k,
    mean,
    reciprocal_rank,
)


def test_first_relevant_rank_case_insensitive():
    urls = ["https://www.kapitalbank.az/en/loans", "https://KAPITALBANK.AZ/en/cards/miles-debet"]
    rank = first_relevant_rank(urls, ["miles-debet"])
    assert rank == 1


def test_first_relevant_rank_returns_none_when_absent():
    assert first_relevant_rank(["a.com", "b.com"], ["nope"]) is None


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
