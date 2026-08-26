"""Unit tests for hybrid retrieval helpers: tokenizer, BM25, RRF, cross-encoder.

Covers the math the README leans on (RRF score, k-damping, logit→probability)
plus the BM25 negative-score guard the comment in ``BM25Index.search`` calls out.
"""
import math

import pytest

from kb_rag.ingest.store import RetrievedChunk
from kb_rag.rag.hybrid import (
    BM25Index,
    CrossEncoderReranker,
    _RRF_K,
    _to_probability,
    reciprocal_rank_fusion,
    tokenize,
)


# --------------------------------------------------------------------------- helpers
def _chunk(text: str, url: str, section: str = "cards", score: float = 0.0) -> RetrievedChunk:
    return RetrievedChunk(
        text=text,
        url=url,
        source_url=url,
        title="t",
        lang="az",
        section=section,
        section_path=section,
        score=score,
    )


# --------------------------------------------------------------------------- tokenizer
def test_tokenize_keeps_azerbaijani_letters():
    assert tokenize("Nağd pul krediti 10.9%-dən") == [
        "nağd", "pul", "krediti", "10", "9", "dən",
    ]


def test_tokenize_keeps_cyrillic_letters():
    assert tokenize("Золотая Корона — переводы") == [
        "золотая", "корона", "переводы",
    ]


def test_tokenize_drops_punctuation_only_input():
    # \w doesn't match punctuation; the empty list must be handled by callers
    assert tokenize("!!! ??? — …") == []


def test_tokenize_is_case_insensitive():
    assert tokenize("BirKart BIRKART birkart") == ["birkart"] * 3


# --------------------------------------------------------------------------- BM25
def test_bm25_ranks_exact_token_match_first():
    chunks = [
        _chunk("Əmanət qutuları üçün şərtlər", "https://x.az/depozit"),
        _chunk("Nağd pul krediti faiz dərəcələri", "https://x.az/kredit"),
        _chunk("Kartların müqavilə şərtləri", "https://x.az/kartlar"),
    ]
    idx = BM25Index(chunks)
    hits = idx.search("nağd pul krediti", limit=3)
    assert hits[0].url == "https://x.az/kredit"
    # the two non-matching docs score 0 and must not be returned
    assert len(hits) == 1


def test_bm25_respects_filter_and_limit():
    chunks = [
        _chunk("kredit şərtləri bir", "https://x.az/1", section="loans"),
        _chunk("kredit şərtləri iki", "https://x.az/2", section="news"),
        _chunk("kredit şərtləri üç", "https://x.az/3", section="loans"),
    ]
    idx = BM25Index(chunks)
    hits = idx.search(
        "kredit şərtləri", limit=1,
        filter_fn=lambda c: c.section != "news",
    )
    assert len(hits) == 1
    assert hits[0].section == "loans"


def test_bm25_empty_query_returns_empty():
    idx = BM25Index([_chunk("söz", "https://x.az/1")])
    assert idx.search("!!!", limit=3) == []


def test_bm25_empty_corpus_search_is_safe():
    # an in-memory BM25 built from nothing must not blow up on a real query
    assert BM25Index([]).search("kredit", limit=5) == []


def test_bm25_keeps_negative_scoring_docs_when_terms_overlap():
    """The overlap-gate (not raw BM25 sign) decides who competes.

    With rank_bm25, scores can go negative on tiny / skewed corpora; without the
    overlap gate, those docs would be dropped even though they share query terms.
    """
    # single very long "ubiquitous term" doc + one short focused doc
    long_filler = ("kredit " * 500).strip()
    chunks = [
        _chunk(long_filler, "https://x.az/filler"),
        _chunk("kredit şərtləri", "https://x.az/focused"),
    ]
    hits = BM25Index(chunks).search("kredit", limit=5)
    urls = [h.url for h in hits]
    assert "https://x.az/focused" in urls
    assert "https://x.az/filler" in urls  # overlap gate keeps it in the running


def test_bm25_from_store_uses_get_all_chunks():
    class _StubStore:
        def __init__(self, chunks):
            self._chunks = chunks

        def get_all_chunks(self):
            return list(self._chunks)

    chunks = [_chunk("kredit", "https://x.az/a")]
    idx = BM25Index.from_store(_StubStore(chunks))
    assert idx.search("kredit", limit=1)[0].url == "https://x.az/a"


# --------------------------------------------------------------------------- RRF
def test_rrf_promotes_docs_present_in_both_rankings():
    a = _chunk("doc A", "https://x.az/a")
    b = _chunk("doc B", "https://x.az/b")
    c = _chunk("doc C", "https://x.az/c")
    fused = reciprocal_rank_fusion([[a, b], [c, a]])
    keys = [(ch.url) for ch in fused]
    # A appears high in both lists -> top; C appears once but rank 1 there
    assert keys[0] == "https://x.az/a"
    assert set(keys) == {"https://x.az/a", "https://x.az/b", "https://x.az/c"}


def test_rrf_empty_input():
    assert reciprocal_rank_fusion([]) == []
    assert reciprocal_rank_fusion([[], []]) == []


def test_rrf_score_matches_formula():
    """score(d) = sum(1 / (k + rank_i(d))); summed over the rankings it appears in."""
    a = _chunk("a", "https://x.az/a")
    b = _chunk("b", "https://x.az/b")
    # a is rank-0 in list1 (only list) -> 1/(k+0+1)
    only_a = reciprocal_rank_fusion([[a]])
    assert math.isclose(only_a[0].score, 1.0 / (_RRF_K + 1))

    # a rank-0 in list1, rank-1 in list2 -> 1/(k+1) + 1/(k+2)
    two_lists = reciprocal_rank_fusion([[a, b], [b, a]])
    a_score = next(c.score for c in two_lists if c.url == "https://x.az/a")
    b_score = next(c.score for c in two_lists if c.url == "https://x.az/b")
    expected_a = 1.0 / (_RRF_K + 1) + 1.0 / (_RRF_K + 2)
    expected_b = 1.0 / (_RRF_K + 2) + 1.0 / (_RRF_K + 1)  # rank1 + rank0
    assert math.isclose(a_score, expected_a)
    assert math.isclose(b_score, expected_b)
    # and the two are equal — confirms rank-only fusion ignores BM25 magnitudes
    assert math.isclose(a_score, b_score)


def test_rrf_distinguishes_same_url_different_text():
    """URL alone isn't enough — same URL with different text is treated as separate."""
    p1 = _chunk("kredit", "https://x.az/page", score=0.0)
    p2 = _chunk("kredit şərtləri", "https://x.az/page", score=0.0)
    fused = reciprocal_rank_fusion([[p1], [p2]])
    assert len(fused) == 2  # different first-128-char chunks => different keys


# --------------------------------------------------------------------------- cross-encoder
def test_to_probability_passthrough_for_in_unit_scores():
    # some CEs emit raw probabilities; pass them through unchanged
    assert _to_probability(0.0) == 0.0
    assert _to_probability(0.5) == 0.5
    assert _to_probability(1.0) == 1.0


def test_to_probability_sigmoid_for_logits():
    # values inside [0, 1] hit the passthrough branch (probabilities in, out);
    # anything outside is treated as a logit and run through the sigmoid
    assert _to_probability(0.5) == 0.5  # 0.5 in [0,1] -> passthrough, not sigmoid
    assert math.isclose(_to_probability(2.0), 1.0 / (1.0 + math.exp(-2.0)))
    assert math.isclose(_to_probability(-2.0), 1.0 / (1.0 + math.exp(2.0)))
    assert _to_probability(10.0) > 0.99
    assert _to_probability(-10.0) < 0.01


class _FakeCrossEncoder:
    """Mimics sentence_transformers.CrossEncoder just enough for the reranker."""

    def __init__(self, scores, model_name=None, max_length=None):
        self._scores = list(scores)
        self.model_name = model_name
        self.max_length = max_length

    def predict(self, pairs):
        return list(self._scores)


def test_reranker_sorts_descending_and_rescales_logits(monkeypatch):
    # Reranker lazily imports sentence_transformers.CrossEncoder inside the
    # ``model`` property; inject our fake via sys.modules so the import resolves.
    # All three raw scores are >1 or <-1, so they hit the sigmoid branch.
    import sys
    import types

    fake_st = types.ModuleType("sentence_transformers")
    fake_st.CrossEncoder = lambda name, max_length=None: _FakeCrossEncoder(
        [3.0, -3.0, 0.5], model_name=name, max_length=max_length
    )
    monkeypatch.setitem(sys.modules, "sentence_transformers", fake_st)

    rr = CrossEncoderReranker("fake-model", max_length=128)
    chunks = [
        _chunk("first", "https://x.az/a", score=0.0),
        _chunk("second", "https://x.az/b", score=0.0),
        _chunk("third", "https://x.az/c", score=0.0),
    ]
    ranked = rr.rerank("q", chunks)
    # raw 3.0 -> sigmoid ~0.95; raw -3.0 -> sigmoid ~0.05; raw 0.5 is in
    # [0,1] and passes through unchanged as 0.5. Sorted: a (0.95),
    # c (0.5), b (0.05)
    assert [c.url for c in ranked] == [
        "https://x.az/a", "https://x.az/c", "https://x.az/b",
    ]
    assert ranked[0].score > ranked[1].score > ranked[2].score
    assert math.isclose(ranked[0].score, 1.0 / (1.0 + math.exp(-3.0)), rel_tol=1e-6)
    assert math.isclose(ranked[2].score, 1.0 / (1.0 + math.exp(3.0)), rel_tol=1e-6)


def test_reranker_empty_input_is_noop():
    rr = CrossEncoderReranker("fake-model")
    assert rr.rerank("q", []) == []


@pytest.mark.parametrize("model_name", ["BAAI/bge-reranker-v2-m3"])
def test_reranker_model_name_is_stored(model_name):
    rr = CrossEncoderReranker(model_name, max_length=320)
    assert rr.model_name == model_name
    assert rr.max_length == 320
