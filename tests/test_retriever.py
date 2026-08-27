"""Unit tests for retrieval-side helpers: URL dedupe, filters, and end-to-end Retriever.

The Retriever class glues dense ANN, BM25, RRF, rerank, and dedupe together.
We test those glue points with fakes for the embedder / store / cross-encoder so
the suite stays fast and offline.
"""
import pytest

from kb_rag.config import (
    EmbeddingConfig,
    LLMConfig,
    RetrievalConfig,
    Settings,
    VectorStoreConfig,
)
from kb_rag.ingest.store import RetrievedChunk
from kb_rag.rag.retriever import Retriever, dedupe_by_url
from kb_rag.scraper.crawl import is_excluded_url


# --------------------------------------------------------------------------- helpers
def _chunk(url: str, score: float, text: str = "body", section: str = "cards",
           lang: str = "az") -> RetrievedChunk:
    return RetrievedChunk(
        text=text,
        url=url,
        source_url=url,
        title="t",
        lang=lang,
        section=section,
        section_path=section,
        score=score,
    )


def _make_settings(*, enable_bm25: bool = True, rerank: bool = False,
                   query_expansion: bool = False) -> Settings:
    return Settings.model_validate({
        "scraper": {"sitemap_url": "https://x/sitemap.xml"},
        "embedding": EmbeddingConfig().model_dump(),
        "chunking": {"target_chars": 800, "overlap_chars": 120, "min_chars": 200},
        "vector_store": VectorStoreConfig().model_dump(),
        "retrieval": RetrievalConfig(
            enable_bm25=enable_bm25,
            rerank_model="fake-reranker" if rerank else None,
            rerank_candidates=5,
            query_expansion=query_expansion,
        ).model_dump(),
        "llm": LLMConfig().model_dump(),
        "app": {"history_turns": 0},
    })


# --------------------------------------------------------------------------- dedupe_by_url
def test_dedupe_keeps_best_chunk_per_url():
    ranked = [
        _chunk("https://x.az/a", 0.9),
        _chunk("https://x.az/b", 0.8),
        _chunk("https://x.az/a", 0.7),   # duplicate page — must be dropped
        _chunk("https://x.az/c", 0.6),
        _chunk("https://x.az/b", 0.5),   # duplicate page — must be dropped
    ]
    kept = dedupe_by_url(ranked)
    assert [c.url for c in kept] == ["https://x.az/a", "https://x.az/b", "https://x.az/c"]
    assert kept[0].score == 0.9  # first occurrence wins (results are score-ordered)


def test_dedupe_max_per_url_two():
    ranked = [
        _chunk("https://x.az/a", 0.9),
        _chunk("https://x.az/a", 0.8),
        _chunk("https://x.az/a", 0.7),  # third chunk from same page — dropped at cap 2
    ]
    kept = dedupe_by_url(ranked, max_per_url=2)
    assert len(kept) == 2


def test_dedupe_empty_input():
    assert dedupe_by_url([]) == []


# --------------------------------------------------------------------------- is_excluded_url
def test_is_excluded_url_case_insensitive():
    assert is_excluded_url("https://CCL.Birbank.az/ru/order/GTKR", ["ccl.birbank.az"])
    assert not is_excluded_url("https://birbank.az/cards", ["ccl.birbank.az"])


@pytest.mark.parametrize("word", ["Nağd", "полезно", "məlumat"])
def test_ftfy_reverses_latin1_mojibake(word):
    # what resp.text produced when a UTF-8 page was decoded as Latin-1
    import ftfy
    broken = word.encode("utf-8").decode("latin-1")
    assert broken != word
    assert ftfy.fix_text(broken) == word


# --------------------------------------------------------------------------- filter builders
class TestBuildWhere:
    def test_no_filters_returns_none(self):
        assert Retriever._build_where(None, None, []) is None

    def test_lang_only(self):
        assert Retriever._build_where("en", None, []) == {"lang": {"$eq": "en"}}

    def test_section_only_string(self):
        # single section is still wrapped in $in for code-path uniformity
        assert Retriever._build_where(None, "cards", []) == {"section": {"$in": ["cards"]}}

    def test_section_only_list(self):
        where = Retriever._build_where(None, ["cards", "loans"], [])
        assert where == {"section": {"$in": ["cards", "loans"]}}

    def test_exclude_sections_only_applied_without_explicit_section(self):
        # news pollution guard — applied when no caller override
        where = Retriever._build_where(None, None, ["news"])
        assert where == {"section": {"$nin": ["news"]}}

    def test_explicit_section_overrides_excludes(self):
        # user explicitly asked for "news" — exclude list must NOT be ANDed in
        where = Retriever._build_where(None, "news", ["news"])
        assert where == {"section": {"$in": ["news"]}}

    def test_lang_and_section_combined_with_and(self):
        where = Retriever._build_where("en", "cards", [])
        assert where == {"$and": [{"lang": {"$eq": "en"}},
                                  {"section": {"$in": ["cards"]}}]}

    def test_lang_and_exclude_when_no_explicit_section(self):
        where = Retriever._build_where("ru", None, ["news", "kampaniyalar"])
        assert where == {"$and": [{"lang": {"$eq": "ru"}},
                                  {"section": {"$nin": ["news", "kampaniyalar"]}}]}


class TestPassesFilter:
    def test_no_filters_passes(self):
        c = _chunk("https://x/a", 0.0, section="news")
        assert Retriever._passes_filter(c, None, None, []) is True

    def test_lang_mismatch(self):
        c = _chunk("https://x/a", 0.0, lang="az")
        assert Retriever._passes_filter(c, "en", None, []) is False

    def test_section_excluded_when_no_explicit_section(self):
        c = _chunk("https://x/a", 0.0, section="news")
        assert Retriever._passes_filter(c, None, None, ["news"]) is False

    def test_explicit_section_bypasses_excludes(self):
        c = _chunk("https://x/a", 0.0, section="news")
        assert Retriever._passes_filter(c, None, "news", ["news"]) is True

    def test_section_list_membership(self):
        c = _chunk("https://x/a", 0.0, section="loans")
        assert Retriever._passes_filter(c, None, ["cards", "loans"], []) is True
        assert Retriever._passes_filter(c, None, ["cards"], []) is False

    def test_consistency_with_build_where(self):
        """Where-clause truthiness for every (lang, section, exclude) combo."""
        chunk = _chunk("https://x/a", 0.0, section="loans", lang="en")
        for lang in (None, "en", "ru"):
            for section in (None, "loans", ["cards", "loans"], "news"):
                for excludes in ([], ["news"], ["loans"]):
                    python_keeps = Retriever._passes_filter(chunk, lang, section, excludes)
                    where = Retriever._build_where(lang, section, excludes)
                    # without a real Chroma call we approximate: any of the
                    # where clauses must "match" if the python predicate says so
                    if where is None:
                        assert python_keeps is True
                    elif isinstance(where, dict) and "$and" not in where and "$or" not in where:
                        # single condition — does it match chunk's metadata?
                        key = next(iter(where))
                        op, val = next(iter(where[key].items()))
                        meta_val = chunk.section if key == "section" else chunk.lang
                        if op == "$eq":
                            assert python_keeps is (meta_val == val)
                        elif op == "$in":
                            assert python_keeps is (meta_val in val)
                        elif op == "$nin":
                            assert python_keeps is (meta_val not in val)


# --------------------------------------------------------------------------- end-to-end Retriever
class _FakeEmbedder:
    def __init__(self, mapping):
        # query text -> numpy array; here we just need a 1-d shape
        import numpy as np
        self._mapping = {q: np.zeros(4, dtype="float32") for q in mapping}

    def embed_query(self, query):
        import numpy as np
        return self._mapping.get(query if isinstance(query, str) else query[0],
                                 np.zeros(4, dtype="float32"))


class _FakeStore:
    def __init__(self, dense_results, all_chunks):
        self._dense = dense_results
        self._all = all_chunks

    def query(self, embedding, top_k, where=None):
        return list(self._dense)

    def get_all_chunks(self):
        return list(self._all)


def test_retriever_dense_only_runs_when_bm25_disabled():
    chunks = [
        _chunk("https://x/az-kart", 0.9, section="cards", lang="az"),
        _chunk("https://x/en-card", 0.8, section="cards", lang="en"),
    ]
    store = _FakeStore(dense_results=chunks, all_chunks=chunks)
    embedder = _FakeEmbedder({"anything"})

    settings = _make_settings(enable_bm25=False)
    r = Retriever(settings, embedder, store)

    out = r.retrieve("kredit", top_k=2, lang="az")
    # dense returned both chunks; the lang filter is a Chroma-side where clause
    # that the fake store ignores, so the observable contract here is "BM25
    # was NOT built" (a dense-only call must not touch the BM25 property)
    assert [c.url for c in out] == ["https://x/az-kart", "https://x/en-card"]
    assert r._bm25 is None


def test_retriever_hybrid_fuses_then_dedupes_then_truncates(monkeypatch):
    """End-to-end with a small corpus: dense + BM25 -> RRF -> top-k.

    Two URLs each contribute one chunk in dense and one in BM25, so RRF
    promotes them and dedupe keeps exactly one chunk per page.
    """
    # dense thinks: a is best; b second
    dense = [
        _chunk("https://x/az", 0.9, section="cards", lang="az"),
        _chunk("https://x/b", 0.7, section="cards", lang="az"),
    ]
    # BM25 thinks: b is best; a second
    bm25_chunks = [
        _chunk("https://x/b", 0.0, section="cards", lang="az"),
        _chunk("https://x/az", 0.0, section="cards", lang="az"),
    ]
    # and a duplicate section of the same "az" page — must be deduped out
    dup_chunks = list(dense) + [
        _chunk("https://x/az", 0.6, section="cards", lang="az"),
    ]
    store = _FakeStore(dense_results=dense, all_chunks=dup_chunks)
    embedder = _FakeEmbedder({"q"})

    settings = _make_settings(enable_bm25=True)
    r = Retriever(settings, embedder, store)
    # Avoid actually building a BM25 index for an empty stream — prebuild it.
    r._bm25 = _FakeBM25(bm25_chunks)

    out = r.retrieve("kredit", top_k=2, lang="az")
    urls = [c.url for c in out]
    assert len(out) == 2
    assert len(set(urls)) == 2           # dedupe_by_url did its job
    assert "https://x/az" in urls
    assert "https://x/b" in urls


def test_retriever_uses_python_predicate_for_bm25_when_lang_set():
    """``_passes_filter`` must drop BM25 results that don't match the lang filter."""
    bm25_chunks = [
        _chunk("https://x/en", 0.0, section="cards", lang="en"),
        _chunk("https://x/az", 0.0, section="cards", lang="az"),
    ]
    store = _FakeStore(dense_results=[bm25_chunks[1]], all_chunks=bm25_chunks)
    embedder = _FakeEmbedder({"q"})

    settings = _make_settings(enable_bm25=True)
    r = Retriever(settings, embedder, store)
    r._bm25 = _FakeBM25(bm25_chunks)

    out = r.retrieve("kredit", top_k=3, lang="az")
    # dense alone gave the az chunk; BM25 must NOT sneak the en chunk through
    assert all(c.lang == "az" for c in out)


# --------------------------------------------------------------------------- query expansion (plan 2.2)
class _RecordingStore:
    """Dense results keyed by query text; records every query it was asked.

    The embedder and store are paired (``embed_query`` runs right before
    ``query`` in ``_hybrid_candidates``), so ``last_query`` carries the text.
    """

    def __init__(self, by_query):
        self._by_query = by_query
        self.queries = []
        self.last_query = None

    def query(self, embedding, top_k, where=None):
        q = self.last_query
        self.queries.append(q)
        return list(self._by_query.get(q, []))

    def get_all_chunks(self):
        return []


class _SpyEmbedder:
    """Zero vector for the fake store; tells the paired store which query it is."""

    def __init__(self, store=None):
        self.store = store

    def embed_query(self, query):
        import numpy as np
        if self.store is not None:
            self.store.last_query = query if isinstance(query, str) else query[0]
        return np.zeros(4, dtype="float32")


class _FakeExpander:
    def __init__(self, variants):
        self.variants = variants
        self.calls = 0

    def expand(self, query):
        self.calls += 1
        return self.variants


def test_expansion_disabled_never_calls_expander():
    chunks = [_chunk("https://x/a", 0.9)]
    store = _FakeStore(dense_results=chunks, all_chunks=chunks)
    expander = _FakeExpander(["q", "variant"])
    settings = _make_settings(enable_bm25=False, query_expansion=False)
    r = Retriever(settings, _FakeEmbedder({"q"}), store, expander=expander)
    r.retrieve("q", top_k=1)
    assert expander.calls == 0


def test_expansion_retrieves_per_variant_and_fuses():
    """Each expanded query hits the store; RRF promotes chunks seen in both."""
    a = _chunk("https://x/a", 0.9)
    b = _chunk("https://x/b", 0.8)
    c = _chunk("https://x/c", 0.7)
    store = _RecordingStore({"necə bloklayım": [a, b], "how to block": [b, c]})
    expander = _FakeExpander(["necə bloklayım", "how to block"])

    settings = _make_settings(enable_bm25=False, query_expansion=True)
    r = Retriever(settings, _SpyEmbedder(store), store, expander=expander)
    out = r.retrieve("how to block", top_k=3)

    assert expander.calls == 1
    assert set(store.queries) == {"necə bloklayım", "how to block"}
    # b is rank-1 in list1 + rank-0 in list2 -> fused first; a and c follow
    assert [ch.url for ch in out] == [
        "https://x/b", "https://x/a", "https://x/c",
    ]


def test_expansion_reranker_sees_the_original_query(monkeypatch):
    """Variants widen the pool, but scoring must answer what the user asked."""
    seen = {}

    class _SpyReranker:
        def rerank(self, query, chunks):
            seen["query"] = query
            return list(reversed(chunks))

    a = _chunk("https://x/a", 0.9)
    b = _chunk("https://x/b", 0.8)
    store = _RecordingStore({"orijinal": [a], "en variant": [b]})
    settings = _make_settings(enable_bm25=False, query_expansion=True, rerank=True)
    r = Retriever(settings, _SpyEmbedder(store), store,
                  expander=_FakeExpander(["orijinal", "en variant"]))
    r._reranker = _SpyReranker()
    r.retrieve("orijinal", top_k=2)
    assert seen["query"] == "orijinal"


class _FakeBM25:
    """Records its constructor chunks and exposes a search that just returns them."""

    def __init__(self, chunks):
        self.chunks = chunks
        self._search = BM25IndexStub(chunks)

    def search(self, query, limit, filter_fn=None):
        return self._search.search(query, limit, filter_fn=filter_fn)


class BM25IndexStub:
    """Drops in for BM25Index — no rank_bm25 dependency."""

    def __init__(self, chunks):
        self.chunks = list(chunks)

    def search(self, query, limit, filter_fn=None):
        out = []
        for c in self.chunks:
            if filter_fn and not filter_fn(c):
                continue
            out.append(c)
            if len(out) >= limit:
                break
        return out
