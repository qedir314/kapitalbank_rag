"""Unit tests for LLM query expansion (plan 2.2) — fully offline via a fake client."""
from kb_rag.config import EmbeddingConfig, LLMConfig, RetrievalConfig, Settings
from kb_rag.rag.query_expansion import QueryExpander, _parse_variants


def _settings() -> Settings:
    return Settings.model_validate({
        "scraper": {"sitemap_url": "https://x/sitemap.xml"},
        "embedding": EmbeddingConfig().model_dump(),
        "chunking": {"target_chars": 800, "overlap_chars": 120, "min_chars": 200},
        "vector_store": {"persist_dir": "data/chroma"},
        "retrieval": RetrievalConfig().model_dump(),
        "llm": LLMConfig().model_dump(),
        "app": {"history_turns": 0},
    })


class _FakeClient:
    """Stand-in for DeepSeekClient.complete with a canned response."""

    def __init__(self, response=None, raises=False):
        self.response = response
        self.raises = raises
        self.calls = 0

    def complete(self, messages, **kwargs):
        self.calls += 1
        if self.raises:
            raise RuntimeError("api down")
        return self.response


# --------------------------------------------------------------------------- parsing
def test_parse_variants_plain_json():
    assert _parse_variants('["a", "b", "c"]') == ["a", "b", "c"]


def test_parse_variants_strips_markdown_fences_and_chatter():
    raw = 'Sure:\n```json\n["Sual", "Question", "Вопрос"]\n```'
    assert _parse_variants(raw) == ["Sual", "Question", "Вопрос"]


def test_parse_variants_garbage_is_empty():
    assert _parse_variants("no array here") == []
    assert _parse_variants("[1, 2]") == []  # non-strings dropped
    assert _parse_variants(None) == []


# --------------------------------------------------------------------------- expand
def test_expand_returns_original_first_plus_variants():
    client = _FakeClient('["Kartı necə bloklayım?", "How do I block my card?", "Как заблокировать карту?"]')
    out = QueryExpander(_settings(), client=client).expand("How do I block my card?")
    assert out[0] == "How do I block my card?"  # original leads, always
    assert len(out) == 3  # the English variant equals the original -> deduped
    assert "Kartı necə bloklayım?" in out


def test_expand_dedupes_case_and_whitespace_variants():
    client = _FakeClient('["HOW DO  I BLOCK MY CARD?", "How do I block my card?"]')
    out = QueryExpander(_settings(), client=client).expand("how do I block my card")
    assert out == ["how do I block my card", "HOW DO  I BLOCK MY CARD?"]


def test_expand_api_failure_degrades_to_bare_query():
    out = QueryExpander(_settings(), client=_FakeClient(raises=True)).expand("sual")
    assert out == ["sual"]


def test_expand_unparseable_response_degrades_to_bare_query():
    out = QueryExpander(_settings(), client=_FakeClient("I cannot help")).expand("sual")
    assert out == ["sual"]


def test_expand_caches_per_query():
    client = _FakeClient('["a", "b"]')
    ex = QueryExpander(_settings(), client=client)
    first = ex.expand("q")
    second = ex.expand("q")
    assert first == second
    assert client.calls == 1  # second hit served from cache
