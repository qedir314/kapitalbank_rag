"""Offline tests for conversational query condensing (plan 4.4).

A fake client stands in for DeepSeek so the suite stays network-free; the
contract under test is graceful degradation — condensing must fall back to
the bare question whenever it cannot improve it.
"""
from kb_rag.config import (
    ChunkingConfig,
    EmbeddingConfig,
    LLMConfig,
    RetrievalConfig,
    ScraperConfig,
    Settings,
    VectorStoreConfig,
    ChatConfig,
)
from kb_rag.rag.query_condensing import QueryCondenser

HISTORY = [
    {"role": "user", "content": "Birbank Miles debet kartı nədir?"},
    {"role": "assistant", "content": "Miles qazandıran debet kartıdır."},
]


class _FakeLLM:
    def __init__(self, reply=None, raises=False):
        self.reply = reply
        self.raises = raises
        self.calls = 0
        self.messages = []  # last request, for prompt-content assertions

    def complete(self, messages, **kwargs):
        self.calls += 1
        self.messages = messages
        if self.raises:
            raise RuntimeError("api down")
        return self.reply


def _settings():
    return Settings.model_validate({
        "scraper": ScraperConfig(sitemap_url="https://x/s.xml").model_dump(),
        "embedding": EmbeddingConfig().model_dump(),
        "chunking": ChunkingConfig().model_dump(),
        "vector_store": VectorStoreConfig().model_dump(),
        "retrieval": RetrievalConfig().model_dump(),
        "llm": LLMConfig().model_dump(),
        "app": ChatConfig().model_dump(),
    })


def _condenser(reply=None, raises=False):
    fake = _FakeLLM(reply, raises)
    return QueryCondenser(_settings(), client=fake), fake


def test_no_history_skips_llm_entirely():
    c, fake = _condenser(reply="unused")
    assert c.condense("standalone question", []) == "standalone question"
    assert fake.calls == 0  # single-turn queries must pay zero cost


def test_condense_returns_standalone_query():
    c, fake = _condenser(reply='"Birbank Miles debet kartının qiyməti"')
    out = c.condense("Bəs kartın qiyməti nə qədərdir?", HISTORY)
    assert out == "Birbank Miles debet kartının qiyməti"  # wrapper quotes stripped
    assert fake.calls == 1


def test_api_failure_degrades_to_bare_question():
    c, _ = _condenser(raises=True)
    assert c.condense("bəs faiz?", HISTORY) == "bəs faiz?"


def test_empty_or_rubbish_reply_degrades():
    c, _ = _condenser(reply="   ")
    assert c.condense("bəs faiz?", HISTORY) == "bəs faiz?"
    c2, _ = _condenser(reply="x" * 400)  # runaway completion — reject
    assert c2.condense("bəs faiz?", HISTORY) == "bəs faiz?"


def test_results_cached_per_conversation_state():
    c, fake = _condenser(reply="Birbank Miles kart qiymeti")
    q = "Bəs kartın qiyməti?"
    assert c.condense(q, HISTORY) == c.condense(q, HISTORY)
    assert fake.calls == 1
    # different history = different referent = cache miss
    assert c.condense(q, [{"role": "user", "content": "Kredit kartı nədir?"}])
    assert fake.calls == 2


def test_transcript_and_question_reach_the_prompt():
    c, fake = _condenser(reply="x")
    c.condense("Bəs kartın qiyməti?", HISTORY)
    user_msg = fake.messages[1]["content"]
    assert "Birbank Miles debet kartı nədir?" in user_msg  # history included
    assert "Bəs kartın qiyməti?" in user_msg               # follow-up included
    assert fake.messages[0]["role"] == "system"
