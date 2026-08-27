"""Offline wiring tests for Phase 4 pipeline behavior (plans 4.2–4.4).

RAGPipeline is assembled by hand (no Chroma / no embedding model): a spy
retriever records the query it was given, a fake LLM returns canned text.
This pins the *contracts* — what retrieval runs on, what generation sees,
when the citation report appears, and how freshness reaches the refusal.
"""
from kb_rag.config import (
    ChunkingConfig,
    ChatConfig,
    EmbeddingConfig,
    LLMConfig,
    RetrievalConfig,
    ScraperConfig,
    Settings,
    VectorStoreConfig,
)
from kb_rag.ingest.store import RetrievedChunk
from kb_rag.rag.pipeline import RAGPipeline

HISTORY = [
    {"role": "user", "content": "Birbank Miles debet kartı nədir?"},
    {"role": "assistant", "content": "Mil qazandıran debet kartıdır."},
]


def _settings(*, condensing: bool, verify: bool = True) -> Settings:
    return Settings.model_validate({
        "scraper": ScraperConfig(sitemap_url="https://x/s.xml").model_dump(),
        "embedding": EmbeddingConfig().model_dump(),
        "chunking": ChunkingConfig().model_dump(),
        "vector_store": VectorStoreConfig().model_dump(),
        "retrieval": RetrievalConfig(query_condensing=condensing).model_dump(),
        "llm": LLMConfig().model_dump(),
        "app": ChatConfig(history_turns=6, verify_citations=verify).model_dump(),
    })


def _chunk(n: int = 1, crawled_at: str = "2026-08-26T09:22:33+00:00") -> RetrievedChunk:
    return RetrievedChunk(
        text=f"Birbank debet kartının qiyməti pulsuzdur, sənəd {n}.",
        url=f"https://birbank.az/debet-kartlar-{n}", source_url="",
        title="Debet kartları", lang="az", section="cards",
        section_path="Cards > Debet", score=0.8, crawled_at=crawled_at,
    )


class _SpyRetriever:
    def __init__(self, chunks):
        self.chunks = chunks
        self.queries = []

    def retrieve(self, query, top_k=None, lang=None, section=None):
        self.queries.append(query)
        return self.chunks


class _FakeLLM:
    def __init__(self, reply="Kart pulsuzdur [1]."):
        self.reply = reply
        self.messages = None

    def complete(self, messages, stream=False, **kwargs):
        self.messages = messages
        if stream:
            return iter(list(self.reply))
        return self.reply


class _SpyCondenser:
    def __init__(self, standalone="Birbank Miles debet kartının qiyməti"):
        self.standalone = standalone
        self.calls = []

    def condense(self, question, history):
        self.calls.append((question, tuple(history)))
        return self.standalone


class _FakeStore:
    def __init__(self, count=5):
        self._count = count

    def count(self):
        return self._count

    def latest_crawled_at(self):
        return "2026-08-26T09:22:33+00:00"


def _pipeline(*, condensing: bool, chunks=None, verify: bool = True) -> RAGPipeline:
    p = object.__new__(RAGPipeline)          # bypass __init__: no Chroma/model
    p.settings = _settings(condensing=condensing, verify=verify)
    p.store = _FakeStore(count=len(chunks) if chunks else 5)
    p.retriever = _SpyRetriever(chunks if chunks is not None else [_chunk()])
    p._llm = _FakeLLM()
    p._as_of = None
    p.condenser = _SpyCondenser() if condensing else None
    return p


# ------------------------------------------------------------------ 4.4 condensing
def test_history_triggers_condensing_for_retrieval_only():
    p = _pipeline(condensing=True)
    ans = p.answer("Bəs kartın qiyməti nə qədərdir?", history=HISTORY)
    assert p.retriever.queries == ["Birbank Miles debet kartının qiyməti"]
    assert p.condenser.calls and p.condenser.calls[0][0] == "Bəs kartın qiyməti nə qədərdir?"
    assert ans.retrieval_query == "Birbank Miles debet kartının qiyməti"
    # generation must still see the USER's wording as the final message
    assert p._llm.messages[-1]["content"] == "Bəs kartın qiyməti nə qədərdir?"
    # ...and the conversation history as preceding turns
    assert [m["role"] for m in p._llm.messages] == ["system", "user", "assistant", "user"]


def test_no_history_never_calls_condenser():
    p = _pipeline(condensing=True)
    ans = p.answer("Birbank Miles debet kartı nədir?")
    assert p.condenser.calls == []
    assert p.retriever.queries == ["Birbank Miles debet kartı nədir?"]
    assert ans.retrieval_query is None


def test_condensing_off_retrieves_bare_followup():
    p = _pipeline(condensing=False)
    ans = p.answer("Bəs kartın qiyməti nə qədərdir?", history=HISTORY)
    assert p.retriever.queries == ["Bəs kartın qiyməti nə qədərdir?"]
    assert ans.retrieval_query is None


# ------------------------------------------------------------------ 4.2 citations
def test_citation_report_attached_to_final_answer():
    p = _pipeline(condensing=False)
    ans = p.answer("kart qiyməti?")
    assert ans.citations is not None
    assert ans.citations.ok  # generated text cites [1], passage 1 supports it


def test_citation_report_populated_when_stream_is_consumed():
    p = _pipeline(condensing=False)
    ans = p.answer("kart qiyməti?", stream=True)
    assert ans.citations is None              # not yet — tokens still streaming
    text = "".join(ans.text_stream)
    assert text == "Kart pulsuzdur [1]."
    assert ans.citations is not None and ans.citations.ok  # filled at completion


def test_flagged_citation_survives_streaming():
    p = _pipeline(condensing=False)
    p._llm = _FakeLLM(reply="Filial sayı 500-dür [9].")   # passage 9 doesn't exist
    ans = p.answer("neçə filial var?", stream=True)
    "".join(ans.text_stream)
    assert 9 in ans.citations.invalid


def test_verification_can_be_disabled():
    p = _pipeline(condensing=False, verify=False)
    ans = p.answer("kart qiyməti?")
    assert ans.citations is None


# ------------------------------------------------------------------ 4.3 freshness
def test_sources_carry_crawled_at():
    p = _pipeline(condensing=False, chunks=[_chunk(1, crawled_at="2026-08-26T09:22:33+00:00")])
    ans = p.answer("kart qiyməti?")
    assert ans.sources[0].crawled_at == "2026-08-26T09:22:33+00:00"


def test_empty_retrieval_refusal_mentions_content_date():
    p = _pipeline(condensing=False, chunks=[])
    ans = p.answer("bilinməyən şey?")
    assert "as of 2026-08-26" in ans.text
