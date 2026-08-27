"""End-to-end RAG orchestration: retrieve -> ground -> generate.

``RAGPipeline.answer`` returns an ``Answer`` carrying the retrieved sources
plus either the full answer text (eval/batch mode) or a token iterator
(Streamlit streaming).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from functools import lru_cache
from typing import Iterator

from kb_rag.config import Settings, get_settings
from kb_rag.ingest.embeddings import E5Embedder
from kb_rag.ingest.store import VectorStore
from kb_rag.rag.citations import CitationReport, verify_citations
from kb_rag.rag.llm import DeepSeekClient
from kb_rag.rag.prompts import build_context_block, build_messages, build_system_prompt
from kb_rag.rag.query_condensing import QueryCondenser
from kb_rag.rag.query_expansion import QueryExpander
from kb_rag.rag.retriever import Retriever


@dataclass(frozen=True)
class Source:
    index: int  # 1-based, matches [n] citations in the answer text
    title: str
    url: str           # final URL (birbank.az) shown to users
    source_url: str    # original kapitalbank.az slug
    section_path: str
    score: float
    crawled_at: str = ""  # when the page was scraped — freshness display (plan 4.3)


@dataclass
class Answer:
    sources: list[Source] = field(default_factory=list)
    text: str | None = None                      # set when stream=False
    text_stream: Iterator[str] | None = None     # set when stream=True
    context: str | None = None                   # numbered passages shown to the LLM (for eval)
    retrieval_query: str | None = None           # standalone query used when history condensed (4.4)
    citations: CitationReport | None = None      # runtime citation verification (4.2);
    # for streamed answers this is None until the stream is fully consumed —
    # the wrapper fills it in when the generator completes


class RAGPipeline:
    def __init__(self, settings: Settings):
        self.settings = settings
        self.embedder = E5Embedder(settings.embedding)
        self.store = VectorStore(settings)
        # the expander/condenser resolve the LLM client lazily — retrieval
        # without them must keep working even with no API key present
        expander = (
            QueryExpander(settings, client=None)
            if settings.retrieval.query_expansion
            else None
        )
        self.condenser = (
            QueryCondenser(settings, client=None)
            if settings.retrieval.query_condensing
            else None
        )
        self.retriever = Retriever(settings, self.embedder, self.store, expander=expander)
        self._llm: DeepSeekClient | None = None
        self._as_of: str | None = None  # cached latest crawled_at (plan 4.3)

    @property
    def llm(self) -> DeepSeekClient:
        if self._llm is None:  # lazy: only requires API key when generating
            self._llm = DeepSeekClient(self.settings)
        return self._llm

    @property
    def indexed_chunks(self) -> int:
        return self.store.count()

    def _trim_history(self, history: list[dict]) -> list[dict]:
        turns = max(self.settings.app.history_turns, 0)
        return history[-turns * 2:] if turns else []

    def content_as_of(self) -> str:
        """Latest ``crawled_at`` in the index (cached) — freshness line for
        the sidebar and the no-passages refusal (plan 4.3)."""
        if self._as_of is None:
            self._as_of = self.store.latest_crawled_at()
        return self._as_of

    def answer(
        self,
        question: str,
        history: list[dict] | None = None,
        lang: str | None = None,
        section: str | list[str] | None = None,
        top_k: int | None = None,
        stream: bool = False,
    ) -> Answer:
        trimmed = self._trim_history(history or [])
        # plan 4.4: with condensing on and a live conversation, retrieval runs
        # on a standalone rewrite of the follow-up; generation still sees the
        # user's own wording + history, so the answer stays in the chat's flow
        retrieval_query = question
        if trimmed and self.condenser is not None:
            retrieval_query = self.condenser.condense(question, trimmed)

        chunks = self.retriever.retrieve(
            retrieval_query, top_k=top_k, lang=lang, section=section)
        sources = [
            Source(
                index=i,
                title=c.title or c.section_path,
                url=c.url,
                source_url=c.source_url,
                section_path=c.section_path,
                score=round(c.score, 4),
                crawled_at=c.crawled_at,
            )
            for i, c in enumerate(chunks, start=1)
        ]
        used_condensing = retrieval_query != question

        if not chunks:
            msg = (
                "The knowledge base is empty — run `python -m scripts.build_index` first."
                if self.indexed_chunks == 0
                else "No relevant passages were found for this question and filter combination."
            )
            if self.indexed_chunks and (as_of := self.content_as_of()):
                msg += f" The indexed content is as of {as_of[:10]}."
            return Answer(sources=[], text=msg,
                          retrieval_query=retrieval_query if used_condensing else None)

        context_block = build_context_block(chunks)
        system_prompt = build_system_prompt(context_block)
        messages = build_messages(system_prompt, trimmed, question)

        out = Answer(
            sources=sources,
            context=context_block,
            retrieval_query=retrieval_query if used_condensing else None,
        )
        verify = self.settings.app.verify_citations
        if stream:
            raw_stream = self.llm.complete(messages, stream=True)
            out.text_stream = self._verified_stream(out, raw_stream, context_block, verify)
            return out
        out.text = self.llm.complete(messages, stream=False)
        if verify:
            out.citations = verify_citations(out.text or "", context_block)
        return out

    @staticmethod
    def _verified_stream(
        out: "Answer",
        raw_stream: Iterator[str],
        context_block: str,
        verify: bool,
    ) -> Iterator[str]:
        """Pass tokens through untouched; when the stream completes, run the
        citation checker over the collected text and attach the report to the
        Answer the caller still holds (plan 4.2)."""
        collected: list[str] = []
        for piece in raw_stream:
            collected.append(piece)
            yield piece
        if verify:
            out.citations = verify_citations("".join(collected), context_block)


@lru_cache(maxsize=1)
def get_pipeline() -> RAGPipeline:
    """Process-wide pipeline singleton (loads the embedding model once)."""
    return RAGPipeline(get_settings())
