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
from kb_rag.rag.llm import DeepSeekClient
from kb_rag.rag.prompts import build_context_block, build_messages, build_system_prompt
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


@dataclass
class Answer:
    sources: list[Source] = field(default_factory=list)
    text: str | None = None                      # set when stream=False
    text_stream: Iterator[str] | None = None     # set when stream=True
    context: str | None = None                   # numbered passages shown to the LLM (for eval)


class RAGPipeline:
    def __init__(self, settings: Settings):
        self.settings = settings
        self.embedder = E5Embedder(settings.embedding)
        self.store = VectorStore(settings)
        # the expander resolves the LLM client lazily — retrieval without
        # expansion must keep working even with no API key present
        expander = (
            QueryExpander(settings, client=None)
            if settings.retrieval.query_expansion
            else None
        )
        self.retriever = Retriever(settings, self.embedder, self.store, expander=expander)
        self._llm: DeepSeekClient | None = None

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

    def answer(
        self,
        question: str,
        history: list[dict] | None = None,
        lang: str | None = None,
        section: str | list[str] | None = None,
        top_k: int | None = None,
        stream: bool = False,
    ) -> Answer:
        chunks = self.retriever.retrieve(question, top_k=top_k, lang=lang, section=section)
        sources = [
            Source(
                index=i,
                title=c.title or c.section_path,
                url=c.url,
                source_url=c.source_url,
                section_path=c.section_path,
                score=round(c.score, 4),
            )
            for i, c in enumerate(chunks, start=1)
        ]

        if not chunks:
            msg = (
                "The knowledge base is empty — run `python -m scripts.build_index` first."
                if self.indexed_chunks == 0
                else "No relevant passages were found for this question and filter combination."
            )
            return Answer(sources=[], text=msg)

        context_block = build_context_block(chunks)
        system_prompt = build_system_prompt(context_block)
        messages = build_messages(system_prompt, self._trim_history(history or []), question)

        if stream:
            return Answer(
                sources=sources,
                text_stream=self.llm.complete(messages, stream=True),
                context=context_block,
            )
        return Answer(
            sources=sources,
            text=self.llm.complete(messages, stream=False),
            context=context_block,
        )


@lru_cache(maxsize=1)
def get_pipeline() -> RAGPipeline:
    """Process-wide pipeline singleton (loads the embedding model once)."""
    return RAGPipeline(get_settings())
