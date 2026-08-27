"""Conversational query condensing for multi-turn retrieval (plan 4.4).

Phase 3's multi-turn items exposed the gap: the retriever only ever sees the
bare latest message, so a follow-up like "Bəs kartın qiyməti nə qədərdir?"
("But how much does the card cost?") retrieves against a query missing its
referent (which card?) — a miss no single-turn index can fix. The generation
side already works (the LLM sees chat history); this module gives the
*retrieval* side the same context by folding follow-up + history into one
standalone query before embedding/BM25.

Same machinery and contract as ``query_expansion`` (plan 2.2): one cheap
DeepSeek call, results cached per conversation state, and graceful
degradation — any API failure or empty response falls back to the bare
question, so condensing can never make retrieval worse than the status quo
on infrastructure grounds, only on retrieval grounds (which the A/B measures).
"""

from __future__ import annotations

import hashlib
import re

from kb_rag.config import Settings
from kb_rag.rag.llm import DeepSeekClient

_SYSTEM_PROMPT = (
    "You convert a follow-up question into a standalone search query for an "
    "Azerbaijani bank's knowledge base (Kapital Bank / Birbank). Given the "
    "recent conversation and the latest user message, write ONE query that "
    "resolves every pronoun and Ellipsis (\"it\", \"that\", \"bəs\", \"а "
    "какая\"...) using the conversation — e.g. follow-up \"And its monthly "
    "fee?\" after a Birbank Miles question becomes \"Birbank Miles card "
    "monthly fee\". Write the query in the SAME LANGUAGE as the latest user "
    "message — same script, no translation, even if earlier turns or product "
    "names use another language. If the message is already standalone, output "
    "it unchanged. Reply with ONLY the query text — no quotes, no "
    "explanations."
)

_WS_RE = re.compile(r"\s+")
_MAX_TURNS = 6  # messages (not pairs) of history handed to the LLM


def _clean(raw: str | None, fallback: str) -> str:
    """Normalize the completion: one line, strip wrapper quotes; junk → fallback."""
    text = _WS_RE.sub(" ", (raw or "")).strip().strip("\"'“”«»").strip()
    if not text or len(text) > 300:
        return fallback
    return text


class QueryCondenser:
    """Rewrites follow-up + history into one standalone retrieval query."""

    def __init__(self, settings: Settings, client: DeepSeekClient | None = None):
        self.settings = settings
        self._client = client  # injectable for tests; lazy otherwise (needs API key)
        self._cache: dict[str, str] = {}

    @property
    def client(self) -> DeepSeekClient:
        if self._client is None:
            self._client = DeepSeekClient(self.settings)
        return self._client

    @staticmethod
    def _cache_key(history: list[dict], question: str) -> str:
        payload = "\x1e".join(
            [f"{m.get('role', '')}:{m.get('content', '')}" for m in history[-_MAX_TURNS:]]
            + [question]
        )
        return hashlib.sha1(payload.encode("utf-8")).hexdigest()

    def condense(self, question: str, history: list[dict]) -> str:
        """Standalone query for ``question`` given ``history``; ``question``
        unchanged when there is no history or the LLM path fails."""
        if not history:
            return question
        key = self._cache_key(history, question)
        if key in self._cache:
            return self._cache[key]
        transcript = "\n".join(
            f"{'Customer' if m.get('role') == 'user' else 'Assistant'}: "
            f"{str(m.get('content', ''))[:500]}"
            for m in history[-_MAX_TURNS:]
        )
        try:
            raw = self.client.complete(
                [
                    {"role": "system", "content": _SYSTEM_PROMPT},
                    {"role": "user", "content": f"CONVERSATION:\n{transcript}\n\nLATEST MESSAGE:\n{question}"},
                ],
                temperature=0.0,
                max_tokens=100,
            )
            out = _clean(raw, fallback=question)
        except Exception:
            out = question  # API down → retrieve on the bare follow-up, as before
        if len(self._cache) < 256:
            self._cache[key] = out
        return out
