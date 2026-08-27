"""LLM query expansion for cross-language retrieval (improvement plan 2.2).

The corpus carries the same facts under az/en/ru page trees, but a query only
embeds and tokenizes against its own surface form — an English question about
a product documented mostly in Azerbaijani leans entirely on the multilingual
embedding, and BM25 contributes nothing across the language boundary. One
cheap DeepSeek call rewrites the query into all three languages; each variant
retrieves independently and the candidate lists are fused by RRF (see
``Retriever.retrieve``), so a passage reachable from *any* language gets its
chance at the reranker.

Graceful degradation by design: any API failure or unparseable response falls
back to the bare query, and expansion results are cached per question (the
eval set repeats queries across runs; a chat session repeats follow-ups).
"""

from __future__ import annotations

import json
import re

from kb_rag.config import Settings
from kb_rag.rag.llm import DeepSeekClient

LANGS = ("az", "en", "ru")
_LANG_NAMES = {"az": "Azerbaijani", "en": "English", "ru": "Russian"}

_SYSTEM_PROMPT = (
    "You rewrite banking questions for multilingual retrieval over an "
    "Azerbaijani bank's knowledge base. Reply with ONLY a JSON array of "
    "exactly {n} strings — the question translated into {langs}, in that "
    "order. Keep bank and product names (Kapital Bank, Birbank, BirKart) "
    "as-is. No explanations, no markdown fences."
)

_ARRAY_RE = re.compile(r"\[.*\]", re.DOTALL)
_WS_RE = re.compile(r"\s+")


def _parse_variants(raw: str) -> list[str]:
    """Extract the JSON string array from a completion, ignoring any chatter."""
    match = _ARRAY_RE.search(raw or "")
    if not match:
        return []
    try:
        data = json.loads(match.group(0))
    except json.JSONDecodeError:
        return []
    if not isinstance(data, list):
        return []
    return [item.strip() for item in data if isinstance(item, str) and item.strip()]


def _normalize(text: str) -> str:
    return _WS_RE.sub(" ", text).strip().casefold()


class QueryExpander:
    """Rewrites a query into its az/en/ru variants via one LLM call."""

    def __init__(self, settings: Settings, client: DeepSeekClient | None = None):
        self.settings = settings
        self._client = client  # injectable for tests; lazy otherwise (needs API key)
        self._cache: dict[str, list[str]] = {}

    @property
    def client(self) -> DeepSeekClient:
        if self._client is None:
            self._client = DeepSeekClient(self.settings)
        return self._client

    def expand(self, query: str) -> list[str]:
        """The original query first, then each distinct variant; never empty."""
        if query in self._cache:
            return self._cache[query]
        langs = ", ".join(_LANG_NAMES[code] for code in LANGS)
        system = _SYSTEM_PROMPT.format(n=len(LANGS), langs=langs)
        try:
            raw = self.client.complete(
                [
                    {"role": "system", "content": system},
                    {"role": "user", "content": query},
                ],
                temperature=0.0,
                max_tokens=200,
            )
            variants = _parse_variants(raw)
        except Exception:
            variants = []  # network/API failure degrades to single-query retrieval
        out = [query]
        seen = {_normalize(query)}
        for variant in variants:
            key = _normalize(variant)
            if key not in seen:
                seen.add(key)
                out.append(variant)
        if len(self._cache) < 256:
            self._cache[query] = out
        return out
