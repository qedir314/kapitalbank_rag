"""DeepSeek chat client via its OpenAI-compatible API.

DeepSeek exposes ``https://api.deepseek.com`` with model names ``deepseek-chat``
(V3 family) and ``deepseek-reasoner`` (R1 family). We only use the standard
OpenAI SDK pointed at that base URL — no vendor lock-in in the pipeline code.
"""

from __future__ import annotations

from typing import Iterator

import openai

from kb_rag.config import Settings, get_deepseek_api_key


class DeepSeekClient:
    def __init__(self, settings: Settings, api_key: str | None = None):
        self.settings = settings
        self._client = openai.OpenAI(
            base_url=settings.llm.base_url,
            api_key=api_key or get_deepseek_api_key(),
        )

    def complete(
        self,
        messages: list[dict],
        stream: bool = False,
        model: str | None = None,
        temperature: float | None = None,
        max_tokens: int | None = None,
    ) -> str | Iterator[str]:
        """Generate a completion. Returns a string, or a token iterator when streaming."""
        cfg = self.settings.llm
        try:
            response = self._client.chat.completions.create(
                model=model or cfg.model,
                messages=messages,
                temperature=cfg.temperature if temperature is None else temperature,
                max_tokens=max_tokens or cfg.max_tokens,
                stream=stream,
            )
        except openai.AuthenticationError as exc:
            raise RuntimeError(
                "DeepSeek rejected the API key — check DEEPSEEK_API_KEY in your .env"
            ) from exc

        if not stream:
            return response.choices[0].message.content or ""

        def token_iter() -> Iterator[str]:
            for chunk in response:
                delta = chunk.choices[0].delta if chunk.choices else None
                if delta is not None and delta.content:
                    yield delta.content

        return token_iter()
