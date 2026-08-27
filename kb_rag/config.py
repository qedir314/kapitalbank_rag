"""Typed application settings loaded from ``config.yaml``.

Secrets (DEEPSEEK_API_KEY) never live in Settings — they are read from the
environment on demand to avoid leaking keys through logs/reprs.
"""

from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path

import yaml
from dotenv import load_dotenv
from pydantic import BaseModel

ROOT = Path(__file__).resolve().parents[1]


class ScraperConfig(BaseModel):
    # one sitemap or several (kapitalbank.az + birbank.az + birbank.business)
    sitemap_url: str | list[str]
    user_agent: str = "KapitalRAG-Bot/0.1 (educational RAG project)"
    request_delay_sec: float = 0.5
    timeout_sec: int = 20
    retries: int = 2
    min_text_chars: int = 300
    # skip URLs containing any of these substrings (application widgets, order
    # forms — interactive pages that carry no knowledge-base content)
    exclude_url_patterns: list[str] = []

    @property
    def sitemap_urls(self) -> list[str]:
        urls = self.sitemap_url
        return [urls] if isinstance(urls, str) else list(urls)


class EmbeddingConfig(BaseModel):
    model_name: str = "BAAI/bge-m3"
    batch_size: int = 32
    max_seq_tokens: int = 1024


class ChunkingConfig(BaseModel):
    target_chars: int = 800
    overlap_chars: int = 120
    min_chars: int = 200


class VectorStoreConfig(BaseModel):
    persist_dir: str = "data/chroma"
    collection: str = "kapital_kb"
    distance: str = "cosine"


class RetrievalConfig(BaseModel):
    top_k: int = 6
    exclude_sections: list[str] = []
    # --- hybrid search (dense + BM25 via reciprocal rank fusion) ---
    candidate_pool: int = 60       # candidates fetched per retriever before fusion
    enable_bm25: bool = True
    # --- cross-encoder re-ranking of the fused pool ---
    rerank_model: str | None = "BAAI/bge-reranker-v2-m3"
    rerank_candidates: int = 20
    rerank_max_length: int = 320
    # --- LLM query expansion (plan 2.2): rewrite the query into az/en/ru,
    # retrieve per variant, fuse by RRF. Costs one extra LLM call per query.
    query_expansion: bool = False
    # --- morphology-aware BM25 tokens (plan 2.3): additive Cyrillic→Latin
    # transliteration + one-step Azerbaijani suffix stripping. BM25-only.
    morph_tokens: bool = False
    # --- conversational query condensing (plan 4.4): fold follow-up + chat
    # history into a standalone retrieval query via one LLM call. Only fires
    # when there is history — single-turn queries pay nothing.
    query_condensing: bool = False


class LLMConfig(BaseModel):
    base_url: str = "https://api.deepseek.com"
    model: str = "deepseek-chat"
    temperature: float = 0.2
    max_tokens: int = 1024
    judge_model: str = "deepseek-chat"
    # sampling seed for reproducible generation (eval --seed/--seeds variance
    # runs); None keeps DeepSeek's default nondeterministic sampling
    seed: int | None = None


class ChatConfig(BaseModel):
    history_turns: int = 6
    # plan 4.2: at answer time, re-check every [n] citation marker against the
    # passage it cites (same checker as the eval) and surface unsupported ones
    verify_citations: bool = True


class Settings(BaseModel):
    scraper: ScraperConfig
    embedding: EmbeddingConfig
    chunking: ChunkingConfig
    vector_store: VectorStoreConfig
    retrieval: RetrievalConfig
    llm: LLMConfig
    app: ChatConfig

    # ---- derived paths (always absolute, resolved against project root) ----
    @property
    def chroma_dir(self) -> Path:
        return ROOT / self.vector_store.persist_dir

    @property
    def raw_pages_path(self) -> Path:
        return ROOT / "data" / "raw" / "pages.jsonl"

    @property
    def golden_set_path(self) -> Path:
        return ROOT / "data" / "golden" / "qa.yaml"

    @property
    def eval_results_dir(self) -> Path:
        return ROOT / "eval_results"


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Load settings once per process (safe to call anywhere).

    ``KB_CONFIG`` env var points at an alternate YAML — used for ablation
    configs (model swaps, query-expansion on/off) without touching the live
    ``config.yaml``. Relative paths resolve against the project root.
    """
    load_dotenv(ROOT / ".env")
    cfg_path = Path(os.environ.get("KB_CONFIG") or "config.yaml")
    if not cfg_path.is_absolute():
        cfg_path = ROOT / cfg_path
    with open(cfg_path, encoding="utf-8") as f:
        raw = yaml.safe_load(f)
    return Settings.model_validate(raw)


def get_deepseek_api_key() -> str:
    key = os.environ.get("DEEPSEEK_API_KEY", "").strip()
    if not key:
        raise RuntimeError(
            "DEEPSEEK_API_KEY is not set. Copy .env.example to .env and add "
            "your key from https://platform.deepseek.com/api_keys"
        )
    return key
