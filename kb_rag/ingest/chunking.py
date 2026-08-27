"""Structure-aware chunking.

Strategy: the scraper emits markdown-ish text, so we first split on headings
(keeping a breadcrumb like "Cards > Birbank Miles"), then apply a sliding
window *within* each section so chunks stay semantically coherent. Small
chunks are dropped — they add index noise without retrieval value.
"""

from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass, field

from kb_rag.config import ChunkingConfig

_HEADING_RE = re.compile(r"^(#{1,6})\s+(.+?)\s*#*\s*$")
# sentence enders incl. Azerbaijani/Russian punctuation; also split on newlines
_SENTENCE_RE = re.compile(r"(?<=[.!?…:;])\s+|\n+")


@dataclass
class Chunk:
    chunk_id: str
    text: str
    metadata: dict = field(default_factory=dict)


def split_sections(markdown_text: str) -> list[tuple[str, str]]:
    """Split markdown text into (section_title, body) pairs by ATX headings."""
    sections: list[tuple[str, str]] = []
    current_title, buffer = "", []

    for line in markdown_text.splitlines():
        m = _HEADING_RE.match(line.strip())
        if m:
            if any(s.strip() for s in buffer):
                sections.append((current_title, "\n".join(buffer)))
            current_title, buffer = m.group(2), []
        else:
            buffer.append(line)
    if any(s.strip() for s in buffer):
        sections.append((current_title, "\n".join(buffer)))
    return sections


def _split_units(text: str) -> list[str]:
    """Split text into sentence/paragraph-level units."""
    units = []
    for para in text.split("\n\n"):
        units.extend(u for u in _SENTENCE_RE.split(para) if u and u.strip())
    return [u.strip() for u in units if u.strip()]


def sliding_window(text: str, cfg: ChunkingConfig) -> list[str]:
    """Pack sentence units into windows of ~target_chars with overlap."""
    units = _split_units(text)
    windows: list[str] = []
    current: list[str] = []
    size = 0

    for unit in units:
        # single unit longer than target -> hard-split on characters
        while len(unit) > cfg.target_chars:
            if current:
                windows.append(" ".join(current))
                current, size = [], 0
            piece, unit = unit[: cfg.target_chars], unit[cfg.target_chars - cfg.overlap_chars:]
            windows.append(piece)
        if size + len(unit) > cfg.target_chars and current:
            windows.append(" ".join(current))
            tail = " ".join(current)[-cfg.overlap_chars:]
            current = [tail, unit] if cfg.overlap_chars else [unit]
            size = len(tail) + len(unit) + (1 if cfg.overlap_chars else 0)
        else:
            current.append(unit)
            size += len(unit) + 1
    if current:
        joined = " ".join(current).strip()
        if joined:
            windows.append(joined)
    return windows


def derive_section(url: str) -> str:
    """Coarse topical section from the URL path, used as a metadata filter.

    Uses pattern matching with priority: card/loan/deposit terms map to their
    sections even when the exact slug isn't in the known set.
    """
    path = url.split("://", 1)[-1].lower()
    parts = [p for p in path.split("/") if p]
    if not parts:
        return "other"
    if parts[0] in ("en", "ru", "az"):
        parts = parts[1:]
    # Exact slug match first (fast path)
    known = {
        "cards", "loans", "deposits", "money-transfers", "sigortalar", "insurance",
        "corporate-banking", "birbank", "kampaniyalar", "news", "faq", "how-to",
        "locations", "online-order", "ferdi-bankciliq",
    }
    for part in parts:
        if part in known:
            return part
    # Pattern-based fallback for compound paths
    full_path = "/".join(parts)
    if any(kw in full_path for kw in ["card", "kart", "birbank-miles", "virtual"]):
        return "cards"
    if any(kw in full_path for kw in ["loan", "kredit", "kreditler"]):
        return "loans"
    if any(kw in full_path for kw in ["deposit", "depzit", "depozit", "əmanət"]):
        return "deposits"
    if any(kw in full_path for kw in ["transfer", "pul-kocurmeleri", "western-union", "golden-crown"]):
        return "money-transfers"
    if any(kw in full_path for kw in ["sigorta", "insurance", "sığorta"]):
        return "sigortalar"
    if "faq" in full_path:
        return "faq"
    if "how-to" in full_path or "guide" in full_path:
        return "how-to"
    if "location" in full_path or "branch" in full_path or "filial" in full_path:
        return "locations"
    if "kampaniya" in full_path or "promotion" in full_path or "campaign" in full_path:
        return "kampaniyalar"
    if "news" in full_path:
        return "news"
    return "other"


def build_chunks(page: dict, cfg: ChunkingConfig) -> list[Chunk]:
    """Turn one scraped page record into metadata-rich chunks."""
    chunks: list[Chunk] = []
    page_title = (page.get("title") or "").strip()
    url = page.get("final_url") or page["url"]
    # the original sitemap URL carries the site taxonomy; final URLs may be
    # redirect targets (birbank.az) with a different path structure
    section = derive_section(page["url"])

    for section_title, body in split_sections(page.get("text", "")):
        for window in sliding_window(body, cfg):
            window = window.strip()
            if len(window) < cfg.min_chars:
                continue
            breadcrumb = " > ".join(x for x in (page_title, section_title) if x) or page_title
            idx = len(chunks)
            chunk_id = hashlib.sha1(f"{url}::{idx}".encode()).hexdigest()[:16]
            chunks.append(
                Chunk(
                    chunk_id=chunk_id,
                    text=window,
                    metadata={
                        "url": url,
                        "source_url": page["url"],
                        "lang": page.get("lang", "az"),
                        "title": page_title,
                        "section_path": breadcrumb[:300],
                        "section": section,
                    },
                )
            )
    return chunks
