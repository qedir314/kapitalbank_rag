"""Polite site crawler: fetch pages, extract main content, emit JSONL records.

Uses trafilatura for main-content extraction (drops nav/footer boilerplate) and
keeps markdown headings so the chunker can split on document structure. Some
kapitalbank.az URLs 301-redirect to birbank.az — the final URL is recorded.
"""

from __future__ import annotations

import json
import time
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterator

import requests
import trafilatura
from tqdm import tqdm

from kb_rag.config import Settings
from kb_rag.scraper.sitemap import PageRef, load_page_refs


@dataclass
class PageRecord:
    url: str
    final_url: str
    lang: str
    title: str
    text: str
    status: int
    crawled_at: str


def _decode(resp: requests.Response) -> str:
    """Decode the response body without trusting requests' charset guess.

    ``resp.text`` falls back to Latin-1 when the Content-Type header carries no
    charset — birbank.az serves UTF-8 pages that way, which produced mojibake
    (``NaÄd`` instead of ``Nağd``). Decode as UTF-8 first, sniff only on failure.
    """
    raw = resp.content
    try:
        return raw.decode("utf-8")
    except UnicodeDecodeError:
        fallback = resp.apparent_encoding or "utf-8"
        return raw.decode(fallback, errors="replace")


def is_excluded_url(url: str, patterns: list[str]) -> bool:
    lowered = url.lower()
    return any(pattern.lower() in lowered for pattern in patterns)


def _fetch(
    session: requests.Session,
    url: str,
    settings: Settings,
) -> tuple[str | None, str, str | None, int]:
    """GET with retries. Returns (html|None, final_url, error_reason|None, status)."""
    last_err = None
    for attempt in range(settings.scraper.retries + 1):
        try:
            resp = session.get(url, timeout=settings.scraper.timeout_sec, allow_redirects=True)
            if resp.status_code == 200:
                html = _decode(resp)
                if html:
                    return html, str(resp.url), None, resp.status_code
            last_err = f"HTTP {resp.status_code}"
        except requests.RequestException as exc:
            last_err = f"{type(exc).__name__}: {exc}"
        time.sleep(1.5 * (attempt + 1))
    return None, url, last_err, 0


def _field(doc, name: str) -> str:
    """Read a field off either a dict (old trafilatura) or a Document object (2.x)."""
    if isinstance(doc, dict):
        value = doc.get(name)
    else:
        value = getattr(doc, name, None)
    return (value or "").strip()


def extract_content(html: str) -> tuple[str, str]:
    """Return (title, markdown_text); empty strings when extraction fails.

    bare_extraction is tried first (gives title + markdown together), then
    progressively simpler extract() calls — some thin pages only yield text
    through the plain extractor.
    """
    title, text = "", ""
    doc = trafilatura.bare_extraction(
        html,
        output_format="markdown",
        favor_recall=True,
        include_links=False,
        include_tables=True,
        with_metadata=True,
    )
    if doc:
        title = _field(doc, "title")
        text = _field(doc, "text")
    if not text:
        text = (
            trafilatura.extract(
                html,
                output_format="markdown",
                favor_recall=True,
                include_tables=True,
            )
            or ""
        )
    if not text:
        text = trafilatura.extract(html, favor_recall=True) or ""
    return title, text.strip()


def crawl_site(
    settings: Settings,
    langs: list[str] | None = None,
    limit: int | None = None,
    show_progress: bool = True,
) -> Iterator[PageRecord]:
    """Crawl every configured sitemap, yielding one record per extracted page."""
    refs: list[PageRef] = []
    for sitemap_url in settings.scraper.sitemap_urls:
        refs.extend(load_page_refs(
            sitemap_url,
            langs=langs,
            timeout=settings.scraper.timeout_sec,
            user_agent=settings.scraper.user_agent,
        ))
    if limit:
        refs = refs[:limit]
    session = requests.Session()
    session.headers.update({"User-Agent": settings.scraper.user_agent})

    iterator = tqdm(refs, desc="crawling", unit="page") if show_progress else refs
    crawled_at = datetime.now(timezone.utc).isoformat(timespec="seconds")

    for ref in iterator:
        if is_excluded_url(ref.url, settings.scraper.exclude_url_patterns):
            continue  # application widgets / order forms — not knowledge-base content
        html, final_url, err, status = _fetch(session, ref.url, settings)
        if html is None:
            tqdm.write(f"[skip] {ref.url} -> {err}")
            continue

        title, text = extract_content(html)
        if len(text) < settings.scraper.min_text_chars:
            continue  # JS shell / boilerplate-only page

        yield PageRecord(
            url=ref.url,
            final_url=(final_url or ref.url).rstrip("/"),
            lang=ref.lang,
            title=title,
            text=text,
            status=status,
            crawled_at=crawled_at,
        )
        time.sleep(settings.scraper.request_delay_sec)


def save_jsonl(records: Iterator[PageRecord], path: Path, mode: str = "w") -> dict:
    """Stream records to a JSONL file; returns summary counts.

    ``mode="a"`` appends to an existing crawl (build_index dedupes on
    final_url, so overlapping pages are harmless).
    """
    per_lang: dict[str, int] = {}
    written = 0
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, mode, encoding="utf-8") as f:
        for rec in records:
            f.write(json.dumps(asdict(rec), ensure_ascii=False) + "\n")
            written += 1
            per_lang[rec.lang] = per_lang.get(rec.lang, 0) + 1
    return {"written": written, "per_lang": per_lang}
