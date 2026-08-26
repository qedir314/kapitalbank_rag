"""Sitemap parsing and language detection for kapitalbank.az.

URL shape on the site: Azerbaijani pages sit at the bare path (``/cards``),
English and Russian versions under ``/en/...`` and ``/ru/...``.
"""

from __future__ import annotations

import xml.etree.ElementTree as ET
from dataclasses import dataclass

import requests

SITEMAP_NS = {"sm": "http://www.sitemaps.org/schemas/sitemap/0.9"}
LANG_PREFIXES = ("en", "ru")


@dataclass(frozen=True)
class PageRef:
    url: str
    lang: str


def detect_lang(url: str) -> str:
    """Derive the page language from the first path segment."""
    path = url.split("://", 1)[-1]
    path = path.split("/", 1)[1] if "/" in path else ""
    first = path.split("/", 1)[0].lower()
    return first if first in LANG_PREFIXES else "az"


def _parse_loc_entries(xml_bytes: bytes) -> tuple[list[str], bool]:
    """Return (loc urls, is_sitemap_index)."""
    root = ET.fromstring(xml_bytes)
    tag = root.tag.split("}", 1)[-1]
    locs = [el.text.strip() for el in root.findall(".//sm:loc", SITEMAP_NS) if el.text]
    return locs, tag == "sitemapindex"


def fetch_sitemap_locs(
    sitemap_url: str,
    timeout: int = 20,
    user_agent: str | None = None,
) -> list[str]:
    """Fetch a sitemap (following sitemap-index files one level deep)."""
    headers = {"User-Agent": user_agent} if user_agent else None
    seen: set[str] = set()
    queue = [sitemap_url]
    locs: list[str] = []

    while queue:
        resp = requests.get(queue.pop(0), timeout=timeout, headers=headers)
        resp.raise_for_status()
        entries, is_index = _parse_loc_entries(resp.content)
        if is_index:
            queue.extend(e for e in entries if e not in seen)
            seen.update(entries)
        else:
            locs.extend(entries)

    # de-duplicate, preserving order
    return list(dict.fromkeys(locs))


def load_page_refs(
    sitemap_url: str,
    langs: list[str] | None = None,
    limit: int | None = None,
    timeout: int = 20,
    user_agent: str | None = None,
) -> list[PageRef]:
    """Build the crawl list, optionally filtered by language and capped."""
    refs = [
        PageRef(url=url, lang=detect_lang(url))
        for url in fetch_sitemap_locs(sitemap_url, timeout=timeout, user_agent=user_agent)
    ]
    if langs:
        wanted = {l.lower() for l in langs}
        refs = [r for r in refs if r.lang in wanted]
    if limit:
        refs = refs[:limit]
    return refs
