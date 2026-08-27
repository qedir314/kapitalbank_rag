"""Re-extract FAQ page content from the Next.js JSON embedded in the HTML.

The standard trafilatura pipeline drops FAQ questions (accordion titles are
buttons that trafilatura treats as chrome). The actual Q&A pairs live in the
``__NEXT_DATA__`` JSON blob on every FAQ page, including all language
localizations. This script:

1. Reads the existing pages.jsonl and identifies FAQ URLs.
2. Re-fetches each FAQ page HTML.
3. Parses the embedded JSON, extracts every (question, answer) pair for the
   matching locale, plus the locale's translations of other pairs.
4. Replaces the ``text`` field with structured markdown: one H2 per FAQ group,
   then each question as H3 followed by its answer.
5. Writes updated records to a new pages.jsonl (non-FAQ records pass through).

Usage:
    python -m scripts.fix_faq_pages
"""

from __future__ import annotations

import json
import re
import sys
import time
from pathlib import Path

import requests

# allow running from repo root
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from kb_rag.config import get_settings
from kb_rag.scraper.crawl import _decode, _fetch


FAQ_PATH_SEGMENTS = ("/faq", "/en/faq", "/ru/faq")


def _is_faq_url(url: str) -> bool:
    path = url.split("://", 1)[-1].rstrip("/")
    return path in FAQ_PATH_SEGMENTS or path.endswith("/faq")


def _extract_faq_from_html(html: str, locale: str) -> tuple[str, str, list[dict]]:
    """Return (title, markdown_text, raw_faq_items) from the embedded JSON."""
    match = re.search(r'<script[^>]*id="__NEXT_DATA__"[^>]*>(.*?)</script>', html, re.DOTALL)
    if not match:
        return "", "", []
    try:
        data = json.loads(match.group(1))
    except json.JSONDecodeError:
        return "", "", []

    try:
        faq_state = data["props"]["pageProps"]["initialState"]["faq"]
        faq_groups = faq_state.get("faqGroup", [])
        faq_items = faq_state.get("faqData", [])
    except (KeyError, TypeError):
        return "", "", []

    # Build a mapping: group_id -> group title (for the matching locale)
    group_titles: dict[int, str] = {}
    for g in faq_groups:
        attrs = g.get("attributes", {})
        group_id = g.get("id")
        title = attrs.get("title", "")
        # group titles are only in the page's own locale; use as-is
        group_titles[group_id] = title

    # For each FAQ item, pick the (question, answer, group_id) for the
    # requested locale. The page's own locale is in the top-level attributes;
    # other locales live in localizations.data[].
    pairs_by_group: dict[int, list[tuple[str, str]]] = {}
    for item in faq_items:
        attrs = item.get("attributes", {})
        # group is nested: {data: {id: ..., attributes: {title: ...}}}
        group_data = attrs.get("group", {}).get("data", {})
        group_id = group_data.get("id")
        q, a = attrs.get("question", ""), attrs.get("answer", "")

        # Try to find a localization matching the requested locale
        if locale != "az":  # az is the default page locale; only look elsewhere for en/ru
            for loc in attrs.get("localizations", {}).get("data", []):
                loc_attrs = loc.get("attributes", {})
                if loc_attrs.get("locale") == locale:
                    q = loc_attrs.get("question", q)
                    a = loc_attrs.get("answer", a)
                    break

        if not q:
            continue
        pairs_by_group.setdefault(group_id or 0, []).append((q, a))

    # Render as markdown, sorted by group order
    ordered_groups = sorted(faq_groups, key=lambda g: g.get("attributes", {}).get("order", 0))
    title = ""
    lines: list[str] = []
    try:
        title = data["props"]["pageProps"]["initialState"]["faq"]["meta"]["metatags"]["title"] or \
                data["props"]["pageProps"]["messages"]["faq"]["title"]
    except (KeyError, TypeError):
        pass

    for g in ordered_groups:
        g_id = g.get("id")
        g_title = group_titles.get(g_id, "")
        pairs = pairs_by_group.get(g_id, [])
        if not pairs:
            continue
        lines.append(f"## {g_title}")
        lines.append("")
        for q, a in pairs:
            # Use bold text instead of H3 heading so the question is preserved
            # in chunk text (chunker strips all headings H1-H6 from content)
            lines.append(f"**{q}**")
            lines.append("")
            lines.append(a.strip())
            lines.append("")

    return title, "\n".join(lines).strip(), faq_items


def main() -> None:
    settings = get_settings()
    src = settings.raw_pages_path
    dst = src.with_suffix(".jsonl.faq_fixed")

    session = requests.Session()
    session.headers.update({"User-Agent": settings.scraper.user_agent})

    # First pass: identify FAQ pages and group by final_url
    faq_pages: list[dict] = []
    non_faq_count = 0
    with open(src, encoding="utf-8") as f:
        for line in f:
            p = json.loads(line.strip())
            if _is_faq_url(p.get("final_url", "")) or _is_faq_url(p.get("url", "")):
                faq_pages.append(p)
            else:
                non_faq_count += 1

    print(f"Found {len(faq_pages)} FAQ pages, {non_faq_count} non-FAQ pages")

    # Re-fetch each FAQ page and rebuild the text field
    updated: list[dict] = []
    for page in faq_pages:
        url = page["url"]
        locale = page.get("lang", "az")
        print(f"  Re-fetching {url} (locale={locale})")
        html, final_url, err, status = _fetch(session, url, settings)
        if html is None:
            print(f"    FAILED: {err}, keeping original text")
            updated.append(page)
            continue
        title, new_text, raw_items = _extract_faq_from_html(html, locale)
        print(f"    extracted {len(raw_items)} FAQ items, text {len(new_text)} chars")
        if new_text:
            page["text"] = new_text
            if title:
                page["title"] = title
        updated.append(page)
        time.sleep(settings.scraper.request_delay_sec)

    # Write the merged output: non-FAQ pages from original, then updated FAQ pages
    written_faq = 0
    written_other = 0
    with open(src, encoding="utf-8") as fin, open(dst, "w", encoding="utf-8") as fout:
        for line in fin:
            p = json.loads(line.strip())
            if _is_faq_url(p.get("final_url", "")) or _is_faq_url(p.get("url", "")):
                continue  # will write updated version below
            fout.write(json.dumps(p, ensure_ascii=False) + "\n")
            written_other += 1
        for p in updated:
            fout.write(json.dumps(p, ensure_ascii=False) + "\n")
            written_faq += 1

    print(f"Wrote {dst}: {written_other} non-FAQ + {written_faq} FAQ pages")
    print(f"Review, then rename over {src.name} and rebuild the index.")


if __name__ == "__main__":
    main()
