"""Validate golden-set expectations against actually scraped pages.

Checks that every `expected_sources` fragment matches at least one page's
final/source URL in data/raw/pages.jsonl — catches stale fragments after
re-scrapes (the site 301s most slugs to birbank.az).

Usage:  python -m scripts.validate_golden_set
"""

from __future__ import annotations

import json

from kb_rag.config import get_settings
from kb_rag.evaluation.dataset import load_dataset
from kb_rag.evaluation.retrieval_metrics import first_relevant_rank


def main() -> None:
    settings = get_settings()
    pages = [
        json.loads(line)
        for line in open(settings.raw_pages_path, encoding="utf-8")
        if line.strip()
    ]
    all_urls = [u for p in pages for u in (p["url"], p.get("final_url", ""))]

    items = load_dataset(settings.golden_set_path)
    problems = 0
    for item in items:
        if item.unanswerable:
            continue
        rank = first_relevant_rank(all_urls, item.expected_sources)
        if rank is None:
            problems += 1
            print(f"[MISS] {item.id}: none of {item.expected_sources} matched any scraped URL")
    print(f"\n{len(items)} questions checked against {len(pages)} pages "
          f"({len(all_urls)} URLs) — {problems} unmatched")


if __name__ == "__main__":
    main()
