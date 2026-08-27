"""Validate the golden set against the corpus and the eval schema (plan 3.2).

Checks:
  1. every answerable item's expected_sources fragments match at least one
     scraped page URL (catches stale fragments after re-scrapes — the site
     301s most kapitalbank.az slugs to birbank.az);
  2. unique ids; lang in {az,en,ru}; answerable items carry expectations;
  3. reference_answer coverage for answerable items (correctness judging
     requires one);
  4. stratification: every category has at least --min-per-category (default
     4) ANSWERABLE questions, plus the lang x category matrix for review;
  5. unanswerable items carry no expected_sources (a hit on those would be
     a retrieval false positive scored as a miss).

Usage:  python -m scripts.validate_golden_set [--min-per-category 4]
Exits nonzero when any check fails.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter

from kb_rag.config import get_settings
from kb_rag.evaluation.dataset import load_dataset
from kb_rag.evaluation.retrieval_metrics import first_relevant_rank

VALID_LANGS = {"az", "en", "ru"}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--min-per-category", type=int, default=4)
    args = parser.parse_args()

    settings = get_settings()
    pages = [
        json.loads(line)
        for line in open(settings.raw_pages_path, encoding="utf-8")
        if line.strip()
    ]
    all_urls = [u for p in pages for u in (p["url"], p.get("final_url", ""))]

    items = load_dataset(settings.golden_set_path)
    problems = 0

    def fail(msg: str) -> None:
        nonlocal problems
        problems += 1
        print(f"[FAIL] {msg}")

    # 2. schema basics
    ids = Counter(i.id for i in items)
    for dup, n in ids.items():
        if n > 1:
            fail(f"duplicate id {dup!r} ({n}x)")
    answerable = [i for i in items if not i.unanswerable]
    unanswerable = [i for i in items if i.unanswerable]
    for item in items:
        if item.lang not in VALID_LANGS:
            fail(f"{item.id}: invalid lang {item.lang!r}")
        if item.unanswerable and item.expected_sources:
            fail(f"{item.id}: unanswerable must not list expected_sources")
        if item.history:
            bad = [t for t in item.history if not ({"role", "content"} <= set(t))]
            if bad:
                fail(f"{item.id}: history turns need role+content: {bad}")

    # 1. expectations resolve against the corpus
    for item in answerable:
        if not item.expected_sources:
            fail(f"{item.id}: answerable question has no expected_sources")
            continue
        if first_relevant_rank(all_urls, item.expected_sources) is None:
            fail(f"{item.id}: none of {item.expected_sources} matched any scraped URL")

    # 3. reference coverage
    missing_ref = [i.id for i in answerable if not (i.reference_answer or "").strip()]
    if missing_ref:
        fail(f"{len(missing_ref)} answerable items without reference_answer: "
             f"{', '.join(missing_ref[:10])}")

    # 4. stratification
    cat_counts = Counter(i.category for i in answerable)
    thin = {c: n for c, n in cat_counts.items() if n < args.min_per_category}
    for cat, n in sorted(thin.items()):
        fail(f"category {cat!r} has only {n} answerable questions "
             f"(< {args.min_per_category})")
    lang_counts = Counter(i.lang for i in answerable)

    print(f"\n{len(items)} questions ({len(answerable)} answerable / "
          f"{len(unanswerable)} unanswerable) checked against {len(pages)} pages "
          f"({len(all_urls)} URLs)")
    print("\nAnswerable by category: "
          + ", ".join(f"{c}={n}" for c, n in sorted(cat_counts.items())))
    print("Answerable by lang:     "
          + ", ".join(f"{c}={n}" for c, n in sorted(lang_counts.items())))
    grid = Counter((i.category, i.lang) for i in answerable)
    print("\nlang x category matrix (answerable):")
    print("  " + " " * 16 + "".join(f"{l:>5}" for l in sorted(VALID_LANGS)))
    for cat in sorted(cat_counts):
        print(f"  {cat:<16}" + "".join(f"{grid.get((cat, l), 0):>5}" for l in sorted(VALID_LANGS)))

    print(f"\n{problems} problem(s)." if problems else "\nAll checks passed.")
    sys.exit(1 if problems else 0)


if __name__ == "__main__":
    main()
