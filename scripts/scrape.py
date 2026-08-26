"""Scrape all configured sitemaps into data/raw/pages.jsonl.

Usage:
    python -m scripts.scrape                     # all sitemaps, fresh file
    python -m scripts.scrape --append            # add to the existing crawl
    python -m scripts.scrape --langs az en       # subset of languages
    python -m scripts.scrape --limit 25          # cap pages (smoke runs)
"""

from __future__ import annotations

import argparse

from kb_rag.config import get_settings
from kb_rag.scraper.crawl import crawl_site, save_jsonl


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--langs", nargs="*", default=None, help="az en ru (default: all)")
    parser.add_argument("--limit", type=int, default=None, help="max pages to fetch")
    parser.add_argument("--append", action="store_true",
                        help="add to pages.jsonl instead of overwriting")
    parser.add_argument("--sitemap", action="append", default=None,
                        help="crawl only sitemaps whose URL contains this substring "
                             "(repeatable), e.g. --sitemap birbank")
    args = parser.parse_args()

    settings = get_settings()
    if args.sitemap:
        matched = [u for u in settings.scraper.sitemap_urls
                   if any(sub in u for sub in args.sitemap)]
        settings.scraper = settings.scraper.model_copy(update={"sitemap_url": matched})
        print("Crawling sitemaps:", *matched, sep="\n  ")

    summary = save_jsonl(
        crawl_site(settings, langs=args.langs, limit=args.limit),
        settings.raw_pages_path,
        mode="a" if args.append else "w",
    )
    print(f"Saved {summary['written']} pages -> {settings.raw_pages_path} "
          f"({'appended' if args.append else 'overwritten'})")
    print(f"Per language: {summary['per_lang']}")


if __name__ == "__main__":
    main()
