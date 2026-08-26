"""Chunk scraped pages, embed with multilingual E5, upsert into Chroma.

Usage:
    python -m scripts.build_index            # incremental upsert
    python -m scripts.build_index --reset    # wipe the collection first
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re

from tqdm import tqdm

from kb_rag.config import get_settings
from kb_rag.ingest.chunking import build_chunks
from kb_rag.ingest.embeddings import E5Embedder
from kb_rag.ingest.store import VectorStore
from kb_rag.scraper.crawl import is_excluded_url

_WHITESPACE = re.compile(r"\s+")


def _text_fingerprint(text: str) -> str:
    """Hash of whitespace-normalized text — catches same-content-different-URL."""
    return hashlib.sha1(_WHITESPACE.sub(" ", text).strip().lower().encode("utf-8")).hexdigest()


def load_pages(path):
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                yield json.loads(line)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reset", action="store_true", help="delete and rebuild the collection")
    args = parser.parse_args()

    settings = get_settings()
    store = VectorStore(settings)
    if args.reset:
        store.reset()
        print("Collection reset.")

    embedder = E5Embedder(settings.embedding)

    total_chunks = 0
    skipped_dupes = 0
    skipped_excluded = 0
    skipped_texts = 0
    seen_final_urls: set[str] = set()
    seen_texts: set[str] = set()
    batch: list = []
    BATCH_PAGES = 20  # encode in batches so memory stays flat on large crawls

    def flush(pages):
        nonlocal total_chunks, skipped_texts
        chunks = [c for page in pages for c in build_chunks(page, settings.chunking)]
        fresh = []
        for c in chunks:
            fp = _text_fingerprint(c.text)
            if fp in seen_texts:
                continue  # some birbank pages serve identical text under /en/ and /ru/
            seen_texts.add(fp)
            fresh.append(c)
        skipped_texts += len(chunks) - len(fresh)
        if not fresh:
            return
        embeddings = embedder.embed_passages([c.text for c in fresh])
        store.add_chunks(fresh, embeddings)
        total_chunks += len(fresh)

    patterns = settings.scraper.exclude_url_patterns
    for page in tqdm(load_pages(settings.raw_pages_path), desc="indexing", unit="page"):
        # application widgets / order forms carry no knowledge content
        if is_excluded_url(page.get("final_url", ""), patterns) or \
                is_excluded_url(page.get("url", ""), patterns):
            skipped_excluded += 1
            continue
        # multiple kapitalbank.az slugs can 301 to the same birbank.az page;
        # index each destination once (first language variant wins)
        if page.get("final_url") in seen_final_urls:
            skipped_dupes += 1
            continue
        seen_final_urls.add(page.get("final_url"))
        batch.append(page)
        if len(batch) >= BATCH_PAGES:
            flush(batch)
            batch = []
    flush(batch)

    print(f"Indexed {total_chunks} chunks from {settings.raw_pages_path.name} "
          f"({skipped_dupes} duplicate redirect targets skipped, "
          f"{skipped_excluded} excluded-pattern pages skipped, "
          f"{skipped_texts} duplicate-text chunks skipped, "
          f"collection now holds {store.count()} vectors).")


if __name__ == "__main__":
    main()
