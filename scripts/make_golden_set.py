"""Helper for drafting the golden QA set: browse indexed chunks by section.

Usage:
    python -m scripts.make_golden_set --section faq --sample 10
    python -m scripts.make_golden_set --query "kredit" --top 8
"""

from __future__ import annotations

import argparse
import random
from types import SimpleNamespace

from kb_rag.config import get_settings
from kb_rag.ingest.embeddings import E5Embedder
from kb_rag.ingest.store import VectorStore


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--section", default=None, help="filter by section (e.g. faq, cards)")
    parser.add_argument("--lang", default=None, help="filter by language (az/en/ru)")
    parser.add_argument("--query", default=None, help="semantic search instead of random sample")
    parser.add_argument("--sample", type=int, default=10)
    parser.add_argument("--top", type=int, default=8)
    args = parser.parse_args()

    settings = get_settings()
    store = VectorStore(settings)

    if args.query:
        embedder = E5Embedder(settings.embedding)
        chunks = store.query(embedder.embed_query(args.query), top_k=args.top,
                             where={"lang": {"$eq": args.lang}} if args.lang else None)
    else:
        where = None
        conditions = []
        if args.section:
            conditions.append({"section": {"$eq": args.section}})
        if args.lang:
            conditions.append({"lang": {"$eq": args.lang}})
        if len(conditions) == 1:
            where = conditions[0]
        elif conditions:
            where = {"$and": conditions}
        got = store.collection.get(where=where, include=["documents", "metadatas"], limit=None)
        docs, metas = got["documents"] or [], got["metadatas"] or []
        pairs = list(zip(docs, metas))
        random.shuffle(pairs)
        chunks = [SimpleNamespace(text=d, **(m or {})) for d, m in pairs[: args.sample]]

    print(f"Collection size: {store.count()} chunks\n")
    for i, c in enumerate(chunks, 1):
        print(f"--- #{i} [{getattr(c, 'lang', '?')}] {getattr(c, 'section_path', '')}")
        print(f"    url: {getattr(c, 'url', '')}")
        print(c.text[:400].replace("\n", " ") + ("..." if len(c.text) > 400 else ""))
        print()


if __name__ == "__main__":
    main()
