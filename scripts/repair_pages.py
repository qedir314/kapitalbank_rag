"""Repair mojibake in data/raw/pages.jsonl in place.

The crawler originally decoded some responses with Latin-1 while the pages are
UTF-8 (missing charset header), producing text like ``NaÄd pul krediti``.
ftfy reverses that encoding mistake. A timestamped backup is kept next to the
original file.

Usage:
    python -m scripts.repair_pages            # repair + report
    python -m scripts.repair_pages --dry-run  # count affected records only
"""

from __future__ import annotations

import argparse
import json
import shutil
import time
from pathlib import Path

import ftfy
from tqdm import tqdm

from kb_rag.config import get_settings


def fix_record(record: dict) -> dict:
    record["title"] = ftfy.fix_text(record.get("title") or "")
    record["text"] = ftfy.fix_text(record.get("text") or "")
    return record


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dry-run", action="store_true", help="only count affected records")
    args = parser.parse_args()

    settings = get_settings()
    path: Path = settings.raw_pages_path
    if not path.exists():
        raise SystemExit(f"not found: {path}")

    fixed = 0
    total = 0
    out_path = path.with_suffix(".jsonl.tmp")
    with open(path, encoding="utf-8") as src, \
            open(out_path, "w", encoding="utf-8") as dst:
        for line in tqdm(src, desc="repairing", unit="page"):
            total += 1
            record = json.loads(line)
            original_title, original_text = record.get("title"), record.get("text")
            fix_record(record)
            if record["title"] != original_title or record["text"] != original_text:
                fixed += 1
                if args.dry_run:
                    continue
            dst.write(json.dumps(record, ensure_ascii=False) + "\n")

    print(f"{fixed}/{total} records contain recoverable encoding damage.")

    if args.dry_run:
        out_path.unlink(missing_ok=True)
        return

    backup = path.with_suffix(f".bak.{time.strftime('%Y%m%d_%H%M%S')}")
    shutil.copy2(path, backup)
    out_path.replace(path)
    print(f"Repaired file written. Backup: {backup.name}")


if __name__ == "__main__":
    main()
