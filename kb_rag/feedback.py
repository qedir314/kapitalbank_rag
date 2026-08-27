"""👍/👎 answer feedback capture (improvement plan 4.5).

One JSON line per rating in ``data/feedback.jsonl``. The file is append-only
and deliberately not part of the retrieval path — it's a cheap flywheel: the
review script turns recurring 👎 into draft golden-set candidates, so real
user friction becomes labelled eval questions without a manual authoring pass.

Kept dependency-free and crash-safe: a write failure must never surface in
the chat UI (the user already has their answer), so ``record_feedback`` swallows
I/O errors after one stderr note.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def record_feedback(
    path: Path,
    *,
    question: str,
    answer: str,
    rating: int,               # +1 (👍) or -1 (👎)
    sources: list[Any],        # pipeline.Source objects (need .url/.crawled_at)
    lang: str | None = None,
    sections: list[str] | None = None,
    top_k: int | None = None,
) -> bool:
    """Append one feedback record; returns True on success, False on any I/O error.

    ``answer`` is truncated to 800 chars — enough to spot a wrong/hedged answer
    in review without the log growing unbounded on a chatty session.
    """
    try:
        src = [getattr(s, "url", "") for s in (sources or [])]
        crawled = [getattr(s, "crawled_at", "") for s in (sources or [])]
        rec = {
            "ts": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "question": question,
            "answer": (answer or "")[:800],
            "rating": rating,
            "sources": src,
            "crawled_at": max((c for c in crawled if c), default=""),
            "lang": lang,
            "sections": sections or [],
            "top_k": top_k,
        }
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "a", encoding="utf-8") as f:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
        return True
    except Exception as exc:  # never break the chat for a logging failure
        print(f"[feedback] write failed: {type(exc).__name__}: {exc}")
        return False
