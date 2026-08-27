"""Promote recurring 👎 feedback into draft golden-set candidates (plan 4.5).

The eval set only grows when someone authors questions. This closes the loop:
read ``data/feedback.jsonl``, surface what users actually disliked, and emit
ready-to-edit YAML stubs for ``data/golden/qa.yaml`` — a human still writes the
``reference_answer`` and verifies ``expected_sources`` (the golden set's whole
point is corpus-grounded truth the automated pipeline can't invent).

Usage:
    python -m scripts.review_feedback                 # summary + candidates
    python -m scripts.review_feedback --min-down 1    # lower the repeat bar
    python -m scripts.review_feedback --json          # machine-readable output
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from pathlib import Path

from kb_rag.config import ROOT

_WS = re.compile(r"\s+")


def load_feedback(path: Path) -> list[dict]:
    if not path.exists():
        return []
    out = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line:
            try:
                out.append(json.loads(line))
            except json.JSONDecodeError:
                continue  # tolerate a torn final line
    return out


def _norm(question: str) -> str:
    """Collapse whitespace + case so cosmetic variants group together."""
    return _WS.sub(" ", (question or "").strip()).casefold()


def aggregate(records: list[dict]) -> dict[str, dict]:
    """Per normalized question: {display, up, down, sample_answer, sample_sources}."""
    agg: dict[str, dict] = {}
    for r in records:
        key = _norm(r.get("question", ""))
        if not key:
            continue
        slot = agg.setdefault(key, {
            "display": (r.get("question") or "").strip(),
            "up": 0, "down": 0,
            "sample_answer": "", "sample_sources": [], "langs": set(),
        })
        if r.get("rating", 0) > 0:
            slot["up"] += 1
        elif r.get("rating", 0) < 0:
            slot["down"] += 1
            slot["sample_answer"] = r.get("answer", "")
            slot["sample_sources"] = r.get("sources", [])
            if r.get("lang"):
                slot["langs"].add(r["lang"])
    return agg


def candidates(agg: dict[str, dict], min_down: int) -> list[dict]:
    """Questions with >= min_down 👎 and more down than up — real friction,
    not one contrarian click."""
    out = []
    for slot in agg.values():
        if slot["down"] >= min_down and slot["down"] > slot["up"]:
            out.append(slot)
    out.sort(key=lambda s: -s["down"])
    return out


def _yaml_stub(slot: dict) -> str:
    lang = sorted(slot["langs"])[0] if slot["langs"] else "az"
    src_lines = "".join(f'    # - "{u.split("://", 1)[-1][:60]}"\n'
                        for u in slot["sample_sources"][:3]) or "    # - \n"
    answer = slot["sample_answer"].replace("\n", " ")[:200]
    return (
        f"- id: TODO-feedback-{hashlib.sha1(slot['display'].encode()).hexdigest()[:5]}\n"
        f'  question: "{slot["display"].replace(chr(34), chr(39))}"\n'
        f"  category: TODO\n"
        f"  lang: {lang}\n"
        f"  expected_sources:\n{src_lines}"
        f"  reference_answer: >-\n"
        f"    TODO — verify against the corpus (a 👎 was cast on this answer:\n"
        f"    \"{answer}\")\n"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--path", type=Path, default=ROOT / "data" / "feedback.jsonl")
    parser.add_argument("--min-down", type=int, default=2,
                        help="👎 count to promote a question as a candidate")
    parser.add_argument("--json", action="store_true", help="machine-readable output")
    args = parser.parse_args()

    records = load_feedback(args.path)
    agg = aggregate(records)
    cands = candidates(agg, args.min_down)
    total_up = sum(s["up"] for s in agg.values())
    total_down = sum(s["down"] for s in agg.values())

    if args.json:
        print(json.dumps({
            "n_records": len(records), "n_questions": len(agg),
            "up": total_up, "down": total_down,
            "candidates": [{"question": c["display"], "down": c["down"],
                            "up": c["up"]} for c in cands],
        }, ensure_ascii=False, indent=2))
        return

    print(f"{len(records)} feedback records over {len(agg)} distinct questions "
          f"({total_up} 👍, {total_down} 👎)")
    if not records:
        print(f"Nothing recorded yet — the app appends to {args.path} when a "
              "visitor clicks 👍/👎.")
        return
    if not cands:
        print(f"\nNo question reached --min-down={args.min_down} 👎 with down>up. "
              "Nothing to promote.")
        return

    print(f"\n{len(cands)} candidate(s) for the golden set — edit, verify against "
          "the corpus, then paste into data/golden/qa.yaml:\n")
    print("# --- draft golden questions (from 👎 feedback) ---")
    for slot in cands:
        print(f"# {slot['down']}👎/{slot['up']}👍  {slot['display']}")
    print()
    for slot in cands:
        print(_yaml_stub(slot))


if __name__ == "__main__":
    main()
