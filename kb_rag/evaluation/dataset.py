"""Golden QA dataset loading.

Schema (data/golden/qa.yaml):

    - id: en-cards-001
      question: "..."
      category: cards            # coarse topic tag for grouped reporting
      lang: en                   # language the question is asked in
      expected_sources:          # URL fragments; a retrieval counts as a hit
        - "kapitalbank.az/en/cards/birkart-miles-debet"   # when any returned URL contains one
      unanswerable: false        # true => correct behavior is an explicit refusal
      reference_answer: |        # optional; used by the judge when present
        ...
"""

from __future__ import annotations

from dataclasses import dataclass, field

import yaml


@dataclass
class GoldenQuestion:
    id: str
    question: str
    category: str
    lang: str
    expected_sources: list[str] = field(default_factory=list)
    unanswerable: bool = False
    reference_answer: str | None = None
    # prior chat turns for multi-turn items (list of {role, content}) —
    # fed to the pipeline so follow-up coreference is actually exercised
    history: list[dict] = field(default_factory=list)
    notes: str | None = None


def load_dataset(path) -> list[GoldenQuestion]:
    with open(path, encoding="utf-8") as f:
        raw = yaml.safe_load(f) or []
    items = []
    for i, entry in enumerate(raw):
        items.append(
            GoldenQuestion(
                id=entry.get("id") or f"q-{i:03d}",
                question=entry["question"],
                category=entry.get("category", "other"),
                lang=entry.get("lang", "az"),
                expected_sources=list(entry.get("expected_sources") or []),
                unanswerable=bool(entry.get("unanswerable", False)),
                reference_answer=entry.get("reference_answer"),
                history=[dict(t) for t in (entry.get("history") or [])],
                notes=entry.get("notes"),
            )
        )
    return items
