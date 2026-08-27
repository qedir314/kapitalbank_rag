"""Runtime citation verification (improvement plan 4.2).

The eval harness already scores ``[n]`` markers with the 3.5 checker; this
module runs the *same* heuristics at answer time so the user can see which
citations the pipeline could actually verify. Design decisions:

- **flag, never strip** — the marker is the user's link to source ``n``;
  silently deleting a wrong-but-plausible ``[3]`` hides the problem, while
  ``[3] ⚠`` plus a caption tells the user exactly which claim to double-check.
- **unjudgeable ≠ unsupported** — very short citing sentences ("Bəli [1].")
  carry too few tokens for a bag-of-words overlap test, so they are left
  unflagged rather than punished (same exclusion the eval applies).
- purely deterministic string statistics — no API call, no latency added.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

from kb_rag.evaluation.citation_metrics import (
    _MARKER_RE,
    _MIN_SENTENCE_TOKENS,
    _SENTENCE_SPLIT_RE,
    SUPPORT_THRESHOLD,
    split_passages,
)
from kb_rag.rag.hybrid import tokenize


@dataclass(frozen=True)
class CitationReport:
    """Which markers the checker could verify against their passages."""
    n_markers: int = 0
    invalid: tuple[int, ...] = ()       # [n] with no passage n in the context
    unsupported: tuple[int, ...] = ()   # [n] whose sentence has no lexical support
    flagged: tuple[int, ...] = ()       # union, sorted — what the UI should warn about

    @property
    def ok(self) -> bool:
        return not self.flagged


def verify_citations(answer_text: str, context: str) -> CitationReport:
    """Check every ``[n]`` marker in ``answer_text`` against the numbered context."""
    passages = split_passages(context)
    invalid: set[int] = set()
    unsupported: set[int] = set()
    n_markers = 0

    for sentence in _SENTENCE_SPLIT_RE.split(answer_text or ""):
        refs = [int(n) for n in _MARKER_RE.findall(sentence)]
        if not refs:
            continue
        n_markers += len(refs)
        invalid.update(n for n in refs if n not in passages)

        tokens = set(tokenize(sentence)) - {str(n) for n in refs}
        if len(tokens) < _MIN_SENTENCE_TOKENS:
            continue  # too short to test — no verdict either way
        valid_refs = [n for n in refs if n in passages]
        if not valid_refs:
            continue  # already counted as invalid
        overlaps = [
            len(tokens & set(tokenize(passages[n]))) / len(tokens) for n in valid_refs
        ]
        if max(overlaps) < SUPPORT_THRESHOLD:
            unsupported.update(valid_refs)

    flagged = tuple(sorted(invalid | unsupported))
    return CitationReport(
        n_markers=n_markers,
        invalid=tuple(sorted(invalid)),
        unsupported=tuple(sorted(unsupported)),
        flagged=flagged,
    )
