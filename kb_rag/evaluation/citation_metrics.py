"""Programmatic citation checks on generated answers (improvement plan 3.5).

The prompt demands ``[n]`` markers on every factual claim, but until now no
metric verified them. This module adds a cheap deterministic layer on top of
the LLM judge:

- **validity** — every ``[n]`` must reference a passage that actually exists
  in the numbered context (models occasionally invent [7] from six passages);
- **support** — the sentence carrying the citation should lexically overlap
  the passage it cites. This is a bag-of-words heuristic: it catches citations
  stapled onto unrelated sentences, but cannot prove entailment — which is
  exactly why the faithfulness judge stays in the loop.
- **coverage** — the share of answer sentences that carry at least one
  citation (the prompt's "every factual claim must have a citation" rule).

All functions are pure string statistics: no network, safe for CI.
"""

from __future__ import annotations

import re

from kb_rag.rag.hybrid import tokenize

_MARKER_RE = re.compile(r"\[(\d{1,2})\]")
_PASSAGE_START_RE = re.compile(r"^\[(\d+)\] ", re.MULTILINE)
# sentence enders — but never split "10.9"-style decimals (bank answers are
# full of them, and a broken sentence would poison both coverage and support)
_SENTENCE_SPLIT_RE = re.compile(r"[!?…\n]+|(?<!\d)\.(?!\d)")
_MIN_SENTENCE_TOKENS = 3     # "Yes [1]." is not judgeable overlap
SUPPORT_THRESHOLD = 0.2      # share of sentence tokens found in the passage


def split_passages(context: str) -> dict[int, str]:
    """Parse the numbered ``build_context_block`` format into {n: passage_text}.

    Each passage starts with a metadata header line (``[n] breadcrumb (url)``)
    which is dropped — the body text is what citations must be supported by.
    """
    marks = list(_PASSAGE_START_RE.finditer(context or ""))
    passages: dict[int, str] = {}
    for i, mark in enumerate(marks):
        end = marks[i + 1].start() if i + 1 < len(marks) else len(context)
        body = context[mark.end():end]
        _, _, rest = body.partition("\n")  # strip the header line
        passages[int(mark.group(1))] = rest.strip()
    return passages


def citation_stats(answer_text: str, context: str) -> dict:
    """Row-level citation metrics; every value None when the answer has no markers.

    Refusals legitimately carry no citations — a refusal row simply shows all
    None and must not be scored down for it (the runner only reports means).
    """
    passages = split_passages(context)
    sentences = [s.strip() for s in _SENTENCE_SPLIT_RE.split(answer_text or "") if s.strip()]
    citing = [s for s in sentences if _MARKER_RE.search(s)]

    stats = {
        "n_citations": sum(len(_MARKER_RE.findall(s)) for s in citing),
        "citation_valid_frac": None,
        "citation_support_frac": None,
        "citation_coverage": None,
    }
    if not citing:
        return stats

    valid = total = 0
    supported = 0
    judged = 0
    for sentence in citing:
        refs = [int(n) for n in _MARKER_RE.findall(sentence)]
        total += len(refs)
        valid += sum(1 for n in refs if n in passages)
        tokens = set(tokenize(sentence)) - {str(n) for n in refs}
        if len(tokens) < _MIN_SENTENCE_TOKENS:
            continue  # too short to test overlap; excluded from support rate
        overlaps = [
            len(tokens & set(tokenize(passages[n]))) / len(tokens)
            for n in refs if n in passages
        ]
        judged += 1
        if overlaps and max(overlaps) >= SUPPORT_THRESHOLD:
            supported += 1

    stats["citation_valid_frac"] = valid / total if total else None
    stats["citation_support_frac"] = supported / judged if judged else None
    stats["citation_coverage"] = len(citing) / len(sentences) if sentences else None
    return stats
