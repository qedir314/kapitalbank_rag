"""Evaluation runner: golden questions -> retrieval + generation scores.

Usage:
    python -m kb_rag.evaluation.runner                     # full dataset
    python -m kb_rag.evaluation.runner --limit 5           # smoke run
    python -m kb_rag.evaluation.runner --dataset PATH
    python -m kb_rag.evaluation.runner --seeds 3           # variance: 3 seeded passes
    python -m kb_rag.evaluation.runner --rejudge CSV --judge-model NAME
                                                           # independent-judge ablation

Per question it measures:
  - retrieval hit@k / reciprocal rank against expected source URLs
  - answer faithfulness & correctness (LLM-as-judge, 1-5, vs reference when set)
  - programmatic citation stats (validity / support / coverage, plan 3.5)
  - refusal correctness for unanswerable questions
Results go to eval_results/run_<timestamp>.csv (+ a .manifest.json recording
the exact config, git SHA and dataset hash that produced every number).
A per-question failure is recorded in the row's ``error`` column instead of
crashing the run mid-way.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import statistics
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
from tqdm import tqdm

from kb_rag.config import ROOT, Settings, get_settings
from kb_rag.evaluation.citation_metrics import citation_stats
from kb_rag.evaluation.dataset import GoldenQuestion, load_dataset
from kb_rag.evaluation.generation_metrics import judge_answer, looks_like_refusal
from kb_rag.evaluation.retrieval_metrics import (
    first_relevant_source_rank,
    hit_at_k,
    mean,
    reciprocal_rank,
)
from kb_rag.rag.llm import DeepSeekClient
from kb_rag.rag.pipeline import get_pipeline
from kb_rag.rag.prompts import build_context_block

def _judge_max_tokens(judge_model: str) -> int:
    """Reasoning models spend part of the completion budget on hidden thinking
    tokens — a 300-token cap starves them into empty answers (observed with
    deepseek-reasoner in the Phase 3.4 ablation)."""
    return 1500 if "reasoner" in judge_model.lower() else 300


# every key a row can carry — keeps the CSV schema stable when errors truncate a row
_ROW_KEYS = [
    "id", "category", "lang", "unanswerable", "hit", "reciprocal_rank",
    "first_hit_rank", "n_sources_returned", "refused", "refusal_correct",
    "faithfulness", "correctness", "judge_rationale", "n_citations",
    "citation_valid_frac", "citation_support_frac", "citation_coverage",
    "retrieval_query", "answer", "error", "judge_error",
]


def _blank_row(item: GoldenQuestion) -> dict:
    row = {k: None for k in _ROW_KEYS}
    row.update(id=item.id, category=item.category, lang=item.lang,
               unanswerable=item.unanswerable)
    return row


def evaluate_question(
    item: GoldenQuestion,
    pipeline,
    judge_client: DeepSeekClient,
    judge_model: str,
    top_k: int,
) -> dict:
    """Score one golden question. Failures land in row columns, never a crash."""
    row = _blank_row(item)
    try:
        # multi-turn items carry prior chat turns; the pipeline trims them to
        # the configured history window exactly like the app does
        answer = pipeline.answer(item.question, history=item.history or None,
                                 lang=None, section=None, top_k=top_k, stream=False)
    except Exception as exc:  # generation/API down: retrieval metrics are lost too
        row["error"] = f"answer: {type(exc).__name__}: {exc}"[:300]
        return row

    # rank over SOURCES (one entry per page), not a flattened URL list — each
    # source still matches on either identity (final birbank.az URL or the
    # original kapitalbank.az slug), but k now means what it says. The old
    # flatten-each-URL approach doubled positions and made hit@6 behave as
    # hit@3 (Phase 4 A/B catch; see retrieval_metrics docstring).
    rank = (first_relevant_source_rank(answer.sources, item.expected_sources)
            if item.expected_sources else None)
    row.update(
        hit=hit_at_k(rank, top_k),
        reciprocal_rank=reciprocal_rank(rank, top_k),
        first_hit_rank=(rank + 1) if rank is not None else None,
        n_sources_returned=len(answer.sources),
        refused=looks_like_refusal(answer.text),
        answer=answer.text or "",
        retrieval_query=answer.retrieval_query,  # condensed standalone query, if any
    )

    if item.unanswerable:
        row["refusal_correct"] = bool(row["refused"])
        # unanswerables are scored on refusal behavior, not on judge rubric
        return row

    # judge must see the SAME passage texts the generator saw — scoring
    # against bare URLs/breadcrumbs would flag every concrete detail as
    # an invention
    try:
        result = judge_answer(
            judge_client,
            judge_model=judge_model,
            question=item.question,
            context=answer.context or "",
            answer=row["answer"],
            reference=item.reference_answer,
            max_tokens=_judge_max_tokens(judge_model),
        )
        row.update(
            faithfulness=result.faithfulness,
            correctness=result.correctness,
            judge_rationale=result.rationale,
        )
    except Exception as exc:  # judge down — retrieval/generation columns survive
        row["judge_error"] = f"{type(exc).__name__}: {exc}"[:300]

    row.update(citation_stats(row["answer"], answer.context or ""))
    return row


def _git_info() -> dict:
    """Best-effort VCS identity for the manifest; None-tolerant outside git."""
    def _run(*args: str) -> str | None:
        try:
            out = subprocess.run(
                ("git", *args), capture_output=True, text=True,
                timeout=15, check=True, cwd=ROOT,
            ).stdout.strip()
            return out or None
        except Exception:
            return None
    porcelain = _run("status", "--porcelain")
    return {
        "sha": _run("rev-parse", "HEAD"),
        "branch": _run("rev-parse", "--abbrev-ref", "HEAD"),
        "dirty": (len(porcelain.splitlines()) > 0) if porcelain is not None else None,
    }


def write_manifest(
    out_path: Path,
    settings: Settings,
    *,
    dataset_path: Path,
    n_items: int,
    top_k: int,
    seed: int | None,
    judge_model: str,
    tag: str | None,
    mode: str,
    df: pd.DataFrame,
) -> Path:
    """Record everything needed to reproduce the run next to its CSV (plan 3.3)."""
    manifest_path = out_path.with_name(out_path.stem + ".manifest.json")
    n_errors = int(df["error"].notna().sum()) if "error" in df else 0
    n_judge_errors = int(df["judge_error"].notna().sum()) if "judge_error" in df else 0
    manifest = {
        "run_csv": out_path.name,
        "mode": mode,
        "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "tag": tag,
        "git": _git_info(),
        "kb_config": os.environ.get("KB_CONFIG") or "config.yaml",
        "top_k": top_k,
        "seed": seed,
        "judge_model": judge_model,
        "dataset": {
            "path": str(Path(dataset_path).relative_to(ROOT)),
            "sha256": hashlib.sha256(Path(dataset_path).read_bytes()).hexdigest(),
            "n_items": n_items,
        },
        "n_errors": n_errors,
        "n_judge_errors": n_judge_errors,
        "environment": {"python": sys.version.split()[0], "platform": platform.platform()},
        "settings": settings.model_dump(),
    }
    manifest_path.write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False, default=str), encoding="utf-8"
    )
    return manifest_path


def summarize(df: pd.DataFrame, top_k: int) -> str:
    answerable = df[~df["unanswerable"]]
    unanswerable = df[df["unanswerable"]]
    lines = [
        f"\n=== RAG evaluation — {len(df)} questions ===",
        f"Retrieval hit@{top_k}: {answerable['hit'].mean():.1%}"
        f"   MRR@{top_k}: {mean(answerable['reciprocal_rank'].dropna().tolist()):.3f}",
    ]
    if len(answerable):
        lines.append(
            f"Judge faithfulness (1-5): {answerable['faithfulness'].mean():.2f}"
            f"   correctness: {answerable['correctness'].mean():.2f}"
        )
        cited = answerable[answerable["citation_valid_frac"].notna()]
        if len(cited):
            lines.append(
                f"Citations: {cited['citation_valid_frac'].mean():.0%} valid"
                f"   {cited['citation_support_frac'].mean():.0%} supported"
                f"   {cited['citation_coverage'].mean():.0%} sentence coverage"
            )
    if len(unanswerable):
        lines.append(f"Refusal accuracy on unanswerables: {unanswerable['refusal_correct'].mean():.1%}")
    n_err = int(df["error"].notna().sum()) if "error" in df else 0
    n_jerr = int(df["judge_error"].notna().sum()) if "judge_error" in df else 0
    if n_err or n_jerr:
        lines.append(f"Errors: {n_err} question failures, {n_jerr} judge failures (rows kept)")

    by_cat = answerable.groupby("category").agg(
        n=("id", "count"),
        hit=(("hit"), "mean"),
        faithfulness=("faithfulness", "mean"),
        correctness=("correctness", "mean"),
    )
    if not by_cat.empty:
        lines.append("\nBy category:\n" + by_cat.to_string(float_format=lambda x: f"{x:.2f}"))
    return "\n".join(lines)


def _apply_settings(no_bm25: bool, no_rerank: bool, seed: int | None,
                    judge_model: str | None) -> Settings:
    """Load settings and apply CLI ablations BEFORE the pipeline singleton builds."""
    get_settings.cache_clear()
    get_pipeline.cache_clear()  # seeds loop: rebuild per pass with fresh settings
    settings = get_settings()
    updates: dict = {}
    if no_bm25:
        updates["enable_bm25"] = False
    if no_rerank:
        updates["rerank_model"] = None
    if updates:
        settings.retrieval = settings.retrieval.model_copy(update=updates)
    llm_updates = {}
    if seed is not None:
        llm_updates["seed"] = seed
    if judge_model:
        llm_updates["judge_model"] = judge_model
    if llm_updates:
        settings.llm = settings.llm.model_copy(update=llm_updates)
    return settings


def run(dataset_path=None, limit: int | None = None, top_k: int | None = None,
        no_bm25: bool = False, no_rerank: bool = False, seed: int | None = None,
        judge_model: str | None = None, tag: str | None = None) -> pd.DataFrame:
    settings = _apply_settings(no_bm25, no_rerank, seed, judge_model)
    k = top_k or settings.retrieval.top_k
    dataset = Path(dataset_path) if dataset_path else settings.golden_set_path
    items = load_dataset(dataset)
    if limit:
        items = items[:limit]

    pipeline = get_pipeline()
    judge_client = DeepSeekClient(settings)

    rows = [
        evaluate_question(item, pipeline, judge_client, settings.llm.judge_model, top_k=k)
        for item in tqdm(items, desc="evaluating", unit="q")
    ]
    df = pd.DataFrame(rows, columns=_ROW_KEYS)

    settings.eval_results_dir.mkdir(parents=True, exist_ok=True)
    out_path = settings.eval_results_dir / f"run_{datetime.now():%Y%m%d_%H%M%S}.csv"
    df.to_csv(out_path, index=False)
    manifest = write_manifest(
        out_path, settings, dataset_path=dataset, n_items=len(items), top_k=k,
        seed=seed, judge_model=settings.llm.judge_model, tag=tag, mode="full", df=df,
    )
    print(summarize(df, k))
    print(f"\nFull results saved to {out_path}")
    print(f"Manifest (config + git SHA + dataset hash) saved to {manifest}")
    return df


def run_seeds(n: int, **kwargs) -> None:
    """N seeded passes, then mean ± std across runs (plan 3.5 variance)."""
    metrics: dict[str, list[float]] = {}
    per_run = []
    base_tag = kwargs.pop("tag", None)
    for i in range(1, n + 1):
        print(f"\n--- seed {i}/{n} ---")
        df = run(seed=i, tag=f"{base_tag or 'seeded'}-seed{i}", **kwargs)
        answerable = df[~df["unanswerable"]]
        unanswerable = df[df["unanswerable"]]
        snapshot = {
            "hit": answerable["hit"].mean(),
            "mrr": answerable["reciprocal_rank"].mean(),
            "faithfulness": answerable["faithfulness"].mean(),
            "correctness": answerable["correctness"].mean(),
            "refusal": unanswerable["refusal_correct"].mean(),
        }
        per_run.append((i, snapshot))
        for key, val in snapshot.items():
            metrics.setdefault(key, []).append(val)
    print(f"\n=== Seed variance over {n} runs ===")
    for key, vals in metrics.items():
        m = statistics.mean(vals)
        s = statistics.stdev(vals) if len(vals) > 1 else 0.0
        print(f"  {key:<14} mean={m:.3f}  std={s:.3f}   runs={['%.3f' % v for v in vals]}")
    for i, snap in per_run:
        print(f"  seed {i}: " + "  ".join(f"{k}={v:.3f}" for k, v in snap.items()))


def rejudge(csv_path: Path, judge_model: str, tag: str | None = None) -> pd.DataFrame:
    """Re-score a previous run's answers with a different judge model (plan 3.4).

    The stored answers are judged against freshly re-retrieved context — the
    index and retrieval are deterministic, so the passages match what the
    generator saw. Emits a per-question agreement report; the stricter judge
    is the one to keep as default.
    """
    settings = _apply_settings(False, False, None, judge_model)
    pipeline = get_pipeline()
    judge_client = DeepSeekClient(settings)
    old = pd.read_csv(csv_path)
    items = {i.id: i for i in load_dataset(settings.golden_set_path)}
    k = settings.retrieval.top_k

    rows = []
    todo = old[(~old["unanswerable"]) & old["answer"].notna() & (old["answer"] != "")]
    for rec in tqdm(todo.to_dict("records"), desc=f"rejudging ({judge_model})", unit="q"):
        item = items.get(rec["id"])
        if item is None:
            continue
        # mirror pipeline.answer's retrieval path, condensing included, so the
        # judge sees the same context the generator saw for multi-turn items
        trimmed = pipeline._trim_history(item.history or [])
        query = item.question
        if trimmed and getattr(pipeline, "condenser", None) is not None:
            query = pipeline.condenser.condense(item.question, trimmed)
        chunks = pipeline.retriever.retrieve(query, top_k=k)
        context = build_context_block(chunks)
        try:
            result = judge_answer(
                judge_client, judge_model=judge_model, question=item.question,
                context=context, answer=rec["answer"], reference=item.reference_answer,
                max_tokens=_judge_max_tokens(judge_model),
            )
        except Exception as exc:
            rows.append({**rec, "rejudge_error": str(exc)[:300]})
            continue
        rows.append({
            **rec,
            "new_faithfulness": result.faithfulness,
            "new_correctness": result.correctness,
            "new_rationale": result.rationale,
            "rejudge_error": None,
        })
    df = pd.DataFrame(rows)
    settings.eval_results_dir.mkdir(parents=True, exist_ok=True)
    out_path = settings.eval_results_dir / (
        f"rejudge_{datetime.now():%Y%m%d_%H%M%S}_{judge_model.replace('/', '_')}.csv"
    )
    df.to_csv(out_path, index=False)

    scored = df[df["new_faithfulness"].notna()]
    print(f"\n=== Independent judge agreement — {judge_model} vs stored ({csv_path.name}) ===")
    print(f"n re-judged: {len(scored)}")
    for dim in ("faithfulness", "correctness"):
        a, b = scored[dim], scored[f"new_{dim}"]
        agree = int((a == b).sum())
        diff = b - a
        big = scored[diff.abs() >= 2][["id", "lang", dim, f"new_{dim}"]]
        print(f"\n{dim}: exact agreement {agree}/{len(scored)} ({agree/len(scored):.0%})"
              f"   mean shift {diff.mean():+.2f}   pearson {a.corr(b):.2f}")
        if len(big):
            print(f"  disagreements |Δ|≥2:\n" + big.to_string(index=False))
    print(f"\nFull comparison saved to {out_path}")
    write_manifest(
        out_path, settings, dataset_path=settings.golden_set_path, n_items=len(df),
        top_k=k, seed=None, judge_model=judge_model, tag=tag, mode="rejudge", df=df,
    )
    return df


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", default=None, help="path to qa.yaml")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--top-k", type=int, default=None)
    parser.add_argument("--no-bm25", action="store_true", help="ablation: dense retrieval only")
    parser.add_argument("--no-rerank", action="store_true", help="ablation: skip cross-encoder re-scoring")
    parser.add_argument("--seed", type=int, default=None, help="generation seed for reproducibility")
    parser.add_argument("--seeds", type=int, default=None,
                        help="run N seeded passes (1..N) and report mean ± std")
    parser.add_argument("--judge-model", default=None,
                        help="override llm.judge_model for this run")
    parser.add_argument("--tag", default=None,
                        help="label recorded in the run manifest (e.g. phase3-canonical)")
    parser.add_argument("--rejudge", type=Path, default=None, metavar="CSV",
                        help="re-score a previous run's CSV with --judge-model and print agreement")
    args = parser.parse_args()

    if args.rejudge:
        if not args.judge_model:
            parser.error("--rejudge requires --judge-model")
        rejudge(args.rejudge, args.judge_model, tag=args.tag)
        return
    kwargs = dict(dataset_path=args.dataset, limit=args.limit, top_k=args.top_k,
                  no_bm25=args.no_bm25, no_rerank=args.no_rerank,
                  judge_model=args.judge_model, tag=args.tag)
    if args.seeds:
        run_seeds(args.seeds, **kwargs)  # seed assigned per pass
    else:
        run(**kwargs, seed=args.seed)


if __name__ == "__main__":
    main()
