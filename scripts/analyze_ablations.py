"""Compare ablation runs of the eval pipeline.

The runner (``kb_rag.evaluation.runner``) writes one CSV per configuration.
Typical workflow:

    python -m kb_rag.evaluation.runner --no-bm25 --no-rerank
    python -m scripts.analyze_ablations --label dense-only \\
        --csv eval_results/run_<timestamp>.csv

Then run the full hybrid setting and compare:

    python -m kb_rag.evaluation.runner
    python -m scripts.analyze_ablations \\
        --label hybrid --csv eval_results/run_<timestamp>.csv \\
        --compare eval_results/run_<dense-only-timestamp>.csv

The analyzer prints retrieval hit@k, MRR@k, judge faithfulness/correctness,
and refusal accuracy side-by-side. Categories (cards, loans, …) are reported
in the by-category breakdown when both runs share the same set.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


def _safe_mean(series: pd.Series) -> float:
    return float(series.dropna().mean()) if not series.dropna().empty else 0.0


def summarize(df: pd.DataFrame) -> dict:
    answerable = df[~df["unanswerable"]]
    unanswerable = df[df["unanswerable"]]
    return {
        "n_total": len(df),
        "n_answerable": len(answerable),
        "n_unanswerable": len(unanswerable),
        "hit@k": _safe_mean(answerable["hit"]),
        "mrr@k": _safe_mean(answerable["reciprocal_rank"]),
        "faithfulness": _safe_mean(answerable["faithfulness"]),
        "correctness": _safe_mean(answerable["correctness"]),
        "refusal_accuracy": _safe_mean(unanswerable["refusal_correct"]) if not unanswerable.empty else None,
    }


def _format(label: str, metrics: dict, top_k: int) -> str:
    refusal = (
        f"{metrics['refusal_accuracy']:.1%}"
        if metrics["refusal_accuracy"] is not None
        else "n/a"
    )
    return (
        f"  {label:<14} "
        f"hit@{top_k}={metrics['hit@k']:.1%}  "
        f"mrr@{top_k}={metrics['mrr@k']:.3f}  "
        f"faith={metrics['faithfulness']:.2f}  "
        f"corr={metrics['correctness']:.2f}  "
        f"refusal={refusal}  "
        f"(n={metrics['n_total']})"
    )


def _per_category(df: pd.DataFrame) -> pd.DataFrame:
    answerable = df[~df["unanswerable"]]
    if answerable.empty:
        return pd.DataFrame()
    by_cat = answerable.groupby("category").agg(
        n=("id", "count"),
        hit=("hit", "mean"),
        mrr=("reciprocal_rank", "mean"),
        faith=("faithfulness", "mean"),
        corr=("correctness", "mean"),
    )
    return by_cat


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--label", required=True, help="display name for this run")
    parser.add_argument("--csv", required=True, type=Path, help="eval CSV produced by the runner")
    parser.add_argument("--top-k", type=int, default=None,
                        help="only needed for the report header (default: 6)")
    parser.add_argument("--compare", type=Path, default=None,
                        help="second CSV to compare against; pass --label too")
    parser.add_argument("--compare-label", default="compare",
                        help="display name for the comparison run")
    args = parser.parse_args()

    df_a = pd.read_csv(args.csv)
    metrics_a = summarize(df_a)
    top_k = args.top_k or 6

    print(f"\n=== Ablation summary (top_k={top_k}) ===")
    print(_format(args.label, metrics_a, top_k))

    cat_a = _per_category(df_a)
    if not cat_a.empty:
        print(f"\nBy category ({args.label}):")
        print(cat_a.to_string(float_format=lambda x: f"{x:.2f}"))

    if args.compare is not None:
        df_b = pd.read_csv(args.compare)
        metrics_b = summarize(df_b)
        print(_format(args.compare_label, metrics_b, top_k))

        # Per-category delta when both runs share the category set
        cat_b = _per_category(df_b)
        common = sorted(set(cat_a.index) & set(cat_b.index))
        if common:
            print("\nDelta per category (this - other, positive = this is better):")
            for cat in common:
                row_a, row_b = cat_a.loc[cat], cat_b.loc[cat]
                d_hit = row_a["hit"] - row_b["hit"]
                d_mrr = row_a["mrr"] - row_b["mrr"]
                d_corr = row_a["corr"] - row_b["corr"]
                print(f"  {cat:<10} hit {d_hit:+.1%}  mrr {d_mrr:+.3f}  corr {d_corr:+.2f}")

        # Headline deltas
        d_hit = metrics_a["hit@k"] - metrics_b["hit@k"]
        d_mrr = metrics_a["mrr@k"] - metrics_b["mrr@k"]
        d_corr = metrics_a["correctness"] - metrics_b["correctness"]
        d_faith = metrics_a["faithfulness"] - metrics_b["faithfulness"]
        print("\nHeadline delta (this - other):")
        print(f"  hit@{top_k} {d_hit:+.1%}   mrr {d_mrr:+.3f}   "
              f"faith {d_faith:+.2f}   corr {d_corr:+.2f}")


if __name__ == "__main__":
    main()
