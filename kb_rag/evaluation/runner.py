"""Evaluation runner: golden questions -> retrieval + generation scores.

Usage:
    python -m kb_rag.evaluation.runner                 # full dataset
    python -m kb_rag.evaluation.runner --limit 5       # smoke run
    python -m kb_rag.evaluation.runner --dataset PATH

Per question it measures:
  - retrieval hit@k / reciprocal rank against expected source URLs
  - answer faithfulness & correctness (LLM-as-judge, 1-5)
  - refusal correctness for unanswerable questions
Results go to eval_results/run_<timestamp>.csv with a console summary.
"""

from __future__ import annotations

import argparse
from datetime import datetime

import pandas as pd
from tqdm import tqdm

from kb_rag.config import get_settings
from kb_rag.evaluation.dataset import GoldenQuestion, load_dataset
from kb_rag.evaluation.generation_metrics import judge_answer, looks_like_refusal
from kb_rag.evaluation.retrieval_metrics import (
    first_relevant_rank,
    hit_at_k,
    mean,
    reciprocal_rank,
)
from kb_rag.rag.llm import DeepSeekClient
from kb_rag.rag.pipeline import get_pipeline


def evaluate_question(
    item: GoldenQuestion,
    pipeline,
    judge_client: DeepSeekClient,
    judge_model: str,
    top_k: int,
) -> dict:
    answer = pipeline.answer(item.question, lang=None, section=None, top_k=top_k, stream=False)
    # a source counts as matching when either its final URL or the original
    # sitemap slug contains an expected fragment (kapitalbank pages 301 to
    # birbank.az, so both identifiers are meaningful)
    retrieved_urls = [
        u for s in answer.sources for u in (s.url, s.source_url) if u
    ]
    rank = first_relevant_rank(retrieved_urls, item.expected_sources) if item.expected_sources else None

    row = {
        "id": item.id,
        "category": item.category,
        "lang": item.lang,
        "unanswerable": item.unanswerable,
        "hit": hit_at_k(rank, top_k),
        "reciprocal_rank": reciprocal_rank(rank, top_k),
        "first_hit_rank": (rank + 1) if rank is not None else None,
        "n_sources_returned": len(retrieved_urls),
        "refused": looks_like_refusal(answer.text),
        "answer": answer.text or "",
    }

    if item.unanswerable:
        row["refusal_correct"] = bool(row["refused"])
        # unanswerables are scored on refusal behavior, not on judge rubric
        row["faithfulness"] = None
        row["correctness"] = None
        row["judge_rationale"] = ""
    else:
        # judge must see the SAME passage texts the generator saw — scoring
        # against bare URLs/breadcrumbs would flag every concrete detail as
        # an invention
        result = judge_answer(
            judge_client,
            judge_model=judge_model,
            question=item.question,
            context=answer.context or "",
            answer=row["answer"],
            reference=item.reference_answer,
        )
        row["refusal_correct"] = None
        row["faithfulness"] = result.faithfulness
        row["correctness"] = result.correctness
        row["judge_rationale"] = result.rationale
    return row


def summarize(df: pd.DataFrame, top_k: int) -> str:
    answerable = df[~df["unanswerable"]]
    unanswerable = df[df["unanswerable"]]
    lines = [
        f"\n=== RAG evaluation — {len(df)} questions ===",
        f"Retrieval hit@{top_k}: {answerable['hit'].mean():.1%}"
        f"   MRR@{top_k}: {mean(answerable['reciprocal_rank'].tolist()):.3f}",
    ]
    if len(answerable):
        lines.append(
            f"Judge faithfulness (1-5): {answerable['faithfulness'].mean():.2f}"
            f"   correctness: {answerable['correctness'].mean():.2f}"
        )
    if len(unanswerable):
        lines.append(f"Refusal accuracy on unanswerables: {unanswerable['refusal_correct'].mean():.1%}")

    by_cat = answerable.groupby("category").agg(
        n=("id", "count"),
        hit=(("hit"), "mean"),
        faithfulness=("faithfulness", "mean"),
        correctness=("correctness", "mean"),
    )
    if not by_cat.empty:
        lines.append("\nBy category:\n" + by_cat.to_string(float_format=lambda x: f"{x:.2f}"))
    return "\n".join(lines)


def run(dataset_path=None, limit: int | None = None, top_k: int | None = None,
        no_bm25: bool = False, no_rerank: bool = False) -> pd.DataFrame:
    settings = get_settings()
    updates: dict = {}
    if no_bm25:
        updates["enable_bm25"] = False
    if no_rerank:
        updates["rerank_model"] = None
    if updates:
        # ablation mode: mutate before get_pipeline() builds the singleton
        settings.retrieval = settings.retrieval.model_copy(update=updates)
    k = top_k or settings.retrieval.top_k
    items = load_dataset(dataset_path or settings.golden_set_path)
    if limit:
        items = items[:limit]

    pipeline = get_pipeline()
    judge_client = DeepSeekClient(settings)

    rows = [
        evaluate_question(item, pipeline, judge_client, settings.llm.judge_model, top_k=k)
        for item in tqdm(items, desc="evaluating", unit="q")
    ]
    df = pd.DataFrame(rows)

    settings.eval_results_dir.mkdir(parents=True, exist_ok=True)
    out_path = settings.eval_results_dir / f"run_{datetime.now():%Y%m%d_%H%M%S}.csv"
    df.to_csv(out_path, index=False)
    print(summarize(df, k))
    print(f"\nFull results saved to {out_path}")
    return df


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", default=None, help="path to qa.yaml")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--top-k", type=int, default=None)
    parser.add_argument("--no-bm25", action="store_true",
                        help="ablation: dense retrieval only")
    parser.add_argument("--no-rerank", action="store_true",
                        help="ablation: skip cross-encoder re-scoring")
    args = parser.parse_args()
    run(dataset_path=args.dataset, limit=args.limit, top_k=args.top_k,
        no_bm25=args.no_bm25, no_rerank=args.no_rerank)


if __name__ == "__main__":
    main()
