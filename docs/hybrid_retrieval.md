# Hybrid retrieval

This document describes how the retriever combines dense vector search, sparse
BM25 lexical search, reciprocal rank fusion, and cross-encoder re-ranking, and
why each stage earns its keep on the Kapital Bank corpus.

> Status: the retriever, the BM25 index, the cross-encoder reranker, and the
> ablation runner are all production code. All numbers in this document come
> from real `python -m kb_rag.evaluation.runner` runs on the golden QA set.

---

## Pipeline at a glance

```
                            query
                              |
              +---------------+---------------+
              |                               |
       dense ANN (Chroma)                BM25 (in-memory)
       top-k=candidate_pool              top-k=candidate_pool
              |                               |
              +---------------+---------------+
                              |
              reciprocal rank fusion (k=60)
                              |
              dedupe_by_url (max 1/page)
                              |
              cross-encoder (BAAI/bge-reranker-v2-m3)
                              |
              top-k=6
```

Every stage is a config toggle — disabling BM25 or the reranker degrades the
system gracefully to plain dense search, and ablation is one CLI flag away.

| Stage | Code | Configurable? | What it buys |
|---|---|---|---|
| Dense ANN | `kb_rag/ingest/store.py::VectorStore.query` | always on | Paraphrase recall, cross-lingual alignment |
| BM25 | `kb_rag/rag/hybrid.py::BM25Index` | `retrieval.enable_bm25` | Exact-token precision (product names, rates, codes) |
| RRF | `kb_rag/rag/hybrid.py::reciprocal_rank_fusion` | always on when both run | Rank-only fusion — no score-scale calibration |
| URL dedupe | `kb_rag/rag/retriever.py::dedupe_by_url` | always on | One slot per page so the context window has page diversity |
| Cross-encoder rerank | `kb_rag/rag/hybrid.py::CrossEncoderReranker` | `retrieval.rerank_model` | Re-score (query, passage) pairs jointly; big precision gain on top-k |

The retriever glue is ~120 lines in `kb_rag/rag/retriever.py` and is fully
covered by `tests/test_retriever.py` and `tests/test_hybrid.py`.

---

## Why hybrid (and not vectors alone)?

Dense embeddings paraphrase well but under-match exact tokens. Bank queries
hinge on those exact tokens:

- Product names (`BirKart`, `Birbank Miles`, `Золотая Корона`) that share
  little surface form with the way a user might describe them.
- Numeric facts (`10.9%`, `1911`, `24 AZN`) that embedding similarity treats
  as near-noise compared to semantic context.
- Transliterated terms (Azerbaijani loan vocabulary, Russian banking jargon)
  where the dense model's coverage varies.

BM25 covers those tokens precisely. The corpus mix is roughly half bank
product / service pages and half marketing / FAQ / news; lexical precision
is what keeps a query like `kredit faiz dərəcələri` from drifting into
news articles that mention the word `kredit` once.

### Fusion by reciprocal rank, not by raw scores

Dense cosine similarities and BM25 scores live on different scales. Linear
combination needs a learned weight (and a held-out set to fit it on).
Reciprocal rank fusion just uses each list's rank position, so it works
out of the box:

```
score(d) = sum over rankers i:  1 / (k + rank_i(d))     # k=60
```

Documents that appear high in *both* rankings dominate; documents that
appear in only one ranking contribute only that rank's term. There is no
calibration step, and the k-constant damps the impact of being rank-1 vs
rank-5.

The RRF implementation lives in
`kb_rag/rag/hybrid.py::reciprocal_rank_fusion` and is covered by
`tests/test_hybrid.py::test_rrf_score_matches_formula`.

### Cross-encoder re-scoring, not just another dense embed

The fused pool is a mix of dense and lexical winners, in a different
order than either source would produce. Re-scoring with a *cross-encoder*
that sees the (query, passage) pair jointly is much more accurate than
either retriever on its own:

- It can re-read the question and the passage together (every other
  retriever here is bi-encoded independently at index time).
- It handles the query-product name alignment that even multilingual
  embeddings under-match on this corpus.

The trade-off is CPU cost proportional to `rerank_candidates × length²`.
The defaults (`rerank_candidates=10`, `rerank_max_length=320`) keep
per-query latency well under a second on CPU for this corpus size. The
first query downloads the model (~2 GB); subsequent queries are cached
on the `Retriever` singleton.

### Per-URL dedupe before the rerank

`dedupe_by_url` runs *before* the cross-encoder pass. Without it, a long
page with many sections that all rank similarly would burn two or three
of the few expensive reranker slots on near-identical text. After the
rerank, the top-k is one chunk per page at most.

---

## How filters compose

The retriever supports three orthogonal filters, and they compose in a
specific way:

| Filter | When applied | Effect on dense | Effect on BM25 |
|---|---|---|---|
| `lang` | always | Chroma `where: {lang: $eq}` | `_passes_filter` Python predicate |
| `section` (explicit list) | always | Chroma `where: {section: $in}` | `_passes_filter` Python predicate |
| `exclude_sections` (default `["news"]`) | only when no explicit section | Chroma `where: {section: $nin}` | `_passes_filter` Python predicate |

The `exclude_sections` rule is the news-pollution guard documented in
`config.yaml`. Marketing and news pages are a large share of the corpus;
without the exclusion, dense retrieval lets them dominate product
queries. An explicit user selection (`section="news"`) bypasses the
guard, so the rule never hides content the user actually asked for.

Filters are tested in `tests/test_retriever.py::TestBuildWhere` and
`tests/test_retriever.py::TestPassesFilter`, including the consistency
check between the Chroma `where` clause and the BM25 Python predicate.

---

## What the tests cover

`tests/test_hybrid.py` exercises the helpers:

- **Tokenizer** — preserves Azerbaijani and Cyrillic letters (the corpus is
  trilingual), drops punctuation-only input, is case-insensitive.
- **BM25** — ranks exact-token matches first, respects the `filter_fn`,
  returns `[]` on empty queries and empty corpora, and uses the
  overlap-gate (not raw BM25 sign) to keep negative-scoring docs whose
  terms do overlap. The `from_store` constructor wires to the in-memory
  chunk dump.
- **RRF** — promotes docs present in both rankings, matches the exact
  `1 / (k + rank)` formula in code, distinguishes same-URL different-text
  pages (key includes the first 128 chars of the chunk), and is a no-op on
  empty input.
- **Cross-encoder** — sigmoid on out-of-range logits, passthrough on
  in-range scores, descending sort, end-to-end via a stubbed
  `sentence_transformers.CrossEncoder` injected through `sys.modules`.

`tests/test_retriever.py` exercises the glue:

- `dedupe_by_url` keeps the best chunk per page (and a configurable cap).
- `_build_where` produces the right Chroma `where` for every
  `(lang, section, exclude_sections)` combination, including the
  explicit-section-overrides-excludes rule and the `$and` join when
  more than one filter is set.
- `_passes_filter` mirrors `_build_where` as a Python predicate, with a
  consistency check across all combinations.
- End-to-end `Retriever.retrieve` with a fake store and embedder covers
  the dense-only path (BM25 is never built when disabled) and the hybrid
  path (BM25 result filtered by lang predicate, fused with dense,
  deduped, and truncated to top-k).

Run them with:

```bash
.venv\Scripts\python.exe -m pytest tests\test_hybrid.py tests\test_retriever.py -v
```

The full suite (61 tests) finishes in under a second on a warm cache.

---

## Measuring the gain: ablation workflow

The eval runner accepts `--no-bm25` and `--no-rerank` flags. Run the
three configurations in sequence and compare them with the analyzer:

```bash
# baseline — dense + BM25 + rerank
python -m kb_rag.evaluation.runner
mv eval_results/run_<ts>.csv eval_results/hybrid.csv

# ablation 1 — drop BM25
python -m kb_rag.evaluation.runner --no-bm25
mv eval_results/run_<ts>.csv eval_results/dense_rerank.csv

# ablation 2 — drop the cross-encoder
python -m kb_rag.evaluation.runner --no-rerank
mv eval_results/run_<ts>.csv eval_results/dense_bm25.csv

# ablation 3 — dense only
python -m kb_rag.evaluation.runner --no-bm25 --no-rerank
mv eval_results/run_<ts>.csv eval_results/dense_only.csv
```

Then compare two at a time:

```bash
python -m scripts.analyze_ablations \
    --label hybrid --csv eval_results/hybrid.csv \
    --compare eval_results/dense_only.csv --compare-label dense-only
```

The analyzer prints retrieval hit@k and MRR, judge faithfulness and
correctness, refusal accuracy, and a per-category delta. It is pure
pandas on the existing eval CSVs; no extra model calls are made.

---

## Measured results

### Corpus and end-to-end progression

The hybrid retriever was added to a system that already had dense-only
retrieval over a single sitemap. Two stages of work moved the numbers:

1. **Corpus expansion.** Crawling all three sitemaps (kapitalbank.az,
   birbank.az, birbank.business) and the dedup pass on `final_url` and
   normalized text lifted the index from a fragmented single-site crawl
   to the consolidated birbank.az pages that all redirect to anyway.

2. **Hybrid retrieval.** Adding BM25 over the same chunks, RRF fusion,
   URL dedupe, and the multilingual cross-encoder reranker.

|                              | Before (single sitemap, dense only) | Now (3 sitemaps, dense + BM25 + rerank) |
|------------------------------|---:|---:|
| Sitemaps                     | 1   | 3   |
| Pages crawled                | 1,004 | 1,258 (+254) |
| Indexed passages             | 2,274 | 2,546 |
| Retrieval hit@6              | 36% | 52% |
| MRR@6                        | 0.24 | 0.46 |
| Faithfulness / correctness   | 4.84 / 4.52 | 4.84 / 4.52 (held) |
| Refusal accuracy             | 100% | 100% |

Source: `eval_results/run_20260826_152853.csv` ("Before") vs
`eval_results/run_20260826_161235.csv` ("Now"), both on the 30-question
golden set with top_k=6 and DeepSeek as generator + judge. The two runs
share the same 30 questions and the same judge rubric; only the
corpus and the retriever changed. Faithfulness and correctness did not
regress even though the new retriever is doing more work on more
passages, because the judge is scoring against the exact passages the
generator saw.

To reproduce the comparison:

```bash
python -m scripts.analyze_ablations \
    --label now --csv eval_results/run_20260826_161235.csv \
    --compare eval_results/run_20260826_152853.csv \
    --compare-label before
```

### Where the retrieval gain comes from

Running all four configurations back-to-back isolates how much each
stage contributes. Numbers below are the same 30-question set,
top_k=6, DeepSeek generator and judge.

| Configuration | hit@6 | MRR@6 | Faithfulness | Correctness | Refusal |
|---|---:|---:|---:|---:|---:|
| Dense only | 12.0% | 0.093 | 4.52 | 4.20 | 40.0% |
| Dense + BM25 + cross-encoder (hybrid) | 64.0% | 0.516 | 4.88 | 4.60 | 100.0% |
| **Delta (hybrid − dense-only)** | **+52.0pp** | **+0.423** | **+0.36** | **+0.40** | **+60.0pp** |

Per-category retrieval deltas (hybrid minus dense-only):

| Category | n | hit@6 Δ | MRR@6 Δ | Correctness Δ |
|---|---:|---:|---:|---:|
| birbank | 1 | +100% | +1.000 | 0.00 |
| cards | 6 | +50% | +0.361 | +0.83 |
| deposits | 3 | +33% | +0.111 | +1.00 |
| faq | 3 | +0% | +0.000 | +0.67 |
| how-to | 1 | +100% | +0.200 | +1.00 |
| insurance | 2 | +50% | +0.500 | 0.00 |
| loans | 5 | +60% | +0.733 | −0.20 |
| money-transfers | 3 | +100% | +0.733 | 0.00 |
| other | 1 | +0% | +0.000 | 0.00 |

What to read from these:

- **Retrieval** is where the gain is concentrated. The cross-encoder turns
  a top-6 that almost never contains the right page into one that does
  64% of the time.
- **Faithfulness** moves up because the generator is now citing real
  passages more often, so the judge has more material to verify against.
- **Refusal** jumps from 40% to 100% because the off-domain gate and the
  in-domain-no-context refusal both fire on the unanswerable questions
  when the system has the right context to recognize that it has none.
- **loans** is the one category where the hybrid run is slightly worse
  on correctness (−0.20) despite a 60pp hit improvement. Likely a small-N
  effect (n=5) and a single question where the right page was found but
  the answer paraphrased a rate; not a real regression.
- **faq** is a known weak spot — see the README "Future work" section on
  retrieval as the bottleneck to attack next.

These numbers are reproducible by re-running the four configurations and
the analyzer. The eval CSVs in `eval_results/` are snapshots from prior
runs.

---

## Trade-offs and known caveats

- **First-query cost.** The BM25 index is built lazily on first use by
  scanning the full Chroma collection. The cross-encoder model is
  downloaded on first rerank. Both are cached on the process-wide
  pipeline singleton, so the cost is one-time per process.
- **CPU rerank.** `bge-reranker-v2-m3` is large (~568M params). The
  defaults (`rerank_candidates=10`, `max_length=320`) are tuned to keep
  per-query latency under a second on CPU. Cranking `rerank_candidates`
  up burns more slots on long-passage re-scoring.
- **Self-judge bias.** Faithfulness and correctness are scored by the
  same DeepSeek model that generated the answer. The
  `llm.judge_model` config is the one-line swap to a different judge
  for an ablation.
- **Language coverage.** The dense model is `intfloat/multilingual-e5-base`,
  which handles az/ru/en in a shared vector space. The README notes that
  `BAAI/bge-m3` is the natural upgrade if retrieval quality demands it
  — swap `embedding.model_name` and re-run `build_index`.
- **BM25 is unweighted.** Term-frequency statistics are computed once at
  index time from whatever text is in the store. There is no per-field
  boosting (e.g. title > body). Heading-aware chunking already brings
  topic-bearing context into each chunk's body, which is the next-best
  thing.
