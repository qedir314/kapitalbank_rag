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
        (retrieval.query_expansion, default off — lost the 2.2 A/B)
        one LLM call -> az/en/ru variants, original always first
                              |
                    for each query variant:
              +---------------+---------------+
              |                               |
       dense ANN (Chroma)                BM25 (in-memory)
       top-k=candidate_pool              top-k=candidate_pool
       (bge-m3 embeddings)               (retrieval.morph_tokens:
                                          + transliteration + stems)
              |                               |
              +---------------+---------------+
                              |
              reciprocal rank fusion (k=60) — dense x BM25,
              then across variants when expansion is on
                              |
              dedupe_by_url (max 1/page)
                              |
              cross-encoder (BAAI/bge-reranker-v2-m3)
              — always scored against the ORIGINAL query
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
| Query expansion | `kb_rag/rag/query_expansion.py::QueryExpander` | `retrieval.query_expansion` (off — lost A/B, see below) | Cross-language lexical recall via az/en/ru rewrites |
| Morphology tokens | `kb_rag/rag/hybrid.py::tokenize(morph=True)` | `retrieval.morph_tokens` | Additive Cyrillic→Latin transliteration + az suffix stems for BM25 |

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

The trade-off is compute proportional to `rerank_candidates × length²`.
The defaults (`rerank_candidates=20`, `rerank_max_length=320`) were
chosen by the Phase 1 pool ablation (plan 1.4: 10→20 won ~2 pp of hit@6)
and stay ~1 s/query on CPU; the reranker runs on CUDA when a GPU is
present. The first query downloads the model (~2 GB); subsequent queries
are cached on the `Retriever` singleton.

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

The full suite (93 tests) finishes in a few seconds on a warm cache.

---

## Measuring the gain: ablation workflow

Beyond `--no-bm25`/`--no-rerank`, the runner (and `build_index`) honor a
`KB_CONFIG` environment variable: point it at an alternate YAML and the
whole process runs with that config, leaving `config.yaml` untouched.
The measured ablations below map to checked-in configs:

| Config | What it changes |
|---|---|
| `config.ablation_e5.yaml` | `multilingual-e5-base` (512-token cap), own Chroma dir `data/chroma_e5` |
| `config.ablation_qe.yaml` | `query_expansion: true`, reuses the main index |
| `config.ablation_morph.yaml` | `morph_tokens: true`, reuses the main index |

```bash
# e5-base index (separate Chroma dir — the main index is never touched)
KB_CONFIG=config.ablation_e5.yaml python -m scripts.build_index --reset
KB_CONFIG=config.ablation_e5.yaml python -m kb_rag.evaluation.runner

# query-side toggles need no rebuild — they run at query time
KB_CONFIG=config.ablation_qe.yaml python -m kb_rag.evaluation.runner
```

Run the three configurations in sequence and compare them with the analyzer:

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

|                              | Before (single sitemap, dense only) | Then (2026-08-26: 3 sitemaps, dense + BM25 + rerank) | Now (2026-08-27: + bge-m3, FAQ fix, taxonomy, pool 20) |
|------------------------------|---:|---:|---:|
| Sitemaps                     | 1   | 3   | 3 |
| Pages crawled                | 1,004 | 1,258 (+254) | 1,258 |
| Indexed passages             | 2,274 | 2,546 | 2,565 |
| Retrieval hit@6              | 36% | 52% | **72%** |
| MRR@6                        | 0.24 | 0.46 | **0.555** |
| Faithfulness / correctness   | 4.84 / 4.52 | 4.84 / 4.52 (held) | 4.88 / 4.56 |
| Refusal accuracy             | 100% | 100% | 100% |

Source: `eval_results/run_20260826_152853.csv` ("Before") vs
`eval_results/run_20260826_161235.csv` ("Then") vs
`eval_results/run_20260827_131038.csv` ("Now" — the Phase 1 + bge-m3
configuration; per-run detail in [improvement_plan.md](improvement_plan.md)),
all on the 30-question
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
top_k=6, DeepSeek generator and judge, **e5-base embeddings era
(2026-08-26)** — superseded for the headline numbers by the Phase 1/2
results in the next sections, but the stage-by-stage deltas still hold.

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
- **faq** was the known weak spot of this era (0/3 across languages) —
  fixed in Phase 1 by the `__NEXT_DATA__` FAQ re-extraction; see the
  embedding-model ablation below for the current faq numbers.

These numbers are reproducible by re-running the four configurations and
the analyzer. The eval CSVs in `eval_results/` are snapshots from prior
runs.

### Embedding model: bge-m3 vs e5-base (plan 2.1, 2026-08-27)

The upgrade the e5-era section above anticipated, finally measured
head-to-head. Identical inputs on both sides: same 2,565-chunk corpus
(post Phase-1 FAQ fix and section taxonomy), same 30-question golden
set, same hybrid config (`candidate_pool=60`, `rerank_candidates=20`,
`bge-reranker-v2-m3`). Only `embedding.model_name` differs — e5 is
capped at 512 positional tokens, bge-m3 embeds 1,024 of each chunk.

| | **BAAI/bge-m3** (current) | intfloat/multilingual-e5-base |
|---|---:|---:|
| Retrieval hit@6 | **72%** | 64% |
| MRR@6 | **0.555** | 0.523 |
| hit@6 by lang (az / en / ru) | 67 / 80 / 67% | 56 / 70 / 67% |
| faq category | **2/3** | 0/3 |
| Faithfulness / correctness | 4.88 / 4.56 | 4.88 / 4.56 |
| Refusal accuracy | 100% | 100% |
| Run | `run_20260827_131038` | `run_20260827_141830` |

The per-question diff is unusually clean: **all of the hit@6 gap lives
in the two FAQ questions** (`az-faq-001`, `en-faq-001`) that bge-m3
recovers and e5 never hits. The rebuilt FAQ pages are long Q&A listings
— e5's 512-token window truncates most of a group's questions out of
the embedding entirely, while bge-m3 sees them. No question is *worse*
under bge-m3. The switch is kept; e5-base remains one config line (and
its own index dir) away via `config.ablation_e5.yaml`.

### Query-side experiments (plan 2.2 + 2.3, 2026-08-27) — both rejected

| Configuration | hit@6 | MRR@6 | Refusal | Verdict |
|---|---:|---:|---:|---|
| Baseline (bge-m3 hybrid) — `…131038` | **72%** | 0.555 | 100% | — |
| + LLM query expansion — `…142300` | 68% | 0.563 | 80% | rejected |
| + morphology-aware BM25 tokens — `…142724` | 68% | 0.536 | 100% | rejected |

**Query expansion** (`retrieval.query_expansion`,
`kb_rag/rag/query_expansion.py`): one temperature-0 DeepSeek call rewrites
the question into az/en/ru variants; each variant runs the full dense +
BM25 + RRF path; the variant candidate lists are fused by a second RRF
before dedupe — the reranker still scores against the *original* query.
Flips: gained `en-deposits-001` (the one whose expected label is an
English `deposits` slug), lost `az-faq-001` and `ru-cards-001`; per-language
hit went to en 90 / az 56 / ru 50% — a *wider* language gap than baseline
(80 / 67 / 67), the opposite of the intent — and the extra cross-language
context cost one refusal (`az-unans-002` answered noise instead of declining).
Reading: bge-m3's shared vector space already does the cross-language
alignment; expanding mostly adds candidate noise that dilutes the fused
ranking. Code + toggle kept, off by default.

**Morphology tokens** (`retrieval.morph_tokens`, `tokenize(morph=True)` in
`hybrid.py`): additive only — every surface token is kept, and Cyrillic
tokens additionally contribute a Latin transliteration, while any token
may contribute one conservative Azerbaijani suffix strip (stem ≥ 4 chars,
longest-suffix-first). No index rebuild needed (BM25 is in-memory). Flips:
exactly one — `ru-deposits-001` squeezed out of top-6; the cross-script
recall wins the design was for ("Карта" ↔ "kart") never converted into a
golden-set hit, while the extra tokens shifted IDF/ranking at the margins.
Code + tests + toggle kept, off by default.

The two rejections share a story: the remaining misses are **not
query-side**. They are the corpus gaps and golden-label defects diagnosed
in Phase 1 — the 80% target belongs to Phase 3 measurement rigor and
corpus work, not to more retrieval cleverness. (Same pattern as Phase 1's
breadcrumb regression: this pipeline is at the point where plausible
"improvements" reliably fail to measure.)

---

## Evaluation rigor (Phase 3)

Everything above was measured on a 30-question set with no ground truth.
Phase 3 rebuilt the measuring instrument — the numbers it produced after
that point are the defensible ones.

### The 74-question set (`data/golden/qa.yaml`)

62 answerable / 12 unanswerable, validated by
`python -m scripts.validate_golden_set` (URL resolution against the corpus,
reference coverage, schema, ≥4-per-category stratification, exit code for CI):

| category | az | en | ru |
|---|---:|---:|---:|
| cards | 5 | 4 | 3 |
| loans | 3 | 4 | 2 |
| deposits | 2 | 3 | 3 |
| money-transfers | 2 | 3 | 2 |
| insurance | 2 | 2 | 2 |
| faq | 3 | 3 | 2 |
| how-to | 1 | 2 | 1 |
| birbank | 1 | 2 | 1 |
| other | 2 | 1 | 1 |
| **answerable** | **21** | **24** | **17** |

Design elements: near-miss confusables (`az-cards-002` asks about a "Birbank
Cashback *debit* card" — Cashback is the installment line; only Visa Infinite
Cashback is a debit card), stale-rate traps (`en-loans-004` couples a campaign
figure with its disqualification conditions), KB-boundary items (the lost-card
FAQ trio, where the honest limited answer is the reference), competitor/
private-data/future-rate off-domain questions, and 3 multi-turn `history`
items. Every answerable question carries a `reference_answer` verified
against `data/raw/pages.jsonl`.

**Composition caveat:** the old 72% and the new 91.9% are not comparable —
the old set contained questions the corpus cannot support (gaps + mislabels,
now fixed or reclassified). On the 23 surviving original questions, with
corrected labels, hit@6 is 87%.

### Canonical results (`run_20260827_150822`, tag `phase3-canonical-74q`)

| Metric | Value | Note |
|---|---:|---|
| Retrieval hit@6 | **91.9%** (MRR 0.808) | 57/62 answerable |
| Judge faithfulness | 4.85 | vs numbered context, as before |
| Judge correctness | 4.55 | **now anchored to reference answers** |
| Refusal accuracy | 91.7% | 11/12; `az-unans-003` answered app steps instead of declining |
| Citation validity | 100% | every `[n]` marker points at a real passage |
| Citation support | 83% | citing sentences lexically overlap the cited passage (`citation_metrics.py`, bag-of-words ≥ 0.2) |
| Citation coverage | 60% | share of answer sentences carrying a marker |

Reference-anchored correctness immediately caught what context-only judging
missed: eight rows of "honest — the context lacks it" answers now score 3
when the fact actually *is* on a page retrieval didn't surface — converting
silent retrieval gaps into visible generation-side symptoms (e.g.
`az-cards-004`: the debit card's free price is on the debet page; the model
correctly said "not in context" and the judge correctly said "incomplete").
The multi-turn item `mt-az-cards-001` also exposes a real gap: the retriever
never sees chat history, so a bare follow-up ("and what does the card cost?")
retrieves nothing — conversational query condensing is the fix (Phase 4.4).

### Seed variance (`--seeds 3`, runs `…151438` / `…151942` / `…152443`)

| Metric | mean | std |
|---|---:|---:|
| hit@6 | 0.919 | 0.000 |
| MRR@6 | 0.808 | 0.000 |
| faithfulness | 4.860 | 0.019 |
| correctness | 4.559 | 0.025 |
| refusal | 0.917 | 0.000 |

Retrieval is deterministic (identical across seeds, as it must be); the
LLM-side metrics wobble by ±0.02–0.03 even at temperature 0 — small enough
that CI can assert on means, and exactly why single-run judge deltas below
~0.05 should not be treated as signal.

### Independent judge ablation (`--rejudge`, `rejudge_20260827_154931_deepseek-reasoner`)

The 62 stored canonical answers were re-scored with `deepseek-reasoner` (R1
family — the API provider is DeepSeek for both, so this is a cross-family but
not cross-vendor bound on self-bias; a true third-party judge is one key +
`llm.judge_model` away).

| Dimension | Exact agreement | Mean shift (R1 − V3) | Pearson | R1 stricter on |
|---|---:|---:|---:|---|
| Faithfulness | 44/59 (75%) | −0.27 | 0.23* | hedged-but-unsupported sentences |
| Correctness | 44/59 (75%) | +0.05 | 0.69 | engagement with false premises |

\* faithfulness correlation is depressed by score saturation (mean 4.85/5 —
almost everything is a 5, so there is little variance to correlate); the
disagreement rows carry the real signal.

| Question | V3 (stored) | R1 | Who is right, reading the reference |
|---|---|---|---|
| `az-cards-002` (fake "Cashback debit" premise) | faith 5 / corr 3 | **faith 1 / corr 1** | R1 — the answer entertained the false premise; the reference demands calling out the confusion |
| `en-cards-004` (ATM commission, flat 2 AZN exists on the landing) | faith 2 / corr 2 | faith 5 | split — V3 called invented framing, R1 rewarded staying-in-context; neither matched the reference-complete fact |
| `en-cards-002` (premium cards) | corr 3 | corr 5 | R1 — accepted the context-sourced list V3 found "incomplete vs reference" |
| `en-faq-001`, `az-transfers-001`, `mt-az-cards-001` | faith 5 | faith 3 | R1 — demotes confident hedging on KB-boundary answers |

Verdict: the two judges agree at the headline level (faithfulness 4.85 vs
4.58, correctness 4.55 vs 4.60 means — differences inside two seed-stds), so
self-bias is **bounded but real at the row level**: R1 catches premise-confusion
answers V3 lets pass. `deepseek-chat` stays the default judge (59/62 parsable at
1,500-token budget after the reasoning-budget fix vs near-total empty-output
failure at 300, ~40× faster, and its leniency is now *documented* rather than
assumed); R1 remains the strictness bound quoted here. The 3 remaining parse
failures are recorded in `rejudge_error` columns, not silently dropped —
exactly the plan 3.5 robustness behavior.

### Traceability

Each run CSV has a `run_*.manifest.json`: full config snapshot, git SHA +
dirty flag, `KB_CONFIG` used, dataset path + sha256, top_k, seed, judge
model, and error counts. The canonical CSVs + manifests cited in this doc
and in README are committed (`.gitignore` negations); `python -m
kb_rag.evaluation.runner --tag NAME` labels new runs. Ad-hoc runs remain
untracked.

---

## Trade-offs and known caveats

- **First-query cost.** The BM25 index is built lazily on first use by
  scanning the full Chroma collection. The cross-encoder model is
  downloaded on first rerank. Both are cached on the process-wide
  pipeline singleton, so the cost is one-time per process.
- **Rerank cost.** `bge-reranker-v2-m3` is large (~568M params). The
  defaults (`rerank_candidates=20`, `max_length=320`) keep per-query
  latency ~1 s on CPU (sub-second on GPU). Cranking
  `rerank_candidates` up burns more slots on long-passage re-scoring.
- **Self-judge bias.** Faithfulness and correctness are scored by the
  same DeepSeek model that generated the answer. The
  `llm.judge_model` config is the one-line swap to a different judge
  for an ablation.
- **Language coverage.** The dense model is `BAAI/bge-m3` (8 k context,
  az/ru/en in one vector space), which replaced
  `intfloat/multilingual-e5-base` after the head-to-head ablation in
  the "Embedding model" section below. The swap is one config line
  (`embedding.model_name`) plus a rebuild.
- **BM25 is unweighted.** Term-frequency statistics are computed once at
  index time from whatever text is in the store. There is no per-field
  boosting (e.g. title > body). Heading-aware chunking already brings
  topic-bearing context into each chunk's body, which is the next-best
  thing.
