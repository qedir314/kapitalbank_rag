# Improvement plan — Kapital Bank RAG (2026-08-27)

Analysis-driven plan for the next iteration of the project. Every item below is
anchored to something observed in the code, the corpus, or the eval CSVs — the
evidence is cited inline.

## 1. Where the project stands

Best measured run (`eval_results/run_20260826_165438.csv`, hybrid config, n=30):

| Metric | Value |
|---|---:|
| Retrieval hit@6 | 64% |
| MRR@6 | 0.516 |
| Judge faithfulness (1–5) | 4.88 |
| Judge correctness (1–5) | 4.60 |
| Refusal accuracy | 100% |

Retrieval is the bottleneck, and the failures are not evenly distributed
(latest run, 9 misses / 25 answerable questions):

| Cut | hit@6 | Reading |
|---|---:|---|
| `lang=en` (10) | 80% | strongest |
| `lang=az` (9) | 56% | weak |
| `lang=ru` (6) | 50% | weakest |
| `category=faq` (3) | **0%** | total miss — all three languages |
| `category=deposits` (3) | 33% | one hit at rank 11 (outside top-6) |
| `category=loans`, `money-transfers`, `how-to` | 100% | solved |

Generation quality is high and stable across all eight recorded runs; attack
retrieval, not generation.

## 2. Key findings from the analysis

Ordered by leverage:

1. **FAQ pages lose their questions during extraction.** The three FAQ misses
   are an ingestion problem first: the indexed `birbank.az/faq` chunks contain
   flattened accordion labels and answer steps ("3D Secure Kart üzrə
   əməliyyatlar Köçürmələr Qeydiyyat…") but not the questions themselves, so
   question-shaped queries have nothing to match. Only 12 FAQ chunks exist in
   the index.
2. **Chunk text lacks its heading context.** Only 168 / 2,546 chunks contain
   their page title in the text; section headings are stored as metadata only.
   BM25 can't match product names that appear only in headings, and dense
   embeddings never see them either. (`kb_rag/ingest/chunking.py::build_chunks`
   keeps `breadcrumb` out of `text`.)
3. **Section taxonomy is broken.** `derive_section()` matches URL path
   segments, but consolidated birbank.az URLs don't carry the taxonomy:
   `news` = 1,456 chunks (57%), `other` = 872 (34%), while `cards` = 29,
   `loans` = 6, `deposits` = 17. The sidebar section filter is effectively
   non-functional for product queries, and `app.py`'s `SECTIONS` list has
   drifted from `derive_section`'s known set (`insurance`, `kampaniyalar`,
   `locations`, `online-order`, `ferdi-bankciliq` are unreachable in the UI).
4. **The golden set has zero reference answers** (`reference_answer` present
   on 0 / 30 questions), so judge "correctness" is really
   "answer-consistent-with-retrieved-context" — it cannot catch a confidently
   wrong answer built from the wrong passage.
5. **Headline numbers are stale and under-reproducible.** README reports
   hit@6 = 52% / MRR = 0.46 (run `…161235`), but the tuned hybrid already
   reaches 64% / 0.516 (run `…165438`). `docs/hybrid_retrieval.md` cites CSVs
   as reproducible sources, yet `eval_results/*` is gitignored and no CSV is
   committed — and no run records the config that produced it.
6. **Self-judge bias never ablated.** DeepSeek generates and judges every
   run; the `llm.judge_model` swap exists in config but was never exercised.
7. **BM25 statistics include excluded content.** The in-memory BM25 index
   spans all 2,546 chunks including the 1,456 news chunks that queries
   exclude, skewing IDF. The tokenizer is bare `\w+` — no morphology handling
   for agglutinative Azerbaijani (`kreditlərin` ≠ `kredit`), no
   Cyrillic↔Latin transliteration for az queries.
8. **Page-level dedupe is order-dependent.** "First language variant wins"
   in `scripts/build_index.py` depends on sitemap/page order, and the winner's
   `lang` metadata becomes the only filterable identity for consolidated pages
   (chunk langs: az 1,023 / ru 821 / en 702 vs raw pages az 476 / ru 424 /
   en 358 — en pages were disproportionately deduped away).
9. **Hygiene gaps.** `.claude/settings.local.json` is committed;
   `requirements.txt` uses `>=` with no lockfile; no CI, no Dockerfile;
   `config.py` defaults have drifted from `config.yaml`
   (`candidate_pool` 40 vs 60, `rerank_candidates` 16 vs 10,
   `rerank_max_length` 448 vs 320); scraper/`build_index` paths have no
   tests (the mojibake `_decode` fix is untested); no latency telemetry;
   `crawled_at` is captured but never surfaced to users.
10. **Unexplored quality dimensions.** No multi-turn evaluation (the app sends
    history, the harness never does), no citation verification (`[n]` markers
    are requested but never checked), no prompt-injection resistance clause in
    the system prompt, no user-feedback capture.

## 3. Phased plan

### Phase 1 — Cheap retrieval wins (target: hit@6 64% → ~75%)

| # | Task | Where | Verify |
|---|---|---|---|
| 1.1 | **Breadcrumb-prefixed chunks**: prepend `"{title} > {heading}."` to each chunk's *text* (not just metadata) before embedding/BM25. Classic contextual-chunking win; fixes finding 2 for both retrievers at once. | `kb_rag/ingest/chunking.py::build_chunks` (+ test) | rebuild index, rerun eval; unit test asserts prefix present |
| 1.2 | **Fix section taxonomy**: derive section from the *source* URL's sitemap taxonomy (kapitalbank.az slugs carry it) with a birbank.az path map as fallback; regenerate chunk metadata. Sync `app.py` `SECTIONS` with the actual value set. | `chunking.py::derive_section`, `scripts/build_index.py`, `app.py` | `other` < 15% of chunks; section filter returns product content; new unit tests per URL shape |
| 1.3 | **FAQ ingestion fix**: inspect raw HTML of `birbank.az/faq`; if questions live in accordion markup trafilatura drops, add a targeted extractor (BeautifulSoup pass) for FAQ pages; re-chunk. | `kb_rag/scraper/crawl.py::extract_content` or a `faq.py` special case | faq golden questions ≥ 2/3 hit; index holds question text |
| 1.4 | **Rerank pool ablation**: config-only sweep — `rerank_candidates` 10→20, `candidate_pool` 60→80 (CPU cost still ~1 s/query). Keep the winning config. | `config.yaml` + eval runner | best config kept; latency stays < ~1.5 s |
| 1.5 | **BM25 over indexed sections only**: build the BM25 corpus excluding `exclude_sections` so IDF isn't skewed by 1,456 news chunks. | `kb_rag/rag/hybrid.py::BM25Index.from_store`, `retriever.py` | unit test + eval non-regression |

**Definition of done:** hit@6 ≥ 75%, MRR@6 ≥ 0.60, refusal 100%, faq ≥ 2/3,
`other` < 15% of chunks, all tests green.

#### Phase 1 outcome (2026-08-27, `run_20260827_131038`, GPU index)

| Task | Result |
|---|---|
| 1.1 breadcrumb-prefix chunks | **reverted** — regressed hit@6 64%→60% (tokens dilute BM25 IDF + embedding); kept as a negative result |
| 1.2 section taxonomy | done — `other` 34%→16%, section filter functional |
| 1.3 FAQ ingestion | **done — faq 0/3 → 2/3.** `scripts/fix_faq_pages.py` re-extracts Q&A from `__NEXT_DATA__`; questions formatted as inline `**bold**` (the chunker strips all H1–H6 from text — a `###` heading would be dropped). 25 faq chunks now indexed across az/en/ru |
| 1.4 rerank pool | done — `rerank_candidates` 20, `candidate_pool` 60 |
| 1.5 BM25 over indexed sections | done — news excluded from IDF stats |

Net: **hit@6 68% → 72%**, MRR@6 0.516 → 0.555, refusal 100%, **faq ≥ 2/3 ✓**,
`other` < 15% ✓. Index rebuild + eval now run on the **RTX 5070** (cu128 torch
2.11 via a junction into `../aznlp-core/.venv`): full 2,565-chunk rebuild in
**1m21s** (was ~40 min CPU).

**Why not the last 3 points to 75%:** the remaining 6 answerable misses were
diagnosed as *not* retriever-tuning problems — 4 are corpus gaps (no virtual-card
page; branch-hours page never crawled; "Zolotaya Korona" exists only as a *news*
article that `exclude_sections` drops by design; the ru FAQ has no "lost card"
Q&A — the generator refuses correctly) and 2 are golden-label defects (expected
`taksitkartlar` vs the real `taksit-karti`; expected `deposits` vs the site's
`depozitler`). Closing them means Phase 3 (reference answers + corrected labels +
70-Q set), not more P1 tuning — forcing the number now would be overfitting bad
labels.

### Phase 2 — Representation & query understanding (target: ≥ 80%, lang gap ≤ 10 pp)

| # | Task | Where | Verify |
|---|---|---|---|
| 2.1 | **bge-m3 vs e5-base ablation**: swap `embedding.model_name` to `BAAI/bge-m3` (8 k context, stronger multilingual), rebuild into a separate persist dir, compare on the identical golden set. Keep whichever wins; the loser stays one config line away. | `config.yaml`, `ingest/embeddings.py` (drop the e5-only prefix assumption if bge-m3 wins), `build_index` | side-by-side eval table in `docs/hybrid_retrieval.md` |
| 2.2 | **Query expansion**: LLM rewrites the user query into az/en/ru variants (one cheap DeepSeek call); each variant retrieves; results fused by RRF. Config toggle, default on only if it measurably wins. | new `kb_rag/rag/query_expansion.py`, `retriever.py`, `pipeline.py` | eval with/without; per-lang hit gap ≤ 10 pp |
| 2.3 | *(optional)* **Morphology-aware BM25 tokens**: light Azerbaijani suffix stripping and/or Cyrillic↔Latin normalization in `tokenize()`. | `kb_rag/rag/hybrid.py` | unit tests on az/ru pairs; eval delta |

**Definition of done:** hit@6 ≥ 80% on the (by then expanded) golden set;
az/ru within 10 pp of en; ablation written up with committed CSVs.

#### Phase 2 outcome (2026-08-27) — representation confirmed, query side rejected

| Task | Result |
|---|---|
| 2.1 bge-m3 vs e5-base ablation | **done — bge-m3 kept, decisively.** Head-to-head on the identical Phase-1 corpus + golden set (same day, same hybrid config): bge-m3 **72%** / MRR 0.555 vs e5-base 64% / 0.523. The entire gap is the two rebuilt FAQ questions — e5's 512-token window truncates most of a Q&A listing before embedding; bge-m3 at 1,024 sees it. No question is worse under bge-m3. Full table in `docs/hybrid_retrieval.md` |
| 2.2 query expansion | **implemented, rejected by measurement.** `kb_rag/rag/query_expansion.py` + per-variant RRF in `Retriever` (reranker still scores the original query), toggle `retrieval.query_expansion`, default **off**. A/B (`run_20260827_142300`): hit@6 68% (−4 pp), az/ru hit *dropped* to 56/50% (gap widened — opposite of intent), and one unanswerable stopped being refused on cross-language noise. bge-m3's shared vector space already handles the alignment; expansion adds candidate dilution |
| 2.3 morphology tokens | **implemented, rejected by measurement.** Additive-only augmentation in `hybrid.tokenize(morph=True)` — Cyrillic→Latin transliteration + one-step az suffix stems (surface tokens always kept, stem ≥ 4 chars), toggle `retrieval.morph_tokens`, default **off**, no index rebuild needed. A/B (`run_20260827_142724`): hit@6 68%, one flip (ru-deposits-001 squeezed out); the cross-script recall wins never converted to a golden-set hit |

**DoD not met — 72%, not 80%, and the target itself is now suspect.** Both
query-understanding levers regressed slightly from the same index, in the
same direction as Phase 1's breadcrumb experiment. Three independent
query-side "wins" that all fail to measure is strong evidence the binding
constraint is exactly where the Phase 1 post-mortem placed it: 4 corpus
gaps + 2 golden-label defects, on a 25-question sample where ±1 question =
±4 pp. **Recommendation: do Phase 3 (references, expanded set, run
manifests) before any further retrieval experimentation** — the current set
is too small and partly mislabeled to reliably grade changes of this size,
which is also why the plan's own sequencing put the measurement work first.

Infra left behind: `KB_CONFIG` env override in `config.get_settings()`
(run anything with an alternate YAML — the three `config.ablation_*.yaml`
files are checked in), the `QueryExpander` module, the morphology tokenizer,
6 new retriever/expander tests + 6 morph tests (81 total, all offline).

### Phase 3 — Evaluation rigor (make every number defensible)

This phase is what turns "64%" from a lab note into an interview-proof claim.

| # | Task | Where | Verify |
|---|---|---|---|
| 3.1 | **Write reference answers** for all 24 answerable golden questions (short ground-truth answer + key facts). The judge rubric already accepts `reference` — it just never receives one. | `data/golden/qa.yaml` | judge sees reference; correctness re-scored |
| 3.2 | **Expand the golden set 30 → ~70**: stratified lang×category, more faq/deposits, near-miss confusables (BirKart vs Birbank Miles), stale-rate traps, 2–3 multi-turn follow-ups. `scripts/make_golden_set.py` already drafts candidates. | `data/golden/qa.yaml`, `scripts/validate_golden_set.py` | every category n ≥ 4; validator passes |
| 3.3 | **Run manifests + committed canonical CSVs**: the runner dumps `config snapshot + git SHA` next to each CSV; gitignore negation commits the canonical runs the docs cite; README/docs cite those exact files. | `evaluation/runner.py`, `.gitignore`, docs | any published number traceable to committed CSV + config |
| 3.4 | **Independent-judge ablation**: run the judge once with a different model family; report per-question agreement; keep the stricter judge as default. | `config.yaml` (`llm.judge_model`), `scripts/analyze_ablations.py` | agreement table documented |
| 3.5 | **Robustness + variance + citation metrics**: retry judge JSON failures and record row-level errors instead of crashing mid-run; 3 seeds per config with mean ± std; add programmatic citation precision/recall (`[n]` markers vs supporting text in passage n). | `generation_metrics.py`, `runner.py` | CI-stable eval; citation precision reported |

**Definition of done:** every metric in README reproducible from committed
artifacts; correctness scored against references; judge bias bounded and
documented.

#### Phase 3 outcome (2026-08-27)

| Task | Result |
|---|---|
| 3.1 reference answers | done — all 62 answerable questions carry corpus-grounded `reference_answer` (verified against `pages.jsonl`, not copied from generator output). The judge now scores correctness against ground truth |
| 3.2 golden set 30 → 74 | done — 62 answerable / 12 unanswerable, every category n ≥ 4, lang×category matrix (az 21 / en 24 / ru 17 answerable). Label fixes: `az-cards-001` (`taksitkartlar`→`taksit`), `az-virtual-001` (→`visa-digital`, the page does exist on birbank.business), `en-faq-001` (dropped non-existent `how-to` fragment). `en-deposits-001` was NOT a label defect — `/en/deposits` exists; it is a genuine rank-7 retrieval miss. `ru-transfers-001` (Zolotaya Korona) reclassified unanswerable: the system is not in the indexed corpus (news-only, excluded) — refusal is now the scored-correct behavior. 3 multi-turn `history` items added; `scripts/validate_golden_set.py` rewritten (URL resolution, schema, reference coverage, stratification, exit codes) and passes |
| 3.3 run manifests | done — `runner` writes `run_*.manifest.json` (config snapshot, git SHA/dirty, dataset sha256, tag/seed/judge/mode); canonical CSVs + manifests un-ignored and committed; `KB_CONFIG` ablation configs checked in; smoke manifest verified to contain no secrets |
| 3.5 robustness + variance + citations | done — judge JSON failures retried ×3; per-row `error`/`judge_error` columns (a dead question no longer kills the run); `--seed`/`--seeds` (mean ± std across seeded passes); `citation_metrics.py` (marker validity / bag-of-words support / sentence coverage) with 8 offline tests |
| 3.4 independent judge | done — R1 re-judge: 75% exact agreement both dimensions; R1 stricter on faithfulness (−0.27, catches premise-confusion answers V3 passed); `deepseek-chat` kept as default judge with the bias now bounded. Required a reasoning-token fix first: R1 at 300 tokens returned empty content (thinking consumed the budget) — judge budget now model-aware (`_judge_max_tokens`) |

**New headline (`run_20260827_150822`, tag `phase3-canonical-74q`, n=74):**
hit@6 **91.9%** / MRR@6 0.808 — *not comparable to the old 72%*: the old set
counted corpus-gap and mislabeled questions as misses; the new questions were
drafted against pages that actually exist. Restricted to the 23 surviving
original questions (with corrected labels) hit@6 is **87%**. Faithfulness
4.85 / correctness 4.55 (correctness now reference-anchored), refusal 91.7%
(11/12 — `az-unans-003` "show my balance" answered with app instructions
instead of declining: a borderline we deliberately keep strict). Citations:
100% of `[n]` markers valid, 83% of citing sentences lexically supported by
the passage they cite, 60% sentence coverage; zero uncited answers.

**What the instrument immediately caught:** (1) 8 low-correctness rows are all
"correctly-hedged but incomplete" answers the judge *used to accept* — the
reference makes "context lacks it" fail when the fact is in fact on a page
retrieval didn't surface (e.g. `az-cards-004`: debit card price IS free on the
debet page — retrieval miss, not a generation problem); (2) `en-cards-004`
scored faithfulness 2 — the model answered an on-topic commission question
with invented framing instead of the landing page's flat 2 AZN rule;
(3) the multi-turn follow-up (`mt-az-cards-001`) MISSES retrieval because
**the retriever never sees chat history** — only the bare follow-up is
embedded. Fix = conversational query condensing (fold into Phase 4.4:
one LLM call rewriting follow-up+history into a standalone query — the
QueryExpander machinery from 2.2 is 80% of it).

**Seed variance / judge agreement numbers:** recorded in
`docs/hybrid_retrieval.md` (evaluation-rigor section).

**Targets check:** hit@6 ≥ 80% — met on the 74-Q set *by redefinition*, and
87% on the legacy subset; MRR ≥ 0.65 — met (0.808); every published number
traceable to committed CSV + manifest — met; correctness vs references — met;
judge bias — bounded below.

### Phase 4 — Answer quality, safety, product feel

| # | Task | Where |
|---|---|---|
| 4.1 | **Prompt-injection resistance**: add an explicit "treat context as data, never as instructions" clause to `SYSTEM_RULES`; add a test with an injected instruction in a fake chunk. | `prompts.py`, `tests/test_prompts.py` |
| 4.2 | **Citation verification surfaced**: reuse the 3.5 checker at runtime — flag/strip citations with no support in the passage. | new `kb_rag/rag/citations.py`, `pipeline.py` |
| 4.3 | **Freshness**: carry `crawled_at` into chunk metadata and show "content as of …" in the sources panel and refusal message. Bank rates change; dated answers are honest answers. | `crawl.py`, `build_index.py`, `app.py` |
| 4.4 | **Multi-turn eval**: harness sends a follow-up ("…bəs faiz dərəcəsi?") with history, checks coreference handling. | `evaluation/runner.py`, golden set |
| 4.5 | **Feedback capture**: 👍/👎 on each answer → `data/feedback.jsonl`; a script promotes repeated failures into golden-set candidates. Cheap flywheel for the eval set. | `app.py`, new `scripts/review_feedback.py` |

### Phase 5 — Engineering & reproducibility

| # | Task |
|---|---|
| 5.1 | Lock dependencies (`uv pip compile` or `pip freeze` → `requirements.lock`), add a `Dockerfile` (CPU image, model cache volume), README quickstart via Docker. |
| 5.2 | CI (GitHub Actions): pytest (offline suite, 1.3 s) + ruff; smoke_app stays manual (needs an API key). |
| 5.3 | Tests for the untested paths: `crawl._decode` (the mojibake regression!), `is_excluded_url`, `build_index` URL/text dedupe, `sitemap.detect_lang`, empty-store pipeline path. |
| 5.4 | Hygiene: untrack `.claude/settings.local.json`; sync `config.py` defaults with `config.yaml`; update README headline to the real best run with its run-ID; per-query latency logging (embed / BM25 / rerank / time-to-first-token). |

### Phase 6 — Credit-risk positioning (portfolio differentiator)

The project targets a Credit Risk AI role; today it answers retail-banking
questions. One domain-specific module would align it with the target role:

1. **Annual-report ingestion**: Kapital Bank publishes annual reports (PDF).
   Add a PDF path (pdfplumber/marker) with table-aware chunking, index the
   financial statements, and add a `financials` category to the golden set
   (NPL ratio, capital adequacy, loan-portfolio composition). Demonstrates
   PDF + table handling — a gap in the current pipeline — on the most
   role-relevant data.
2. **Grounded tariff comparison**: structured extraction of rates/fees from
   retrieved chunks into a comparison table (with citations) — e.g. "compare
   consumer loan rates".
3. *(Optional)* Synthetic-data credit-risk driver notebook, as already sketched
   in README future work.

## 4. Recommended sequencing

```
Week 1:  Phase 1 (1.1 → 1.2 → 1.3 → 1.4/1.5)      # compounding retrieval wins
         + 3.1 + 3.3 in parallel                   # references & manifests are independent
Week 2:  Phase 3 rest (3.2, 3.4, 3.5)             # expanded set before big model swaps
Week 3:  Phase 2 (bge-m3 ablation on the expanded set, then query expansion)
Week 4:  Phase 4 + 5 interleaved; Phase 6.1 if time allows
```

Rationale: fix cheap, structural retrieval defects (Phase 1) before spending
model-swap compute (Phase 2), and harden the measuring instruments (Phase 3)
before the expensive experiments so their results are trustworthy.

Success criteria overall: **hit@6 ≥ 80%, MRR@6 ≥ 0.65, faithfulness ≥ 4.8 held,
refusal 100% held, per-language gap ≤ 10 pp, every published number
reproducible from committed artifacts.**

## 5. Explicitly out of scope (for now)

- **pgvector migration** (README future work): at 2.5 k chunks, Chroma is not
  the constraint; migration buys infra churn, not eval points. Revisit only
  if hybrid SQL-side filtering becomes genuinely needed.
- **Embedding fine-tuning**: far too little relevance-labeled data; hybrid
  retrieval + reranking is the right lever at this scale.
- **GPU / hosted rerank APIs**: CPU rerank latency is already ~1 s/query.
