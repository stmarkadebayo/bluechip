# Implementation Log

Date: 2026-05-19

This log tracks the hardening pass requested after the project status review. It is intentionally operational: each entry states what changed, why it matters for the DSN x BCT submission, and how it was validated.

## Submission Metrics Kept In Scope

- Task A rating quality: MAE and RMSE against temporal holdout reviews.
- Task A generation quality: rating-review consistency and grounded output contract.
- Task B ranking quality: HitRate@K, Recall@K, and NDCG@K.
- Demo readiness: API, UI, sample data, Docker path, and deterministic no-key execution.
- Staff-level evidence: typed contracts, traces, reproducible eval scripts, tests, and explicit data integrity checks.

## Steps Completed

### 1. Restored Local Environment

- Created `.venv`.
- Fixed editable packaging in `pyproject.toml` by explicitly limiting setuptools package discovery to `app*`, `eval*`, and `scripts*`.
- Added `httpx` as a dev dependency because FastAPI/Starlette API contract tests require it.
- Installed the project with `pip install -e ".[dev]"`.

Validation:

- `python -m pytest`
- `ruff check .`
- `make PYTHON=./.venv/bin/python eval`

### 2. Added Embedding-Backed Retrieval Signals

- Added dependency-free hashing embeddings in `app/services/retrieval/embeddings.py`.
- Added `LocalVectorRetriever` in `app/services/retrieval/vector_store.py`.
- Added embeddings to `UserProfile` and `ItemProfile`.
- Fed vector similarity into Task A rating prediction and Task B ranking.

Why this matters:

- The system is no longer purely lexical; it has a deterministic vector signal that can later be swapped for OpenAI, sentence-transformer, or managed vector embeddings without changing ranker contracts.

### 3. Improved Task B Ranking Transparency

- Added `RecommendationWeights` for explicit ranker weighting.
- Added vector, category, novelty, popularity, quality, confidence, and dislike components.
- Added `score_components` to every recommendation response.
- Added a small offline ranker tuning script in `eval/tune_ranker.py`.
- Added a dependency-free pairwise learned-ranker trainer in `eval/train_ranker.py`.
- Added collaborative retrieval artifacts for co-visitation and user-neighbor candidate generation.
- Added category-affinity popularity retrieval as a high-recall source.
- Added candidate source tracking, retrieval scores, and `candidate_diagnostics` to Task B responses.
- Added candidate Recall@50/100, cold-start persona-only, sparse/warm-user, and cross-domain evaluation slices.
- Added candidate miss analysis and a candidate-aware learned-ranker promotion gate.

Current sample metrics:

- Hybrid Task A MAE: `0.3333`
- Hybrid Task A RMSE: `0.5774`
- Adaptive Task A MAE: `0.3333`
- Adaptive Task A RMSE: `0.5774`
- Hybrid Task B NDCG@10: `0.4769`
- Ranker tuning best NDCG@10: `0.877`
- Learned ranker NDCG@10: `1.0`

Current `Subscription_Boxes` metrics on a 250-example real-data slice:

- Task A hybrid MAE: `1.012`
- Task A hybrid RMSE: `1.5987`
- Task A adaptive MAE: `1.056`
- Task A adaptive RMSE: `1.4199`
- Task B hybrid HitRate@10: `0.216`
- Task B hybrid NDCG@10: `0.1195`
- Filtered popularity HitRate@10: `0.216`
- Filtered popularity NDCG@10: `0.1194`

Interpretation:

- The real-data recommendation result is aligned with the filtered-popularity floor while preserving personalization for smaller/contextual candidate sets.
- Unfiltered popularity is higher on the slice because it can recommend items the user has already reviewed; the production agent filters seen items.
- The upgraded eval now separates candidate recall from final ranking, which is the next required evidence before promoting learned Task B weights.
- A 25-example all-category smoke eval over 188,236 items reports base candidate Recall@200 `0.16` and hybrid candidate Recall@200 `0.16`; the balanced source blend avoids collaborative sources crowding out the base pool.
- A 100-example all-category eval with full collaborative artifacts reports candidate Recall@200 `0.20`, Recall@500 `0.25`, Recall@1000 `0.28`.
- After Beauty/sparse retrieval, a 100-example all-category eval reports candidate Recall@1000 `0.29` and candidate misses `71`.
- After review-term and lexical-neighbor retrieval, the same 100-example all-category eval reports candidate Recall@1000 `0.32` and candidate misses `68`.
- On that same 100-example slice, hybrid HitRate@10 is `0.10` and NDCG@10 is `0.0766`, beating filtered popularity at `0.09` and `0.068`.
- The cross-domain slice is materially stronger: HitRate@10 `0.2581`, NDCG@10 `0.2056`.
- Candidate-aware learned-ranker training did not pass promotion: learned NDCG@10 `0.0700` was below current hybrid NDCG@10 `0.0766`, so no runtime weights were written.
- Split learned-ranker training still does not pass promotion after the retrieval upgrade: learned NDCG@10 `0.0788` trails same-slice hybrid NDCG@10 `0.1061`, so runtime ranker weights stay unwired.

### 4. Added Runtime Tracing And Metrics

- Added append-only JSONL traces in `runs/traces/requests.jsonl`.
- Added `/api/metrics`.
- Added `/api/traces`.
- Added `trace_id` to Task A and Task B responses.
- Estimated generation tokens and generation cost locally, with deterministic template generation cost recorded as `0`.

Why this matters:

- The demo can show how an agent reached its answer, and the project has a clear path to production observability.

### 5. Built The Demo UI

- Added `ui/index.html`.
- Mounted the UI at `/ui/`.
- Redirected `/` to `/ui/`.
- Included tabs for review simulation, recommendation, and runtime metrics.
- Updated the Dockerfile to include `ui/`.
- Verified the UI in the Codex in-app browser against FastAPI on localhost.

Why this matters:

- The project now has a usable first-screen demo instead of only API docs.

Browser verification:

- `/ui/` loads and reports `API healthy`.
- Review Simulation returns a grounded rating/review response.
- Recommend ranks `Mainland Grill` first for the default conversation-friendly, not-too-expensive dinner scenario.
- Metrics tab reads `/api/metrics` and `/api/traces`.

### 6. Expanded Tests

- Added API contract tests for health, metrics, review simulation, recommendation, trace IDs, and score components.
- Added vector retrieval behavior tests.
- Added downloader integrity status tests.
- Existing recommendation filtering and temporal split tests still pass.

Current test count:

- 31 tests passing.

### 7. Task A Rating Optimization Pass

- Expanded Task A profile features with recent rating tendency, rating volatility, positive/negative rating shares, and rating trend.
- Expanded rating stats with user/item/category standard deviations, star shares, recent user means, and user-category priors.
- Added interaction features for preference-affinity, dislike-risk, vector-affinity, reliability gaps, and user/item prior disagreement.
- Trained a Task A candidate matrix over compact/full feature sets, MSE, Huber, and MAE losses.
- Added ordinal star policies to the candidate matrix: no star policy, validation-calibrated thresholds, and ordinary rounded-star thresholds.
- Selected the first saved artifact by validation MAE, then added an RMSE-selected serving artifact for the rubric metric.
- Added `--workers` to `eval/train_task_a_model.py`; process workers are used when the local runtime allows them, with an in-sandbox fallback.
- Added RMSE-first Task A selection as the default, plus a saved `task_a_rating_stats.json` runtime artifact so serving can load the trained model without rebuilding corpus stats.
- Added `eval/promote_task_a.py`; current serving policy promotes `calibrated_profile` with RMSE `1.2654`.
- Added direct DeepSeek generation support through `DEEPSEEK_API_KEY` / `LLM_PROVIDER=deepseek`.
- Added external-provider data handling for generation eval: non-sample datasets default to `deny`, `redact` sends synthetic prompts, and `allow` is reserved for explicit data-export approval.
- DeepSeek redacted strict smoke passed: generation success `1.0`, provider failure `0.0`, validation consistency `1.0`.
- Added human-eval scoring table generation for Task A and Task B.

Current selected Task A artifact:

- Path: `data/processed/all_categories/task_a_model_rmse.json`
- Selected model: `full_mse_rmse_ensemble`
- Training rows: `46,941`
- Validation rows: `8,283`
- Features: `35`

Current Task A training report:

- Validation MAE: `0.3531`
- Validation RMSE: `0.5880`
- 5,000-example eval MAE from saved RMSE artifact: `0.9136`
- 5,000-example eval RMSE from saved RMSE artifact: `1.3378`

Task A restart lesson:

- Use temporal user splits from the start.
- Make rating prediction a trained, measurable model; keep LLM generation downstream of the fixed rating.
- Build shrinkage user/item/category priors, recent user tendency, volatility, star shares, and user-category features before text generation.
- Select candidates by the submission metric and explicitly test ordinal star policies.
- Report slice metrics so light users, warm users, sparse items, and weak categories are visible.

## Dataset Status

Complete locally:

- `All_Beauty.jsonl`
- `meta_All_Beauty.jsonl`
- `Digital_Music.jsonl`
- `meta_Digital_Music.jsonl`
- `Gift_Cards.jsonl`
- `meta_Gift_Cards.jsonl`
- `Magazine_Subscriptions.jsonl`
- `meta_Magazine_Subscriptions.jsonl`
- `Subscription_Boxes.jsonl`
- `meta_Subscription_Boxes.jsonl`

Processed artifacts built:

- Per-category processed datasets under `data/processed/categories/`.
- Combined five-category dataset under `data/processed/all_categories/`.
- Combined temporal split:
  - reviews: `1,071,963`
  - train: `978,425`
  - test_task_a: `93,538`
  - test_task_b: `93,538`
  - items: `188,236`

The downloader is resumable and byte-size checked. Run:

```bash
./.venv/bin/python scripts/download_amazon_hf.py --with-metadata --check-only --strict
```

The downloader now supports all 34 Amazon Reviews 2023 review categories:

```bash
./.venv/bin/python scripts/download_amazon_hf.py --all-categories --with-metadata
```

That path is intentionally opt-in because the full source is hundreds of GB.

## Verification Snapshot

Last successful local checks during this pass:

```bash
./.venv/bin/ruff check .
./.venv/bin/pytest
./.venv/bin/python -m compileall app eval scripts tests
make PYTHON=./.venv/bin/python eval
```

Known caveat:

- Full all-category Task B eval over every held-out user is intentionally expensive; the current logged all-category Task B metric is a bounded 100-example slice over the full item corpus.

Current all-category slice metrics:

- Task A, 5,000 examples: adaptive MAE `0.9274`, adaptive RMSE `1.3866`.
- Task A, 5,000 examples: hybrid MAE `0.9236`, hybrid RMSE `1.5262`.
- Task A trained model artifact: `data/processed/all_categories/task_a_model.json`.
- Task A RMSE-trained runtime artifact: `data/processed/all_categories/task_a_model_rmse.json`.
- Task A runtime stats artifact: `data/processed/all_categories/task_a_rating_stats.json`.
- Task A serving policy artifact: `data/processed/all_categories/task_a_serving_policy.json`, currently `calibrated_profile`.
- Task A trained model, 5,000 examples: validation-selected calibrated-star MAE `0.9124`, RMSE `1.487`.
- Task A RMSE-trained model, 5,000 examples: MAE `0.9136`, RMSE `1.3378`.
- Task A promoted serving head, 5,000 examples: calibrated profile RMSE `1.2654`.
- Task A trained model raw continuous score, 5,000 examples: MAE `0.9177`, RMSE `1.3395`.
- Task A trained model rounded raw score, 5,000 examples: MAE `0.8834`, RMSE `1.3795`.
- Task B, 100 examples over 188,236 items after Beauty/sparse retrieval: hybrid HitRate@10 `0.10`, hybrid NDCG@10 `0.0766`, candidate Recall@1000 `0.29`.
- Task B, 100 examples over 188,236 items after review-term retrieval and lexical-neighbor retrieval: hybrid HitRate@10 `0.10`, hybrid NDCG@10 `0.0766`, candidate Recall@1000 `0.32`, candidate misses `68`.
- Task B review-term retrieval artifact: `data/processed/all_categories/review_term_retrieval.json` (`421M`), covering all `188,236` items with item terms and term postings.
- Task B learned-ranker gate after the review-term retrieval upgrade was not promoted: learned holdout NDCG@10 `0.0788` vs same-slice hybrid `0.1061`.
- Task B graph-walk ablation was implemented and measured but not enabled by default: it kept candidate Recall@1000 `0.32`, HitRate@10 `0.10`, and NDCG@10 `0.0766` while making evaluation slower.
- Task B Beauty taxonomy ablation was measured on a 25-example smoke slice and gated off because it crowded out stronger sources.
- Added context-category guarding in the ranker so explicit Beauty/music/gift contexts suppress off-topic global-popular items.
- Generated a judge-ready contextual human-eval pack at `docs/human_eval_task_b_contextual.md` with real histories, contexts, top-10 recommendations, source traces, and blank human-score columns.
