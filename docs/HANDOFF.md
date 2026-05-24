# Bluechip Handoff

This document summarizes how the project got to its current state, what is implemented, what was validated, and what should happen next.

The submission scope is now frozen. Use [SUBMISSION_FREEZE.md](SUBMISSION_FREEZE.md) as the source of truth for the eight-step path to final submission.

Current final-submission sequence:

1. Freeze scope around the current evidence-first hybrid agent.
2. Keep the completed Task B contextual human-eval summary and Task A review pack.
3. Keep the completed `implicit` ALS/BPR/item-item baseline report.
4. Run final validation.
5. Finalize the 4-8 page solution paper.
6. Package the repo safely.
7. Demo-check the API/UI.
8. Submit the repo, paper PDF, architecture diagram, app/API instructions, and eval summary.

## Project Context

The hackathon brief asks for an LLM agent that solves two related tasks:

- **Task A: User modeling / review simulation**: infer a user's likely rating and generate a review for a target item.
- **Task B: Personalized recommendation**: rank candidate items for a user and explain why they fit.

We interpreted "Build an agent" as an orchestration layer over deterministic, testable tools rather than a single prompt. The agent profiles the user, profiles items, retrieves candidates, ranks or predicts ratings, generates text, validates outputs, and returns an explicit trace.

## How We Got Here

1. We read the hackathon brief and narrowed the goal to the core build: a shared user-intelligence engine with two serving heads.
2. We clarified that "staff-level" means scalable, measurable, reproducible, and not just prompt-based.
3. We did a research pass across recommender systems, user modeling, agent architecture, retrieval, and industry patterns.
4. We kept one local API with clear internal boundaries:
   - offline ingestion and feature building
   - typed profile and item contracts
   - aspect-aware evidence intelligence
   - retrieval before ranking
   - deterministic scoring before LLM generation
   - review planning before final text generation
   - optional LLM only at the final language step
   - eval scripts as first-class project artifacts
5. We implemented the repository from scratch, then reviewed it for correctness, scalability, and GitHub readiness.
6. We downloaded and byte-validated five Amazon Reviews 2023 categories plus metadata.
7. We built a combined all-category processed corpus and trained Task A models selected for both validation MAE and validation RMSE.
8. We upgraded the architecture toward an evidence-first recommendation system while keeping it local and reproducible.

## Submission Position

The strongest truthful story for the current submission is evidence-first behavior-aware personalization:

- Task A is rating-first review simulation: predict the rating from user/item evidence, then generate and validate the review.
- Task B is retrieval before ranking: build a source-attributed candidate pool, rank it with explicit components, and explain from visible evidence.
- Sparse and cross-domain behavior are measured separately because they are high-value rubric areas.
- LLM output is downstream of profiling, retrieval, ranking, and validation.
- Neural sequence models should stay out of the current runtime unless fixed evals beat the current hybrid baseline.

Latest bounded all-category Task B metrics after evidence graph work and the popularity-rank floor:

| Metric | Value |
| --- | ---: |
| `hybrid_candidate_recall@50` | `0.13` |
| `hybrid_candidate_recall@100` | `0.18` |
| `hybrid_candidate_recall@1000` | `0.34` |
| `hybrid_ranker_hit_rate@10` | `0.10` |
| `hybrid_ranker_ndcg@10` | `0.0766` |
| Sparse candidate recall@1000 | `0.3611` |
| Cross-domain candidate recall@1000 | `0.5484` |
| Vector source recall | `0.0` |

Vector retrieval is present as a deterministic diagnostic hook, but current measured vector source recall is `0.0`; do not describe it as a recall win.

## Response To Latest Architecture Review

The last teammate feedback was directionally right: the previous risk was optimizing for semantic fit while missing co-engagement behavior. A content encoder can look semantically relevant and still fail candidate recall, so the repository now treats Task B as a multi-objective retrieval and ranking problem.

What changed to address it:

- Added a local MTMH-style retrieval shape: multiple retrieval heads now contribute candidates instead of relying on one semantic/vector path.
- Added co-engagement heads: item co-visitation and user-neighbor collaborative retrieval are first-class candidate sources.
- Added semantic/evidence heads: BM25 profile/context retrieval, review-term retrieval, lexical item-neighbor retrieval, aspect evidence graph retrieval, and deterministic vector diagnostics.
- Added source attribution: responses and eval reports expose `candidate_sources`, per-source retrieval scores, source counts, and candidate diagnostics.
- Added multi-objective scoring: the ranker blends preference, context, category, aspect, sequential, evidence graph, Nigerian context, collaborative, retrieval, source diversity, item quality, popularity, novelty, and confidence features.
- Added recall-first measurement: Task B eval now separates candidate Recall@50/100/1000 from HitRate@10 and NDCG@10, with sparse-user and cross-domain slices.
- Added production boundaries: `app/platform/feature_store.py`, `app/platform/model_registry.py`, and `app/serving/orchestrators/` make it easier to swap local heads for stronger learned models later.

What is intentionally not claimed:

- We have not trained MTMH as a neural multi-task model in this repo.
- We have not implemented HSTU as the online ranker.
- We have not proven vector retrieval quality; vector source recall is currently `0.0`, so it remains a diagnostic/extensibility hook.

Recommended interpretation for a teammate or judge:

The current code implements the practical architecture around the feedback: multi-head retrieval, co-engagement plus semantic evidence, multi-objective ranking, source diagnostics, and fixed eval gates. MTMH and HSTU remain the next production-model upgrades after candidate recall improves and same-slice offline evals justify the added complexity.

## What Is Built

### Application

- FastAPI app in `app/main.py`.
- API routes in `app/api/routes.py`.
- Endpoints:
  - `GET /api/health`
  - `POST /api/profile-user`
  - `POST /api/simulate-review`
  - `POST /api/recommend`
  - `POST /api/conversation/turn`
  - `POST /api/conversation/{conversation_id}/feedback`
  - `GET /api/conversation/{conversation_id}`
  - `GET /api/conversations`
  - `POST /api/infer-cold-start`
  - `POST /api/transfer-cross-domain`
  - `POST /api/nigerian/context`
  - `GET /api/metrics`
  - `GET /api/traces`
  - `GET /api/runtime/registry`
  - `GET /api/runtime/feature-store`
- Browser demo mounted at `/ui/`.

### Agent Workflows

- `ReviewSimulationAgent` in `app/serving/orchestrators/review_simulation.py`
  - builds user profile
  - builds target item profile
  - predicts rating
  - generates review
  - validates rating-review consistency
  - returns an agent trace

- `RecommendationAgent` in `app/serving/orchestrators/recommendation.py`
  - builds user profile
  - retrieves candidate items
  - filters seen/history items
  - ranks candidates
  - generates grounded explanations
  - returns an agent trace

### Core Services

- User profiling: `app/services/profiling/user_profile.py`
- Item profiling: `app/services/profiling/item_profile.py`
- Aspect intelligence: `app/services/intelligence/aspects.py`
- BM25 retrieval: `app/services/retrieval/text.py`
- Item co-occurrence retrieval: `app/services/retrieval/item_similarity.py`
- Evidence graph retrieval: `app/services/retrieval/evidence_graph.py`
- Candidate generation: `app/services/retrieval/candidates.py`
- Rating prediction: `app/services/ranking/rating.py`
- Recommendation ranking: `app/services/ranking/recommendation.py`
- Dependency-free vector retrieval: `app/services/retrieval/vector_store.py`
- Deterministic hashing embeddings: `app/services/retrieval/embeddings.py`
- Generation provider abstraction: `app/services/generation/providers.py`
- Review/recommendation text generation: `app/services/generation/generator.py`
- Review planning: `app/services/generation/review_plan.py`
- Evidence critic: `app/services/validation/evidence_critic.py`
- Output validation: `app/services/validation/critic.py`
- Runtime trace store: `app/stores/trace_store.py`
- Local feature store: `app/platform/feature_store.py`
- Local model/index registry: `app/platform/model_registry.py`

### Data and Evaluation

- Amazon JSONL ingestion: `scripts/ingest_amazon.py`
- Temporal split builder: `scripts/build_splits.py`
- Retrieval index builder: `scripts/build_retrieval_index.py`
- Evidence graph builder: `scripts/build_evidence_graph.py`
- Dataset downloader/checker: `scripts/download_amazon_hf.py`
- Shared eval helpers: `eval/common.py`
- Task A eval: `eval/eval_task_a.py`
- Evidence intelligence eval: `eval/eval_evidence_intelligence.py`
- Task B eval: `eval/eval_task_b.py`
- Metrics: `eval/metrics.py`

### Documentation

- Main setup and architecture: `README.md`
- Literature and industry review: `research/literature_review.md`
- Production architecture notes: `infra/production_architecture.md`
- Solution paper: `paper/solution_paper.md`
- Prompt contracts: `prompts/`
- Hardening log: `docs/IMPLEMENTATION_LOG.md`
- This handoff: `docs/HANDOFF.md`

### DevOps

- Dockerfile and docker-compose are included.
- GitHub Actions CI is included at `.github/workflows/ci.yml`.
- Raw datasets, processed datasets, eval outputs, virtualenvs, and caches are ignored.
- `.env.example` documents optional LLM provider settings without committing secrets.

## Current Dataset State

Raw datasets are intentionally ignored by Git. Teammates should download them locally.

Complete locally:

- `data/raw/All_Beauty.jsonl`
- `data/raw/meta_All_Beauty.jsonl`
- `data/raw/Digital_Music.jsonl`
- `data/raw/meta_Digital_Music.jsonl`
- `data/raw/Gift_Cards.jsonl`
- `data/raw/meta_Gift_Cards.jsonl`
- `data/raw/Magazine_Subscriptions.jsonl`
- `data/raw/meta_Magazine_Subscriptions.jsonl`
- `data/raw/Subscription_Boxes.jsonl`
- `data/raw/meta_Subscription_Boxes.jsonl`

The combined processed dataset under `data/processed/all_categories` contains:

- 1,071,963 reviews
- 188,236 items
- 923,201 users
- 978,425 train reviews
- 93,538 Task A holdout reviews
- 93,538 Task B holdout reviews

Check dataset integrity:

```bash
python scripts/download_amazon_hf.py --with-metadata --check-only --strict
```

Resume downloads:

```bash
python scripts/download_amazon_hf.py --with-metadata
```

Opt into every Amazon Reviews 2023 review category:

```bash
python scripts/download_amazon_hf.py --all-categories --with-metadata
```

The full source is hundreds of GB, so the default remains the five locally validated categories.

Use one category for a fast real-data run:

```bash
python scripts/ingest_amazon.py \
  --reviews data/raw/Subscription_Boxes.jsonl \
  --metadata data/raw/meta_Subscription_Boxes.jsonl \
  --category Subscription_Boxes \
  --output-dir data/processed
```

Build the combined processed dataset after raw downloads:

```bash
python scripts/build_combined_processed.py \
  --input-root data/processed/categories \
  --output-dir data/processed/all_categories
```

## Local Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
```

Run checks:

```bash
./.venv/bin/ruff check .
./.venv/bin/pytest
./.venv/bin/python -m compileall app eval scripts tests
make PYTHON=./.venv/bin/python eval
```

Run API:

```bash
uvicorn app.main:app --reload
```

Build the local model/index registry:

```bash
python scripts/build_model_registry.py --output data/processed/model_registry.json
```

Build the evidence graph artifact directly:

```bash
python scripts/build_evidence_graph.py \
  --train data/processed/train.jsonl \
  --items data/processed/items.jsonl \
  --output data/processed/evidence_graph_retrieval.json
```

`scripts/build_retrieval_index.py` also writes `evidence_graph_retrieval.json` beside the collaborative and review-term retrieval artifacts.

Open:

```text
http://127.0.0.1:8000/docs
http://127.0.0.1:8000/ui/
```

Optional OpenAI-backed generation:

```bash
LLM_PROVIDER=openai
LLM_MODEL=gpt-5
OPENAI_API_KEY=...
```

Without provider variables, the system uses deterministic fallback text generation. `.env` and `.env.*` files are ignored; keep real API keys local.

## What Was Validated

These checks passed locally:

- Python compile check across `app`, `eval`, `scripts`, and `tests`.
- Current hardening pass: 43 tests pass, lint passes, compile passes, sample eval passes, and bounded all-category Task B eval runs.
- FastAPI import smoke test passed.
- `Subscription_Boxes` real-data ingestion passed.
- `Subscription_Boxes` temporal split and retrieval index build passed.
- `Subscription_Boxes` Task A and Task B eval scripts ran successfully.

Current sample eval signal:

- Task A hybrid profile scorer beats the simple sample baselines on MAE/RMSE.
- Task B hybrid ranker passes the sample eval and ranks held-out items well on the tiny sample.
- Ranker tuning reaches NDCG@10 `0.877` on the sample split.
- Learned ranker training reaches NDCG@10 `1.0` on the sample split.
- Task B eval now reports candidate Recall@50/100, hybrid candidate recall, cold-start persona-only ranking, sparse/warm-user slices, and cross-domain slices.
- Evidence-intelligence code paths are covered by focused tests for aspect extraction, graph retrieval, and review-plan fallback generation.

Current real-data signal:

- Task A runs end to end on `Subscription_Boxes`.
- On a 250-example `Subscription_Boxes` slice, Task A hybrid MAE is `1.012`; adaptive star MAE is `1.056`.
- On the same slice, Task B hybrid HitRate@10 is `0.216` and NDCG@10 is `0.1195`.
- Task B now matches the filtered-popularity floor on that slice while preserving seen-item filtering. Unfiltered popularity is higher because it can recommend previously reviewed items.
- A trained Task A linear model artifact exists at `data/processed/all_categories/task_a_model.json`.
- An RMSE-selected Task A runtime artifact exists at `data/processed/all_categories/task_a_model_rmse.json`.
- Runtime rating stats are precomputed at `data/processed/all_categories/task_a_rating_stats.json`.
- Task A serving policy is written to `data/processed/all_categories/task_a_serving_policy.json`; current promoted head is `calibrated_profile`.
- The original Task A artifact was selected by validation MAE from a candidate matrix over compact/full features, MSE/Huber/MAE losses, and calibrated/rounded star policies.
- The selected model is `full_mse_calibrated_star` with 35 engineered features.
- On validation, selected Task A MAE is `0.2826` and RMSE is `0.6121`.
- The RMSE-selected serving model is `full_mse_rmse_ensemble`; validation MAE is `0.3531` and validation RMSE is `0.5880`.
- On a 5,000-example all-category slice, adaptive MAE is `0.9274`, hybrid MAE is `0.9236`, and the saved Task A artifact MAE is `0.9124`.
- On that same 5,000-example slice, raw continuous model MAE is `0.9177`; ordinary rounded raw-score MAE is `0.8834`.
- On a 5,000-example all-category slice, the RMSE-selected serving artifact reports MAE `0.9136` and RMSE `1.3378`.
- The Task A promotion gate selects `calibrated_profile` for serving because it has the best 5,000-example RMSE: `1.2654`.
- Earlier all-category Task B slices established the filtered-popularity floor and candidate-recall bottleneck.
- Full all-category collaborative artifacts were built locally:
  - `data/processed/all_categories/collaborative_retrieval.json` (`237M`)
  - `data/processed/all_categories/item_neighbors.json` (`25M`)
  - `data/processed/all_categories/review_term_retrieval.json` (`421M`)
- Latest bounded all-category Task B eval after evidence graph work and the popularity-rank floor:
  - hybrid candidate Recall@50 `0.13`
  - hybrid candidate Recall@100 `0.18`
  - hybrid candidate Recall@1000 `0.34`
  - hybrid HitRate@10 `0.10`
  - hybrid NDCG@10 `0.0766`
  - sparse candidate Recall@1000 `0.3611`
  - cross-domain candidate Recall@1000 `0.5484`
  - vector source recall `0.0`
- Beauty taxonomy retrieval is enabled as a measured candidate-generation source.
- Graph-walk retrieval is not enabled because it added latency without improving the fixed all-category metrics.
- Context-category ranking guards are enabled for explicit Beauty, music, and gift contexts; this improves the human-eval/contextual demo path without changing empty-context offline metrics.
- Contextual human-eval pack: `docs/human_eval_task_b_contextual.md`.

## Important Fixes Already Made

- Added explicit `seen_item_ids` to user profiles.
- Filtered already-reviewed/history items out of recommendations.
- Added regression tests for seen-item filtering and temporal splits.
- Made ingestion stream reviews first and filter metadata afterward instead of loading all metadata upfront.
- Added dataset byte-size checks so partial downloads are not treated as valid.
- Improved candidate generation to blend BM25, neighbors, and popularity fallback.
- Improved ranking to account for sparse-user cases with popularity and item quality priors.
- Added GitHub-safe ignore rules so large data and generated artifacts are not committed.
- Fixed editable Python packaging so `pip install -e ".[dev]"` works.
- Added local hashing embeddings and vector retrieval as a deterministic diagnostic hook; current vector source recall is `0.0`, so it is not a quality claim.
- Added score component breakdowns for recommendations.
- Added collaborative retrieval artifacts for item co-visitation and user-neighbor candidate generation.
- Added category-affinity popularity as a high-recall Task B source.
- Added review-term retrieval and lexical-neighbor retrieval for Task B candidate generation.
- Added aspect-aware evidence extraction for personas, review history, and item metadata.
- Added evidence graph retrieval for aspect-to-item, category-aspect, item-transition, and category-transition candidate paths.
- Added evidence-aware ranking features for aspect match, sequential match, evidence graph match, and Nigerian-context match.
- Added plan-then-write review generation for Task A.
- Added evidence critic checks for grounding and sensitive-inference risk.
- Added Beauty taxonomy retrieval and evidence graph retrieval as default candidate sources after measurement.
- Added context-category ranker guards and regression coverage for contextual recommendation relevance.
- Added candidate source tracking, retrieval scores, and response-level `candidate_diagnostics`.
- Added Task B candidate recall metrics and slice reporting so retrieval can be optimized before ranker promotion.
- Added miss-analysis JSON output and Markdown reporting for candidate misses.
- Added JSONL runtime traces, `/api/metrics`, `/api/traces`, and response `trace_id`s.
- Added an operational `/ui/` demo with Review Simulation, Recommend, and Metrics tabs.
- Added API contract, vector retrieval, and downloader integrity tests.
- Added Task A adaptive star prediction, slice metrics, and tuning via `make tune-task-a`.
- Added Task A model training via `eval/train_task_a_model.py`; trained artifacts are saved under ignored processed data directories.
- Added direct DeepSeek generation support through `DEEPSEEK_API_KEY` and `LLM_PROVIDER=deepseek`; `.env` remains ignored.
- Added external-data guards for DeepSeek/OpenRouter/OpenAI generation eval and generated human-eval scoring tables in `docs/`.
- Added richer Task A features: recency, volatility, star shares, user-category priors, reliability gaps, and semantic/profile interactions.
- Added Task A candidate matrix training with `--workers` support for local CPU utilization where process pools are allowed.

## Task A Restart Notes

If we restarted Task A, the proven path is:

1. Use temporal user splits from day one: earlier reviews as history, later reviews as targets.
2. Treat rating prediction as the core model; generation should happen after the rating is fixed.
3. Build user/item/category statistics first: shrinkage means, recent tendency, volatility, star shares, and user-category priors.
4. Add semantic/profile features as supporting signals, not the whole model.
5. Train candidate models and select by the submission metric, with RMSE as the current default.
6. Handle 1-5 stars as ordinal outputs with calibrated or rounded star policies.
7. Report overall MAE plus light/medium/warm-user, category, sparse-item, and rating-tendency slices.
8. Generate the review from predicted rating, user preferences, item facts, and prior review style.

Avoid pure prompting for the rating, random train/test splits, and lexical-only matching.

## Known Gaps

- Full all-category Task B evaluation is expensive and currently represented by a bounded slice.
- Task B candidate recall is still the main bottleneck: latest bounded all-category hybrid candidate Recall@1000 is `0.34`, while Recall@50 is only `0.13`.
- The 24 May positive-recommendation candidate-recall proof improved the retrieval diagnostic to Recall@50 `0.151`, Recall@100 `0.1823`, Recall@1000 `0.3986`, sparse Recall@1000 `0.3973`, and cross-domain Recall@1000 `0.6081`. It is not a final ranker promotion.
- Miss analysis should be rerun after each retrieval change; earlier reports showed Beauty-heavy misses, sparse-history misses, and items with no neighbor path.
- Evidence graph retrieval is enabled as a local candidate source, but source-level attribution should be reported carefully. Vector source recall is currently `0.0`.
- Current embeddings are deterministic hashing embeddings, not neural model embeddings.
- Current generation is optional and only used at the final text step.
- Test coverage is improved but still intentionally focused on core contracts.
- No external tracing/cost dashboard is wired; local JSONL metrics are implemented.
- Task B contextual human-eval scores are summarized in `docs/evaluation/HUMAN_EVAL_TASK_B_CONTEXTUAL_RESULTS.md`; the Task A review pack remains available for judge-facing behavioural review.

## Recommended Next Steps

1. Decide whether to download more Amazon Reviews 2023 categories; use `--all-categories` only if storage/time budget allows it.
2. Improve Task B on the completed real-data corpus:
   - continue Beauty-specific retrieval work to improve the dominant miss category
   - add sparse-user fallbacks beyond global popularity
   - rerun Task B with candidate Recall@200/500/1000 and miss reports
  - rerun same-target ranker training only after submission or if a short cached run is available
   - add diversity controls after relevance improves
3. Improve Task A:
   - reconcile validation-selected calibrated thresholds with slice-level rounded-star behavior
   - add cached feature rows for faster full-matrix retraining
   - add optional LightGBM/CatBoost/sklearn rankers when dependency policy allows it
   - review text quality metrics after rating quality stabilizes
4. Add deeper tests:
   - ingestion with metadata
   - more ingestion edge cases
   - real-data eval smoke checks
   - trace persistence failure modes
5. Keep the solution paper synced as larger same-slice evals replace the current bounded smoke metrics.

## Possible Improvements

- Promote neural embeddings only after same-slice all-category evals show lift over the current hybrid gate.
- Extend matrix factorization baselines only if they beat the completed `implicit` ALS/BPR/item-item report.
- Add sequence-aware recommendation for users with enough history.
- Cache profile and item features to avoid recomputation on every request.
- Persist traces to external observability storage for experiment comparison.
- Add LLM-as-judge only as a secondary qualitative metric, not the main score.

## GitHub Push Notes

The repo has been initialized and committed locally.

The repo has a configured `origin` remote:

```text
https://github.com/stmarkadebayo/bluechip.git
```

Recommended pre-push commands:

```bash
git status --short
./.venv/bin/ruff check .
./.venv/bin/pytest
./.venv/bin/python -m compileall app eval scripts tests
git add .env.example .gitignore Dockerfile Makefile README.md app docs eval paper pyproject.toml research scripts tests ui
git status --short
git commit -m "Tighten Bluechip agent for teammate handoff"
git push origin main
```

If pushing to a new remote instead:

```bash
git remote add origin https://github.com/YOUR_USERNAME/YOUR_REPO.git
git push -u origin main
```

Do not force-add `data/raw`, `data/processed`, `.env`, `.env.*`, or `runs/eval`.
