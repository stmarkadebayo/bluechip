# Bluechip Handoff

This document summarizes how the project got to its current state, what is implemented, what was validated, and what should happen next.

## Project Context

The hackathon brief asks for an LLM agent that solves two related tasks:

- **Task A: User modeling / review simulation**: infer a user's likely rating and generate a review for a target item.
- **Task B: Personalized recommendation**: rank candidate items for a user and explain why they fit.

We interpreted "Build an agent" as an orchestration layer over deterministic, testable tools rather than a single prompt. The agent profiles the user, profiles items, retrieves candidates, ranks or predicts ratings, generates text, validates outputs, and returns an explicit trace.

## How We Got Here

1. We read the hackathon brief and narrowed the goal to the core build: a shared user-intelligence engine with two serving heads.
2. We clarified that "staff-level" means scalable, measurable, reproducible, and not just prompt-based.
3. We did a research pass across recommender systems, user modeling, agent architecture, retrieval, and industry patterns.
4. We chose a production-shaped architecture:
   - offline ingestion and feature building
   - typed profile and item contracts
   - retrieval before ranking
   - deterministic scoring before LLM generation
   - optional LLM only at the final language step
   - eval scripts as first-class project artifacts
5. We implemented the repository from scratch, then reviewed it for correctness, scalability, and GitHub readiness.
6. We downloaded and byte-validated five Amazon Reviews 2023 categories plus metadata.
7. We built a combined all-category processed corpus and trained Task A models selected for both validation MAE and validation RMSE.

## What Is Built

### Application

- FastAPI app in `app/main.py`.
- API routes in `app/api/routes.py`.
- Endpoints:
  - `GET /api/health`
  - `POST /api/profile-user`
  - `POST /api/simulate-review`
  - `POST /api/recommend`
  - `GET /api/metrics`
  - `GET /api/traces`
- Browser demo mounted at `/ui/`.

### Agent Workflows

- `ReviewSimulationAgent`
  - builds user profile
  - builds target item profile
  - predicts rating
  - generates review
  - validates rating-review consistency
  - returns an agent trace

- `RecommendationAgent`
  - builds user profile
  - retrieves candidate items
  - filters seen/history items
  - ranks candidates
  - generates grounded explanations
  - returns an agent trace

### Core Services

- User profiling: `app/services/profiling/user_profile.py`
- Item profiling: `app/services/profiling/item_profile.py`
- BM25 retrieval: `app/services/retrieval/text.py`
- Item co-occurrence retrieval: `app/services/retrieval/item_similarity.py`
- Candidate generation: `app/services/retrieval/candidates.py`
- Rating prediction: `app/services/ranking/rating.py`
- Recommendation ranking: `app/services/ranking/recommendation.py`
- Dependency-free vector retrieval: `app/services/retrieval/vector_store.py`
- Deterministic hashing embeddings: `app/services/retrieval/embeddings.py`
- Generation provider abstraction: `app/services/generation/providers.py`
- Review/recommendation text generation: `app/services/generation/generator.py`
- Output validation: `app/services/validation/critic.py`
- Runtime trace store: `app/stores/trace_store.py`

### Data and Evaluation

- Amazon JSONL ingestion: `scripts/ingest_amazon.py`
- Temporal split builder: `scripts/build_splits.py`
- Retrieval index builder: `scripts/build_retrieval_index.py`
- Dataset downloader/checker: `scripts/download_amazon_hf.py`
- Shared eval helpers: `eval/common.py`
- Task A eval: `eval/eval_task_a.py`
- Task B eval: `eval/eval_task_b.py`
- Ranker tuning: `eval/tune_ranker.py`
- Learned ranker training: `eval/train_ranker.py`
- Metrics: `eval/metrics.py`

### Documentation

- Main setup and architecture: `README.md`
- Literature and industry review: `research/literature_review.md`
- Production architecture notes: `infra/production_architecture.md`
- Solution paper draft: `paper/solution_paper.md`
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
- Current hardening pass: 31 tests pass, lint passes, compile passes, sample eval passes, and bounded all-category Task B eval runs.
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
- On a 100-example all-category Task B slice over 188,236 items, hybrid HitRate@10 is `0.09` and NDCG@10 is `0.068`, matching filtered popularity.
- After the candidate-recall upgrade, a 25-example all-category smoke eval reports base and hybrid candidate Recall@200 both at `0.16`; hybrid no longer crowds out the base candidate pool on that slice.
- Full all-category collaborative artifacts were built locally:
  - `data/processed/all_categories/collaborative_retrieval.json` (`237M`)
  - `data/processed/all_categories/item_neighbors.json` (`25M`)
  - `data/processed/all_categories/review_term_retrieval.json` (`421M`)
- Updated 100-example all-category Task B eval over 188,236 items:
  - candidate Recall@200 `0.20`
  - candidate Recall@500 `0.25`
  - candidate Recall@1000 `0.28`
  - hybrid HitRate@10 `0.10` vs filtered popularity `0.09`
  - hybrid NDCG@10 `0.0766` vs filtered popularity `0.068`
  - cross-domain HitRate@10 `0.2581`, NDCG@10 `0.2056`
- Beauty/sparse retrieval adds item-title terms, aspect retrieval, and sparse category-tail exploration:
  - candidate Recall@1000 `0.29`
  - candidate misses `71`
  - hybrid HitRate@10 `0.10`, NDCG@10 `0.0766`
- Review-term retrieval and lexical-neighbor retrieval add positive-review language and item-term postings:
  - candidate Recall@1000 `0.32`
  - candidate misses `68`
  - hybrid HitRate@10 `0.10`, NDCG@10 `0.0766`
  - ranker promotion remains blocked: learned holdout NDCG@10 `0.0788` vs same-slice hybrid `0.1061`
- Graph-walk retrieval was implemented and measured as an ablation, but is not enabled in default candidate generation because it added latency without improving the 100-example all-category metrics.
- Beauty taxonomy retrieval was measured on a 25-example smoke slice and gated off because it crowded out stronger candidate sources.
- Context-category ranking guards are enabled for explicit Beauty, music, and gift contexts; this improves the human-eval/contextual demo path without changing empty-context offline metrics.
- Contextual human-eval pack: `docs/human_eval_task_b_contextual.md`.
- Candidate-aware learned ranker trained on the same 100-example all-category slice was not promoted because learned NDCG@10 `0.0700` trailed current hybrid NDCG@10 `0.0766`.
- Split learned-ranker training now uses a holdout slice and is rejected unless it beats same-slice hybrid. Latest split result was not promoted: learned NDCG@10 `0.0788` vs hybrid `0.1061`.

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
- Added local hashing embeddings and vector retrieval as a deterministic semantic signal.
- Added score component breakdowns for recommendations.
- Added collaborative retrieval artifacts for item co-visitation and user-neighbor candidate generation.
- Added category-affinity popularity as a high-recall Task B source.
- Added review-term retrieval and lexical-neighbor retrieval for Task B candidate generation.
- Added graph-walk and Beauty-taxonomy ablation code paths, but did not enable them by default after measurement.
- Added context-category ranker guards and regression coverage for contextual recommendation relevance.
- Added candidate source tracking, retrieval scores, and response-level `candidate_diagnostics`.
- Added Task B candidate recall metrics and slice reporting so retrieval can be optimized before ranker promotion.
- Added miss-analysis JSON output and Markdown reporting for candidate misses.
- Added a candidate-aware ranker trainer plus `eval/promote_ranker.py` promotion gate.
- Added JSONL runtime traces, `/api/metrics`, `/api/traces`, and response `trace_id`s.
- Added an operational `/ui/` demo with Review Simulation, Recommend, and Metrics tabs.
- Added API contract, vector retrieval, and downloader integrity tests.
- Added ranker hyperparameter search via `make tune`.
- Added a dependency-free pairwise learned-ranker training path via `make train-ranker`.
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
- Task B candidate recall is still the main bottleneck: Recall@1000 is `0.32` on the bounded all-category slice after review-term retrieval.
- Miss analysis shows the dominant miss categories are `All_Beauty` (`60` of `68` misses at candidate-limit `1000`), sparse history (`48` misses), and items with no item-neighbor path (`37` misses).
- Current embeddings are deterministic hashing embeddings, not neural model embeddings.
- Current generation is optional and only used at the final text step.
- Test coverage is improved but still intentionally focused on core contracts.
- No external tracing/cost dashboard is wired; local JSONL metrics are implemented.
- Learned-ranker training is implemented, but learned weights should not be promoted until real-data evaluation is complete.
- Contextual human-eval scores still need actual human labels; the judge-ready table is generated but intentionally unscored.

## Recommended Next Steps

1. Decide whether to download more Amazon Reviews 2023 categories; use `--all-categories` only if storage/time budget allows it.
2. Improve Task B on the completed real-data corpus:
   - continue Beauty-specific retrieval work to improve the dominant miss category
   - add sparse-user fallbacks beyond global popularity
   - rerun Task B with candidate Recall@200/500/1000 and miss reports
   - promote learned ranker weights only after `eval/promote_ranker.py` passes
   - fill human scores in `docs/human_eval_task_b_contextual.md`
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
5. Run `make tune` on real processed data and freeze better ranker weights.
6. Run `make train-ranker` on real processed data and compare against tuned heuristic weights.
7. Update the solution paper with real metrics and ablation tables.

## Possible Improvements

- Replace deterministic hashing embeddings with stronger neural embeddings when credentials/runtime allow it.
- Use a two-stage recommender: candidate retrieval, then learned ranker.
- Add matrix factorization or implicit-feedback ALS as a baseline.
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
