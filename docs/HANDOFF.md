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
6. We attempted to download five real Amazon Reviews 2023 categories. One completed; four remain partial because network transfers reset or stalled.
7. We initialized Git, added ignore rules, CI, and committed the project so it can be pushed for collaboration.

## What Is Built

### Application

- FastAPI app in `app/main.py`.
- API routes in `app/api/routes.py`.
- Endpoints:
  - `GET /api/health`
  - `POST /api/profile-user`
  - `POST /api/simulate-review`
  - `POST /api/recommend`

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
- Generation provider abstraction: `app/services/generation/providers.py`
- Review/recommendation text generation: `app/services/generation/generator.py`
- Output validation: `app/services/validation/critic.py`

### Data and Evaluation

- Amazon JSONL ingestion: `scripts/ingest_amazon.py`
- Temporal split builder: `scripts/build_splits.py`
- Retrieval index builder: `scripts/build_retrieval_index.py`
- Dataset downloader/checker: `scripts/download_amazon_hf.py`
- Shared eval helpers: `eval/common.py`
- Task A eval: `eval/eval_task_a.py`
- Task B eval: `eval/eval_task_b.py`
- Metrics: `eval/metrics.py`

### Documentation

- Main setup and architecture: `README.md`
- Literature and industry review: `research/literature_review.md`
- Production architecture notes: `infra/production_architecture.md`
- Solution paper draft: `paper/solution_paper.md`
- Prompt contracts: `prompts/`
- This handoff: `docs/HANDOFF.md`

### DevOps

- Dockerfile and docker-compose are included.
- GitHub Actions CI is included at `.github/workflows/ci.yml`.
- Raw datasets, processed datasets, eval outputs, virtualenvs, and caches are ignored.
- `.env.example` documents optional LLM provider settings without committing secrets.

## Current Dataset State

Raw datasets are intentionally ignored by Git. Teammates should download them locally.

Complete locally:

- `data/raw/Subscription_Boxes.jsonl`
- `data/raw/meta_Subscription_Boxes.jsonl`

The completed `Subscription_Boxes` dataset ingests to:

- 16,215 reviews
- 641 items
- 15,236 users

Partial local downloads:

- `All_Beauty.jsonl`: partial, expected 326,611,506 bytes
- `Digital_Music.jsonl`: partial, expected 78,823,304 bytes
- `Magazine_Subscriptions.jsonl`: partial, expected 33,297,013 bytes
- `Gift_Cards.jsonl`: partial, expected 50,231,035 bytes

Metadata still missing for those four partial categories.

Check dataset integrity:

```bash
python scripts/download_amazon_hf.py --with-metadata --check-only
```

Resume downloads:

```bash
python scripts/download_amazon_hf.py --with-metadata
```

Use one category for a fast real-data run:

```bash
python scripts/ingest_amazon.py \
  --reviews data/raw/Subscription_Boxes.jsonl \
  --metadata data/raw/meta_Subscription_Boxes.jsonl \
  --category Subscription_Boxes \
  --output-dir data/processed
```

## Local Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
```

Run checks:

```bash
python3 -m pytest
make eval
```

Run API:

```bash
uvicorn app.main:app --reload
```

Open:

```text
http://127.0.0.1:8000/docs
```

Optional OpenAI-backed generation:

```bash
LLM_PROVIDER=openai
LLM_MODEL=gpt-5
OPENAI_API_KEY=...
```

Without these variables, the system uses deterministic fallback text generation.

## What Was Validated

These checks passed locally:

- Python compile check across `app`, `eval`, `scripts`, and `tests`.
- `python3 -m pytest`: 2 tests passed.
- `make eval`: sample Task A and Task B eval passed.
- FastAPI import smoke test passed.
- `Subscription_Boxes` real-data ingestion passed.
- `Subscription_Boxes` temporal split and retrieval index build passed.
- `Subscription_Boxes` Task A and Task B eval scripts ran successfully.

Current sample eval signal:

- Task A hybrid profile scorer beats the simple sample baselines on MAE/RMSE.
- Task B hybrid ranker passes the sample eval and ranks held-out items well on the tiny sample.

Current real-data signal:

- Task A runs end to end on `Subscription_Boxes`.
- Task B runs end to end, but sparse-user ranking needs improvement. Popularity remains a strong baseline on the real dataset.

## Important Fixes Already Made

- Added explicit `seen_item_ids` to user profiles.
- Filtered already-reviewed/history items out of recommendations.
- Added regression tests for seen-item filtering and temporal splits.
- Made ingestion stream reviews first and filter metadata afterward instead of loading all metadata upfront.
- Added dataset byte-size checks so partial downloads are not treated as valid.
- Improved candidate generation to blend BM25, neighbors, and popularity fallback.
- Improved ranking to account for sparse-user cases with popularity and item quality priors.
- Added GitHub-safe ignore rules so large data and generated artifacts are not committed.

## Known Gaps

- Four of five intended Amazon categories are not fully downloaded yet.
- Task B ranking needs a stronger real-data strategy.
- Current profiles are lexical and heuristic, not embedding-based.
- Current generation is optional and only used at the final text step.
- No UI is implemented yet beyond API docs.
- Test coverage is still minimal.
- No model tracing/cost dashboard yet.
- No learned ranker or offline hyperparameter search yet.

## Recommended Next Steps

1. Finish downloads for the remaining categories.
2. Ingest each category separately and build combined splits.
3. Improve Task B before polishing the demo:
   - stronger item-item co-visitation
   - category-aware popularity priors
   - diversity controls
   - better candidate source blending
   - ablation metrics by retrieval source
4. Improve Task A:
   - user strictness calibration
   - item/user bias blending
   - cold-start vs warm-user eval slices
   - review text quality metrics after rating quality stabilizes
5. Add more tests:
   - ingestion with metadata
   - downloader integrity states
   - candidate generation excluding history
   - API contract tests
6. Build a small demo UI after ranking quality is more defensible.
7. Update the solution paper with real metrics and ablation tables.

## Possible Improvements

- Replace heuristic text profiles with embedding-backed user and item representations.
- Use a two-stage recommender: candidate retrieval, then learned ranker.
- Add matrix factorization or implicit-feedback ALS as a baseline.
- Add sequence-aware recommendation for users with enough history.
- Cache profile and item features to avoid recomputation on every request.
- Persist traces and evaluation runs for experiment comparison.
- Add LLM-as-judge only as a secondary qualitative metric, not the main score.
- Add a Streamlit, Next.js, or simple HTML demo that shows traces and evidence.

## GitHub Push Notes

The repo has been initialized and committed locally.

Current commit:

```text
ddbd2f3 Prepare Bluechip agent project
```

To push:

```bash
git remote add origin https://github.com/YOUR_USERNAME/YOUR_REPO.git
git push -u origin main
```

Do not force-add `data/raw`, `data/processed`, `.env`, or `runs/eval`.
