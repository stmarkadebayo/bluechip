# Bluechip User Intelligence Agent

Evidence-first review simulation and recommendation for the DSN x BCT LLM Agent Challenge.

Bluechip solves both required tasks with one shared user-intelligence layer:

```text
persona / history / context
  -> user and item evidence
  -> profiling, retrieval, ranking, generation, validation
  -> Task A review simulation or Task B recommendation
```

The LLM is optional and late in the pipeline. Rating prediction, retrieval, ranking, validation, metrics, and traces remain runnable without external API keys.

## Submission Status

The hack brief requires:

- Task A containerized app/API: user persona and product details in, rating and review out.
- Task B containerized app/API: user persona in, personalized recommendations out.
- Solution paper, 4-8 pages.
- Clean reproducible code repository.

This repository provides a FastAPI app, browser demo, Dockerfile, docker-compose, sample data, evaluation scripts, documentation, and deterministic fallbacks.

Judge-facing docs:

- [Solution paper](paper/solution_paper.md)
- [Submission freeze plan](docs/SUBMISSION_FREEZE.md)
- [Submission evaluation summary](docs/evaluation/SUBMISSION_EVAL_SUMMARY.md)
- [Dataset EDA](docs/evaluation/DATASET_EDA.md)
- [Implicit baseline results](docs/evaluation/IMPLICIT_BASELINE_RESULTS.md)
- [Quality review and source pruning](docs/evaluation/QUALITY_REVIEW_PRUNING.md)
- [Task B contextual human eval results](docs/evaluation/HUMAN_EVAL_TASK_B_CONTEXTUAL_RESULTS.md)
- [System architecture](docs/architecture/SYSTEM_ARCHITECTURE.md)
- [Demo script](docs/product/DEMO_SCRIPT.md)
- [Implementation log](docs/IMPLEMENTATION_LOG.md)

Frozen submission scope:

- Keep the runtime claim evidence-first and hybrid.
- Add only `implicit` ALS/BPR/item-item baselines as new model work before submission.
- Do not start LightGCN, SASRec, HSTU, PETER, PEPLER, NARRE, or a trained Wide & Deep model for this deadline.
- Task B contextual human eval is complete and summarized; Task A behavioural human eval can still be added if time allows.
- Run final validation and paper polish after any last human-eval-driven ranking tweaks.

The detailed eight-step plan is in [docs/SUBMISSION_FREEZE.md](docs/SUBMISSION_FREEZE.md).

## Current Metric Snapshot

These are bounded local metrics, not inflated full-product claims.

| Area | Metric | Current value |
| --- | --- | ---: |
| Task A | Latest documented 5,000-example all-category RMSE gate | `1.2654` |
| Task B | `hybrid_candidate_recall@50` | `0.13` |
| Task B | `hybrid_candidate_recall@100` | `0.18` |
| Task B | `hybrid_candidate_recall@1000` | `0.34` |
| Task B | `hybrid_ranker_hit_rate@10` | `0.10` |
| Task B | `hybrid_ranker_ndcg@10` | `0.0766` |
| Task B | Sparse candidate recall@1000 | `0.3611` |
| Task B | Cross-domain candidate recall@1000 | `0.5484` |
| Task B | Vector source recall | `0.0` |

Interpretation:

- Task A is rating-first: predict the star rating, then generate and validate the review.
- Task B is retrieval-before-ranking: candidate recall is measured separately from top-10 ranking.
- Cross-domain candidate retrieval is currently the strongest Task B slice.
- Vector retrieval is a diagnostic/extensibility hook today, not a promoted quality claim.
- Retrieval source metadata, default disabling, family labels, and score calibration are centralized in `app/services/retrieval/source_registry.py`.

## Architecture

```text
app/
  api/                  FastAPI routes
  serving/              task orchestrators and trace boundaries
  models/               typed request/response schemas
  platform/             local feature store and model/index registry
  services/
    profiling/          user and item behavioral profiles
    intelligence/       aspect-aware evidence extraction
    retrieval/          multi-head retrieval and source diagnostics
    ranking/            rating prediction and recommendation scoring
    generation/         review/reason text generation
    validation/         consistency and grounding checks
    nigerian/           Nigerian context and voice signals
    conversation/       multi-turn state
  stores/               trace/profile stores

data/
  sample/               tiny reproducible sample data
  raw/                  local raw datasets, not committed
  processed/            local processed artifacts, not committed

eval/                   task and evidence evaluation scripts
docs/                   judge-facing documentation
paper/                  solution paper
prompts/                generation prompt contracts
scripts/                ingestion, split, index, and registry builders
ui/                     browser demo mounted at /ui/
```

Task A flow:

```text
POST /api/simulate-review
  -> build user profile
  -> build target item profile
  -> predict rating
  -> generate rating-conditioned review
  -> validate consistency and grounding
  -> return rating, review, signals, validation, trace
```

Task B flow:

```text
POST /api/recommend
  -> build user profile
  -> retrieve candidates from multiple source heads
  -> rank with explicit score components
  -> generate reasons and tradeoffs
  -> return recommendations, source diagnostics, trace
```

The lean serving path disables noisy vector/sparse-tail sources by default through the shared source registry. Set `BLUECHIP_DISABLED_RETRIEVAL_SOURCES` only when deliberately running an ablation or experiment.

## Quick Start

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
uvicorn app.main:app --reload
```

Open:

```text
http://127.0.0.1:8000/ui/
http://127.0.0.1:8000/docs
http://127.0.0.1:8000/api/health
```

Run with Docker:

```bash
docker compose up --build
```

The API is exposed on:

```text
http://127.0.0.1:8000
```

## Deploy

The repo includes `render.yaml` for a Docker-based Render deployment. For a same-origin demo, share the hosted `/ui/` path rather than the local `file://` UI.

Required hosted settings:

```bash
LLM_PROVIDER=openrouter
OPENROUTER_MODEL=deepseek/deepseek-v4-flash:free
OPENROUTER_API_KEY=...
BLUECHIP_PROFILE_ENHANCER=true
BLUECHIP_ALLOW_MODEL_DOWNLOAD=false
DATABASE_URL=postgresql://...
BLUECHIP_RATE_LIMIT_ENABLED=true
```

The Render Blueprint uses the free web plan and expects an external Postgres URL for durable traces and conversations. Neon works well for this path: create a Neon database, copy its connection string, and set it as `DATABASE_URL` in Render. If `DATABASE_URL` is not set, the app falls back to local SQLite, which is fine locally but ephemeral on free hosting. The Dockerfile binds to `${PORT:-8000}`, so it works locally and on hosts that inject `PORT`.

Build or refresh the lean artifact bundle before committing a deploy:

```bash
python scripts/build_deploy_artifacts.py
```

That command copies the small sample/index artifacts into `data/deploy/processed`, generates deploy-safe Task A stats/policy, and avoids packaging the full local `data/processed` directory.

Before sharing a free-tier deployment, warm the service once:

```bash
python tests/api_smoke.py --base-url https://your-host.example
```

Free-tier instances can sleep. For live judging, prewarm with the smoke test and keep `DATABASE_URL` configured so conversation state and traces survive container restarts.

## Environment

The app runs without a `.env` file. Copy `.env.example` to `.env` only when using external generation providers or local full-data artifacts.

```bash
cp .env.example .env
```

Supported provider modes:

```bash
# deterministic local mock useful for demos/tests
LLM_PROVIDER=mock

# template fallback, no external calls
LLM_PROVIDER=template

# OpenRouter
LLM_PROVIDER=openrouter
OPENROUTER_MODEL=deepseek/deepseek-v4-flash:free
OPENROUTER_API_KEY=...

# DeepSeek direct
LLM_PROVIDER=deepseek
DEEPSEEK_MODEL=deepseek-v4-flash
DEEPSEEK_API_KEY=...

# OpenAI Responses API
LLM_PROVIDER=openai
LLM_MODEL=gpt-5
OPENAI_API_KEY=...
```

If `OPENROUTER_API_KEY` or `DEEPSEEK_API_KEY` is set and `LLM_PROVIDER` is blank, the runtime auto-selects that provider. Without credentials, deterministic fallback remains available.

Optional profile enhancement:

```bash
# off by default so offline evals are deterministic
BLUECHIP_PROFILE_ENHANCER=false
```

Requests can also opt in per call with `"enhance_with_llm": true`. The enhancer is bounded: it only adds a few inferred terms/aspects/categories, records provenance in `profile_enhancement`, and falls back to the deterministic profile on provider failure.

Optional local artifact paths:

```bash
TASK_A_MODEL_PATH=data/processed/all_categories/task_a_model_rmse.json
TASK_A_STATS_PATH=data/processed/all_categories/task_a_rating_stats.json
TASK_A_SERVING_POLICY=data/processed/all_categories/task_a_serving_policy.json
TASK_B_RETRIEVAL_INDEX=data/processed/all_categories/collaborative_retrieval.json
TASK_B_EVIDENCE_GRAPH_INDEX=data/processed/all_categories/evidence_graph_retrieval.json
TASK_B_RANKER_WEIGHTS=
```

Optional SQLite feature store:

```bash
make sqlite-feature-store PROCESSED_DIR=data/processed/all_categories
BLUECHIP_FEATURE_STORE_SQLITE=data/processed/all_categories/feature_store.sqlite
```

The SQLite backend preserves JSON payloads but adds indexed point lookups for items and user histories. If `BLUECHIP_FEATURE_STORE_SQLITE` is unset, the app uses the existing JSONL feature store.

## API Examples

### Health

```bash
curl http://127.0.0.1:8000/api/health
```

### Task A: Simulate Review

```bash
curl -X POST http://127.0.0.1:8000/api/simulate-review \
  -H "Content-Type: application/json" \
  -d '{
    "user_persona": "A Lagos student who likes affordable products, rates slow delivery harshly, and prefers practical details.",
    "user_history": [
      {
        "item_id": "beauty_001",
        "item_name": "Shea Moisture Cream",
        "rating": 5,
        "review": "Good value, original product, and delivery was fast.",
        "category": "All_Beauty"
      }
    ],
    "target_item": {
      "item_id": "beauty_010",
      "name": "Coconut Hair Cream",
      "category": "All_Beauty",
      "metadata": {"price": "affordable", "delivery": "fast"},
      "summary": "Moisturizing coconut hair cream for natural hair.",
      "average_rating": 4.2
    },
    "locale": "Nigeria"
  }'
```

Response includes `predicted_rating`, `review`, `confidence`, `user_signals`, `item_signals`, `validation`, `agent_trace`, and `trace_id`.

### Task B: Recommend

```bash
curl -X POST http://127.0.0.1:8000/api/recommend \
  -H "Content-Type: application/json" \
  -d '{
    "user_persona": "A Lagos student who likes affordable natural hair products and cares about original quality.",
    "user_history": [],
    "context": "Need a reliable hair product under a tight budget.",
    "candidate_items": [
      {
        "item_id": "beauty_010",
        "name": "Coconut Hair Cream",
        "category": "All_Beauty",
        "metadata": {"price": "affordable", "delivery": "fast"},
        "summary": "Moisturizing coconut hair cream for natural hair.",
        "average_rating": 4.2
      },
      {
        "item_id": "beauty_011",
        "name": "Luxury Perfume Oil",
        "category": "All_Beauty",
        "metadata": {"price": "premium"},
        "summary": "Long lasting fragrance oil.",
        "average_rating": 4.5
      }
    ],
    "locale": "Nigeria",
    "limit": 2
  }'
```

Response includes ranked `recommendations`, `candidate_sources`, `retrieval_scores`, `score_components`, `candidate_diagnostics`, `agent_trace`, and `trace_id`.

### Multi-Turn And Diagnostics

```text
POST /api/conversation/turn
POST /api/conversation/{conversation_id}/feedback
GET  /api/conversation/{conversation_id}
GET  /api/conversations
POST /api/infer-cold-start
POST /api/transfer-cross-domain
POST /api/nigerian/context
GET  /api/metrics
GET  /api/traces
GET  /api/runtime/registry
GET  /api/runtime/feature-store
```

Use `http://127.0.0.1:8000/docs` for exact schemas.

## Evaluation

### Fast Sample Evaluation

```bash
make eval
```

Equivalent explicit commands:

```bash
python scripts/build_splits.py \
  --reviews data/sample/reviews.jsonl \
  --items data/sample/items.jsonl \
  --output-dir data/processed

python scripts/build_retrieval_index.py \
  --train data/processed/train.jsonl \
  --items data/processed/items.jsonl \
  --output-dir data/processed

python eval/eval_task_a.py
python eval/eval_task_b.py
```

### Quality Gates

```bash
ruff check .
pytest
python -m compileall app eval scripts tests
```

### Task A Rating Gate

Use this when full processed data exists under `data/processed/all_categories`:

```bash
python eval/eval_task_a.py \
  --reviews data/processed/all_categories/reviews.jsonl \
  --items data/processed/all_categories/items.jsonl \
  --processed-dir data/processed/all_categories \
  --output runs/eval/all_categories_task_a_with_model_rmse_5000.json \
  --max-examples 5000 \
  --model-path data/processed/all_categories/task_a_model_rmse.json

python eval/promote_task_a.py \
  --task-a-report runs/eval/all_categories_task_a_with_model_rmse_5000.json \
  --output runs/eval/all_categories_task_a_serving_promotion.json \
  --policy-output data/processed/all_categories/task_a_serving_policy.json
```

### Task A Generation Smoke

```bash
python eval/eval_task_a_generation.py \
  --strict-provider \
  --external-data-policy redact \
  --max-examples 25
```

Generation eval reports provider failures separately from quality and tracks fallback rate, validation consistency, rating mention, item grounding, sentiment alignment, ROUGE-L F1, and unigram F1.

### Task B Candidate And Ranking Gate

```bash
python eval/eval_task_b.py \
  --processed-dir data/processed/all_categories \
  --output runs/eval/submission_task_b_100x1000.json \
  --miss-output runs/eval/submission_task_b_100x1000_misses.json \
  --max-examples 100 \
  --candidate-limit 1000
```

Lean/pruned gate:

```bash
python eval/eval_task_b.py \
  --processed-dir data/processed/all_categories \
  --output runs/eval/task_b_pruned_100x1000.json \
  --miss-output runs/eval/task_b_pruned_100x1000_misses.json \
  --max-examples 100 \
  --candidate-limit 1000 \
  --hybrid-only \
  --disabled-sources vector_profile,bm25_profile,beauty_sparse_tail,sparse_category_tail,neural_vector
```

Source ablations:

```bash
python eval/run_task_b_source_ablation.py \
  --processed-dir data/processed/all_categories \
  --max-examples 100 \
  --candidate-limit 1000
```

### Contextual Human-Eval Pack

```bash
python eval/create_task_b_contextual_eval.py \
  --processed-dir data/processed/all_categories \
  --output docs/human_eval_task_b_contextual.md \
  --max-examples 20 \
  --candidate-limit 1000
```

## Data Pipeline

The repo includes tiny sample data for reproducibility. Full Amazon Reviews 2023 data is downloaded locally and is not committed.

Download/check the configured Amazon subset:

```bash
python scripts/download_amazon_hf.py --with-metadata
python scripts/download_amazon_hf.py --with-metadata --check-only --strict
```

Ingest one category:

```bash
python scripts/ingest_amazon.py \
  --reviews data/raw/All_Beauty.jsonl \
  --metadata data/raw/meta_All_Beauty.jsonl \
  --category All_Beauty \
  --output-dir data/processed/categories/All_Beauty
```

Build combined processed data:

```bash
python scripts/build_combined_processed.py \
  --input-root data/processed/categories \
  --output-dir data/processed/all_categories

python scripts/build_splits.py \
  --reviews data/processed/all_categories/reviews.jsonl \
  --items data/processed/all_categories/items.jsonl \
  --output-dir data/processed/all_categories

python scripts/build_retrieval_index.py \
  --train data/processed/all_categories/train.jsonl \
  --items data/processed/all_categories/items.jsonl \
  --output-dir data/processed/all_categories \
  --top-k 100

python scripts/build_evidence_graph.py \
  --train data/processed/all_categories/train.jsonl \
  --items data/processed/all_categories/items.jsonl \
  --output data/processed/all_categories/evidence_graph_retrieval.json

python scripts/build_model_registry.py \
  --output data/processed/all_categories/model_registry.json
```

Build a neural FAISS index only when dependencies and storage are available:

```bash
python scripts/build_neural_index.py \
  --items data/processed/all_categories/items.jsonl \
  --output-dir data/processed/all_categories
```

Treat neural/vector results as experimental unless fixed eval reports show a lift.

## Demo Path

1. Start the app with `uvicorn app.main:app --reload` or `docker compose up --build`.
2. Open `http://127.0.0.1:8000/ui/`.
3. Run Task B with a Nigerian cold-start or cross-domain context.
4. Show candidate sources, score components, diagnostics, and trace ID.
5. Run Task A with a visible user history and target item.
6. Show predicted rating, generated review, validation status, and trace.
7. Open `/api/metrics` and `/api/traces`.

Use [docs/product/DEMO_SCRIPT.md](docs/product/DEMO_SCRIPT.md) for the full talk track.

## Design Principles

- Shared user model, two task heads.
- Retrieval before ranking; ranking before explanation.
- Rating before review generation.
- LLMs enhance language, not hidden scoring.
- Nigerian context is applied only when persona, locale, or item evidence supports it.
- Every major claim needs a metric, ablation, trace, or documented limitation.
- Deterministic fallback keeps the repo runnable for judges without secrets.
