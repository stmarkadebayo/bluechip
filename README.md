# Bluechip User Intelligence Agent

Scalable reference implementation for the DSN x BCT LLM Agent Challenge.

This project is organized around one shared capability: converting user evidence into a reusable behavioral profile, then using that profile for both hackathon tasks.

```text
User history / persona
  -> user intelligence engine
  -> review simulation
  -> personalized recommendation
```

## What We Are Building

The hackathon has two required tasks:

1. **Task A: User Modeling**
   Given a user persona/history and a target item, predict the user's likely star rating and generate a review in that user's behavioral style.

2. **Task B: Recommendation**
   Given a user persona/history and optional current context, rank personalized recommendations and explain why they fit.

The repository treats both tasks as different serving heads on top of the same profiling, retrieval, ranking, generation, and validation pipeline.

For teammate handoff and current dataset status, see [docs/HANDOFF.md](docs/HANDOFF.md).
For GitHub push readiness, see [docs/PUSH_READY.md](docs/PUSH_READY.md).
For the latest hardening pass and validation log, see
[docs/IMPLEMENTATION_LOG.md](docs/IMPLEMENTATION_LOG.md).

## Architecture

```text
Raw reviews and metadata
  -> ingestion and normalization
  -> user profile extraction
  -> item profile extraction
  -> retrieval index
  -> ranking / rating prediction
  -> grounded generation
  -> validation
  -> API / UI
```

The design is grounded in the literature and industry survey in
[research/literature_review.md](research/literature_review.md).

Local hackathon implementation:

```text
app/
  api/                  FastAPI routes
  core/                 configuration and runtime settings
  models/               typed request/response schemas
  services/
    profiling/          user and item behavioral profiles
    retrieval/          local candidate retrieval interface
    ranking/            rating prediction and recommendation scoring
    generation/         final review/explanation generation
    validation/         consistency and grounding checks
  stores/               storage interfaces and local stores

data/
  raw/                  original datasets, not committed
  processed/            normalized local artifacts, not committed
  sample/               tiny reproducible sample data

eval/                   task-specific evaluation scripts
infra/                  production mapping and deployment notes
paper/                  solution paper draft
prompts/                prompt contracts for generation/critic steps
tests/                  unit and integration tests
ui/                     operational browser demo
```

Production mapping:

| Local component | Scalable equivalent |
| --- | --- |
| JSONL files | Object storage data lake |
| Python preprocessing | Spark, Ray, Beam, or scheduled batch jobs |
| SQLite/local metadata | Postgres, BigQuery, Snowflake, or feature store |
| Local retrieval | FAISS, pgvector, Milvus, Pinecone, or Weaviate |
| In-process ranker | Dedicated ranking service |
| FastAPI | Containerized API on ECS, Kubernetes, Cloud Run, or similar |
| Local eval scripts | CI quality gates and batch evaluation jobs |
| Prompt logs | Tracing, model observability, and cost monitoring |

## Request Flow

### Task A: Review Simulation

```text
POST /api/simulate-review

1. ReviewSimulationAgent inspects the request
2. Build or load user profile
3. Build or load target item profile
4. Predict rating with deterministic scorer
5. Generate review with optional LLM provider or deterministic fallback
6. Validate rating-review consistency
7. Return structured response and agent trace
```

### Task B: Recommendation

```text
POST /api/recommend

1. RecommendationAgent inspects history/context
2. Build or load user profile
3. Generate candidates from co-visitation, user-neighbor CF, review-term retrieval, lexical item-neighbor retrieval, BM25, vector retrieval, category-affinity popularity, and global popularity fallback
4. Track candidate sources and retrieval scores for observability
5. Rank candidates using hybrid scoring
6. Apply explicit context-category guards for clear Beauty, music, and gift contexts
7. Generate concise explanations with optional LLM provider or deterministic fallback
8. Return ranked recommendations, candidate diagnostics, and agent trace
```

## API Shape

### `POST /api/simulate-review`

Input:

```json
{
  "user_persona": "A Lagos-based student who likes affordable, quiet restaurants and rates slow service harshly.",
  "user_history": [
    {
      "item_id": "rest_001",
      "item_name": "Campus Bistro",
      "rating": 5,
      "review": "Affordable food, calm space, and the staff did not waste time."
    }
  ],
  "target_item": {
    "item_id": "rest_010",
    "name": "Mainland Grill",
    "category": "restaurant",
    "metadata": {
      "price": "medium",
      "ambience": "quiet",
      "service": "fast"
    },
    "summary": "Known for grilled food, calm seating, and reliable service."
  },
  "locale": "Nigeria"
}
```

Output:

```json
{
  "predicted_rating": 4,
  "review": "I would give Mainland Grill a solid 4 stars...",
  "confidence": 0.74,
  "user_signals": ["likes quiet ambience", "values fast service"],
  "item_signals": ["quiet ambience", "fast service"],
  "validation": {
    "is_consistent": true,
    "issues": []
  }
}
```

### `POST /api/recommend`

Input:

```json
{
  "user_persona": "A Lagos-based student who likes affordable, quiet restaurants and spicy food.",
  "user_history": [],
  "context": "Dinner with two friends, not too expensive, somewhere conversation-friendly.",
  "candidate_items": [
    {
      "item_id": "rest_010",
      "name": "Mainland Grill",
      "category": "restaurant",
      "metadata": {
        "price": "medium",
        "ambience": "quiet"
      },
      "summary": "Grilled food, calm seating, and reliable service."
    }
  ],
  "locale": "Nigeria",
  "limit": 5
}
```

Output:

```json
{
  "recommendations": [
    {
      "rank": 1,
      "item_id": "rest_010",
      "name": "Mainland Grill",
      "score": 0.81,
      "reason": "Matches the user's preference for calm, affordable dinner spots.",
      "tradeoffs": "Medium price, so it may not be the cheapest option.",
      "candidate_sources": ["bm25_profile", "vector_profile"],
      "retrieval_scores": {"bm25_profile": 1.0, "vector_profile": 0.42},
      "score_components": {
        "preference_match": 0.5,
        "context_match": 0.33,
        "collaborative_match": 0.0,
        "popularity": 1.0
      }
    }
  ],
  "candidate_diagnostics": {
    "strategy": "cold_start",
    "input_count": 1,
    "candidate_count": 1,
    "source_counts": {"bm25_profile": 1, "vector_profile": 1},
    "used_collaborative": false
  }
}
```

## Scoring Strategy

The system keeps the expensive model out of the critical path until late in the pipeline.

Recommendation score:

```text
score =
  base_quality_popularity_prior
  blended with
  preference_match
+ context_match
+ category_match
+ vector_match
+ collaborative_match
+ retrieval_match
+ source_diversity
+ item_quality
+ popularity
+ novelty
+ confidence
- dislike/context penalties
```

Rating prediction:

```text
predicted_rating =
  trained Task A model over
  user/item/category shrinkage priors
+ recent user tendency and rating volatility
+ positive/negative star-share features
+ user-category affinity
+ semantic/profile overlap
+ reliability and prior-gap interactions
```

Task A training compares compact/full feature sets, MSE/Huber/MAE losses, and calibrated/rounded star policies, then saves the validation-RMSE winner by default. Both serving heads remain transparent so they can be evaluated, ablated, and replaced with stronger models later.

## Evaluation Plan

Task A:

- RMSE and MAE for rating prediction
- ROUGE or BERTScore for generated review similarity
- sentiment-rating consistency
- human review of behavioral fidelity
- ablations: no profile, no item profile, no validation

Task B:

- HitRate@10
- NDCG@10
- Recall@K and candidate Recall@50/100
- cold-start subset performance
- sparse/warm-user and cross-domain slices
- contextual human-eval pack with source traces
- source ablations: popularity, filtered popularity, BM25, vector, hybrid candidates, hybrid ranker, cold-start persona-only

## Quick Start

Create a virtual environment and install dependencies:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
```

Run the API:

```bash
uvicorn app.main:app --reload
```

Open:

```text
http://127.0.0.1:8000/docs
http://127.0.0.1:8000/ui/
```

Build local sample artifacts and run evaluation:

```bash
python scripts/build_splits.py --reviews data/sample/reviews.jsonl --items data/sample/items.jsonl --output-dir data/processed
python scripts/build_retrieval_index.py \
  --train data/processed/train.jsonl \
  --items data/processed/items.jsonl \
  --output-dir data/processed
python eval/eval_task_a.py
python eval/eval_task_b.py
```

Or:

```bash
make eval
make eval-generation
make tune-task-a
make train-task-a
make promote-task-a
make tune
make train-ranker
make promote-ranker
```

Train and evaluate Task A on the combined local corpus:

```bash
python eval/train_task_a_model.py \
  --reviews data/processed/all_categories/reviews.jsonl \
  --items data/processed/all_categories/items.jsonl \
  --processed-dir data/processed/all_categories \
  --output-model data/processed/all_categories/task_a_model_rmse.json \
  --output-stats data/processed/all_categories/task_a_rating_stats.json \
  --output-report runs/eval/all_categories_task_a_model_training_rmse.json \
  --candidate-dir runs/eval/task_a_candidates_rmse \
  --selection-metric rmse \
  --ensemble \
  --workers 10
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

Normalize a real Amazon Reviews 2023 subset:

```bash
python scripts/download_amazon_hf.py --with-metadata
python scripts/download_amazon_hf.py --with-metadata --check-only --strict
```

Download every Amazon Reviews 2023 category only when you have the storage/time budget:

```bash
python scripts/download_amazon_hf.py --all-categories --with-metadata
```

```bash
python scripts/ingest_amazon.py \
  --reviews data/raw/All_Beauty.jsonl \
  --metadata data/raw/meta_All_Beauty.jsonl \
  --category All_Beauty \
  --output-dir data/processed \
  --limit 50000
python scripts/build_splits.py --reviews data/processed/reviews.jsonl --items data/processed/items.jsonl --output-dir data/processed
python scripts/build_retrieval_index.py \
  --train data/processed/train.jsonl \
  --items data/processed/items.jsonl \
  --output-dir data/processed
```

Combine all processed categories after per-category ingestion:

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
```

Run Task B candidate-recall sweeps and miss analysis:

```bash
python eval/eval_task_b.py \
  --processed-dir data/processed/all_categories \
  --output runs/eval/all_categories_task_b_review_terms_stopword_1000.json \
  --miss-output runs/eval/all_categories_task_b_review_terms_stopword_misses_1000.json \
  --max-examples 100 \
  --candidate-limit 1000
python eval/train_ranker.py \
  --processed-dir data/processed/all_categories \
  --output runs/eval/all_categories_learned_ranker_review_terms_split_1000.json \
  --max-examples 100 \
  --candidate-limit 1000 \
  --epochs 8 \
  --max-negatives 80 \
  --validation-fraction 0.5
python eval/promote_ranker.py \
  --task-b-report runs/eval/all_categories_task_b_review_terms_stopword_1000.json \
  --learned-ranker-report runs/eval/all_categories_learned_ranker_review_terms_split_1000.json \
  --output runs/eval/all_categories_task_b_review_terms_ranker_promotion.json \
  --weights-output data/processed/all_categories/task_b_ranker_weights.json \
  --candidate-limit 1000
python eval/create_task_b_contextual_eval.py \
  --processed-dir data/processed/all_categories \
  --output docs/human_eval_task_b_contextual.md \
  --max-examples 20 \
  --candidate-limit 1000
```

Run with Docker:

```bash
docker compose up --build
```

## Environment

Copy `.env.example` to `.env` when adding model providers.

OpenRouter is the preferred generation provider for the submission demo. The deterministic template path remains only as a reproducibility fallback so the core pipeline can still run if credentials or credits are unavailable.

To enable OpenAI-backed generation:

```bash
LLM_PROVIDER=openai
LLM_MODEL=gpt-5
OPENAI_API_KEY=...
```

To enable direct DeepSeek-backed generation:

```bash
LLM_PROVIDER=deepseek
DEEPSEEK_MODEL=deepseek-v4-flash
DEEPSEEK_API_KEY=...
```

To enable OpenRouter-backed generation with DeepSeek V4 Flash free:

```bash
LLM_PROVIDER=openrouter
OPENROUTER_MODEL=deepseek/deepseek-v4-flash:free
OPENROUTER_API_KEY=...
```

If `OPENROUTER_API_KEY` is set and `LLM_PROVIDER` is left blank, the app uses
OpenRouter with `deepseek/deepseek-v4-flash:free` by default.
If `DEEPSEEK_API_KEY` is set and `LLM_PROVIDER` is left blank, the app uses
direct DeepSeek first.

Evaluate generated review quality through the configured provider:

```bash
python eval/eval_task_a_generation.py --strict-provider --external-data-policy redact --max-examples 25
```

This reports provider failures separately from review quality and tracks rating mention, item grounding, validation consistency, sentiment alignment, ROUGE-L F1, and unigram F1. External providers default to `deny` on non-sample datasets; use `redact` for provider smoke tests, or `allow` only after explicit approval to export eval-row content.

LLM output is used only for final review/explanation text. Profiling, retrieval, scoring, validation, and evaluation remain reproducible.

Task B runtime can optionally load local retrieval/ranker artifacts:

```bash
TASK_A_MODEL_PATH=data/processed/all_categories/task_a_model_rmse.json
TASK_A_STATS_PATH=data/processed/all_categories/task_a_rating_stats.json
TASK_A_SERVING_POLICY=data/processed/all_categories/task_a_serving_policy.json
TASK_B_RETRIEVAL_INDEX=data/processed/all_categories/collaborative_retrieval.json
TASK_B_RANKER_WEIGHTS=data/processed/all_categories/task_b_ranker_weights.json
```

When `review_term_retrieval.json` exists next to the configured Task B retrieval index, runtime and eval attach it automatically for review-term and lexical-neighbor candidate sources.

Only set `TASK_B_RANKER_WEIGHTS` after `eval/promote_ranker.py` passes.

## Staff-Level Design Principles

- **Shared engine, two heads:** one user model powers both review simulation and recommendation.
- **Offline before online:** expensive profile, summary, and embedding work should be precomputed.
- **LLM as a component, not the system:** retrieval and ranking remain measurable and reproducible.
- **Vector signal without vendor lock-in:** local hashing embeddings provide deterministic semantic retrieval and ranking features that can be replaced by managed embeddings later.
- **Typed intermediate contracts:** every service exchanges structured schemas, not loose prompt blobs.
- **Graceful degradation:** local deterministic fallback keeps the app runnable without external APIs.
- **Evaluation first:** each major quality claim should have a metric, ablation, or qualitative example.
- **Learned ranking path:** a candidate-aware pairwise trainer plus promotion gate prevents weaker learned weights from being wired into runtime.
- **Observable agent behavior:** Task A and Task B responses include trace IDs, and `/api/metrics` plus `/api/traces` expose local runtime evidence.
- **Production-shaped boundaries:** local modules map cleanly to scalable services later.

## Submission Checklist

- Containerized API or web app
- Review simulation endpoint
- Recommendation endpoint
- Reproducible sample data
- Evaluation scripts
- README with setup and architecture
- 4-8 page solution paper
- Clean code repository
