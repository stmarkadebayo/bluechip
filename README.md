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
ui/                     demo UI placeholder
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
3. Generate candidates from BM25, item-neighbor retrieval, and popularity fallback
4. Rank candidates using hybrid scoring
5. Generate concise explanations with optional LLM provider or deterministic fallback
6. Return ranked recommendations and agent trace
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
      "tradeoffs": "Medium price, so it may not be the cheapest option."
    }
  ]
}
```

## Scoring Strategy

The system keeps the expensive model out of the critical path until late in the pipeline.

Recommendation score:

```text
score =
  0.35 * preference_match
+ 0.25 * context_match
+ 0.20 * item_quality
+ 0.10 * novelty
+ 0.10 * confidence
```

Rating prediction:

```text
predicted_rating =
  user_mean
+ item_quality_adjustment
+ category_preference_adjustment
+ theme_match_adjustment
```

Both are intentionally transparent so they can be evaluated, ablated, and replaced with stronger models later.

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
- Recall@K
- cold-start subset performance
- qualitative cross-domain examples
- ablations: popularity baseline, embedding retrieval, hybrid ranker, hybrid ranker plus LLM reasoning

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
```

Build local sample artifacts and run evaluation:

```bash
python scripts/build_splits.py --reviews data/sample/reviews.jsonl --items data/sample/items.jsonl --output-dir data/processed
python scripts/build_retrieval_index.py --train data/processed/train.jsonl --output-dir data/processed
python eval/eval_task_a.py
python eval/eval_task_b.py
```

Or:

```bash
make eval
```

Normalize a real Amazon Reviews 2023 subset:

```bash
python scripts/download_amazon_hf.py --with-metadata
python scripts/download_amazon_hf.py --with-metadata --check-only --strict
```

```bash
python scripts/ingest_amazon.py \
  --reviews data/raw/All_Beauty.jsonl \
  --metadata data/raw/meta_All_Beauty.jsonl \
  --category All_Beauty \
  --output-dir data/processed \
  --limit 50000
python scripts/build_splits.py --reviews data/processed/reviews.jsonl --items data/processed/items.jsonl --output-dir data/processed
python scripts/build_retrieval_index.py --train data/processed/train.jsonl --output-dir data/processed
```

Run with Docker:

```bash
docker compose up --build
```

## Environment

Copy `.env.example` to `.env` when adding model providers.

The current scaffold works without an LLM key by using deterministic scoring and template generation. That is deliberate: judges should be able to run the core pipeline even without private credentials.

To enable OpenAI-backed generation:

```bash
LLM_PROVIDER=openai
LLM_MODEL=gpt-5
OPENAI_API_KEY=...
```

LLM output is used only for final review/explanation text. Profiling, retrieval, scoring, validation, and evaluation remain reproducible.

## Staff-Level Design Principles

- **Shared engine, two heads:** one user model powers both review simulation and recommendation.
- **Offline before online:** expensive profile, summary, and embedding work should be precomputed.
- **LLM as a component, not the system:** retrieval and ranking remain measurable and reproducible.
- **Typed intermediate contracts:** every service exchanges structured schemas, not loose prompt blobs.
- **Graceful degradation:** local deterministic fallback keeps the app runnable without external APIs.
- **Evaluation first:** each major quality claim should have a metric, ablation, or qualitative example.
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
