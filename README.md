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
For the submission metric snapshot, see
[docs/evaluation/SUBMISSION_EVAL_SUMMARY.md](docs/evaluation/SUBMISSION_EVAL_SUMMARY.md).
For the judge demo script, see [docs/product/DEMO_SCRIPT.md](docs/product/DEMO_SCRIPT.md).

## Submission Snapshot

The DSN x BCT brief requires both tasks, a containerized app/API, a 4-8 page solution paper, and a clean reproducible repo. Task A predicts likely ratings and reviews. Task B recommends personalized items; its rubric weights Ranking Quality at 30 points, Cold-Start & Cross-Domain at 25, Contextual Relevance human eval at 20, Solution Paper at 15, and Code Reproducibility at 10. Nigerian contextualization earns extra marks when it is grounded in visible user/context evidence.

The strongest truthful story is evidence-first behavior-aware personalization:

- Retrieval, ranking, rating prediction, validation, and traceability are measurable before any LLM writes prose.
- The app is Amazon-first: product metadata, review history, candidate retrieval, ranking diagnostics, and grounded explanations.
- Sparse and cross-domain cases are treated as first-class eval slices, not demo-only claims.
- LLMs are downstream generators/explainers. They do not choose directly from the full product catalog.
- Neural sequence models are out of scope for the current submission unless fixed evals beat the current hybrid baseline.

Current bounded all-category Task B metrics after evidence graph work and the popularity-rank floor:

| Metric | Current value | Submission read |
| --- | ---: | --- |
| `hybrid_candidate_recall@50` | `0.13` | Early recall is still sparse; useful for diagnosing top-pool coverage. |
| `hybrid_candidate_recall@100` | `0.18` | Candidate generation is improving but remains the main bottleneck. |
| `hybrid_candidate_recall@1000` | `0.34` | Best current overall candidate-recall signal. |
| `hybrid_ranker_hit_rate@10` | `0.10` | Top-10 ranking beats only a low bar and needs promotion discipline. |
| `hybrid_ranker_ndcg@10` | `0.0766` | Ranking quality is not overstated; next gate is better same-slice NDCG. |
| Sparse candidate recall@1000 | `0.3611` | Sparse-user handling is a credible focus area. |
| Cross-domain candidate recall@1000 | `0.5484` | Cross-domain candidate retrieval is the strongest Task B slice. |
| Vector source recall | `0.0` | Vector retrieval is present as a diagnostic hook, not a quality claim. |

Current Task A serving evidence remains rating-first: the promoted serving head is selected by fixed evaluation, with the latest documented 5,000-example all-category RMSE gate at `1.2654`. Review generation is downstream of the fixed rating and validated for rating-review consistency and grounding.

## Response To Architecture Feedback

The latest review correctly identified the core Task B problem: semantic relevance alone is not enough. The current bounded metrics show that the system must optimize candidate recall, co-engagement, semantic fit, and top-rank quality together instead of treating vector/semantic similarity as the whole retrieval solution.

What has been implemented in this pass:

- **Multi-head retrieval, local MTMH-style shape**: candidate generation now combines co-visitation, user-neighbor collaborative retrieval, review-term retrieval, lexical item-neighbor retrieval, evidence graph retrieval, BM25 profile/context retrieval, category-affinity popularity, global popularity fallback, and deterministic vector diagnostics. Each candidate keeps source attribution and per-source retrieval scores.
- **Multi-objective ranking**: recommendation scoring now blends preference, context, category, aspect, sequential, evidence graph, Nigerian-context, collaborative, retrieval, diversity, popularity, novelty, item-quality, and confidence signals with explicit score components in the API response.
- **Recall-first evaluation discipline**: Task B eval reports candidate Recall@50/100/1000 separately from HitRate@10 and NDCG@10, plus sparse-user and cross-domain slices. This prevents a top-rank metric from hiding a weak retrieval pool.
- **Diversity and multi-tower behavior without adding a heavyweight model**: user profile, item profile, aspect evidence, context, collaborative history, and popularity operate as separate signal towers that feed retrieval and ranking. Source-diversity features and context-category guards reduce over-reliance on a single semantic path.
- **Production path for MTMH and HSTU**: the current repo does not claim to train MTMH or HSTU neural models. It creates the serving, feature, source attribution, and promotion boundaries needed to replace local retrieval/ranking heads with MTMH-style multi-task multi-head retrieval and an HSTU-style sequential ranker after fixed offline evals prove a lift.

How this addresses the feedback:

| Feedback | Current implementation | Remaining work |
| --- | --- | --- |
| Semantic relevance and co-engagement are disconnected | Retrieval now uses both semantic/lexical paths and co-engagement paths, with source-level diagnostics. Vector recall is reported truthfully as `0.0` today. | Train or integrate a true MTMH retrieval model once enough fixed eval evidence and infrastructure are available. |
| Need MTMH-style retrieval | The local candidate generator is multi-head and multi-objective, with recall reporting and candidate-source attribution. | Replace or augment deterministic heads with a learned multi-task retriever optimized for recall plus semantic relevance. |
| Need stronger ranking such as HSTU | Current ranker exposes sequential, aspect, graph, collaborative, context, and popularity features and promotion gates. | Add an HSTU or sequence-aware ranker only after candidate Recall@K improves enough for ranking gains to matter. |
| Need multi-tower diversity | Separate user, item, aspect, context, collaborative, graph, and popularity signals feed scoring; source diversity is part of the ranker. | Promote learned tower weights or neural towers after same-slice eval beats the hybrid baseline. |

## Architecture

```text
Raw reviews and metadata
  -> ingestion and normalization
  -> aspect-aware evidence intelligence
  -> user profile extraction
  -> item profile extraction
  -> retrieval indexes and evidence graph
  -> ranking / rating prediction
  -> review planning
  -> grounded generation
  -> evidence validation
  -> API / UI
```

The design is grounded in the literature and industry survey in
[research/literature_review.md](research/literature_review.md).

Local hackathon implementation:

```text
app/
  api/                  FastAPI routes
  serving/              API orchestration, trace metadata, and serving workflow boundaries
  core/                 configuration and runtime settings
  models/               typed request/response schemas
  platform/             local feature store and model/index registry abstractions
  services/
    intelligence/       aspect-aware evidence extraction
    profiling/          user and item behavioral profiles
    retrieval/          multi-head local candidate retrieval and source diagnostics
    ranking/            multi-objective rating prediction and recommendation scoring
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
| Evidence graph artifact | Graph service, graph database, or offline graph retrieval index |
| Local feature store | SageMaker Feature Store, Feast, DynamoDB, Redis, or warehouse-backed features |
| Local model registry | SageMaker Model Registry, MLflow, Vertex AI Model Registry, or internal artifact registry |
| In-process ranker | Dedicated ranking service |
| FastAPI | Containerized API on ECS, Kubernetes, Cloud Run, or similar |
| Local eval scripts | CI quality gates and batch evaluation jobs |
| Prompt logs | Tracing, model observability, and cost monitoring |

## Request Flow

### Task A: Review Simulation

```text
POST /api/simulate-review

1. Serving `ReviewSimulationAgent` inspects the request
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

1. Serving `RecommendationAgent` inspects history/context
2. Build or load user profile
3. Generate candidates from co-visitation, user-neighbor CF, review-term retrieval, lexical item-neighbor retrieval, evidence graph retrieval, BM25, deterministic vector diagnostics, category-affinity popularity, and global popularity fallback
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
      "candidate_sources": ["bm25_profile", "category_popularity"],
      "retrieval_scores": {"bm25_profile": 1.0, "category_popularity": 0.42},
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
+ aspect_match
+ sequential_match
+ evidence_graph_match
+ nigerian_context_match
+ diagnostic_vector_match
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
- Recall@K and candidate Recall@50/100/1000
- evidence candidate Recall@K and evidence source mix
- cold-start subset performance
- sparse/warm-user and cross-domain slices
- contextual human-eval pack with source traces
- source diagnostics: popularity, filtered popularity, BM25, diagnostic vector retrieval, evidence graph, hybrid candidates, hybrid ranker, cold-start persona-only

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

Suggested submission demo path:

```text
1. Open http://127.0.0.1:8000/ui/
2. Run Task B Recommend with the Lagos student / conversation-friendly dinner example.
3. Point out candidate sources, score components, Nigerian context match, and trace ID.
4. Run the cross-domain or cold-start example and explain the candidate diagnostics.
5. Run Task A Review and show predicted rating, generated review, validation status, and trace.
6. Open /api/metrics and /api/traces from the Metrics tab.
```

Build local sample artifacts and run evaluation:

```bash
python scripts/build_splits.py --reviews data/sample/reviews.jsonl --items data/sample/items.jsonl --output-dir data/processed
python scripts/build_retrieval_index.py \
  --train data/processed/train.jsonl \
  --items data/processed/items.jsonl \
  --output-dir data/processed
python eval/eval_evidence_intelligence.py
python eval/eval_task_a.py
python eval/eval_task_b.py
python scripts/build_model_registry.py --output data/processed/model_registry.json
```

Or:

```bash
make eval
make registry
make eval-generation
make eval-evidence
make tune-task-a
make train-task-a
make promote-task-a
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
TASK_B_EVIDENCE_GRAPH_INDEX=data/processed/all_categories/evidence_graph_retrieval.json
```

When `review_term_retrieval.json` or `evidence_graph_retrieval.json` exists next to the configured Task B retrieval index, runtime and eval attach it automatically for review-term, lexical-neighbor, aspect graph, and sequential candidate sources.

## Staff-Level Design Principles

- **Shared engine, two heads:** one user model powers both review simulation and recommendation.
- **Offline before online:** expensive profile, summary, and embedding work should be precomputed.
- **LLM as a component, not the system:** retrieval and ranking remain measurable and reproducible.
- **Vector diagnostics without vendor lock-in:** local hashing embeddings provide a deterministic hook that can be replaced by stronger embeddings later; current vector source recall is `0.0`, so it is not presented as a retrieval-quality win.
- **Evidence graph signal without cloud lock-in:** local aspect and sequential graph artifacts can later map to a graph service or feature store.
- **Typed intermediate contracts:** every service exchanges structured schemas, not loose prompt blobs.
- **Graceful degradation:** local deterministic fallback keeps the app runnable without external APIs.
- **Evaluation first:** each major quality claim should have a metric, ablation, or qualitative example.
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
