# Feature Store 10/10 Implementation Plan

This plan upgrades the current local feature store from a reproducible artifact resolver into a production-shaped offline/online feature platform while keeping the hackathon app runnable without cloud services.

## Current Rating: 6.5/10

Strengths:

- Versioned artifact discovery for processed data, models, and indexes.
- Point lookup for items and user histories.
- Runtime summaries exposed through `/api/runtime/feature-store`.
- Deterministic local fallback suitable for judges.

Gaps:

- JSONL scans are too slow for larger online serving.
- No point-in-time correctness guarantees.
- No typed feature definitions or ownership metadata.
- No online/offline parity checks.
- No feature freshness, quality, or drift monitoring.
- No promotion gates tying feature changes to eval deltas.

## Phase 1: SQLite Online Store

Status: implemented as the first bounded step.

- Build `feature_store.sqlite` from processed JSONL artifacts.
- Store canonical item and review rows as JSON payloads.
- Add indexed point lookup by `item_id`, `user_id`, `timestamp`, and category.
- Select backend with `BLUECHIP_FEATURE_STORE_SQLITE`.
- Keep JSONL backend as the default fallback.

Commands:

```bash
make sqlite-feature-store PROCESSED_DIR=data/processed/all_categories
BLUECHIP_FEATURE_STORE_SQLITE=data/processed/all_categories/feature_store.sqlite
```

## Phase 2: Typed Feature Views

Create explicit feature view definitions:

| Feature View | Entity | Examples |
| --- | --- | --- |
| `user_rating_behavior` | user | average rating, strictness, variance, trend |
| `user_preference_terms` | user | preferred terms, disliked terms, recent terms |
| `user_aspect_profile` | user | aspect scores, positive/negative aspects |
| `item_quality_profile` | item | quality score, popularity, average rating |
| `item_aspect_profile` | item | aspect evidence and terms |
| `retrieval_edges` | item/category/aspect | co-visitation, transitions, evidence graph |

Each feature view should define:

- owner;
- entity keys;
- source artifacts;
- freshness SLA;
- transformation code path;
- validation checks;
- online/offline storage target;
- evaluator that consumes it.

## Phase 3: Point-In-Time Training Sets

Add a materializer that builds training rows using only evidence available before the label timestamp.

Required outputs:

- `task_a_training_features.parquet` or SQLite table;
- `task_b_training_features.parquet` or SQLite table;
- leakage report showing excluded future interactions;
- split manifest with data versions and time windows.

Promotion rule: no model/index promotion without a point-in-time manifest.

## Phase 4: Online/Offline Parity Tests

For a fixed sample of users/items:

1. Load features from the offline materialized table.
2. Load the same features through the online feature store API.
3. Compare field-by-field with tolerances.
4. Fail CI on parity drift.

Target: `>= 99.5%` exact parity for categorical/list features and bounded tolerance for floats.

## Phase 5: Feature Quality Gates

Add automated checks:

- null rate;
- out-of-range values;
- list cardinality;
- duplicate entity rows;
- stale timestamps;
- category distribution drift;
- top-term drift;
- train/test leakage;
- missing online rows for eval users/items.

Expose the report at:

```text
GET /api/runtime/feature-store/quality
```

## Phase 6: Feature Registry And Lineage

Extend the local model registry into a feature registry:

- feature view name;
- semantic version;
- source files and checksums;
- transformation code hash;
- generated artifact hash;
- dependent models/evals;
- promotion status.

This makes every recommendation trace explain which feature version influenced it.

## Phase 7: Retrieval Artifact Integration

Treat retrieval artifacts as feature views:

- collaborative neighbors;
- review-term index;
- evidence graph;
- FAISS index;
- FAISS item-id companion map;
- BM25/token indexes.

Each artifact should have:

- source dataset version;
- build command;
- row/vector counts;
- source recall diagnostics;
- promotion gate.

## Phase 8: Monitoring And Product Analytics

Keep offline eval in `eval/`. Use analytics only for deployed product behavior.

PostHog can help with:

- demo funnel;
- accepted/rejected recommendations;
- context edits;
- click-through;
- repeat usage;
- latency events.

It should not replace:

- RMSE;
- HitRate/NDCG;
- candidate recall;
- ROUGE/BERTScore;
- human eval labels.

## Target 10/10 Bar

The feature store deserves a 10/10 when it has:

- indexed online lookup;
- point-in-time offline features;
- online/offline parity tests;
- feature registry and lineage;
- feature quality reports;
- retrieval artifact versioning;
- CI promotion gates tied to eval metrics;
- runtime traces that identify feature versions per request;
- human-readable docs for every feature view.

The current SQLite backend is Phase 1. It improves serving practicality, but the real 10/10 jump comes from typed feature views, point-in-time materialization, and parity gates.
