# System Architecture

## Purpose

This document shows Bluechip's current local architecture and the production mapping behind it. The design principle is consistent across both: user and item evidence feed profiling, retrieval, ranking, generation, validation, evaluation, and observability. LLMs are downstream synthesis components, not the source of ranking truth.

## Current Local Architecture

```mermaid
flowchart LR
  user["User / Demo / API Client"] --> api["FastAPI Routes\napp/api/routes.py"]
  api --> review_agent["Serving Orchestrator\nReviewSimulationAgent"]
  api --> rec_agent["Serving Orchestrator\nRecommendationAgent"]

  review_agent --> user_profile["User Profiling"]
  review_agent --> item_profile["Item Profiling"]
  review_agent --> rating["Rating Prediction"]
  review_agent --> review_plan["Review Planning"]
  review_plan --> review_gen["Review Generation"]
  review_agent --> validation["Evidence Validation"]

  rec_agent --> user_profile
  rec_agent --> retrieval["Candidate Retrieval"]
  rec_agent --> ranking["Recommendation Ranking"]
  rec_agent --> rec_gen["Recommendation Explanation"]

  data["JSONL Data\nsample/raw/processed"] --> user_profile
  data --> item_profile
  data --> retrieval
  data --> rating
  data --> ranking
  feature_store["Local Feature Store\napp/platform/feature_store.py"] --> data
  aspects["Aspect Intelligence\napp/services/intelligence"] --> user_profile
  aspects --> item_profile
  aspects --> retrieval

  indexes["Local Retrieval Artifacts\ncollaborative, item-neighbor,\nreview-term, vector"] --> retrieval
  evidence_graph["Evidence Graph Artifact"] --> retrieval
  models["Local Model Artifacts\nTask A policy/model/stats,\nTask B weights"] --> rating
  models --> ranking
  registry["Local Model/Index Registry\napp/platform/model_registry.py"] --> models
  registry --> indexes

  review_gen --> providers["Generation Providers\ntemplate/mock/OpenAI/OpenRouter/DeepSeek"]
  rec_gen --> providers
  validation --> response["Structured API Response"]
  rec_gen --> response
  rating --> response
  ranking --> response

  review_agent --> traces["JSONL Trace Store\n/api/metrics /api/traces"]
  rec_agent --> traces
  retrieval --> traces
  ranking --> traces
  providers --> traces
  validation --> traces
```

Current implementation notes:

- `app/serving/orchestrators/` orchestrates task workflows for API serving.
- `app/platform/feature_store.py` is the local feature-store abstraction over processed artifacts.
- `app/platform/model_registry.py` is the local model/index registry abstraction.
- `app/services/intelligence/` extracts aspect-aware evidence used by profiling, retrieval, ranking, and generation.
- `app/services/retrieval/` implements a local multi-head retrieval layer: collaborative co-engagement, user-neighbor retrieval, review-term retrieval, lexical item-neighbor retrieval, evidence graph retrieval, BM25, category-affinity popularity, global fallback, and deterministic vector diagnostics.
- `app/services/retrieval/evidence_graph.py` adds aspect graph and sequential retrieval paths.
- `app/services/ranking/` implements multi-objective scoring over preference, context, category, aspect, sequential, evidence graph, Nigerian-context, collaborative, retrieval, diversity, popularity, novelty, quality, and confidence features.
- `app/services/generation/review_plan.py` implements plan-then-write review generation.
- `app/services/validation/evidence_critic.py` validates grounding and sensitive-inference risk.
- `app/agents/` remains as a compatibility shim for older imports.
- `app/services/` contains profiling, retrieval, ranking, generation, and validation tools.
- `eval/` contains offline metrics, training, and promotion scripts.
- `scripts/` contains ingestion, split building, retrieval-index building, and dataset utilities.
- `runs/traces/requests.jsonl` stores local observability records.

## MTMH, HSTU, And Multi-Tower Alignment

The architecture review called out a real gap between semantic relevance and co-engagement recall. The local system now addresses that gap structurally, while keeping heavier neural models behind promotion gates.

| Architecture concept | Local implementation now | Production upgrade path |
| --- | --- | --- |
| MTMH multi-head retrieval | Multiple retrieval heads feed one attributed candidate pool: co-visitation, user-neighbor CF, review-term, lexical-neighbor, evidence graph, BM25, vector diagnostic, category-affinity popularity, and global fallback. | Train a true multi-task multi-head retriever that jointly optimizes candidate recall and semantic relevance. |
| Multi-task objective | Eval reports candidate Recall@50/100/1000 separately from HitRate@10 and NDCG@10; ranker components expose relevance, context, graph, collaborative, and diversity signals. | Promote learned retrieval/ranking artifacts only when fixed same-slice recall and ranking metrics improve together. |
| HSTU-style sequential ranking | Current ranker has sequential and evidence graph features, but not an HSTU neural sequence model. | Add HSTU or another sequence-aware ranker after retrieval recall improves enough for top-rank modeling to matter. |
| Multi-tower diversity | User, item, aspect, context, collaborative, graph, and popularity signals are separate towers feeding scoring; source diversity is explicit. | Learn tower weights or neural tower representations once the hybrid baseline is beaten by fixed eval. |

This wording is deliberate: the repo has implemented the serving boundaries, retrieval heads, features, diagnostics, and promotion discipline required by the feedback. It does not overclaim that MTMH or HSTU has already been trained.

## Amazon-Scale Target Architecture

```mermaid
flowchart TB
  subgraph offline["Offline Data And Evaluation Plane"]
    raw["Raw Reviews / Purchases / Metadata"] --> lake["Object Storage Data Lake"]
    lake --> ingest["Ingestion And Normalization"]
    ingest --> quality["Data Quality And Leakage Checks"]
    quality --> features["Feature Pipelines"]
    features --> feature_store["Feature Store"]
    features --> vector_store["Vector Store / ANN Indexes"]
    features --> graph_store["Item/User Graph Indexes"]
    feature_store --> evals["Replayable Offline Evals"]
    vector_store --> evals
    graph_store --> evals
    evals --> promotion["Promotion Gates"]
    human["Human Eval Packs"] --> promotion
    promotion --> registry["Model / Index Registry"]
  end

  subgraph online["Online Serving Plane"]
    client["Shopping Surface / Assistant / API"] --> gateway["API Gateway"]
    gateway --> orchestrator["Serving Orchestrator"]
    orchestrator --> profile_svc["Profile Service"]
    orchestrator --> item_svc["Item Intelligence Service"]
    profile_svc --> evidence_svc["Evidence Intelligence Service"]
    item_svc --> evidence_svc
    evidence_svc --> retrieval_svc["Retrieval Service"]
    profile_svc --> retrieval_svc["Retrieval Service"]
    item_svc --> retrieval_svc
    retrieval_svc --> rank_svc["Ranking Service"]
    rank_svc --> gen_svc["Grounded Generation Service"]
    gen_svc --> validate_svc["Validation And Safety Service"]
    validate_svc --> gateway
  end

  registry --> profile_svc
  registry --> retrieval_svc
  registry --> rank_svc
  registry --> gen_svc
  feature_store --> profile_svc
  feature_store --> item_svc
  vector_store --> retrieval_svc
  graph_store --> retrieval_svc

  orchestrator --> obs["Tracing / Metrics / Cost / Alerts"]
  retrieval_svc --> obs
  rank_svc --> obs
  gen_svc --> obs
  validate_svc --> obs
```

Target implementation notes:

- Offline systems own data correctness, features, indexes, evals, and promotion.
- Online systems own latency, fallback behavior, traceability, and API contracts.
- Model and index promotion is gated by fixed evaluation reports.
- Runtime traces include model versions, index versions, retrieval source mix, latency, cost, fallback reason, and validation status.

## Online Request Flow

```mermaid
sequenceDiagram
  participant Client
  participant API as FastAPI / Gateway
  participant Orchestrator
  participant Profile as Profile Service
  participant Retrieval as Retrieval Service
  participant Ranking as Ranking Service
  participant Generation as Generation Service
  participant Validation as Validation Service
  participant Trace as Trace Store

  Client->>API: POST simulate-review or recommend
  API->>Orchestrator: Validate request contract
  Orchestrator->>Profile: Build/load user profile
  Profile-->>Orchestrator: User signals, confidence, seen items

  alt Task A: Review Simulation
    Orchestrator->>Ranking: Predict rating for target item
    Ranking-->>Orchestrator: Rating, score, model name
    Orchestrator->>Generation: Build review plan and generate rating-conditioned review
    Generation-->>Orchestrator: Grounded review text
    Orchestrator->>Validation: Check consistency and grounding
    Validation-->>Orchestrator: Validation status and issues
  else Task B: Recommendation
    Orchestrator->>Retrieval: Generate candidate pool
    Retrieval-->>Orchestrator: Candidates, sources, retrieval scores
    Orchestrator->>Ranking: Rank candidates
    Ranking-->>Orchestrator: Ranked items and score components
    Orchestrator->>Generation: Explain selected recommendations
    Generation-->>Orchestrator: Grounded reasons and tradeoffs
  end

  Orchestrator->>Trace: Append trace metadata
  Orchestrator-->>API: Structured response with trace_id
  API-->>Client: JSON response
```

## Offline Data, Evaluation, And Promotion Flow

```mermaid
flowchart LR
  raw["Raw Amazon Review Data"] --> ingest["scripts/ingest_amazon.py"]
  ingest --> processed["Processed Reviews And Items"]
  processed --> splits["scripts/build_splits.py\nTemporal Holdouts"]
  splits --> index["scripts/build_retrieval_index.py"]
  index --> evidence_graph["Evidence Graph Retrieval Index"]
  splits --> implicit_index["scripts/build_implicit_item_index.py\nSQLite Item-Item Index"]
  source_registry["retrieval/source_registry.py\nSource Families And Defaults"] --> task_b_eval

  splits --> task_a_train["eval/train_task_a_model.py"]
  splits --> task_a_eval["eval/eval_task_a.py"]
  task_a_train --> task_a_model["Task A Model Artifact"]
  task_a_model --> task_a_eval
  task_a_eval --> task_a_promote["eval/promote_task_a.py"]
  task_a_promote --> task_a_policy["Task A Serving Policy"]

  splits --> task_a_gen["eval/eval_task_a_generation.py"]
  task_a_gen --> gen_report["Task A Generation Report"]

  index --> task_b_eval["eval/eval_task_b.py"]
  evidence_graph --> task_b_eval
  implicit_index --> task_b_eval
  splits --> task_b_eval
  task_b_eval --> ablation["eval/run_task_b_source_ablation.py"]
  evidence_graph --> evidence_eval["eval/eval_evidence_intelligence.py"]
  task_b_eval --> miss_report["Candidate Recall And Miss Analysis"]

  human_tables["eval/create_human_eval_tables.py"] --> human_reports["Human Eval Packs"]
  human_reports --> promotion_policy["Promotion Policy"]
  task_a_promote --> promotion_policy
```

Promotion rules:

- Task A rating artifacts must beat the current serving policy on fixed RMSE-first evaluation.
- Task A generation changes must preserve validation, grounding, and text-quality metrics.
- Task B ranker artifacts must beat filtered popularity and the current hybrid ranker on the same holdout.
- Retrieval changes must report candidate Recall@K and miss analysis before ranker changes are trusted.
- Retrieval sources must be promoted through the source registry and ablation runner so serving defaults, ranking features, and eval diagnostics stay in sync.

## Service Ownership Map

| Area | Current Local Modules | Target Owner | Primary Metrics |
| --- | --- | --- | --- |
| Product narrative | `README.md`, `docs/product/`, `paper/solution_paper.md` | Product / Principal Engineer | launch metric, guardrails, non-goals |
| Data ingestion | `scripts/ingest_amazon.py`, `scripts/build_splits.py` | Data Platform | data validity, leakage checks, split reproducibility |
| User intelligence | `app/services/profiling/`, `app/services/intelligence/` | Personalization ML | profile confidence, aspect coverage, slice performance |
| Item intelligence | `app/services/profiling/item_profile.py` | Personalization ML | metadata coverage, item quality signals |
| Retrieval | `app/services/retrieval/`, `scripts/build_retrieval_index.py`, `scripts/build_evidence_graph.py` | Retrieval ML | candidate Recall@K, source coverage, miss causes |
| Ranking | `app/services/ranking/`, `eval/eval_task_b.py` | Ranking ML | RMSE, NDCG@10, HitRate@10 |
| Generation | `app/services/generation/`, `prompts/` | Generation Quality | grounding, text quality, fallback rate |
| Validation | `app/services/validation/` | Trust And Safety | consistency rate, unsafe explanation rate |
| Evaluation | `eval/`, `docs/evaluation/` | Eval Quality | fixed reports, human scores, promotion decisions |
| Serving | `app/api/`, `app/serving/orchestrators/` | Platform | latency, API stability, fallback behavior |
| Feature store and registry | `app/platform/`, `scripts/build_model_registry.py` | ML Platform | artifact versions, point lookups, registry completeness |
| Observability | `app/stores/trace_store.py`, `/api/metrics`, `/api/traces` | Infra / Observability | trace completeness, model/index versions, cost |

## Current Gaps Against Target

- Architecture is local by default, with one FastAPI app and internal service boundaries.
- Feature store and model registry are local implementations; managed cloud equivalents remain deployment work.
- Vector and graph services are still local artifacts rather than managed infrastructure.
- Cloud provisioning is intentionally out of scope for the current repo-local implementation.
- Task B contextual human eval has been scored and summarized; Task A behavioural human eval can still be added for a stronger final paper.
- Dashboards and rollout controls are not implemented yet; `/api/metrics` and `/api/traces` are local substitutes.
- Privacy and safety rules are documented, but broader automated enforcement is still a future step.
