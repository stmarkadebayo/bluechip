# Production Architecture Notes

The hackathon repository runs locally, but the module boundaries are designed to map to a scalable serving architecture.

## Offline Layer

- Ingest review data from object storage.
- Normalize users, items, reviews, ratings, and metadata.
- Compute user profiles, item profiles, aspect evidence, embeddings, graph features, and aggregate features.
- Persist features to a feature store and vector store.
- Persist aspect-to-item, category-aspect, item-transition, and category-transition indexes to a graph store or retrieval artifact.
- Run scheduled evaluation jobs against fixed validation splits.
- Register promoted model, policy, and index artifacts in a model/index registry.

## Online Layer

- API receives review simulation or recommendation requests.
- Profile service loads precomputed profiles or builds cold-start profiles.
- Evidence intelligence service extracts aspect and localized context signals.
- Retrieval service fetches candidates from collaborative co-engagement, user-neighbor, lexical, review-term, evidence graph, vector diagnostic, category-affinity, and popularity paths.
- Ranking service scores candidates or predicts ratings with evidence-aware, context-aware, sequence-aware, collaborative, diversity, popularity, and confidence features.
- Generation service builds a review/explanation plan, then produces user-facing text from grounded evidence.
- Validation service checks consistency, grounding, and sensitive-inference risk before response.
- Runtime service exposes metrics, traces, feature-store status, and model/index registry status.

## Model Evolution Path

- Near term: keep the local hybrid system as the reproducible baseline because it exposes source attribution, candidate recall, NDCG, HitRate, sparse-user, and cross-domain metrics.
- Retrieval upgrade: replace or augment local heads with MTMH-style multi-task multi-head retrieval when a fixed offline report proves recall and semantic relevance improve together.
- Ranking upgrade: evaluate HSTU or another sequence-aware ranker only after retrieval Recall@K improves; weak candidate pools cap any top-rank model.
- Diversity upgrade: promote learned multi-tower weights after they beat the current source-diverse hybrid ranker on the same holdout and preserve contextual relevance.

## Scale Controls

- Cache hot user profiles and item profiles.
- Limit LLM usage to final synthesis or reranking of small candidate sets.
- Track latency, token cost, fallback rate, validation failures, retrieval source mix, model/index versions, and quality metrics.
- Keep ranking and generation models swappable behind stable interfaces.
- Keep cloud provisioning out of this repo-local pass; managed services are target mappings, not prerequisites.

## Local Entry Point

- Monolith: `uvicorn app.main:app --reload`

The local feature store reads processed artifacts through `app/platform/feature_store.py`. The local model/index registry resolves promoted artifacts through `app/platform/model_registry.py` and can be materialized with `scripts/build_model_registry.py`.

The local evidence-intelligence layer reads aspect evidence through `app/services/intelligence/aspects.py`, graph retrieval through `app/services/retrieval/evidence_graph.py`, review plans through `app/services/generation/review_plan.py`, and critic checks through `app/services/validation/evidence_critic.py`.
