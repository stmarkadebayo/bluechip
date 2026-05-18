# Production Architecture Notes

The hackathon repository runs locally, but the module boundaries are designed to map to a scalable serving architecture.

## Offline Layer

- Ingest review data from object storage.
- Normalize users, items, reviews, ratings, and metadata.
- Compute user profiles, item profiles, embeddings, and aggregate features.
- Persist features to a feature store and vector store.
- Run scheduled evaluation jobs against fixed validation splits.

## Online Layer

- API receives review simulation or recommendation requests.
- Profile service loads precomputed profiles or builds cold-start profiles.
- Retrieval service fetches candidates or similar evidence.
- Ranking service scores candidates or predicts ratings.
- Generation service produces user-facing text from grounded evidence.
- Validation service checks consistency and grounding before response.

## Scale Controls

- Cache hot user profiles and item profiles.
- Limit LLM usage to final synthesis or reranking of small candidate sets.
- Track latency, token cost, fallback rate, validation failures, and quality metrics.
- Keep ranking and generation models swappable behind stable interfaces.

