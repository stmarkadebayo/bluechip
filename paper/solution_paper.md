# Solution Paper Draft

## 1. Problem Framing

Online reviews are behavioral traces. The system treats review history as evidence for user preferences, rating behavior, and voice, then reuses that behavioral profile for both review simulation and recommendation.

## 2. System Overview

The architecture has one shared user intelligence engine with two task-specific heads:

- Review simulation
- Personalized recommendation

## Related Work

The system follows the production recommender pattern of separating candidate generation and ranking, as seen in YouTube, Pinterest, Airbnb, Netflix, Amazon, Spotify, and Uber engineering writeups. For review generation, the closest research line is explainable recommendation, especially NRT, PETER, PEPLER, and recent multi-aspect prompt learning work. For LLM-based recommendation, we use LLMs as profile extractors, explainers, generators, and critics, while keeping retrieval and ranking measurable and reproducible.

## 3. Task A: User Modeling

The pipeline builds a user profile, builds a target item profile, predicts a rating with a transparent scorer, generates a grounded review, and validates consistency.

## 4. Task B: Recommendation

The pipeline builds a user profile, retrieves or accepts candidate items, ranks them with hybrid scoring, and generates concise explanations grounded in candidate evidence. Candidate generation is intentionally multi-source: positive item co-visitation, user-neighbor collaborative filtering, review-term retrieval from positive review language, lexical item-neighbor retrieval from item terms, BM25 profile search, deterministic vector retrieval, category-affinity popularity, and global popularity fallback. Each recommendation includes candidate sources, retrieval scores, score components, tradeoffs, and a trace ID.

## 5. Agent Workflow

The system separates profiling, evidence retrieval, scoring, generation, and validation. This makes the agent behavior observable and easier to evaluate than a single prompt.

At serving time, the API invokes task-specific agents rather than calling services directly. `ReviewSimulationAgent` profiles the user and item, predicts a rating, generates a rating-conditioned review, and validates consistency. `RecommendationAgent` decides whether the request is cold-start, history-aware, or contextual-history-aware, generates a source-attributed candidate pool, ranks it, explains the top results, and records candidate diagnostics.

## 6. Experiments

Planned comparisons:

- popularity baseline
- filtered popularity baseline
- candidate recall before and after collaborative sources
- BM25 and vector retrieval
- cold-start persona-only ranking
- profile-only model
- item-profile model
- hybrid ranker
- hybrid ranker with generation and validation
- candidate-aware learned ranker with promotion gate

Current local eval reports are generated under `runs/eval/` from the sample temporal split. Full-dataset results should replace these before final submission.

Current bounded all-category Task B evidence uses 100 held-out users over 188,236 candidate items. After adding review-term and lexical-neighbor retrieval, candidate Recall@1000 improved from 0.29 to 0.32 and candidate misses dropped from 71 to 68. Final hybrid HitRate@10 remains 0.10 and NDCG@10 remains 0.0766, so retrieval improved but ranking quality is still the active bottleneck. The learned ranker is not promoted because its holdout NDCG@10 is 0.0788 versus 0.1061 for the same-slice hybrid ranker. A graph-walk retrieval ablation did not improve these metrics and was not promoted into default serving.

For contextual relevance, the ranker applies explicit context-category guards for Beauty, music, and gift-seeking scenarios so global-popular items do not dominate when the user gives a clear intent. The generated contextual human-eval pack contains 20 real histories, contexts, top-10 recommendations, source traces, and blank score columns for human judging.

## 7. Evaluation

Task A:

- RMSE / MAE for rating prediction
- ROUGE or BERTScore for review text
- rating-review consistency
- human behavioral fidelity review

Task B:

- HitRate@10
- NDCG@10
- Recall@K and candidate Recall@50/100
- cold-start subset analysis
- sparse-user, warm-user, and cross-domain slices
- contextual human evaluation
- miss analysis by category, history depth, train popularity, and neighbor-path availability

## 8. Limitations

- Sparse user histories reduce confidence.
- Generated text can overfit to style while missing deeper preference signals.
- Cultural localization must be grounded in persona evidence and not forced.
- Local demo data is smaller than the intended production-scale dataset.

## 9. Future Work

- learned ranking model
- better item and user embeddings
- LLM reranking on top candidates
- richer cold-start interviews
- longitudinal memory and feedback loops
