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

The pipeline builds a user profile, retrieves or accepts candidate items, ranks them with hybrid scoring, and generates concise explanations grounded in candidate evidence.

## 5. Agent Workflow

The system separates profiling, evidence retrieval, scoring, generation, and validation. This makes the agent behavior observable and easier to evaluate than a single prompt.

At serving time, the API invokes task-specific agents rather than calling services directly. `ReviewSimulationAgent` profiles the user and item, predicts a rating, generates a rating-conditioned review, and validates consistency. `RecommendationAgent` decides whether the request is cold-start or history-aware, generates candidates, ranks them, and explains the top results.

## 6. Experiments

Planned comparisons:

- popularity baseline
- profile-only model
- item-profile model
- hybrid ranker
- hybrid ranker with generation and validation

Current local eval reports are generated under `runs/eval/` from the sample temporal split. Full-dataset results should replace these before final submission.

## 7. Evaluation

Task A:

- RMSE / MAE for rating prediction
- ROUGE or BERTScore for review text
- rating-review consistency
- human behavioral fidelity review

Task B:

- HitRate@10
- NDCG@10
- Recall@K
- cold-start subset analysis

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
