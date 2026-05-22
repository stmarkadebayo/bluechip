# Bluechip Solution Paper Draft

## 1. Problem Framing

The DSN x BCT challenge asks for agents that understand behavior, preferences, and what a user may choose next. Bluechip frames that as an Amazon-first personalization problem: a user's review history, ratings, product interactions, and current shopping context are behavioral evidence. The system turns that evidence into reusable profiles, then uses the profiles for two serving heads:

- **Task A: Review and rating simulation.** Given a user persona or history and a target product, predict the likely star rating and generate a review grounded in the user's visible behavior and the product facts.
- **Task B: Recommendation.** Given a user persona or history and optional current context, retrieve and rank relevant products, then explain the recommendations with traceable evidence.

The central design choice is evidence-first agent behavior. The agent does not ask an LLM to invent preferences or directly choose from a catalog. It profiles users and items, retrieves candidate products, scores them, validates outputs, and only then uses generation to write reviews or explanations.

## 2. Architecture

Bluechip uses one shared user-intelligence engine with two task-specific heads:

```text
user history / persona
  -> aspect-aware user profile
  -> item profile and catalog evidence
  -> candidate retrieval or target-item scoring
  -> rating prediction / recommendation ranking
  -> grounded generation
  -> validation and trace logging
```

The API is containerized with FastAPI and exposes:

- `POST /api/simulate-review`
- `POST /api/recommend`
- `POST /api/profile-user`
- `GET /api/metrics`
- `GET /api/traces`
- `/ui/` for the browser demo console

The implementation keeps intermediate contracts typed and observable. Recommendation responses include candidate sources, retrieval scores, score components, tradeoffs, and trace IDs. Review responses include predicted rating, grounded review text, confidence, validation status, and trace IDs.

## 3. Task A: Rating-First Review Simulation

Task A is treated as a rating prediction problem before it is treated as a writing problem. The pipeline builds a user profile from ratings, review text, recent behavior, preference aspects, disliked aspects, and category history. It builds an item profile from metadata, item text, category, and review-derived signals. A deterministic rating model then predicts the likely star rating.

Only after the rating is fixed does generation write the review. This prevents the language model from drifting into a review that sounds plausible but contradicts the measured rating behavior. The generated review is checked for rating-review consistency, target-item grounding, and unsafe sensitive inference.

Current documented Task A evidence:

- Trained and heuristic serving heads are compared through fixed eval reports.
- The promoted serving policy is selected by RMSE, the rubric-relevant metric.
- The latest documented 5,000-example all-category gate promotes `calibrated_profile` with RMSE `1.2654`.
- External model providers are optional and guarded by data-export policy; deterministic fallback keeps the repo reproducible without secrets.

## 4. Task B: Retrieval Before Ranking

Task B is where the system invests most of its measurable personalization work. Candidate generation is intentionally multi-source because sparse histories and cross-domain movement are common in review datasets. Current sources include:

- positive item co-visitation
- user-neighbor collaborative filtering
- review-term retrieval from positive review language
- lexical item-neighbor retrieval from item terms
- aspect and sequential evidence graph retrieval
- BM25 profile search
- deterministic vector retrieval as a diagnostic hook
- category-affinity popularity
- global popularity fallback

The final ranker blends preference match, context match, category match, aspect match, sequential match, evidence graph match, Nigerian context match, collaborative match, retrieval source diversity, item quality, popularity, novelty, and confidence. Seen items are filtered from recommendations.

The important limitation is explicit: vector retrieval is not currently a proven recall contributor. Its measured source recall is `0.0`, so it is documented as a swappable diagnostic path rather than a quality claim.

## 5. Current Measured Results

The strongest current Task B evidence comes from bounded all-category evaluation after evidence graph work and the popularity-rank floor. The full all-category held-out set is expensive, so these values are reported as bounded evaluation results and not as full-corpus final scores.

| Metric | Current value | Interpretation |
| --- | ---: | --- |
| `hybrid_candidate_recall@50` | `0.13` | Top candidate pool still misses many positives. |
| `hybrid_candidate_recall@100` | `0.18` | Candidate generation is improving but remains the bottleneck. |
| `hybrid_candidate_recall@1000` | `0.34` | Best current overall candidate-recall signal. |
| `hybrid_ranker_hit_rate@10` | `0.10` | Ranking quality is measurable but not overstated. |
| `hybrid_ranker_ndcg@10` | `0.0766` | Next promotion gate is better same-slice NDCG@10. |
| Sparse candidate recall@1000 | `0.3611` | Sparse-user handling is a credible differentiator. |
| Cross-domain candidate recall@1000 | `0.5484` | Cross-domain retrieval is the strongest measured Task B slice. |
| Vector source recall | `0.0` | Do not claim vector retrieval improved recall yet. |

The result is not presented as "solved recommendation." It shows a measured retrieval and ranking system with clear source diagnostics, slice reporting, and promotion gates. The current quality gate is same-slice NDCG@10 and HitRate@10 without reducing candidate recall.

## 6. Ablations And Rejected Paths

The submission story is stronger because weaker ideas are recorded instead of hidden.

- The learned-ranker experiment was not promoted because it did not beat the same-slice hybrid ranker; the submission runtime keeps the measured hybrid ranker.
- Graph-walk retrieval was measured and not treated as a default quality win when it added cost without improving the logged ranking metrics.
- Beauty taxonomy retrieval is retained because it improves measured candidate coverage.
- Vector retrieval remains a diagnostic hook because measured vector source recall is `0.0`.
- Larger neural sequence models are out of scope, not claimed as implemented submission performance.

## 7. Contextual Relevance And Nigerian Grounding

The brief awards extra credit for Nigerian contextualization. Bluechip uses this only when grounded in user input, item metadata, or explicit context. For example, a Lagos student persona looking for an affordable conversation-friendly dinner can affect context matching, explanation wording, and tradeoff reporting. The system should not invent ethnicity, income, religion, politics, health status, or psychological traits.

For Task B, contextual human evaluation is supported through generated judge packs with user history, current context, top recommendations, source traces, and score columns. The intended rubric checks:

- whether recommendations match the user and context
- whether tradeoffs are honest
- whether explanations cite visible behavior or product facts
- whether Nigerian context is useful and not forced

## 8. Reproducibility

The repo is designed to run without private data or paid model credentials. Sample data is committed, raw Amazon data and generated artifacts are ignored, and optional provider keys stay local.

Core commands:

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
```

Narrow validation commands:

```bash
./.venv/bin/ruff check .
./.venv/bin/pytest
./.venv/bin/python eval/eval_task_b.py --processed-dir data/processed/all_categories --max-examples 100 --candidate-limit 1000 --output runs/eval/submission_task_b.json --miss-output runs/eval/submission_task_b_misses.json
./.venv/bin/python eval/create_task_b_contextual_eval.py --processed-dir data/processed/all_categories --output docs/human_eval_task_b_contextual.md --max-examples 20 --candidate-limit 1000
```

## 9. Rubric Alignment

Task B scoring weights are addressed directly:

- **Ranking Quality, 30 points:** measured by HitRate@10 and NDCG@10 against fixed baselines.
- **Cold-Start & Cross-Domain, 25 points:** measured by sparse and cross-domain candidate-recall slices.
- **Contextual Relevance human eval, 20 points:** supported by UI scenarios and generated human-eval tables with source traces.
- **Solution Paper, 15 points:** this paper explains architecture decisions, experiments, source diagnostics, limitations, and next gates.
- **Code Reproducibility, 10 points:** Docker, FastAPI, sample data, eval commands, tests, and no-key fallback are included.

## 10. Limitations And Next Gates

The main limitation is ranking quality after retrieval. Candidate recall at 1000 is better than early-pool recall, and cross-domain candidate retrieval is strong, but top-10 quality still needs improvement. The remaining gates are:

1. Improve candidate recall@50 and recall@100 without reducing cross-domain recall.
2. Accept ranking changes only if same-slice NDCG@10 and HitRate@10 beat the current hybrid ranker.
3. Score the contextual human-eval pack with actual human labels.
4. Replace hashing embeddings with stronger neural embeddings only after source-level recall improves.
5. Keep Nigerian localization grounded in explicit context and visible behavior.
