# Submission Evaluation Summary

## Scope

This page is the judge-facing metric snapshot for the DSN x BCT submission package. It should be read as bounded local evidence, not a claim that every full-corpus eval has been exhaustively run.

Challenge deliverables:

- Containerized app/API for Task A review/rating simulation.
- Containerized app/API for Task B personalized recommendation.
- 4-8 page solution paper.
- Clean, documented, reproducible repository.

Task B scoring from the brief:

| Area | Points |
| --- | ---: |
| Ranking Quality: NDCG@10 / Hit Rate | 30 |
| Cold-Start & Cross-Domain | 25 |
| Contextual Relevance human eval | 20 |
| Solution Paper | 15 |
| Code Reproducibility | 10 |

Additional marks are available when Nigerian contextualization is useful and grounded.

## Current Task B Metrics

Latest bounded all-category Task B metrics after evidence graph work and the popularity-rank floor:

| Metric | Value | Status |
| --- | ---: | --- |
| `hybrid_candidate_recall@50` | `0.13` | Needs improvement. |
| `hybrid_candidate_recall@100` | `0.18` | Needs improvement. |
| `hybrid_candidate_recall@1000` | `0.34` | Best current overall recall signal. |
| `hybrid_ranker_hit_rate@10` | `0.10` | Current top-10 ranking gate. |
| `hybrid_ranker_ndcg@10` | `0.0766` | Current top-10 ranking gate. |
| Sparse candidate recall@1000 | `0.3611` | Positive sparse-user signal. |
| Cross-domain candidate recall@1000 | `0.5484` | Strongest Task B slice. |
| Vector source recall | `0.0` | Diagnostic only; do not oversell. |

Interpretation:

- Candidate generation is the main Task B bottleneck, especially at @50 and @100.
- Cross-domain retrieval is the strongest measured slice and should be highlighted.
- Ranking is still gated by fixed evals; the runtime uses the measured hybrid ranker.
- Vector retrieval exists for a swappable embedding path, but current measured source recall is `0.0`.

## Current Task A Signal

Task A is rating-first:

- The review text is generated after rating prediction, not before.
- The serving head is selected by fixed eval and promotion policy.
- The latest documented 5,000-example all-category RMSE gate promotes `calibrated_profile` with RMSE `1.2654`.
- Generation eval tracks provider failures, fallback rate, validation consistency, rating mention, item grounding, sentiment alignment, ROUGE-L F1, and unigram F1.

## Reproduction Commands

Install and run locally:

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

Core checks:

```bash
./.venv/bin/ruff check .
./.venv/bin/pytest
./.venv/bin/python -m compileall app eval scripts tests
```

Task A rating gate:

```bash
./.venv/bin/python eval/eval_task_a.py \
  --reviews data/processed/all_categories/reviews.jsonl \
  --items data/processed/all_categories/items.jsonl \
  --processed-dir data/processed/all_categories \
  --output runs/eval/all_categories_task_a_with_model_rmse_5000.json \
  --max-examples 5000 \
  --model-path data/processed/all_categories/task_a_model_rmse.json

./.venv/bin/python eval/promote_task_a.py \
  --task-a-report runs/eval/all_categories_task_a_with_model_rmse_5000.json \
  --output runs/eval/all_categories_task_a_serving_promotion.json \
  --policy-output data/processed/all_categories/task_a_serving_policy.json
```

Task B candidate/ranker gate:

```bash
./.venv/bin/python eval/eval_task_b.py \
  --processed-dir data/processed/all_categories \
  --output runs/eval/submission_task_b_100x1000.json \
  --miss-output runs/eval/submission_task_b_100x1000_misses.json \
  --max-examples 100 \
  --candidate-limit 1000
```

Contextual human-eval pack:

```bash
./.venv/bin/python eval/create_task_b_contextual_eval.py \
  --processed-dir data/processed/all_categories \
  --output docs/human_eval_task_b_contextual.md \
  --max-examples 20 \
  --candidate-limit 1000
```

Provider-backed Task A generation smoke:

```bash
./.venv/bin/python eval/eval_task_a_generation.py \
  --strict-provider \
  --external-data-policy redact \
  --max-examples 25
```

## Next Gates

- Improve candidate recall@50 and recall@100 while maintaining recall@1000.
- Preserve or improve sparse recall@1000 and cross-domain recall@1000.
- Accept ranking changes only when same-slice NDCG@10 and HitRate@10 beat the current hybrid ranker.
- Add scored human labels to the contextual relevance pack.
- Treat stronger neural embeddings and sequence models as out of scope until source-level recall improves.
