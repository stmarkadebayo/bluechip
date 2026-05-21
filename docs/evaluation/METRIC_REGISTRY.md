# Metric Registry

## Purpose

This registry defines the metrics Bluechip uses before claiming progress. Each change should name the primary metric it expects to improve and the guardrail metrics it must not regress.

## Submission Snapshot

Current bounded all-category Task B metrics after evidence graph work and the popularity-rank floor:

| Metric | Value | Use |
| --- | ---: | --- |
| `hybrid_candidate_recall@50` | `0.13` | Early-pool retrieval diagnostic. |
| `hybrid_candidate_recall@100` | `0.18` | Early-pool retrieval diagnostic. |
| `hybrid_candidate_recall@1000` | `0.34` | Primary current candidate-recall snapshot. |
| `hybrid_ranker_hit_rate@10` | `0.10` | Primary ranking snapshot. |
| `hybrid_ranker_ndcg@10` | `0.0766` | Primary ranking snapshot. |
| Sparse candidate recall@1000 | `0.3611` | Cold/sparse slice signal. |
| Cross-domain candidate recall@1000 | `0.5484` | Cross-domain slice signal. |
| Vector source recall | `0.0` | Diagnostic only; not a quality claim. |

Task B rubric weights from the brief:

| Rubric area | Points |
| --- | ---: |
| Ranking Quality | 30 |
| Cold-Start & Cross-Domain | 25 |
| Contextual Relevance human eval | 20 |
| Solution Paper | 15 |
| Code Reproducibility | 10 |

## Task A: Review Simulation

Primary metric:
- `rmse`: rating prediction error on a fixed temporal holdout.

Diagnostic metrics:
- `mae`: easier-to-read rating error.
- `generation_success_rate`: share of examples that produced review text.
- `provider_failure_rate`: external provider failures when strict provider mode is enabled.
- `fallback_rate`: share of generated examples that used fallback text.
- `validation_consistency_rate`: rating-review consistency and grounding pass rate.
- `rating_mention_rate`: generated review mentions the predicted rating.
- `item_mention_rate`: generated review mentions the target item.
- `sentiment_alignment_rate`: generated review sentiment matches predicted rating direction.
- `rouge_l_f1`: dependency-free lexical quality proxy.
- `unigram_f1`: dependency-free overlap quality proxy.
- `bertscore_f1`: optional semantic quality metric when `bert-score` dependencies and a local/downloaded model are available.

Required slices:
- cold or no history
- light history
- medium history
- warm history
- category

## Task B: Recommendation

Primary retrieval metric:
- `hybrid_candidate_recall@50`, `hybrid_candidate_recall@100`, and `hybrid_candidate_recall@1000`: whether the held-out positive item appears in the candidate pool before final ranking.

Primary ranking metrics:
- `hybrid_ranker_ndcg@10`
- `hybrid_ranker_hit_rate@10`

Required baselines:
- `filtered_popularity`
- current `hybrid_ranker`
- `base_candidate_recall`

Required diagnostics:
- retrieval source counts
- evidence graph source counts
- vector source recall, currently `0.0`
- candidate miss cause counts
- miss category counts
- sparse, medium, warm user slices
- cross-domain slice
- candidate recall by slice

## Evidence Intelligence Layer

Primary diagnostic metrics:
- `evidence_candidate_recall@K`: held-out item coverage when the aspect-aware evidence graph is attached as a candidate source.
- `evidence_candidate_hit_rate@10`: whether evidence-layer retrieval contributes top-10 recoverability.
- `user_aspect_coverage`: share of examples where user history/persona produced structured aspect scores.
- `nigerian_context_example_share`: share of examples where localized Nigerian context terms were detected.

Required diagnostics:
- retrieval source counts for `aspect_evidence_graph`, `category_aspect_graph`, `sequential_transition`, and `category_transition`
- slice comparison against `eval/eval_task_b.py` candidate recall
- example-level inspection when graph sources dominate or regress recall
- vector retrieval must be reported separately from graph sources because current vector source recall is `0.0`

Acceptance use:
- Evidence metrics are diagnostic and must not replace Task B promotion gates.
- A graph or aspect change is useful only if it improves candidate recall, improves weak slices, or adds grounded explanation evidence without regressing NDCG@10/HitRate@10.

## Production Metrics

Required for serving:
- request count by endpoint
- average latency
- generation provider mix
- estimated generation tokens
- estimated generation cost
- validation failure rate
- fallback rate
- retrieval source mix
- model version counts
- index version counts
- validation status and fallback reason counts
- evidence graph/index version mix
- profile aspect coverage and low-confidence recommendation rate

## Reporting Rules

- Sample-data metrics are smoke tests only.
- Real-data claims must include dataset path, example count, metric values, and command.
- Promotion claims must name the previous baseline and the selected artifact.
- Human eval claims must include the rubric and scored examples, not only summary prose.
