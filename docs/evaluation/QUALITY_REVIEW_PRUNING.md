# Quality Review And Source Pruning

Date: 2026-05-22

This review was triggered after the implicit item-item baseline was promoted into the Task B candidate generator. The goal was not to add more components. The goal was to keep only the parts that improve competition quality or engineering reliability.

## What Changed

- Added `implicit_item_item` as a real Task B retrieval source.
- Built `data/processed/all_categories/implicit_item_neighbors.sqlite` as a lazy SQLite item-neighbor artifact.
- Added `scripts/build_implicit_item_index.py` and `make implicit-item-index`.
- Added `--disabled-sources` to `eval/eval_task_b.py` for pruning and ablation.
- Added `--hybrid-only` and `--candidate-recall-only` for faster targeted evals.
- Added `--sample-strategy stride` to reduce first-N evaluation bias when desired.
- Split the Task B evaluator into `EvalDatasetBuilder`, `RecommendationEvaluator`, and `EvalRunner` while preserving the report contract.
- Added `eval/run_task_b_source_ablation.py` and `make task-b-source-ablation`.
- Centralized retrieval source metadata in `app/services/retrieval/source_registry.py`.
- Removed stale `graph_walk` wiring because it was not called by production candidate generation.
- Calibrated candidate/ranker source scoring by source family instead of treating every raw source score as equivalent.
- Extracted candidate balancing into `CandidateMixer`, keeping source floors and exploration budgets measurable in one place.
- Removed the arbitrary `source_diversity` ranking reward.
- Reduced raw retrieval-score weight because source scores are not calibrated across retrievers.
- Moved contextual hair/skincare/nail/gift intent rules to `app/services/ranking/context_intents.json`.
- Updated the contextual human-eval generator to use the same lean retrieval policy as serving.
- Fixed serving LLM rerank ordering so returned LLM order is respected.
- Prevented LLM review reasoning from overriding the promoted Task A rating head.
- Removed diagnostic item-profile signals such as `quality score` and `category:` from Task A review prose evidence.

## Pruned Sources

The current lean submission mask disables these sources for the no-context all-category Task B gate:

```text
vector_profile,bm25_profile,beauty_sparse_tail,sparse_category_tail,neural_vector
```

Reason:

- `vector_profile` had `0.0` all-category source recall in the bounded gate and was expensive.
- `bm25_profile` standalone HitRate@10 was weak on the no-context gate.
- `sparse_category_tail` had `0.0` recall.
- `beauty_sparse_tail` had very low recall and occupied protected exploration space.
- `neural_vector` remains disabled by default until it beats the same-slice gate; current neural checks are runtime validation, not a quality promotion.

Serving uses a lean default through `BLUECHIP_DISABLED_RETRIEVAL_SOURCES`. If the env var is unset, serving disables local vector, neural vector, and sparse-tail sources; it also disables `bm25_profile` when the request has no explicit context.

## Ablation Command

Run source-family ablations against the same Task B gate:

```bash
python eval/run_task_b_source_ablation.py \
  --processed-dir data/processed/all_categories \
  --max-examples 100 \
  --candidate-limit 1000
```

For faster retrieval-only checks:

```bash
python eval/run_task_b_source_ablation.py \
  --processed-dir data/processed/all_categories \
  --max-examples 250 \
  --candidate-limit 1000 \
  --candidate-recall-only
```

## Results

Comparable 100x1000 gate after adding implicit item-item:

| Run | HitRate@10 | NDCG@10 | Recall@50 | Recall@100 | Recall@1000 |
| --- | ---: | ---: | ---: | ---: | ---: |
| Hybrid with implicit item-item | `0.10` | `0.0766` | `0.13` | `0.18` | `0.34` |
| Lean pruned hybrid | `0.10` | `0.0766` | `0.13` | `0.18` | `0.34` |
| Lean pruned, no source diversity | `0.10` | `0.0766` | `0.13` | `0.18` | `0.34` |

The pruning did not improve the 100-row headline metric, but it removed weak/noisy sources without hurting the gate. That is still a useful promotion because the runtime path is simpler and cheaper.

Larger candidate-recall-only diagnostic, first 250 examples:

| Metric | Value |
| --- | ---: |
| Recall@50 | `0.104` |
| Recall@100 | `0.136` |
| Recall@1000 | `0.28` |
| Sparse Recall@1000 | `0.2697` |
| Medium Recall@1000 | `0.3846` |
| Warm Recall@1000 | `0.2121` |
| Cross-domain Recall@1000 | `0.4615` |

This larger slice is lower than the first 100-row gate, so the earlier `0.34` should be treated as a bounded gate, not a full-corpus quality claim.

## Source Findings

On the 250-row pruned candidate-only run:

| Source family | Recall@1000 | Read |
| --- | ---: | --- |
| popularity/fallback | `0.24` | Strongest broad recall source. |
| aspect/evidence | `0.132` | Useful backfill, but very large source volume. |
| collaborative/co-engagement | `0.096` | Helps warm/collab cases; item-item is useful but not dominant. |
| lexical/review-term | `0.052` | Keep bounded; do not let it dominate. |
| semantic/vector | `0.0` | Disabled for lean run. |

`implicit_item_item` contributed `0.044` source recall@1000 on the 250-row diagnostic. It is not a silver bullet, but it is a real conventional co-engagement signal and should stay as a bounded retrieval head.

## Scalability Findings

The current full hybrid candidate-generation path is still too slow for large sweeps:

- 100x1000 ranked eval completes in about 70 seconds after pruning/cache work.
- 250x1000 candidate-only completes in about 2 minutes.
- 1000x1000 and 5000x1000 candidate-only were stopped because the Python path was too slow.

Primary cause:

- Several sources scan or sort large Python structures per user.
- Source diagnostics are useful but expensive.
- Candidate generation is still object-heavy and not fully indexed for batch eval.

Needed next optimization:

- Move review-term, evidence-graph, and category-popularity retrieval to SQLite or compact top-k arrays with bounded per-source reads.
- Add progress and per-source timing to eval.
- Add incremental lift diagnostics: source-only, source-removed, and new hits over a baseline.

## Task A Findings

Task A quality review found two metric risks:

- The trained rating model validation likely leaks because `RatingStats` are built from all train rows before validation feature creation.
- The serving review simulator previously allowed LLM reasoning to override the promoted metric-backed rating.

Immediate fix completed:

- Serving now keeps the promoted rating head as the numeric rating owner. LLM reasoning can shape prose but does not override the RMSE-validated rating.

Still needed:

- Rework Task A training validation so stats are built only from fit rows.
- Treat `calibrated_profile` as the official rating path unless a retrained model beats it on held-out Task A.
- Align generation eval with the serving rating policy.

## Human Eval Follow-Up

The completed Task B contextual human-eval CSV has 20 scored examples. The composite average is `2.2125 / 5`, which linearly maps to about `8.85 / 20` for the contextual relevance judging bucket. Every example had `target_in_top10 = no`.

The notes consistently call out poor context fit:

- recommendations are generic;
- recommendations do not match hair/styling, skincare, nail-care, or gift intent tightly enough;
- generic beauty popularity can dominate the more specific request.

Immediate fix completed:

- Added a contextual sub-intent guard in the ranker. Hair/styling, skincare, nail-care, and gift contexts now apply an explicit boost for matching items and a penalty for same-category but wrong-subintent items.
- Moved the rule terms and weights to `app/services/ranking/context_intents.json` so human-eval feedback can tune the guard without changing ranker code.
- Regenerated smoke contextual recommendations show the top hair and nail examples now contain specific hair/nail products rather than generic beauty popularity lists.

This does not change the no-context Task B leaderboard gate, but it should improve the human contextual relevance surface.

## Submission Position

Use the lean hybrid as the current submission path:

- It preserves the known 100x1000 Task B gate.
- It removes low-value sources.
- It adds a conventional implicit item-item source without requiring the heavy `implicit` package at serving time.
- It avoids LLM rating override.
- It fixes LLM rerank ordering when external generation is enabled.

Do not claim that every added source improves quality. Claim that the system is evidence-first and that source diagnostics were used to prune weak sources.
