# Task B Fast Proof Run - 2026-05-24

## Purpose

This run tests whether the new Task B retrieval work improves the main bottleneck: finding the held-out target item before final ranking.

The brief scores Task B on NDCG@10 / Hit Rate, cold-start and cross-domain quality, contextual relevance, solution paper, and reproducibility. Candidate recall is not the final rubric score, but it is the promotion diagnostic that determines whether longer ranker training is worth running.

## Run Scope

- Source dataset: `data/processed/all_categories`
- Isolated proof artifact directory: `data/processed/task_b_fast_proof_20260524`
- Output directory: `runs/eval/task_b_fast_proof_20260524`
- Retrieval artifact mode: `all_rating_weighted`
- Primary proof metric: `hybrid_candidate_recall@1000`
- Early-pool diagnostics: `hybrid_candidate_recall@50`, `hybrid_candidate_recall@100`
- Required slices: all, sparse users, cross-domain, and Beauty-heavy misses
- Ranking scope: candidate-recall only; top-10 ranker promotion still requires a separate full ranking run

## Baseline Snapshot

Current documented all-category Task B retrieval/ranking snapshot:

| Metric | Baseline |
| --- | ---: |
| `hybrid_candidate_recall@50` | `0.13` |
| `hybrid_candidate_recall@100` | `0.18` |
| `hybrid_candidate_recall@1000` | `0.34` |
| Sparse candidate recall@1000 | `0.3611` |
| Cross-domain candidate recall@1000 | `0.5484` |
| `hybrid_ranker_hit_rate@10` | `0.10` |
| `hybrid_ranker_ndcg@10` | `0.0766` |

## Commands

Artifact rebuild:

```bash
./.venv/bin/python -u scripts/build_retrieval_index.py \
  --train data/processed/task_b_fast_proof_20260524/train.jsonl \
  --items data/processed/task_b_fast_proof_20260524/items.jsonl \
  --output-dir data/processed/task_b_fast_proof_20260524 \
  --interaction-mode all_rating_weighted \
  --top-k 100 \
  --max-users-per-item 700 \
  --max-positive-items-per-user 80 \
  --review-term-max-terms-per-item 24 \
  --review-term-max-items-per-term 350
```

Artifact rebuild result:

- Runtime: `285.37` seconds.
- Proof artifact directory size: about `1.0G`.
- Baseline artifacts in `data/processed/all_categories` were not overwritten.

Smoke validation:

```bash
./.venv/bin/python -u eval/eval_task_b.py \
  --processed-dir data/processed/task_b_fast_proof_20260524 \
  --max-examples 300 \
  --sample-strategy stride \
  --candidate-limit 1000 \
  --hybrid-only \
  --candidate-recall-only \
  --disabled-sources vector_profile,neural_vector \
  --target-mode all_interactions \
  --output runs/eval/task_b_fast_proof_20260524/all_interactions_candidate_recall_smoke300.json \
  --miss-output runs/eval/task_b_fast_proof_20260524/all_interactions_candidate_recall_smoke300_misses.json \
  --max-misses 100 \
  --progress-every 100
```

Smoke result:

| Metric | 300-row smoke |
| --- | ---: |
| `hybrid_candidate_recall@50` | `0.1467` |
| `hybrid_candidate_recall@100` | `0.1867` |
| `hybrid_candidate_recall@1000` | `0.39` |
| Sparse candidate recall@1000 | `0.3936` |
| Cross-domain candidate recall@1000 | `0.6203` |

Full all-interaction eval was run as two deterministic shards to avoid loading the larger retrieval artifacts four times on the local laptop:

```bash
./.venv/bin/python -u eval/eval_task_b.py \
  --processed-dir data/processed/task_b_fast_proof_20260524 \
  --candidate-limit 1000 \
  --hybrid-only \
  --candidate-recall-only \
  --disabled-sources vector_profile,neural_vector \
  --target-mode all_interactions \
  --shard-count 2 \
  --shard-index 0 \
  --row-cache runs/eval/task_b_fast_proof_20260524/all_interactions_candidate_recall_shard0of2_cache.jsonl \
  --rebuild-row-cache \
  --output runs/eval/task_b_fast_proof_20260524/all_interactions_candidate_recall_shard0of2.json \
  --miss-output runs/eval/task_b_fast_proof_20260524/all_interactions_candidate_recall_shard0of2_misses.json \
  --max-misses 500 \
  --progress-every 1000
```

The second shard used `--shard-index 1` with matching output/cache paths. Aggregate command:

```bash
./.venv/bin/python eval/aggregate_task_b_reports.py \
  --inputs \
    runs/eval/task_b_fast_proof_20260524/all_interactions_candidate_recall_shard0of2.json \
    runs/eval/task_b_fast_proof_20260524/all_interactions_candidate_recall_shard1of2.json \
  --output runs/eval/task_b_fast_proof_20260524/all_interactions_candidate_recall.json
```

The positive-recommendation report was derived from the row caches, avoiding a second 2.8-hour retrieval pass:

```bash
./.venv/bin/python -u eval/report_task_b_from_row_cache.py \
  --processed-dir data/processed/task_b_fast_proof_20260524 \
  --row-caches \
    runs/eval/task_b_fast_proof_20260524/all_interactions_candidate_recall_shard0of2_cache.jsonl \
    runs/eval/task_b_fast_proof_20260524/all_interactions_candidate_recall_shard1of2_cache.jsonl \
  --target-mode positive_recommendation \
  --candidate-limit 1000 \
  --output runs/eval/task_b_fast_proof_20260524/positive_recommendation_candidate_recall.json \
  --miss-output runs/eval/task_b_fast_proof_20260524/positive_recommendation_candidate_recall_misses.json \
  --max-misses 500
```

## Runtime

| Step | Wall time |
| --- | ---: |
| Retrieval artifact rebuild | `285.37s` |
| 300-row smoke | `113.39s` |
| Full all-interaction shard 0/2 | `10061.13s` |
| Full all-interaction shard 1/2 | `10060.58s` |
| Positive report from row caches | `201.96s` |

The full sharded candidate-recall pass took about `2.8` hours wall-clock. The row caches occupy about `13G` under `runs/eval/task_b_fast_proof_20260524`; they are local run artifacts and are not part of the repo submission.

## Results

### All Interactions Target

This target mode treats every held-out next interaction as correct, including low ratings.

Target rating mix: `19,839` rows rated `1-3` (`0.2121`) and `73,699` rows rated `4-5` (`0.7879`).

| Metric | Fast proof | Baseline | Read |
| --- | ---: | ---: | --- |
| Examples | `93538` | - | Full Task B holdout. |
| `hybrid_candidate_recall@50` | `0.1302` | `0.13` | Essentially flat. |
| `hybrid_candidate_recall@100` | `0.1589` | `0.18` | Regression in early pool. |
| `hybrid_candidate_recall@1000` | `0.362` | `0.34` | Improvement in broad pool. |
| Sparse candidate recall@1000 | `0.3603` | `0.3611` | Slight miss. |
| Cross-domain candidate recall@1000 | `0.5752` | `0.5484` | Improvement. |

Promotion read: not a clean all-interactions promotion. The run improves broad Recall@1000 and cross-domain recall, but it regresses Recall@100 and narrowly misses the sparse-user gate.

### Positive-Recommendation Target

This target mode excludes held-out rows with rating below `4`, matching the recommendation interpretation more closely.

Target rating mix: `73,699` rows rated `4-5`.

| Metric | Fast proof | Baseline | Read |
| --- | ---: | ---: | --- |
| Examples | `73699` | - | Rating `4-5` holdout rows. |
| `hybrid_candidate_recall@50` | `0.151` | `0.13` | Passes early-pool gate. |
| `hybrid_candidate_recall@100` | `0.1823` | `0.18` | Passes early-pool gate. |
| `hybrid_candidate_recall@1000` | `0.3986` | `0.34` | Strong broad-pool lift. |
| Sparse candidate recall@1000 | `0.3973` | `0.3611` | Strong sparse-user lift. |
| Cross-domain candidate recall@1000 | `0.6081` | `0.5484` | Strong cross-domain lift. |

Promotion read: the retrieval gates pass for the positive-recommendation interpretation, but the run is still candidate-recall only. It cannot pass final Task B promotion until the ranker metrics are rerun on the same target mode.

## Miss Analysis

All-interactions misses:

- Candidate misses: `59673`
- Largest miss categories: `All_Beauty` `42264`, `Digital_Music` `10585`, `Subscription_Boxes` `709`, `For Him` `699`, `Chanukah` `632`
- Miss history buckets: sparse `53771`, medium `5145`, warm `757`

Positive-recommendation misses:

- Candidate misses: `44325`
- Largest miss categories: `All_Beauty` `29521`, `Digital_Music` `9324`, `For Him` `626`, `Chanukah` `606`, `Subscription_Boxes` `457`
- Miss history buckets: sparse `39661`, medium `4042`, warm `622`

The main remaining retrieval weakness is still sparse-user Beauty and Digital Music rows. The positive-target lift means rating/objective alignment is worth taking seriously before more ranker work.

## Engineering Notes

- The proof artifacts were built in an isolated directory, so the current submission artifacts under `data/processed/all_categories` were not overwritten.
- Four parallel eval shards caused memory pressure on the base M3 Pro MacBook Pro. Two shards were the stable throughput point.
- `eval/eval_task_b.py` now supports deterministic shard partitioning through `--shard-count` and `--shard-index`.
- `eval/aggregate_task_b_reports.py` aggregates shard reports with weighted metrics and combined promotion-gate checks.
- `eval/report_task_b_from_row_cache.py` derives alternate target-mode reports from cached eval rows without rerunning retrieval.

## Next Decision

Use the positive-recommendation target for the next serious Task B training/eval pass, because it aligns better with a recommendation benchmark and passes all candidate-recall gates in this proof. The next required run is a same-target ranker evaluation/training pass so HitRate@10 and NDCG@10 can be compared against the existing `0.10` / `0.0766` gate.
