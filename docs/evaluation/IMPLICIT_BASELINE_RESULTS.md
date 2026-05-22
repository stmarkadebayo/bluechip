# Implicit Baseline Results

Date: 2026-05-22

This report records the conventional collaborative-filtering baseline run using the `implicit` Python package. It is a benchmark for Task B, not a replacement for the Bluechip contextual hybrid ranker.

## Command

```bash
./.venv/bin/python eval/eval_implicit_baselines.py \
  --processed-dir data/processed/all_categories \
  --output runs/eval/implicit_baselines_all_categories_30000.json \
  --max-examples 30000 \
  --candidate-limit 1000 \
  --als-factors 64 \
  --als-iterations 10 \
  --bpr-factors 64 \
  --bpr-iterations 10 \
  --item-neighbors 100
```

## Data

| Field | Value |
| --- | ---: |
| Train interactions | 978,425 |
| Task B examples evaluated | 30,000 |
| Users in train matrix | 923,201 |
| Items in train matrix | 188,236 |
| Matrix nonzeros | 975,058 |
| Matrix build time | 1.2445s |

## Overall Metrics

| Model | HitRate@10 | NDCG@10 | Recall@50 | Recall@100 | Recall@1000 | Fit seconds | Recommend seconds |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| ALS | 0.0115 | 0.0066 | 0.0298 | 0.0441 | 0.1129 | 22.1413 | 56.3393 |
| BPR | 0.0009 | 0.0004 | 0.0030 | 0.0059 | 0.0364 | 2.9165 | 57.9213 |
| Item-item cosine | 0.0583 | 0.0348 | 0.1067 | 0.1314 | 0.2863 | 0.3125 | 42.1182 |

## Slice Metrics

| Model | Slice | Examples | HitRate@10 | NDCG@10 | Recall@1000 |
| --- | --- | ---: | ---: | ---: | ---: |
| ALS | sparse_history_1_2 | 26,905 | 0.0112 | 0.0065 | 0.1073 |
| ALS | medium_history_3_7 | 2,776 | 0.0140 | 0.0075 | 0.1556 |
| ALS | warm_history_8_plus | 319 | 0.0125 | 0.0065 | 0.2132 |
| ALS | cross_domain | 6,675 | 0.0243 | 0.0138 | 0.1907 |
| BPR | sparse_history_1_2 | 26,905 | 0.0010 | 0.0004 | 0.0375 |
| BPR | medium_history_3_7 | 2,776 | 0.0000 | 0.0000 | 0.0259 |
| BPR | warm_history_8_plus | 319 | 0.0000 | 0.0000 | 0.0313 |
| BPR | cross_domain | 6,675 | 0.0003 | 0.0001 | 0.0427 |
| Item-item cosine | sparse_history_1_2 | 26,905 | 0.0605 | 0.0363 | 0.2852 |
| Item-item cosine | medium_history_3_7 | 2,776 | 0.0432 | 0.0246 | 0.3019 |
| Item-item cosine | warm_history_8_plus | 319 | 0.0031 | 0.0014 | 0.2445 |
| Item-item cosine | cross_domain | 6,675 | 0.1086 | 0.0500 | 0.4440 |

## Interpretation

- Item-item cosine is the only useful `implicit` baseline from this pass.
- Item-item Recall@1000 `0.2863` is below the current hybrid candidate Recall@1000 gate of `0.34`, but it is strong enough to cite as a conventional baseline.
- Item-item cross-domain Recall@1000 `0.4440` is directionally close to the current hybrid cross-domain Recall@1000 `0.5484`, so it supports the paper's claim that co-engagement/item-neighbor evidence is valuable.
- ALS is weaker than item-item for this sparse, mostly one-to-few-history split.
- BPR did not learn a useful ranking under this quick configuration. Treat it as a baseline attempt, not a promoted model.

## Paper Wording

Use conservative wording:

> We trained `implicit` ALS, BPR, and item-item collaborative-filtering baselines on the all-category training split. The item-item baseline was strongest, reaching HitRate@10 `0.0583`, NDCG@10 `0.0348`, and Recall@1000 `0.2863` on a 30,000-example all-category Task B slice. The hybrid Bluechip candidate stack remains stronger on the currently logged all-category Recall@1000 gate (`0.34`) and cross-domain Recall@1000 (`0.5484`), while item-item provides a useful conventional comparison.

Do not claim that ALS or BPR improved the final system.
