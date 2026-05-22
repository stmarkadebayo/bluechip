# Implicit Baseline Results

Date: 2026-05-22

This report records the first conventional collaborative-filtering baseline run using the `implicit` Python package. It is a benchmark for Task B, not a replacement for the Bluechip contextual hybrid ranker.

## Command

```bash
./.venv/bin/python eval/eval_implicit_baselines.py \
  --processed-dir data/processed/all_categories \
  --output runs/eval/implicit_baselines_all_categories_5000.json \
  --max-examples 5000 \
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
| Task B examples evaluated | 5,000 |
| Users in train matrix | 923,201 |
| Items in train matrix | 188,236 |
| Matrix nonzeros | 975,058 |
| Matrix build time | 2.9415s |

## Overall Metrics

| Model | HitRate@10 | NDCG@10 | Recall@50 | Recall@100 | Recall@1000 | Fit seconds | Recommend seconds |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| ALS | 0.0104 | 0.0059 | 0.0294 | 0.0468 | 0.1280 | 32.2130 | 15.5789 |
| BPR | 0.0002 | 0.0001 | 0.0022 | 0.0046 | 0.0356 | 5.1418 | 18.4849 |
| Item-item cosine | 0.0516 | 0.0280 | 0.1030 | 0.1254 | 0.2914 | 0.5001 | 14.7338 |

## Slice Metrics

| Model | Slice | Examples | HitRate@10 | NDCG@10 | Recall@1000 |
| --- | --- | ---: | ---: | ---: | ---: |
| ALS | sparse_history_1_2 | 4,264 | 0.0108 | 0.0063 | 0.1180 |
| ALS | medium_history_3_7 | 593 | 0.0101 | 0.0045 | 0.1636 |
| ALS | warm_history_8_plus | 143 | 0.0000 | 0.0000 | 0.2797 |
| ALS | cross_domain | 1,236 | 0.0202 | 0.0120 | 0.2087 |
| BPR | sparse_history_1_2 | 4,264 | 0.0002 | 0.0001 | 0.0382 |
| BPR | medium_history_3_7 | 593 | 0.0000 | 0.0000 | 0.0152 |
| BPR | warm_history_8_plus | 143 | 0.0000 | 0.0000 | 0.0420 |
| BPR | cross_domain | 1,236 | 0.0000 | 0.0000 | 0.0372 |
| Item-item cosine | sparse_history_1_2 | 4,264 | 0.0551 | 0.0302 | 0.2920 |
| Item-item cosine | medium_history_3_7 | 593 | 0.0371 | 0.0187 | 0.2934 |
| Item-item cosine | warm_history_8_plus | 143 | 0.0070 | 0.0030 | 0.2657 |
| Item-item cosine | cross_domain | 1,236 | 0.1068 | 0.0479 | 0.4717 |

## Interpretation

- Item-item cosine is the only useful `implicit` baseline from this pass.
- Item-item Recall@1000 `0.2914` is below the current hybrid candidate Recall@1000 gate of `0.34`, but it is strong enough to cite as a conventional baseline.
- Item-item cross-domain Recall@1000 `0.4717` is close to the current hybrid cross-domain Recall@1000 `0.5484`, so it supports the paper's claim that co-engagement/item-neighbor evidence is valuable.
- ALS is weaker than item-item for this sparse, mostly one-to-few-history split.
- BPR did not learn a useful ranking under this quick configuration. Treat it as a baseline attempt, not a promoted model.

## Paper Wording

Use conservative wording:

> We trained `implicit` ALS, BPR, and item-item collaborative-filtering baselines on the all-category training split. The item-item baseline was strongest, reaching HitRate@10 `0.0516`, NDCG@10 `0.0280`, and Recall@1000 `0.2914` on a 5,000-example all-category Task B slice. The hybrid Bluechip candidate stack remains stronger on the currently logged all-category Recall@1000 gate (`0.34`) and cross-domain Recall@1000 (`0.5484`), while item-item provides a useful conventional comparison.

Do not claim that ALS or BPR improved the final system.
