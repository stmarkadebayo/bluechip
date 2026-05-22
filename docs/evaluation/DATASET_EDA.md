# Dataset EDA

Processed directory: `data/processed/all_categories`

## Corpus Summary

| Metric | Value |
| --- | ---: |
| Reviews | 1,071,963 |
| Users | 923,201 |
| Items | 188,236 |
| Observed user-item pairs | 1,059,332 |
| Matrix density | 0.00000610 |
| Matrix sparsity | 0.99999390 |
| Missing review text rows after ingestion | 709 |
| Missing item names | 0 |
| Missing item summaries | 56 |

## Splits

| Split | Rows |
| --- | ---: |
| Train | 978,425 |
| Task A holdout | 93,538 |
| Task B holdout | 93,538 |

## Ratings

| Rating | Reviews |
| --- | ---: |
| 1 | 132,793 |
| 2 | 53,509 |
| 3 | 72,750 |
| 4 | 110,058 |
| 5 | 702,853 |

## Top Review Categories

| Category | Reviews |
| --- | ---: |
| All_Beauty | 701,421 |
| Digital_Music | 130,388 |
| For Him | 66,885 |
| Subscription_Boxes | 16,215 |
| Chanukah | 14,912 |
| Restaurants | 12,164 |
| Christmas | 11,695 |
| Amazon Incentives Brand Guidelines | 8,346 |
| Clothing, Shoes & Accessories | 8,307 |
| Specialty Cards | 7,386 |

## Top Item Categories

| Category | Items |
| --- | ---: |
| All_Beauty | 112,559 |
| Digital_Music | 70,501 |
| Magazine_Subscriptions | 834 |
| Subscription_Boxes | 641 |
| Restaurants | 297 |
| For Him | 231 |
| Midwest | 205 |
| Baseball | 198 |
| Design & Decoration | 148 |
| Specialty Cards | 140 |

## Distribution Summaries

| Distribution | Min | P50 | Mean | P90 | P95 | P99 | Max |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Reviews per user | 1.0 | 1.0 | 1.1611 | 2.0 | 2.0 | 4.0 | 341.0 |
| Reviews per item | 1.0 | 1.0 | 5.6948 | 8.0 | 16.0 | 63.0 | 36859.0 |
| Metadata rating_number | 1.0 | 1.0 | 5.6948 | 8.0 | 16.0 | 63.0 | 36859.0 |
| Item average_rating | 1.0 | 4.667 | 4.0876 | 5.0 | 5.0 | 5.0 | 5.0 |

## Task B Slice Shape

| Slice | Examples | Share |
| --- | ---: | ---: |
| Sparse history, 1-2 train reviews | 84,053 | 0.8986 |
| Medium history, 3-7 train reviews | 8,372 | 0.0895 |
| Warm history, 8+ train reviews | 1,113 | 0.0119 |
| Cross-domain examples | 23,671 | 0.2531 |

## Notes

- Rows with missing user_id, missing item_id, empty source review text, or invalid ratings are filtered during ingestion; residual missing-like normalized text is tracked in this report.
- Temporal splits keep each eligible user's latest review as the Task A and Task B holdout target.
- Task A and Task B currently share the same latest-review holdout rows because the brief evaluates different outputs over the same behavioral prediction setup.
