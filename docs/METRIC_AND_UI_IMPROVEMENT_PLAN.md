# Metric And UI Improvement Plan

## Metric Improvements

### Task A: Review Simulation

- Add shrinkage-calibrated rating prediction that blends global, user, item, and category priors.
- Keep profile signals as adjustments instead of letting sparse lexical overlap dominate the rating.
- Report warm/cold slices separately:
  - cold: 0 history items
  - light: 1-2 history items
  - medium: 3-9 history items
  - warm: 10+ history items
- Report category slices so weak domains are visible.
- Add a dependency-free training/tuning path for Task A weights.
- Train a Task A candidate matrix across compact/full feature sets, MSE/Huber/MAE losses, and calibrated/rounded star policies.
- Select the saved artifact by validation MAE, then report independent holdout slices separately.
- Keep generation grounded in the predicted rating, item facts, and user profile.
- Validate rating-review consistency and item mention in tests.

Current Task A status:

- Selected model: `full_mse_calibrated_star`
- Validation MAE: `0.2826`
- 5,000-example all-category saved-artifact MAE: `0.9124`
- 5,000-example all-category rounded raw-score MAE: `0.8834`

### Task B: Recommendation

- Measure candidate recall before final ranking.
- Build stronger item-item co-visitation from real data.
- Add neural embeddings later when runtime/credentials allow it.
- Train the ranker on sampled positives and negatives from the real corpus.
- Report cold-start, sparse-user, warm-user, category, and candidate-source slices.

## UI Improvements

- Add preset examples so judges can run the demo without editing JSON.
- Convert recommendation output into cards with score bars and tradeoff badges.
- Add evidence chips for user signals, item signals, and matched signals.
- Add a trace timeline: profile -> retrieve -> rank -> explain -> validate.
- Add metric cards for Task A and Task B submission numbers.
- Add a submission readiness panel showing dataset size, eval status, and latest metrics.
- Keep raw JSON available, but make it secondary to human-readable evidence.

## Current UI Rating

The current UI is about `6.5/10` visually: clean, readable, and operational, but still closer to an internal developer tool than a polished judging demo.
