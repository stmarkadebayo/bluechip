# Scoring Rubric And Current Status

Date: 2026-05-19

This document tracks the visible judging rubric from the challenge deck against the current implementation.

## Task A: User Modeling

Rubric items:

- Review text quality: ROUGE / BERTScore
- Rating accuracy: RMSE
- Behavioural fidelity: human evaluation
- Solution paper
- Code reproducibility

Current status:

- Rating prediction is implemented and trained. The latest optimization pass now selects by validation RMSE, writes a runtime stats artifact, and serves the model automatically when local artifacts are present.
- A Task A promotion gate now serves the lowest-RMSE head from the eval report. Current policy selects `calibrated_profile` because it beats the trained heads on the 5,000-example slice.
- Latest 5,000-example all-category Task A RMSE:
  - saved validation-MAE artifact: `1.487`
  - saved validation-RMSE artifact: `1.3378`
  - raw continuous trained model: `1.3395`
  - calibrated profile baseline: `1.2654`
- Review generation works, but ROUGE/BERTScore evaluation is not implemented yet.
- Behavioural fidelity is supported by profile-conditioned generation and validation. Human-eval scoring tables now exist, but they still need human scores.
- Code reproducibility is strong: local venv, tests, lint, Docker path, deterministic no-key generation, ignored large data, and documented commands.

Task A priority correction:

1. Add review text quality metrics: ROUGE-style overlap now, optional BERTScore later if dependency/runtime budget allows.
2. Continue RMSE work by reconciling the trained model with the lower-RMSE calibrated profile baseline.
3. Add a behavioural fidelity review rubric and a small scored eval set.
4. Update the solution paper with Task A metric tables and ablations.

## Task B: Recommendation

Rubric items:

- Ranking quality: NDCG@10 / Hit Rate, 30 points
- Cold-start and cross-domain, 25 points
- Contextual relevance: human evaluation, 20 points
- Solution paper, 15 points
- Code reproducibility, 10 points

Current status:

- Ranking endpoint, collaborative candidate generation, seen-item filtering, score components, source diagnostics, traces, and explanations are implemented.
- Eval now separates candidate recall from final rank quality and reports sparse/warm-user, cross-domain, and cold-start persona-only views.
- Current real-data ranking quality is weak:
  - `Subscription_Boxes`, 250 examples: hybrid HitRate@10 `0.216`, NDCG@10 `0.1195`
  - all-category, 100 examples over 188,236 items: hybrid HitRate@10 `0.09`, NDCG@10 `0.068`
  - all-category, 25-example candidate-recall smoke: base and hybrid candidate Recall@200 both `0.16`
- Updated all-category, 100-example, 188,236-item run after collaborative/category-affinity retrieval:
  - candidate Recall@200 `0.20`, Recall@500 `0.25`, Recall@1000 `0.28`
  - hybrid HitRate@10 `0.10` vs filtered popularity `0.09`
  - hybrid NDCG@10 `0.0766` vs filtered popularity `0.068`
  - cross-domain slice HitRate@10 `0.2581`, NDCG@10 `0.2056`
- Updated Beauty/sparse retrieval run:
  - candidate Recall@1000 `0.29`
  - hybrid HitRate@10 `0.10`, NDCG@10 `0.0766`
  - candidate misses `71`, down from `72`
- Review-term retrieval and lexical-neighbor retrieval are now wired into index build, eval, train, and runtime:
  - candidate Recall@1000 `0.32`
  - hybrid HitRate@10 `0.10`, NDCG@10 `0.0766`
  - candidate misses `68`, down from `71`
  - artifact: `data/processed/all_categories/review_term_retrieval.json`
- Candidate-aware learned ranker was trained on the same 100-example all-category slice but was not promoted: learned NDCG@10 `0.0700` was below current hybrid NDCG@10 `0.0766`.
- The learned ranker now uses a holdout split and is rejected unless it beats current hybrid on the same holdout slice. Latest split run after the retrieval upgrade was not promoted: learned NDCG@10 `0.0788` vs same-slice hybrid `0.1061`.
- Cold-start and cross-domain behavior exists structurally through category/user profiles and all-category data, and the eval now exposes first-class slice hooks.
- Contextual relevance exists in the demo and ranker features. Human-eval tables are generated, but scores are not populated yet.
- Contextual relevance is now backed by `docs/human_eval_task_b_contextual.md`, a 20-example judge-ready pack with real histories, explicit contexts, top-10 recommendations, source traces, and blank human-score columns.
- The ranker now applies context-category guards for explicit Beauty, music, and gift contexts, reducing off-topic global-popular recommendations in human-eval scenarios without changing empty-context offline metrics.
- Code reproducibility is strong, but Task B needs stronger retrieval and evaluation to be competitive.

Task B priority correction:

1. Continue improving Beauty-specific retrieval; most remaining candidate misses are still `All_Beauty`.
2. Improve sparse/warm-user retrieval separately; warm-user HitRate@10 is currently `0.0` on the bounded slice.
3. Get human scores populated for the contextual eval pack.
4. Promote learned ranker weights only when `eval/promote_ranker.py` passes on the same dataset/report.
5. Update the solution paper with final Task B ranking, cold-start, cross-domain, contextual relevance, and miss-analysis evidence.
