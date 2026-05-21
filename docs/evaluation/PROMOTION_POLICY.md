# Promotion Policy

## Purpose

Bluechip promotes model, ranker, and generation behavior only when fixed evaluation evidence beats the current baseline and guardrails remain acceptable.

## General Rules

- Define the primary metric before running the experiment.
- Use temporal holdouts when available.
- Do not tune and promote on the same examples.
- Include slice metrics for known weak areas.
- Write a promotion or rejection report for every candidate artifact.
- Keep deterministic fallback behavior available for local development and CI.
- Treat sample-data wins as smoke tests only; submission claims need fixed real-data or bounded all-category reports with commands.

## Task A Rating Promotion

Promotion command:

```bash
./.venv/bin/python eval/promote_task_a.py
```

Acceptance bar:
- Candidate serving head must beat the current serving policy on the selected metric, defaulting to RMSE.
- Report must include candidate heads and metric values.
- Runtime policy must point to the selected serving head only after the gate passes.

Guardrails:
- Review generation must still pass validation consistency checks.
- External provider evaluation must respect the configured data policy.

## Task A Generation Promotion

Evaluation command:

```bash
./.venv/bin/python eval/eval_task_a_generation.py
```

Acceptance bar:
- Generation success rate must stay high.
- Validation consistency, item grounding, rating mention, sentiment alignment, and lexical quality metrics must not regress.
- Optional BERTScore may be used for semantic comparison when dependencies and local model access are available.

Guardrails:
- LLM output must remain downstream of fixed rating prediction.
- Non-sample private data must not be sent to external providers without explicit approval.

## Task B Ranking Acceptance

Acceptance bar:
- Hybrid candidate recall must not regress against base candidate recall.
- Hybrid ranker must beat filtered popularity on HitRate@10 and NDCG@10.
- The current bounded all-category ranking snapshot is HitRate@10 `0.10` and NDCG@10 `0.0766`; use same-slice comparisons before promoting.
- Candidate recall should be reported at @50, @100, and @1000. Current snapshot: `0.13`, `0.18`, and `0.34`.
- Sparse and cross-domain candidate recall must be reported. Current snapshot: sparse@1000 `0.3611`, cross-domain@1000 `0.5484`.

Guardrails:
- Seen-item filtering must remain enabled unless an eval explicitly requires repeat-item prediction.
- Candidate miss analysis must be reviewed when recall is weak.
- Weak slices such as Beauty, sparse-user, warm-user, and cross-domain scenarios must be reported.
- Vector retrieval must not be cited as a promotion reason while vector source recall remains `0.0`.

## Runtime Rollback

Rollback triggers:
- Primary metric regresses on the fixed report.
- Validation failure rate rises above the launch guardrail.
- Fallback rate or provider failure rate spikes.
- Trace data shows missing model/index versions after promotion.
- Privacy review finds unsafe explanation behavior.

Rollback actions:
- Remove or unset promoted artifact environment variables.
- Revert serving policy to the previous accepted artifact.
- Rebuild retrieval indexes only from approved data.
- Re-run the relevant eval and promotion command before re-enabling the artifact.
