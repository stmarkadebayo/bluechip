# Submission Freeze Plan

Date: 2026-05-22

This is the frozen path from the current Bluechip repo state to final submission. Scope is intentionally narrow: protect the working app, finish human evidence, add only the fastest conventional Task B baseline if it runs cleanly, then do final validation.

## Frozen Scope

In scope before submission:

- current FastAPI app, browser demo, Docker packaging, and typed API contracts;
- Task A rating-first review simulation;
- Task B multi-head retrieval, hybrid ranking, explanations, and source diagnostics;
- bounded LLM profile enhancement, review generation, recommendation reasoning, and validation with deterministic fallback;
- DeepSeek/OpenRouter/OpenAI provider support, with DeepSeek already smoke-tested locally;
- SQLite local feature-store backend;
- all-category `implicit` baselines only: ALS, BPR, and item-item;
- human-eval scoring from the generated CSV packs;
- final paper, README, evaluation summary, and push hygiene.

Out of scope before submission:

- LightGCN, SASRec, HSTU, PETER, PEPLER, NARRE, or a trained Wide & Deep model;
- a new agent framework such as LangGraph, CrewAI, or AutoGen;
- PostHog as an eval replacement;
- unproven quality claims for neural FAISS, graph evidence, or LLM reranking;
- large UI expansion.

The paper claim stays conservative: Bluechip is an evidence-first hybrid user-intelligence agent. Deterministic systems handle profiling, retrieval, rating, ranking, validation, metrics, and fallbacks. LLMs are downstream helpers for bounded enrichment, generation, explanation, and critique.

## Eight-Step Submission Plan

1. Freeze scope.

Only `implicit` baselines are allowed as new model work. All other neural/modeling ideas move to future work unless they are already implemented and evaluated.

2. Human eval.

Import the CSV packs into Google Sheets, score Task A behavioral fidelity and Task B contextual relevance, average scores by dimension, and add the results to the paper.

3. Optional `implicit` baseline.

Train/evaluate ALS, BPR, and item-item on `data/processed/all_categories`. Report Recall@50/100/1000, HitRate@10, and NDCG@10. Use results in the paper only if the run is clean and reproducible.

4. Final evaluation run.

Run `ruff check .`, `pytest`, `python -m compileall app eval scripts tests`, Task A eval/generation smoke, Task B bounded all-category eval, SQLite feature-store smoke, DeepSeek smoke, and API smoke.

5. Paper finalization.

Keep the solution paper within 4-8 pages. Include architecture, Task A flow, Task B flow, metrics, baselines, human eval, Nigerian contextualization, limitations, and next work.

6. Repo packaging.

Confirm `.env`, secrets, large data, processed artifacts, run outputs, and virtualenv files are ignored. Confirm README quickstart, Docker, and eval instructions are usable by a teammate or judge.

7. Demo check.

Start the API and verify `/api/health`, `/docs`, `/ui/`, `/api/simulate-review`, `/api/recommend`, `/api/runtime/feature-store`, `/api/metrics`, and `/api/traces`.

8. Submission assets.

Prepare the GitHub repo link, solution paper PDF, architecture diagram, eval summary, human-eval summary, app/API instructions, and any caveats about ignored local full-data artifacts.

## Current Readiness Call

- Without human eval: not final-submission ready because two rubric rows are human-scored.
- With human eval and final validation: about 85% ready.
- With a clean `implicit` baseline report added: about 90% ready.

The remaining critical blockers are human-eval labels, final validation, paper polish, repo hygiene, and packaging the final submission assets.
