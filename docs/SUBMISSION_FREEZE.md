# Submission Freeze Plan

Date: 2026-05-22

This is the frozen path from the current Bluechip repo state to final submission. Scope is intentionally narrow: protect the working app, report only validated evidence, package the repo cleanly, and keep long-running model work out of the final hours.

## Frozen Scope

In scope before submission:

- current FastAPI app, browser demo, Docker packaging, and typed API contracts;
- Task A rating-first review simulation;
- Task B multi-head retrieval, hybrid ranking, explanations, and source diagnostics;
- bounded LLM profile enhancement, review generation, recommendation reasoning, and validation with deterministic fallback;
- DeepSeek/OpenRouter/OpenAI provider support, with DeepSeek already smoke-tested locally;
- SQLite local feature-store backend;
- completed all-category `implicit` ALS/BPR/item-item baseline report;
- completed Task B contextual human-eval scoring and Task A human-eval review pack;
- 24 May Task B positive-recommendation candidate-recall proof;
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

Only already-validated work is allowed into the final submission. Long-running ranker training and new model families move to future work.

2. Human eval.

Task B contextual relevance has a scored CSV summary. The Task A behavioural review pack is included as judge-facing evidence support, while the official human score remains judge-run.

3. Optional `implicit` baseline.

The ALS/BPR/item-item baseline report is complete. Do not rerun it during packaging.

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

The repo is submission-ready after final validation, paper PDF generation, secret/artifact hygiene, and live smoke testing. The only major Task B caveat is explicitly documented: the positive-recommendation run proves candidate-recall lift, while same-target learned-ranker HitRate@10/NDCG@10 remains next work.
