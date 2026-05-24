# Push Readiness Notes

Date: 2026-05-22

This repo is ready to share with a teammate as source code, tests, docs, and reproducible sample data. Large local datasets, processed artifacts, run outputs, virtualenvs, and secrets are intentionally ignored. The active final-submission plan is [docs/SUBMISSION_FREEZE.md](SUBMISSION_FREEZE.md).

## Frozen Scope

- Keep the evidence-first hybrid architecture.
- Keep the completed Task B contextual human-eval summary and Task A review pack.
- Keep the completed `implicit` ALS/BPR/item-item baseline report.
- Do not start LightGCN, SASRec, HSTU, PETER, PEPLER, NARRE, or trained Wide & Deep now.
- Run the final validation suite after paper PDF generation and repo hygiene checks.

## What Changed Recently

- Added Task A RMSE-focused training, promotion, and runtime serving policy.
- Added DeepSeek/OpenRouter/OpenAI generation provider support with external-data guards.
- Added Task B candidate recall reporting, miss analysis, and slice metrics.
- Added collaborative, review-term, lexical item-neighbor, Beauty/sparse, vector, BM25, category, and popularity retrieval sources.
- Added learned-ranker tooling behind explicit eval/promotion gates; the runtime uses the measured hybrid ranker unless an artifact is configured.
- Added context-category guards for explicit Beauty, music, and gift recommendation contexts.
- Added contextual Task B human-eval pack: `docs/human_eval_task_b_contextual.md`.
- Added Google Sheets-ready CSV exports for human eval.
- Added API contract, generation, retrieval, downloader, Task A rating, and recommendation regression tests.
- Added the 24 May Task B fast proof showing positive-target candidate-recall lift.

## Current Validation

Run before pushing:

```bash
./.venv/bin/ruff check .
./.venv/bin/pytest
./.venv/bin/python -m compileall app eval scripts tests
```

Last recorded full validation result before this documentation pass:

- Ruff: passed
- Pytest: passed
- Compileall: passed

## Git Hygiene

Ignored on purpose:

- `.env`, `.env.*`
- `.venv/`
- `data/raw/*`
- `data/processed/*`
- `runs/*`
- caches and local coverage files

Commit source/docs/tests only:

```bash
git add .env.example .gitignore Dockerfile Makefile README.md app docs eval paper pyproject.toml research scripts tests ui
git status --short
git commit -m "Tighten Bluechip agent for teammate handoff"
git push origin main
```

Do not use `git add -f` on ignored data or run artifacts.

## Teammate Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
make PYTHON=./.venv/bin/python eval
uvicorn app.main:app --reload
```

Open:

```text
http://127.0.0.1:8000/docs
http://127.0.0.1:8000/ui/
```

For real-data work, teammates must download/build ignored local artifacts:

```bash
python scripts/download_amazon_hf.py --with-metadata
python scripts/download_amazon_hf.py --with-metadata --check-only --strict
```

Then follow the all-category commands in `README.md` and `docs/HANDOFF.md`.
