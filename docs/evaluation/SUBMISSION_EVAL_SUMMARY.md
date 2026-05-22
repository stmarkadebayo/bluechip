# Submission Evaluation Summary

This is the judge-facing metric snapshot for the DSN x BCT submission package. It is bounded local evidence, not a claim that every full-corpus experiment has been exhaustively run.

The active scope freeze and eight-step submission path are documented in [../SUBMISSION_FREEZE.md](../SUBMISSION_FREEZE.md).

## Current Submission Sequence

1. Freeze scope around the current evidence-first hybrid agent.
2. Complete human eval from the CSV packs.
3. Run the optional `implicit` ALS/BPR/item-item baseline on all categories.
4. Run final validation.
5. Finalize the 4-8 page solution paper.
6. Package the repo safely.
7. Demo-check the API/UI.
8. Submit the repo, paper PDF, architecture diagram, app/API instructions, and eval summary.

## Brief Alignment

Required deliverables:

- Containerized app/API for Task A review and rating simulation.
- Containerized app/API for Task B personalized recommendation.
- Solution paper, 4-8 pages.
- Clean, documented, reproducible code repository.

Task A scoring areas from the brief:

| Area | Points |
| --- | ---: |
| Review Text Quality: ROUGE / BERTScore | 30 |
| Rating Accuracy: RMSE | 15 |
| Behavioural Fidelity: human eval | 20 |
| Solution Paper | 15 |
| Code Reproducibility | 10 |
| Cross-domain | 10 |

Task B scoring areas from the brief:

| Area | Points |
| --- | ---: |
| Ranking Quality: NDCG@10 / Hit Rate | 30 |
| Cold-Start & Cross-Domain | 25 |
| Contextual Relevance: human eval | 20 |
| Solution Paper | 15 |
| Code Reproducibility | 10 |

Additional marks are available when Nigerian contextualization is useful, grounded, and authentic.

## Current Task A Signal

Task A is rating-first:

- The review text is generated after rating prediction, not before.
- The serving head is selected by fixed eval and promotion policy.
- The latest documented 5,000-example all-category RMSE gate promotes `calibrated_profile` with RMSE `1.2654`.
- Final all-category generation smoke: 25 examples, validation consistency `1.0`, rating mention `1.0`, item grounding `1.0`, sentiment alignment `1.0`, ROUGE-L F1 `0.0836`, unigram F1 `0.1185`.
- Generation eval tracks provider failures, fallback rate, validation consistency, rating mention, item grounding, sentiment alignment, ROUGE-L F1, and unigram F1.
- Behavioural fidelity still needs scored human labels; the repo provides the structure and examples needed for that review.

## Current Task B Metrics

Latest bounded all-category Task B metrics after evidence graph work and the popularity-rank floor:

| Metric | Value | Submission read |
| --- | ---: | --- |
| `hybrid_candidate_recall@50` | `0.13` | Early candidate pool remains weak. |
| `hybrid_candidate_recall@100` | `0.18` | Candidate recall is still the main bottleneck. |
| `hybrid_candidate_recall@1000` | `0.34` | Best current overall candidate-recall signal. |
| `hybrid_ranker_hit_rate@10` | `0.10` | Current top-10 ranking gate. |
| `hybrid_ranker_ndcg@10` | `0.0766` | Current top-10 ranking gate. |
| Sparse candidate recall@1000 | `0.3611` | Credible sparse-user signal, not solved. |
| Cross-domain candidate recall@1000 | `0.5484` | Strongest measured Task B slice. |
| Vector source recall | `0.0` | Diagnostic only; do not claim a quality lift. |

Lean-pruning review, 22 May 2026:

- `implicit_item_item` is now a real retrieval source backed by a lazy SQLite artifact.
- The lean run disables `vector_profile`, `bm25_profile`, `beauty_sparse_tail`, `sparse_category_tail`, and `neural_vector` for the no-context all-category gate.
- Retrieval source family metadata now lives in one registry used by serving, ranking, and eval diagnostics.
- `eval_task_b.py` is split into dataset-builder, evaluator, and runner layers while preserving the existing JSON report contract.
- The lean 100x1000 gate preserved HitRate@10 `0.10`, NDCG@10 `0.0766`, Recall@50 `0.13`, Recall@100 `0.18`, and Recall@1000 `0.34`.
- A larger 250-example candidate-only diagnostic produced Recall@50 `0.104`, Recall@100 `0.136`, Recall@1000 `0.28`, and cross-domain Recall@1000 `0.4615`.
- Details and engineering decisions are recorded in [QUALITY_REVIEW_PRUNING.md](QUALITY_REVIEW_PRUNING.md).

## Human Evaluation

Task B contextual human eval is recorded in [HUMAN_EVAL_TASK_B_CONTEXTUAL_RESULTS.md](HUMAN_EVAL_TASK_B_CONTEXTUAL_RESULTS.md), generated from [../human_eval_task_b_contextual.csv](../human_eval_task_b_contextual.csv).

| Human-eval metric | Value |
| --- | ---: |
| Examples scored | `20` |
| Top-10 relevance mean | `2.15 / 5` |
| Context fit mean | `2.25 / 5` |
| Diversity mean | `2.35 / 5` |
| Explanation quality mean | `2.10 / 5` |
| Composite mean | `2.2125 / 5` |
| Linear contextual relevance estimate | `8.85 / 20` |
| Target in top-10 | `0 / 20` |

Interpretation:

- Human eval confirms the main Task B weakness is contextual precision, not just broad candidate recall.
- The strongest recurring issue is that generic same-category beauty items can outrank the specific sub-intent requested by the context.
- A follow-up ranker patch adds explicit hair/skincare/nail/gift sub-intent boosts and penalties for contextual requests, with terms and weights in `app/services/ranking/context_intents.json`.

Final validation smoke, 22 May 2026:

| Run | Result |
| --- | --- |
| Task B all-categories legacy, 50 examples, candidate-limit 100 | HitRate@10 `0.06`, NDCG@10 `0.0471`, candidate Recall@50 `0.08`, candidate Recall@100 `0.12`, cross-domain candidate Recall@100 `0.2857`. |
| Task B all-categories neural FAISS, 5 examples, candidate-limit 50 | Prebuilt `188,236`-vector FAISS index loaded with companion item-id map; `neural_vector` contributed 63 candidates; held-out hits were `0/5`, so this is runtime validation only. |
| Task B sample-data neural smoke | Neural FAISS active on 11 items; `neural_vector` candidate recall@100 `1.0`. |
| Task A all-categories agentic generation, 25 examples | Deterministic fallback succeeded; validation consistency, item mention, rating mention, and sentiment alignment were all `1.0`. |

## Implicit Baseline

The first conventional collaborative-filtering baseline run is recorded in [IMPLICIT_BASELINE_RESULTS.md](IMPLICIT_BASELINE_RESULTS.md). It trained ALS, BPR, and item-item cosine with `implicit` on the all-category training split and evaluated 30,000 Task B examples at candidate-limit `1000`.

| Model | HitRate@10 | NDCG@10 | Recall@50 | Recall@100 | Recall@1000 |
| --- | ---: | ---: | ---: | ---: | ---: |
| ALS | `0.0115` | `0.0066` | `0.0298` | `0.0441` | `0.1129` |
| BPR | `0.0009` | `0.0004` | `0.0030` | `0.0059` | `0.0364` |
| Item-item cosine | `0.0583` | `0.0348` | `0.1067` | `0.1314` | `0.2863` |

Interpretation:

- Item-item cosine is the useful conventional baseline from this run.
- The current hybrid candidate Recall@1000 gate remains stronger at `0.34`, but item-item is close enough to validate that co-engagement/item-neighbor evidence is worth keeping.
- ALS and BPR should be reported as attempted baselines, not promoted models.

Interpretation:

- Candidate generation is the main Task B bottleneck, especially at @50 and @100.
- Cross-domain retrieval is the strongest measured slice and should be highlighted.
- Ranking changes should be promoted only when same-slice NDCG@10 and HitRate@10 beat the current hybrid gate.
- Neural/vector retrieval exists for a swappable embedding path, but current all-category smoke evidence is runtime validation rather than a proven quality lift.

## Reproduction Commands

Install and run locally:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
uvicorn app.main:app --reload
```

Open:

```text
http://127.0.0.1:8000/ui/
http://127.0.0.1:8000/docs
http://127.0.0.1:8000/api/health
```

Run with Docker:

```bash
docker compose up --build
```

Core checks:

```bash
ruff check .
pytest
python -m compileall app eval scripts tests
```

Fast sample eval:

```bash
make eval
```

Task A rating gate:

```bash
python eval/eval_task_a.py \
  --reviews data/processed/all_categories/reviews.jsonl \
  --items data/processed/all_categories/items.jsonl \
  --processed-dir data/processed/all_categories \
  --output runs/eval/all_categories_task_a_with_model_rmse_5000.json \
  --max-examples 5000 \
  --model-path data/processed/all_categories/task_a_model_rmse.json

python eval/promote_task_a.py \
  --task-a-report runs/eval/all_categories_task_a_with_model_rmse_5000.json \
  --output runs/eval/all_categories_task_a_serving_promotion.json \
  --policy-output data/processed/all_categories/task_a_serving_policy.json
```

Task B candidate/ranker gate:

```bash
python eval/eval_task_b.py \
  --processed-dir data/processed/all_categories \
  --output runs/eval/submission_task_b_100x1000.json \
  --miss-output runs/eval/submission_task_b_100x1000_misses.json \
  --max-examples 100 \
  --candidate-limit 1000
```

Lean Task B gate:

```bash
python eval/eval_task_b.py \
  --processed-dir data/processed/all_categories \
  --output runs/eval/task_b_pruned_nodiversity_100x1000.json \
  --miss-output runs/eval/task_b_pruned_nodiversity_100x1000_misses.json \
  --max-examples 100 \
  --candidate-limit 1000 \
  --hybrid-only \
  --disabled-sources vector_profile,bm25_profile,beauty_sparse_tail,sparse_category_tail,neural_vector
```

Task B source ablation gate:

```bash
python eval/run_task_b_source_ablation.py \
  --processed-dir data/processed/all_categories \
  --max-examples 100 \
  --candidate-limit 1000
```

Build the implicit item-item retrieval artifact:

```bash
python scripts/build_implicit_item_index.py \
  --processed-dir data/processed/all_categories \
  --output data/processed/all_categories/implicit_item_neighbors.sqlite \
  --neighbors 100
```

Contextual human-eval pack:

```bash
python eval/create_task_b_contextual_eval.py \
  --processed-dir data/processed/all_categories \
  --output docs/human_eval_task_b_contextual.md \
  --max-examples 20 \
  --candidate-limit 1000
```

Provider-backed Task A generation smoke:

```bash
python eval/eval_task_a_generation.py \
  --strict-provider \
  --external-data-policy redact \
  --max-examples 25
```

Neural FAISS index support:

```bash
python scripts/build_neural_index.py \
  --items data/processed/all_categories/items.jsonl \
  --output data/processed/all_categories/neural_index.faiss

# If the FAISS file already exists, regenerate only the companion id map:
python scripts/build_neural_index.py \
  --items data/processed/all_categories/items.jsonl \
  --output data/processed/all_categories/neural_index.faiss \
  --ids-only
```

## Promotion Discipline

- Improve candidate Recall@50 and Recall@100 while maintaining Recall@1000.
- Preserve or improve sparse recall@1000 and cross-domain recall@1000.
- Accept ranking changes only when same-slice NDCG@10 and HitRate@10 beat the current hybrid ranker.
- Add scored human labels to the contextual relevance and behavioural fidelity packs.
- Treat stronger neural embeddings and sequence models as out of scope until fixed source-level recall improves.
