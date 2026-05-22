# Evidence Intelligence Layer

## Summary

The Evidence Intelligence Layer is the local personalization layer for Bluechip. It combines aspect-aware user and item modeling, graph retrieval, sequential signals, evidence-aware ranking, plan-then-write review generation, and critic validation.

The goal is to make the system evidence-first while staying local and reproducible.

## Chosen Stack

The implementation combines the highest-leverage local ideas for the hackathon time box:

- **Aspect-aware evidence graph**: represents user/item fit by product aspects, category-aspect paths, and visible evidence terms.
- **Sequential retrieval**: uses recent positive history to recover likely next items through item and category transitions.
- **Lightweight evidence reranker**: adds aspect, graph, sequential, vector, collaborative, quality, novelty, and context features without requiring a heavy serving model.
- **Plan/critic generation**: builds a structured review plan before final text and validates generated text for grounding and sensitive-inference risk.

This is the default path. Larger neural embeddings, graph neural networks, or deep sequence recommenders can be added later, but only behind the same retrieval, ranking, evaluation, and promotion contracts.

## Components

| Component | Implementation | Purpose |
| --- | --- | --- |
| Aspect extraction | `app/services/intelligence/aspects.py` | Converts personas, reviews, and item metadata into structured aspect scores and evidence terms. |
| Evidence graph retrieval | `app/services/retrieval/evidence_graph.py` | Builds aspect-to-item, category-aspect, item-transition, and category-transition candidate paths. |
| Graph builder | `scripts/build_evidence_graph.py` | Builds `evidence_graph_retrieval.json` as a standalone artifact. |
| Retrieval integration | `app/services/retrieval/candidates.py` | Adds evidence graph candidates to the retrieval source portfolio. |
| Ranking features | `app/services/ranking/features.py` | Adds aspect, sequential, evidence graph, and Nigerian-context features. |
| Review planning | `app/services/generation/review_plan.py` | Creates a structured review plan before text generation. |
| Evidence critic | `app/services/validation/evidence_critic.py` | Checks generated text for grounding and sensitive inference issues. |
| Evidence eval | `eval/eval_evidence_intelligence.py` | Reports evidence-layer candidate recall, source counts, aspect coverage, and Nigerian-context coverage. |

## Data Flow

```mermaid
flowchart LR
  reviews["User Reviews / Persona"] --> aspects["Aspect Intelligence"]
  items["Item Metadata"] --> aspects
  aspects --> profiles["User And Item Profiles"]
  profiles --> graph["Evidence Graph"]
  graph --> retrieval["Candidate Retrieval"]
  retrieval --> ranking["Evidence-Aware Ranking"]
  ranking --> review_plan["Review Plan"]
  review_plan --> generation["Grounded Generation"]
  generation --> critic["Evidence Critic"]
  critic --> response["Response + Trace"]
```

## Retrieval Paths

- `aspect_evidence_graph`: user aspect preferences to matching items.
- `category_aspect_graph`: preferred category plus matching aspect to item.
- `sequential_transition`: recent liked item to next likely item.
- `category_transition`: recent liked category to next likely item.

These sources complement BM25, vector, review-term, lexical-neighbor, category-affinity, co-visitation, user-neighbor, and popularity sources.

## Ranking Signals

The ranker now includes:

- `aspect_match`
- `sequential_match`
- `evidence_graph_match`
- `nigerian_context_match`

These features are additive. Existing ranker behavior still works when the evidence graph artifact is missing.

## Review Generation

Task A review generation now follows:

```text
rating prediction -> review plan -> grounded text -> evidence critic
```

The review plan contains predicted rating, verdict, voice style, locale tone, positive evidence, negative evidence, aspect scores, and required mentions.

## Commands

Build the evidence graph:

```bash
python scripts/build_evidence_graph.py \
  --train data/processed/train.jsonl \
  --items data/processed/items.jsonl \
  --output data/processed/evidence_graph_retrieval.json
```

Run the evidence-layer eval:

```bash
python eval/eval_evidence_intelligence.py
```

The full retrieval index builder also writes `evidence_graph_retrieval.json`:

```bash
python scripts/build_retrieval_index.py \
  --train data/processed/train.jsonl \
  --items data/processed/items.jsonl \
  --output-dir data/processed
```

## Acceptance Use

This layer is not a replacement for Task B evaluation. Use it as an evidence source and diagnostic report. Final Task B quality claims still depend on `eval/eval_task_b.py`.
