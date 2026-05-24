# Submission Demo Script

## Setup

Run the app:

```bash
uvicorn app.main:app --reload
```

Open:

```text
http://127.0.0.1:8000/ui/
```

Fallback path if the UI is not available:

```text
http://127.0.0.1:8000/docs
```

## Talk Track

Bluechip is an evidence-first personalization agent for the DSN x BCT challenge. It solves Task A and Task B with the same user-intelligence layer: profile visible behavior, retrieve or score product evidence, rank or predict, generate only after the decision, validate the output, and return a trace.

## Flow 1: Task B Recommendation

1. Open the Recommend tab.
2. Use the Lagos student / conversation-friendly dinner scenario or another Nigerian-context example already in the UI.
3. Run the recommendation.
4. Point out:
   - ranked products
   - candidate sources
   - score components
   - tradeoffs
   - candidate diagnostics
   - trace ID

What to say:

```text
The agent is not asking an LLM to choose products from the catalog. It retrieves candidates from behavior, review terms, evidence graph paths, BM25, popularity floors, and diagnostics, then ranks the candidate pool with explicit score components.
```

## Flow 2: Cold-Start Or Cross-Domain

1. Switch to a cold-start or cross-domain example.
2. Run the recommendation.
3. Highlight that sparse and cross-domain slices are measured separately.

Use the current metric snapshot:

| Metric | Value |
| --- | ---: |
| Sparse candidate recall@1000 | `0.3611` |
| Cross-domain candidate recall@1000 | `0.5484` |
| Positive-target sparse candidate recall@1000 | `0.3973` |
| Positive-target cross-domain candidate recall@1000 | `0.6081` |

What to say:

```text
Cross-domain candidate retrieval is currently the strongest Task B slice. Sparse users are still hard, but we measure them directly instead of hiding them inside an average.
```

## Flow 3: Task A Review Simulation

1. Open the Review tab.
2. Run the strict-reviewer or visible-history example.
3. Point out:
   - predicted rating
   - generated review
   - user signals
   - item signals
   - validation status
   - trace steps

What to say:

```text
Task A is rating-first. The system predicts the star rating from user and item evidence, then writes a rating-conditioned review and validates that the review is consistent with the rating and item facts.
```

## Flow 4: Runtime Evidence

1. Open the Metrics tab.
2. Refresh metrics and traces.
3. Show:
   - endpoint request counts
   - generation provider/fallback mix
   - validation status
   - trace IDs

What to say:

```text
Every demo output is traceable. The submission is designed so judges can inspect how a recommendation or review was produced.
```

## Current Metrics To Quote

| Metric | Value |
| --- | ---: |
| `hybrid_candidate_recall@50` | `0.13` |
| `hybrid_candidate_recall@100` | `0.18` |
| `hybrid_candidate_recall@1000` | `0.34` |
| `hybrid_ranker_hit_rate@10` | `0.10` |
| `hybrid_ranker_ndcg@10` | `0.0766` |
| Sparse candidate recall@1000 | `0.3611` |
| Cross-domain candidate recall@1000 | `0.5484` |
| Vector source recall | `0.0` |
| Positive-target candidate recall@1000 | `0.3986` |
| Positive-target sparse recall@1000 | `0.3973` |
| Positive-target cross-domain recall@1000 | `0.6081` |

Do not claim vector retrieval improved recall. It is present as a deterministic swappable hook for future embeddings.

Do not claim the positive-target proof is a final ranker promotion. It is candidate-recall-only evidence showing that objective alignment improves the retrieval bottleneck.

## Close

The honest submission position is:

- Task A is measurable rating-first generation.
- Task B is a traceable retrieval/ranking agent with source diagnostics.
- Cross-domain and sparse-user behavior are measured explicitly.
- Same-target ranker training remains the next gate.
- Nigerian context is included when the user or scenario gives evidence for it.
