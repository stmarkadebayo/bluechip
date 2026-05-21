# Bluechip PR/FAQ

## Press Release

Bluechip helps shoppers discover products that match their demonstrated preferences and understand why each recommendation fits. It turns review history, product metadata, and current shopping context into a reusable user-intelligence profile, then uses that profile to predict likely reactions, rank relevant products, and generate grounded explanations.

## Customer Problem

Online shoppers face large catalogs, noisy reviews, and unclear recommendation reasons. A recommendation is less useful when it cannot explain the evidence behind the match, misses a user's known preferences, or over-personalizes from weak signals.

## Customer Promise

Given a user's visible behavior and optional context, Bluechip will recommend or simulate reactions to products using explicit evidence: user history, item facts, aspect evidence, candidate retrieval signals, ranking components, and validation checks.

## Primary Customers

- Shoppers who want relevant products and understandable explanations.
- Product and marketplace teams that need measurable personalization quality.
- ML and platform teams that need replayable evals, promotion gates, and traceable decisions.

## Success Metrics

- Task A rating quality: RMSE on fixed temporal holdouts.
- Task A text quality: ROUGE-L or unigram overlap now, optional BERTScore when dependencies are available, plus human behavioral fidelity.
- Task B retrieval quality: candidate Recall@K before ranking.
- Evidence intelligence quality: aspect coverage, evidence candidate Recall@K, and evidence graph source mix.
- Task B ranking quality: NDCG@10 and HitRate@10 against filtered popularity and current hybrid baselines.
- Trust and operations: validation failure rate, fallback rate, latency, estimated cost, and privacy-safe explanation rate.

Current Task B submission snapshot:

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

The launch metric for the next Task B gate is same-slice NDCG@10 with HitRate@10 and candidate Recall@50/100/1000 as guardrails. Roll back or reject a ranker if NDCG@10 improves by overfitting while candidate recall, sparse recall, cross-domain recall, validation safety, or latency regresses.

## Guardrails

- Do not use an LLM to choose from a large product catalog.
- Do not expose sensitive inferred attributes in explanations.
- Do not promote model or ranker artifacts without a fixed evaluation report.
- Do not change public API contracts without a migration plan.
- Do not commit secrets, raw private data, or large generated artifacts.
- Do not claim vector retrieval as a quality win while measured vector source recall is `0.0`.
- Do not force Nigerian localization when the user, item, or context does not support it.

## Non-Goals

- Bluechip is not a general chatbot.
- Bluechip is not a prompt-only recommender.
- Bluechip does not claim production infrastructure until dashboards, rollout controls, and privacy review are implemented.
- Bluechip does not optimize polished text at the expense of retrieval, ranking, or measurement quality.

## FAQ

### Why Amazon-first?

The current repo already models reviews, products, ratings, candidate retrieval, and recommendation explanations. Amazon-style shopping personalization is the clearest production analogy and keeps the roadmap focused.

### Where do feed-ranking ideas still apply?

Feed and content-ranking concepts are useful for experimentation, engagement metrics, cold-start handling, and safety review. They are secondary patterns, not the primary product surface.

### What makes Bluechip different from a single LLM prompt?

The LLM is downstream. Bluechip builds aspect-aware profiles, retrieves candidates through lexical/collaborative/vector/evidence graph sources, ranks or predicts ratings, generates grounded text, validates outputs, and records traces.

### What is the current quality bar?

The credible bar is not a larger prompt. It is a stronger evaluation spine: Task A text metrics, Task B candidate recall/miss analysis, scored human eval packs, promotion policies, and richer runtime traces.

### How should we talk about Nigerian contextualization?

Use it when grounded in explicit context, such as Lagos, affordability, delivery reliability, local shopping constraints, or social dining context. Do not infer sensitive traits or add cultural flavor when the evidence is not present.
