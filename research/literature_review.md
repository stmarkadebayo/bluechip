# Bluechip Literature Review and Implementation Map

Search date: 2026-05-22

This review maps the DSN x BCT hackathon brief, the current Bluechip codebase, and the recommendation / LLM-agent literature. The goal is to be useful for the submission paper and engineering direction, not to cite papers decoratively.

## Executive Answer

The brief says the judges conduct the official human evaluation during judging. Our internal labels do not replace that official score, but they help us tune the submission, catch weak examples before judges see them, and report a bounded human study in the solution paper.

The brief assigns human scoring to both tasks:

- Task A: Behavioural Fidelity, 20 points.
- Task B: Contextual Relevance, 20 points.

The repo has generated human-eval packs and a scored Task B contextual summary:

- `docs/human_eval_task_a.md`
- `docs/human_eval_task_b.md`
- `docs/human_eval_task_b_contextual.md`
- `docs/evaluation/HUMAN_EVAL_TASK_B_CONTEXTUAL_RESULTS.md`

Minimum defensible path:

Task B contextual relevance is scored on 20 examples and summarized in the evaluation docs. Task A has a judge-facing review pack, but we should not overclaim a scored Task A human study unless the labels are filled.

## What We Have Actually Implemented

The current system is not a clone of one paper. It is a pragmatic hybrid recommender and review-simulation system that implements the architecture patterns behind several strong papers and industry systems.

| Area | Status | Main code | Honest claim |
| --- | --- | --- | --- |
| Multi-stage recommendation | Implemented | `app/services/retrieval/candidates.py`, `app/services/ranking/recommendation.py` | Candidate generation and ranking are separate, matching the large-scale two-stage pattern. |
| Item-to-item / co-engagement retrieval | Implemented | `app/services/retrieval/candidates.py`, retrieval artifacts | We use item-neighbor and co-engagement style retrieval. |
| Review-text user and item profiles | Implemented | `app/services/profiling/`, `app/services/intelligence/` | We use review text to build user and item signals. |
| Evidence graph retrieval | Implemented as lightweight graph features | `app/services/retrieval/evidence_graph.py`, `scripts/build_evidence_graph.py` | We use graph-like co-engagement, aspect, and sequence evidence. We do not train a neural GNN. |
| Semantic retrieval | Implemented | `app/services/retrieval/neural_embeddings.py`, `app/services/retrieval/vector_store.py` | We use sentence-transformers plus FAISS as a swappable vector retrieval path. |
| Rating-first review generation | Implemented | `app/serving/orchestrators/review_simulation.py`, `app/services/generation/` | Rating is predicted before review text is generated, keeping text aligned with the metric decision. |
| Grounded LLM generation and explanation | Implemented with fallback | `app/services/generation/`, `app/services/agentic/` | LLMs write reviews, explanations, profile enrichments, and reasoning summaries after deterministic decisions exist. |
| Bounded LLM profile enrichment | Implemented | `app/services/profiling/profile_enhancer.py` | LLM-inferred fields merge into deterministic profiles with caps, confidence, and fallback. |
| SQLite feature store | Implemented as Phase 1 | `app/platform/feature_store.py`, `scripts/build_sqlite_feature_store.py` | SQLite improves local serving and point lookups; it is not yet a 10/10 production feature store. |
| Evaluation spine | Implemented | `eval/`, `docs/evaluation/` | We measure RMSE, ROUGE-L, optional BERTScore, candidate Recall@K, HitRate@K, NDCG@K, sparse and cross-domain slices. |
| Human eval | Task B contextual scored; Task A review pack generated | `docs/human_eval_*.md`, `docs/evaluation/HUMAN_EVAL_TASK_B_CONTEXTUAL_RESULTS.md` | Report Task B human eval as bounded evidence; do not claim scored Task A behavioural labels unless added. |

## Papers And Systems We Can Honestly Say Influenced The Implementation

### Multi-stage Recommender Architecture

[Deep Neural Networks for YouTube Recommendations](https://research.google.com/pubs/pub45530.html?authuser=1) is the clearest industry precedent for splitting recommendation into candidate generation and ranking. The paper describes a two-stage system: a candidate generator first narrows the catalog, then a ranking model scores the smaller set.

Bluechip implements this pattern directly:

- retrieval heads generate candidates;
- source attribution is preserved;
- ranking happens after retrieval;
- explanations happen after ranking;
- eval separates candidate recall from top-10 ranking quality.

This is one of the strongest architectural alignments in the repo.

### Item-to-item Collaborative Filtering

[Amazon's recommendation history](https://www.amazon.science/the-history-of-amazons-recommendation-algorithm) explains why item-to-item collaborative filtering scales well: use recent user history to fetch related products, then weight candidates by relatedness.

Bluechip implements the same idea locally through:

- co-visitation retrieval;
- lexical item-neighbor retrieval;
- category-affinity popular fallback;
- held-out future interaction evaluation.

We should cite this as an architectural and retrieval-pattern influence, not as a claim that we implemented Amazon's exact production algorithm.

### BPR And Implicit-feedback Ranking

[BPR: Bayesian Personalized Ranking from Implicit Feedback](https://arxiv.org/abs/1205.2618) gives the classic pairwise ranking objective for implicit interactions.

Current status:

- Not implemented as a trained model.
- The eval frame is compatible with BPR-style ranking.
- It is the best next baseline to add through `implicit` or RecBole.

Use this phrasing in the paper:

> We evaluate with ranking metrics suitable for implicit recommendation and leave trained BPR/ALS baselines as a next reproducibility extension.

Do not claim that Bluechip currently trains BPR.

### Wide & Deep / Hybrid Memorization And Generalization

[Wide & Deep Learning for Recommender Systems](https://arxiv.org/abs/1606.07792) argues for combining memorization from sparse/crossed features with generalization from dense embeddings.

Bluechip's current scorer is not a trained Wide & Deep model, but it follows the same practical motivation:

- collaborative and co-engagement signals handle memorized behavior;
- profile, aspect, category, context, and vector signals generalize to sparse/cold-start cases;
- a single ranker combines these components.

Honest claim:

> The ranker is a transparent hybrid scorer inspired by Wide & Deep's memorization/generalization split, not a trained Wide & Deep neural model.

### Review-aware User And Item Modeling

[DeepCoNN](https://arxiv.org/abs/1701.04783) jointly models users and items from review text. It uses parallel networks over user reviews and item reviews.

Bluechip implements the non-neural equivalent:

- user profiles are built from prior reviews, ratings, categories, aspects, and terms;
- item profiles are built from metadata and review-derived evidence;
- both profiles feed rating prediction, ranking, and generation.

Honest claim:

> We implement review-aware user and item profiling in the spirit of DeepCoNN, but not its CNN architecture.

### Review Usefulness And Explanation Evidence

[NARRE](https://doi.org/10.1145/3178876.3186070) uses review-level attention to weight useful reviews for rating prediction and explanation.

Bluechip partially mirrors the motivation:

- it extracts aspect and term evidence;
- it preserves source families for recommendations;
- it validates groundedness and sensitive-inference risk;
- it exposes candidate sources in contextual human-eval packs.

Honest claim:

> We use explicit evidence selection and source attribution, not neural review-level attention.

### Personalized Explanation And Review Generation

[PETER](https://arxiv.org/abs/2105.11601) and [PEPLER](https://arxiv.org/abs/2202.07371) are the closest match for Task A and recommendation explanations because they combine personalization with natural-language generation.

Bluechip implements the serving pattern, not the trained models:

- predict or rank first;
- build a grounded plan;
- generate review/explanation text conditioned on the user, item, rating, and evidence;
- validate consistency after generation.

Honest claim:

> We implement a rating-conditioned, evidence-grounded generation pipeline influenced by explainable recommendation work; we do not train PETER or PEPLER.

### Retrieval-augmented Generation

[RAG](https://arxiv.org/abs/2005.11401) combines model generation with non-parametric retrieved memory to improve specificity and factuality.

Bluechip uses the RAG pattern throughout:

- retrieve user/item/context evidence first;
- pass only bounded evidence into the LLM;
- require deterministic fallback;
- validate grounding and consistency.

This is safe to claim as implemented at the system-pattern level.

### Agentic Reasoning

[ReAct](https://arxiv.org/abs/2210.03629) combines language-model reasoning traces with actions against external tools or environments.

Bluechip is ReAct-adjacent, not a full ReAct agent:

- the orchestrators call explicit profiling, retrieval, ranking, generation, and validation tools;
- LLM calls can produce structured reasoning summaries;
- traces capture the path taken.

But the system does not run an open-ended thought/action loop. This is intentional for reproducibility, latency, and hackathon demo reliability.

### Sequential Recommendation

[SASRec](https://huggingface.co/papers/1808.09781) models user action histories with self-attention.

Current status:

- Bluechip has sequential and recent-history features.
- Bluechip does not train a SASRec model.
- SASRec is a strong next baseline if we want a true sequence model.

The original [SASRec GitHub repo](https://github.com/kang205/SASRec) is useful for understanding data formatting, but RecBole is likely faster to integrate for our repo.

### HSTU / Generative Recommenders

[Actions Speak Louder than Words: Trillion-Parameter Sequential Transducers for Generative Recommendations](https://arxiv.org/abs/2402.17152) introduces HSTU for large-scale sequential generative recommendation.

Current status:

- We do not implement HSTU.
- We do have a target architecture slot for sequence-aware ranking.
- HSTU is overkill for the hackathon deadline and our current bottleneck.

Reason:

Task B's measured bottleneck is candidate recall. A heavy sequence model will not help if the held-out item is not in the candidate pool often enough.

Use [meta-recsys/generative-recommenders](https://github.com/facebookresearch/generative-recommenders) as a reference implementation only.

### Graph Recommendation

[LightGCN](https://arxiv.org/abs/2002.02126) simplifies graph convolution for collaborative filtering by focusing on neighborhood aggregation in the user-item graph.

Bluechip currently has graph techniques, but not a GNN:

- co-visitation/item-neighbor candidate edges;
- aspect evidence graph;
- sequential transition evidence;
- source-family diagnostics;
- graph-derived ranking features.

Would a GNN be better?

Not immediately. A LightGCN or PinSage-style model can help once the data split and candidate recall evaluation are stable, but the current system should first build on the measured retrieval/objective-alignment gains. A GNN is only worth adding if it beats same-slice Recall@50/100/1000 and does not hurt HitRate@10/NDCG@10.

## Evaluation Literature And Metrics

### Text Generation Metrics

[ROUGE](https://aclanthology.org/W04-1013/) is implemented as a dependency-free lexical proxy in `eval/eval_task_a_generation.py`.

[BERTScore](https://arxiv.org/abs/1904.09675) is supported as optional because it requires heavier dependencies and local/downloaded models. This is the right tradeoff for reproducibility: judges can run the core eval without model downloads, and a provider/model-enabled environment can run semantic scoring.

### Ranking Metrics

HitRate@K, Recall@K, and NDCG@K are the right shape for Task B because the target is a held-out future item or interaction.

Bluechip's most important evaluation design is separation of:

- candidate Recall@50/100/1000;
- ranker HitRate@10 and NDCG@10;
- sparse-user slices;
- cross-domain slices;
- source-family diagnostics.

This prevents ranker tuning from hiding retrieval failure.

### Human Evaluation

Human eval is not optional for the brief's scoring, even though the official evaluation is judge-run. Our internal pass is still worth doing because it exposes whether the submitted outputs look behaviorally faithful and contextually relevant to humans.

Recommended rubric:

Task A:

- rating fit;
- voice fit;
- groundedness;
- specificity.

Task B:

- top-10 relevance;
- context fit;
- diversity;
- explanation quality.

Suggested reporting:

```text
Task A human eval: n examples, r reviewers, mean score by dimension, mean overall.
Task B contextual eval: n examples, r reviewers, mean score by dimension, mean overall.
Disagreements were averaged; no reviewer saw model internals beyond the provided histories and outputs.
```

## Technical Blogs And System Writeups That Should Shape The Next Version

| Source | Why it matters | Bluechip implication |
| --- | --- | --- |
| [YouTube DNN recommendations](https://research.google.com/pubs/pub45530.html?authuser=1) | Candidate generation and ranking are separate models. | Keep improving recall before over-tuning ranker. |
| [Pinterest Pixie](https://medium.com/pinterest-engineering/introducing-pixie-an-advanced-graph-based-recommendation-system-e7b4229b664b) | Real-time graph candidate generation with biased random walks. | Add a small Pixie-style random-walk retrieval head over our evidence graph. |
| [Airbnb Embedding-Based Retrieval](https://airbnb.tech/ai-ml/embedding-based-retrieval-for-airbnb-search/) | ANN retrieval should be trained/evaluated as candidate narrowing, not final ranking. | Fine-tune or calibrate embeddings only if candidate recall improves on fixed slices. |
| [Amazon Science recommender history](https://www.amazon.science/the-history-of-amazons-recommendation-algorithm) | Item-to-item similarity is scalable and practical. | Strengthen item-neighbor and co-visitation artifacts before heavier neural models. |
| [Feast docs](https://docs.feast.dev/) | Feature stores need offline/online parity and low-latency serving. | Our SQLite store is Phase 1; 10/10 requires typed feature views, lineage, point-in-time training sets, and parity tests. |
| [Feast point-in-time joins](https://docs.feast.dev/v0.17-branch/getting-started/concepts/point-in-time-joins) | Prevents target leakage when building training sets. | Add timestamped feature materialization before training learned rankers. |
| [OpenTelemetry Python](https://opentelemetry.io/docs/languages/python/) | Standard tracing for services. | Keep JSONL traces for the hackathon; map them to OpenTelemetry later. |
| [PostHog](https://posthog.com/) | Product analytics, flags, experiments, session replay. | Useful after launch for product analytics, not a replacement for offline eval or human eval. |

## GitHub Implementations Worth Using

| Repo | Use it for | Recommendation |
| --- | --- | --- |
| [benfred/implicit](https://github.com/benfred/implicit) | ALS, BPR, logistic MF, item-item nearest neighbors for implicit feedback. | Highest-leverage next baseline. It is smaller than RecBole and matches our current data better. |
| [RUCAIBox/RecBole](https://github.com/RUCAIBox/RecBole) | Standardized recommender baselines including sequential and graph models. | Use for benchmark scripts, not as the serving stack. |
| [recommenders-team/recommenders](https://github.com/recommenders-team/recommenders) | Microsoft best-practice notebooks and evaluation patterns. | Borrow eval/report structure and baseline examples. |
| [meta-recsys/generative-recommenders](https://github.com/facebookresearch/generative-recommenders) | HSTU reference implementation. | Research reference only; too heavy for the hackathon app. |
| [kang205/SASRec](https://github.com/kang205/SASRec) | Original SASRec implementation and sequential data format. | Useful reference; RecBole is likely easier for a modern baseline. |
| [chenchongthu/NARRE](https://github.com/chenchongthu/NARRE) | Neural attentional rating regression with review-level explanations. | Useful reference for evidence weighting; too old to import directly without modernization. |
| [lileipisces/PETER](https://github.com/lileipisces/PETER) | Personalized transformer explanation generation. | Use to improve Task A paper framing and eval format, not as immediate dependency. |
| [lileipisces/PEPLER](https://github.com/lileipisces/PEPLER) | Personalized prompt learning for explainable recommendation. | Stretch reference for trainable personalized generation. |
| [feast-dev/feast](https://github.com/feast-dev/feast) | Feature store concepts, registry, online/offline separation. | Adopt concepts now; consider dependency only after hackathon. |
| [huggingface/sentence-transformers](https://github.com/UKPLab/sentence-transformers) | Semantic embeddings and rerankers. | Already used for embeddings; next step is domain-specific fine-tuning or cross-encoder reranking only on top candidates. |
| [NVIDIA Merlin](https://github.com/NVIDIA-Merlin/Merlin) | Production-scale GPU recommender stack. | Good architecture reference, too heavy for this repo now. |

## Frozen Next Work Before Submission

The active submission freeze is in `docs/SUBMISSION_FREEZE.md`. The final submission keeps the system evidence-first and reports only completed model work: the implicit ALS/BPR/item-item baseline, the Task B fast proof, and the fixed Task A/Task B evaluation snapshots.

LightGCN, SASRec, HSTU, PETER, PEPLER, NARRE, and trained Wide & Deep remain future work for this deadline. They are useful research references, but starting them now would add too much integration and validation risk.

## What To Add Next, In Priority Order

1. Complete human eval scoring.

This is the highest-score-per-hour work left. It directly addresses Behavioural Fidelity and Contextual Relevance.

2. Add an `implicit` baseline runner.

Train ALS/BPR/item-item baselines on the same split and report Recall@50/100/1000, HitRate@10, and NDCG@10. This gives the solution paper a strong conventional baseline.

3. Add a Pixie-style graph retrieval head.

Use the existing evidence graph and run bounded random walks or Personalized PageRank from a user's positive items/aspects. Promote only if candidate recall improves on fixed slices.

4. Add point-in-time feature materialization.

SQLite is fine for local serving, but the feature store becomes strong when it can produce timestamp-correct Task A and Task B training frames with no leakage.

5. Add a learned ranker only after recall improves.

A LightGBM/XGBoost ranker over current score components is likely more useful than a neural sequence model right now. Promote it only with same-slice HitRate@10 and NDCG@10 gains.

6. Treat LLM reranking as optional.

LLM reranking is optional because it is expensive, non-deterministic, hard to evaluate at catalog scale, and does not fix candidate recall. Keep it limited to reranking or explaining a small top candidate set, with deterministic fallback.

7. Keep the custom lightweight agent framework.

The current layers are explicit, inspectable, and testable. LangGraph/CrewAI/AutoGen would add orchestration features but also more dependency and debugging surface. Use a framework later only if we need durable graph execution, human approval nodes, long-running workflows, or distributed tool calls.

## Paper Positioning

Strong, honest claim:

> Bluechip implements an evidence-first user intelligence system. It separates candidate retrieval, ranking/rating, LLM generation, validation, and evaluation. The design is influenced by multi-stage industrial recommenders, review-aware user/item modeling, explainable recommendation, RAG, and graph retrieval, while keeping exact neural baselines such as BPR, LightGCN, SASRec, HSTU, PETER, and PEPLER as explicit future or benchmark work unless implemented and evaluated.

What not to claim:

- Do not say we trained BPR, LightGCN, SASRec, HSTU, PETER, or PEPLER.
- Do not say FAISS/neural retrieval improves quality until same-slice eval proves lift.
- Do not call the evidence graph a GNN.
- Do not say PostHog is the eval layer. It is product analytics and experimentation, not offline metric evaluation.

## Submission Checklist From This Review

- Fill `docs/human_eval_task_a.md`.
- Fill `docs/human_eval_task_b_contextual.md`.
- Add human-eval means to `paper/solution_paper.md`.
- Cite YouTube, Amazon, DeepCoNN, PETER/PEPLER, RAG, ROUGE, BERTScore, BPR, LightGCN/SASRec/HSTU as related work with exact implemented/not-implemented wording.
- Mention `docs/FEATURE_STORE_10_PLAN.md` as the feature-store upgrade plan.
- Keep claims conservative around graph and neural methods.
