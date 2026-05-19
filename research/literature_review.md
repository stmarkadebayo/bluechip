# Literature Review and Technical Direction

This document captures the research and industry patterns that should shape the Bluechip hackathon system.

The goal is not to cite papers for decoration. The goal is to choose an implementation strategy that is defensible, measurable, scalable, and aligned with the challenge brief.

## Executive Takeaways

1. Build one shared user-intelligence engine, not two unrelated demos.
2. Use a multi-stage recommender architecture: candidate generation, filtering, ranking, optional reranking, explanation.
3. Use the LLM late in the pipeline for reasoning, review generation, explanations, and validation. Do not ask it to rank an entire catalog.
4. Treat Task A as an explainable recommendation / personalized review generation problem: rating prediction and text generation must be jointly consistent.
5. Treat Task B as a hybrid recommender problem: behavior signals, item metadata, text embeddings, context, and popularity all matter.
6. Make baselines strong. Recommender literature has a reproducibility problem; tuned baselines often beat flashy models.
7. Precompute profiles, embeddings, summaries, and indexes offline. Keep request-time serving cheap.
8. Evaluate with held-out interactions, not just demos.

## Foundational Papers

| Area | Source | Core Idea | What We Should Use |
| --- | --- | --- | --- |
| Collaborative filtering | [BPR: Bayesian Personalized Ranking from Implicit Feedback](https://arxiv.org/abs/1205.2618) | Optimize pairwise ranking from implicit signals. | Use BPR/implicit-feedback baselines for Task B. |
| Neural collaborative filtering | [Neural Collaborative Filtering](https://arxiv.org/abs/1708.05031) | Replace inner product user-item interaction with learned neural interaction. | Cite as a stronger learned ranker path, but do not start here. |
| Large-scale hybrid ranking | [Wide & Deep Learning for Recommender Systems](https://arxiv.org/abs/1606.07792) | Combine memorization and generalization for production ranking. | Use this as justification for hybrid feature scoring. |
| Feature interaction ranking | [DeepFM](https://arxiv.org/abs/1703.04247) | Learn low- and high-order feature interactions without heavy manual feature engineering. | Future learned ranker direction. |
| Production deep recsys | [DLRM](https://arxiv.org/abs/1906.00091) | Embeddings plus dense feature interactions for personalization. | Cite in scalability section; not needed for MVP. |
| Sequential recommendation | [SASRec](https://arxiv.org/abs/1808.09781) | Use self-attention over recent user actions. | Use sequence-aware features: recent positives should weigh more than old history. |
| Sequential masked modeling | [BERT4Rec](https://arxiv.org/abs/1904.06690) | Bidirectional transformer-style sequence modeling for next-item prediction. | Stretch goal for sequence modeling, not MVP. |
| Graph recommendation | [LightGCN](https://arxiv.org/abs/2002.02126) | Simplifies graph convolution for collaborative filtering. | Useful if we model users/items as a graph later. |
| Reproducibility warning | [Are We Really Making Much Progress?](https://arxiv.org/abs/1907.06902) | Many neural recsys papers are hard to reproduce and weak against baselines. | We need honest baselines and ablations in the paper. |

## Review Generation and Explainable Recommendation

Task A maps directly to this literature. The right framing is not "make an LLM write a review." The right framing is "predict a rating and generate a personalized textual explanation/review consistent with the user and item."

| Source | Core Idea | Implementation Implication |
| --- | --- | --- |
| [NRT: Neural Rating Regression with Abstractive Tips Generation](https://arxiv.org/abs/1708.00154) | Jointly predict rating and generate short text reflecting user experience. | Keep rating prediction and review generation coupled. |
| [PETER: Personalized Transformer for Explainable Recommendation](https://arxiv.org/abs/2105.11601) | Personalized generation can use user and item identities plus text generation objectives. | Our prompt context should include user profile, item profile, examples, and rating. |
| [PEPLER: Personalized Prompt Learning for Explainable Recommendation](https://arxiv.org/abs/2202.07371) | Prompt learning can fuse user/item identifiers with pretrained language models. | If we add trainable components, use soft prompts/adapters before full fine-tuning. |
| [MAPLE: Multi-Aspect Prompt Learning](https://arxiv.org/abs/2408.09865) | Multi-aspect prompts improve review generation coherence and factual relevance. | Represent item aspects like price, service, ambience, durability, quality, etc. |

Design decision:

```text
Task A = rating predictor + grounded review generator + consistency critic
```

The generator should not choose the rating. The rating model should predict first, then generation should explain that rating in the user's likely voice.

## LLM Recommender and Agent Papers

LLM work is useful, but the scalable pattern is to augment conventional recommenders rather than replacing them.

| Source | Core Idea | What We Should Use |
| --- | --- | --- |
| [RecMind: LLM Powered Agent for Recommendation](https://arxiv.org/abs/2308.14296) | Recommendation agents can use tools, reasoning, and external knowledge. | Present our runtime as profiler, retriever, scorer, generator, critic tools. |
| [RecAgent](https://huggingface.co/papers/2306.02552) | LLM agents can simulate user behavior for recommender research. | Supports the user-behavior simulation story for Task A. |
| [LLMRec: Graph Augmentation for Recommendation](https://arxiv.org/abs/2311.00423) | LLMs can improve graph features/data rather than serve every request online. | Use LLMs offline for profile/aspect enrichment when possible. |
| [Large Language Models meet Collaborative Filtering](https://arxiv.org/abs/2404.11343) | LLMs can work with collaborative filtering efficiently. | Keep collaborative filtering and LLM reasoning separate. |
| [Large Language Models for Generative Recommendation: Survey](https://huggingface.co/papers/2309.01157) | Surveys LLM roles in generative recommendation. | Cite for taxonomy and tradeoffs. |
| [LLM4Rec Survey](https://www.mdpi.com/1999-5903/17/6/252) | Broad survey of LLM integration in recommenders. | Use for related work, but prioritize primary papers. |

Design decision:

```text
LLM roles:
  - infer readable user profile from evidence
  - extract item aspects from text/metadata
  - explain top ranked candidates
  - generate review text conditioned on predicted rating
  - critique hallucination and rating-review mismatch

Non-LLM roles:
  - catalog candidate generation
  - ranking thousands of candidates
  - metric evaluation
  - deterministic fallbacks
```

## Industry Systems

The industry pattern is consistent: large systems use multiple stages and precompute expensive work.

| Company / Source | Pattern | What We Should Copy |
| --- | --- | --- |
| [YouTube DNN recommendations](https://research.google.com/pubs/pub45530.html?authuser=1) | Two-stage system: candidate generation followed by ranking. | Build candidate generation and ranking as separate services. |
| [Pinterest Pixie](https://medium.com/pinterest-engineering/introducing-pixie-an-advanced-graph-based-recommendation-system-e7b4229b664b) and [paper](https://arxiv.org/abs/1711.07601) | Real-time graph retrieval over massive item graph. | Add graph/item-neighbor retrieval as a candidate generator. |
| [Airbnb embedding-based retrieval](https://airbnb.tech/ai-ml/embedding-based-retrieval-for-airbnb-search/) | Precompute item embeddings offline; only compute query/user side online. | Build embeddings offline and cache indexes. |
| [Netflix recommendation overview](https://help.netflix.com/en/node/100639) | Combines user interactions, similar users, and item metadata. | Use hybrid signals rather than a single technique. |
| [Netflix foundation model for personalization](https://netflixtechblog.com/foundation-model-for-personalized-recommendation-1a0bd8e02d39) | Unified personalization representation across many use cases. | Our shared UserProfile is the local version of this idea. |
| [Amazon recommendation history](https://www.amazon.science/the-history-of-amazons-recommendation-algorithm) | Item-level similarity and evaluation tied to future behavior. | Implement item-to-item and held-out future interaction eval. |
| [Spotify generalized user representations](https://research.atspotify.com/2025/9/generalized-user-representations-for-large-scale-recommendations) | Stable user embeddings reused across retrieval, ranking, and generation. | Create user profile embeddings and behavioral profile fields. |
| [Uber two-tower embeddings](https://www.uber.com/blog/innovative-recommendation-applications-using-two-tower-embeddings/) | Two-tower retrieval for matching entities at scale. | Future learned candidate generator. |

Design decision:

```text
Runtime Task B flow:
  1. retrieve 200-500 candidates cheaply
  2. blend collaborative, lexical, vector, category, and popularity sources
  3. filter hard mismatches and previously seen items
  4. rank with hybrid score
  5. evaluate candidate recall before tuning final rank quality
  6. optionally LLM-rerank/explain top 10 only
```

## Useful Open-Source Repositories

| Repo | Why It Matters | How We Should Use It |
| --- | --- | --- |
| [RecBole](https://github.com/RUCAIBox/RecBole) | Broad benchmark framework with many recommender algorithms and evaluation protocols. | Use for baseline reference or optional benchmark runner. |
| [Microsoft Recommenders](https://github.com/recommenders-team/recommenders) | Practical examples and best practices for recommendation systems. | Borrow evaluation structure and baseline patterns. |
| [NVIDIA Merlin](https://github.com/NVIDIA-Merlin/Merlin) | End-to-end GPU-accelerated recommender stack. | Cite as production-scale analog; too heavy for MVP. |
| [implicit](https://github.com/benfred/implicit) | Fast ALS/BPR/nearest-neighbor models for implicit feedback. | Strong local baseline candidate for Task B. |
| [Cornac](https://github.com/PreferredAI/cornac) | Multimodal recommender framework with rating and ranking metrics. | Good if we want text-aware baselines and standard metrics. |
| [PETER](https://github.com/lileipisces/PETER) | Reference implementation for personalized transformer explanation generation. | Study data format and eval choices for Task A. |
| [PEPLER](https://github.com/lileipisces/PEPLER) | Personalized prompt learning reference. | Stretch direction for trainable generation. |
| [LightFM](https://github.com/lyst/lightfm) | Hybrid matrix factorization with user/item features. | Good baseline for cold-start if install works. |

Recommendation:

- Use our own simple pipeline for the submitted app.
- Use `implicit` or `Cornac` for a serious baseline if dependency/time allows.
- Use RecBole as a benchmark reference, not as the main app dependency.

## Dataset Direction

Hackathon brief allows Yelp, Amazon Reviews, and Goodreads.

| Dataset | Strength | Risk |
| --- | --- | --- |
| [Amazon Reviews 2023](https://amazon-reviews-2023.github.io/) | Public, recent, rich reviews, item metadata, categories, timestamps, standard splits. | Some categories are large; choose one manageable category. |
| [Yelp Open Dataset](https://www.yelp.com/dataset) | Great fit for restaurants, service, ambience, local context. | Download/license flow can be slower; data is geographically non-Nigerian. |
| [Goodreads datasets](https://mengtingwan.github.io/data/goodreads.html) | Rich taste modeling for books and cross-domain demos. | Older data; access links may need care. |
| [McAuley Lab datasets](https://cseweb.ucsd.edu/~jmcauley/datasets.html) | Central source for Amazon/Goodreads-style recommendation datasets. | Need select subset to keep reproducible. |

Recommended primary dataset:

```text
Amazon Reviews 2023 - All_Beauty or Digital_Music subset
```

Rationale:

- direct user reviews and ratings
- item metadata
- manageable compared with full Amazon categories
- supports Task A and Task B from the same normalized schema
- easy to explain in the paper

Yelp can be a secondary demo if we manually download it in time, especially for restaurant-style Nigerian-context UI examples.

## Target Architecture After Research

```text
Offline:
  raw reviews
    -> normalize users/items/reviews
    -> timestamp split
    -> user profile features
    -> item aspect profiles
    -> text embeddings
    -> item-item neighbors
    -> baseline model artifacts
    -> eval fixtures

Online:
  request
    -> profile lookup or cold-start profile builder
    -> candidate generation
    -> ranking / rating prediction
    -> LLM generation or explanation
    -> validation critic
    -> structured response
```

## Task A Implementation Plan

1. Use held-out last review per user as the test target.
2. Build profile from earlier user reviews only.
3. Predict rating with baselines:
   - global mean
   - user mean
   - item mean
   - user plus item bias
   - hybrid profile scorer
4. Generate review only after rating prediction.
5. Condition generation on:
   - predicted rating
   - user preference profile
   - user voice examples
   - item metadata/aspects
   - retrieved similar user reviews
6. Validate:
   - rating mentioned
   - sentiment matches rating
   - item facts are grounded
   - no unsupported claims

Metrics:

- RMSE / MAE for rating
- ROUGE-L / BERTScore if feasible for text
- sentiment-rating consistency
- human examples for behavioral fidelity

## Task B Implementation Plan

1. Use held-out future item as positive.
2. Generate candidates using multiple sources:
   - popular by category
   - item-item similarity from user positives
   - text embedding similarity
   - profile/context keyword match
3. Rank candidates with explicit hybrid features:
   - preference match
   - context match
   - item quality
   - novelty
   - confidence/evidence quantity
4. Explain only top results.

Metrics:

- HitRate@10
- NDCG@10
- Recall@K
- cold-start subset performance
- ablations by candidate generator and ranker feature set

## What To Build Next

Highest-leverage next engineering tasks:

1. Add dataset ingestion for one Amazon Reviews 2023 category.
2. Add timestamp split and normalized local artifacts.
3. Add baseline rating and ranking metrics.
4. Add BM25 or embedding retrieval.
5. Add a model-provider interface for LLM generation.
6. Add eval report output under `runs/eval/`.
7. Update the solution paper with real results and related work.

Do not build a larger UI before this. A beautiful demo without real evaluation will be easy to dismiss.

## Paper Positioning

Suggested claim:

> We build a scalable user-intelligence engine that turns review history into a reusable behavioral profile. The same profile supports review simulation through rating-conditioned generation and personalized recommendation through multi-stage retrieval, hybrid ranking, and grounded explanation.

This claim is strong because it is:

- aligned with the brief
- supported by recommender-system literature
- compatible with industry architecture
- feasible within the hackathon deadline
- measurable through standard offline metrics
