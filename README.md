# 🌟 Bluechip-Marvid: Evidence-First User Intelligence Agent

[![FastAPI](https://img.shields.io/badge/FastAPI-009688?style=for-the-badge&logo=fastapi&logoColor=white)](https://fastapi.tiangolo.com)
[![Python](https://img.shields.io/badge/Python_3.11-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://python.org)
[![Docker](https://img.shields.io/badge/Docker-2496ED?style=for-the-badge&logo=docker&logoColor=white)](https://docker.com)
[![Nigerian Context](https://img.shields.io/badge/Context-Nigerian_Grounding-green?style=for-the-badge&logo=accenture&logoColor=white)]()

> A production-grade, highly-deterministic AI agent architecture built for the **DSN x BCT LLM Agent Challenge**. Bluechip-Marvid tackles ratings/review simulation (Task A) and high-recall personalized recommendations (Task B) using a robust, evidence-first framework.

🔗 **Live Console Demo:** [bluechip-marvid.onrender.com](https://bluechip-marvid.onrender.com)

---

## 🚀 Why Bluechip-Marvid?

Most agentic solutions query LLMs blindly, leading to **hallucinations, rating-text mismatch, high latency, and complete failures when APIs crash**. 

Bluechip-Marvid is designed with an **"Evidence-First, Deterministic-Fallback"** philosophy:
1. **Zero-Key Fallback:** If LLM keys or internet connections are missing, the system automatically uses robust local statistical rating engines and semantic caches, registering a $1.26$ RMSE offline!
2. **Decoupled retrieval & ranking:** Mirrors production-grade search engines (like Netflix/YouTube), using database graphs for high-recall candidates and linear features for ranking, reserving LLMs *only* for reasoning explanations.
3. **Traceability:** Every decision is recorded step-by-step in an inspectable trace timeline.

---

## 🛠️ The Architecture

```
                  ┌───────────────────────────────┐
                  │   Raw History & Active Intent │
                  └───────────────┬───────────────┘
                                  ▼
                  ┌───────────────────────────────┐
                  │    Nigerian Context Engine    │  ◄── Injects regional markers
                  └───────────────┬───────────────┘
                                  ▼
         ┌────────────────────────┴────────────────────────┐
         ▼                                                 ▼
┌─────────────────┐                               ┌─────────────────┐
│     TASK A      │                               │     TASK B      │
│ User Modeling   │                               │ Recommendation  │
└────────┬────────┘                               └────────┬────────┘
         ▼                                                 ▼
┌─────────────────┐                               ┌─────────────────┐
│ Aspect Profiles │                               │ Multi-Head Pool │
└────────┬────────┘                               └────────┬────────┘
         ▼                                                 ▼
┌─────────────────┐                               ┌─────────────────┐
│ Heuristic Rating│                               │ 16-Aspect Ranker│
└────────┬────────┘                               └────────┬────────┘
         ▼                                                 ▼
┌─────────────────┐                               ┌─────────────────┐
│ Critic Filter   │ (Consistency Gate)            │ Explainer & LLM │
└─────────────────┘                               └─────────────────┘
```

### 🟩 Task A: Review & Rating Simulation
- **Aspect Profiles:** Ingests user transaction histories and categorizes affinity metrics.
- **Star Predictor:** Computes predicted star ratings ($[1-5]$) with a bounded offline RMSE.
- **Language Critic:** Runs a zero-LLM validation gate ensuring the generated text directly aligns with the predicted star rating.

### 🟨 Task B: Personalized Recommender
- **Multi-Head Candidate Retrieval:** Runs co-visitation, CF-neighbors, FAISS vectors, and BM25 lexical heads in parallel.
- **16-Feature Hybrid Ranker:** Scores candidates based on category fit, context terms, price elasticity, and dislikes.
- **Late Explanation:** Conversational reasoner creates clear explainability traces.

---

## 🇳🇬 The Nigerian Context Engine
Built as an evidence-aware context processor rather than a simple prompt wrapper:
* **Locale groundings:** Models regional biases across 20+ cities (e.g. fast-paced Lagos Mainland commuting vs Abuja premium lifestyles).
* **Naira sensitivity:** Adjusts price friction rules according to local market norms.
* **Naija Voice Modulation:** Modulates Nigerian English and Pidgin vocabulary intensity levels based on user profile indicators.

---

## 🧠 Techniques Implemented

The runtime is not a prompt-only wrapper. It implements the practical parts of the literature review and engineering research that were safe to validate before submission:

| Technique | Implemented as | Why it matters |
| :--- | :--- | :--- |
| YouTube-style two-stage recommendation | Candidate retrieval first, hybrid ranking second | Keeps recall failures separate from ranking failures. |
| Amazon-style item-to-item retrieval | Co-visitation, item-neighbor, and implicit item-item evidence | Strong sparse-user baseline and scalable retrieval pattern. |
| Review-aware modeling | User/item aspect profiles from ratings, review text, categories, and metadata | Makes Task A and Task B personalized before generation. |
| RAG-style grounded generation | Bounded evidence passed to review/reason generators | Reduces unsupported claims and keeps LLMs late in the pipeline. |
| Wide & Deep-inspired hybrid scoring | Explicit collaborative, lexical, aspect, context, popularity, novelty, and Nigerian-context components | Combines memorized behavior with generalization for cold-start cases. |
| Graph/evidence retrieval | Evidence graph, source families, and graph-derived ranking features | Adds explainable non-neural graph signal without overclaiming a GNN. |
| FAISS/vector path | Sentence-transformer index with item-id mapping | Runtime-extensible semantic retrieval; reported conservatively as diagnostic until same-slice lift is proven. |
| Evaluation gates | RMSE, ROUGE-L, Recall@K, HitRate@10, NDCG@10, sparse/cross-domain slices, human-eval packet | Makes the repo auditable rather than demo-only. |

Related work and implementation mapping live in [`research/literature_review.md`](research/literature_review.md).

---

## 📊 Offline Performance & Benchmarks

We operate under honest, multi-sliced performance metrics tested on standard evaluation slices:

| Task / Slice | Metric | Offline Score Gate |
| :--- | :--- | :--- |
| **Task A** | Rating Heuristics RMSE | **1.2654** |
| **Task A** | Text-Rating Consistency Rate | **1.0000** |
| **Task B** | Multi-Head Candidate Recall@1000 | **0.3400** |
| **Task B** | Sparse User Recall@1000 | **0.3611** |
| **Task B** | Cross-Domain Recall@1000 | **0.5484** |
| **Task B** | Positive-Target Candidate Recall@1000 | **0.3986** |

---

## ⚡ 1-Minute Developer Quick-Start

Launch the FastAPI agent server and access the interactive console locally in seconds:

```bash
# Clone the repository
git clone https://github.com/stmarkadebayo/bluechip.git && cd bluechip

# Start the server (runs on port 8000)
docker compose up --build -d
```

No Docker? Use standard python:
```bash
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"
uvicorn app.main:app --reload
```
Now visit **`http://localhost:8000/ui/`** in your browser to experience the **Bluechip-Marvid Agent Console**.

---

## 📄 Submission Solution Papers
For full academic explanations, equations, and TikZ architecture diagrams:
- 📖 [Task A: User Modeling Solution Paper](./paper/user_model_paper.tex)
- 📖 [Task B: Recommendation Agent Solution Paper](./paper/recommendation_agent_paper.tex)

Additional judge-facing evidence:
- 📊 [Submission evaluation summary](./docs/evaluation/SUBMISSION_EVAL_SUMMARY.md)
- 🧪 [Task B fast proof run](./docs/evaluation/TASK_B_FAST_PROOF_20260524.md)
- 📈 [Implicit baseline results](./docs/evaluation/IMPLICIT_BASELINE_RESULTS.md)
- 🧭 [System architecture](./docs/architecture/SYSTEM_ARCHITECTURE.md)
- 🔎 [Literature and implementation map](./research/literature_review.md)

*Developed for the DSN x BCT LLM Agent Challenge.*
