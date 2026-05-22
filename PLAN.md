# PLAN.md — Bluechip LLM Agent Hackathon Implementation Plan

> **Competition**: DSN x BCT LLM Agent Challenge  
> **Deadline**: 24 May 2026, end of day  
> **Team**: Bluechip User Intelligence Agent  
> **Status**: Scope frozen for submission; core implementation complete; FAISS item mapping fixed; bounded tests and eval smokes completed

See [docs/SUBMISSION_FREEZE.md](docs/SUBMISSION_FREEZE.md) for the frozen eight-step path to final submission.

---

## 1. Competition Summary

Two tasks, one ambition — build LLM agents that understand human behaviour:

| Task | Description | Key Metrics |
|---|---|---|
| **Task A — User Modeling** | Simulate reviews: predict star ratings and generate authentic reviews for unseen items, capturing tone, rating behaviour, and contextual nuance. | ROUGE/BERTScore (30), RMSE (15), Behavioural Fidelity (20), Paper (15), Reproducibility (10), Cross-domain (10) |
| **Task B — Recommendation** | Personalised recommendation with cold-start handling, cross-domain transfer, multi-turn scenarios. Agentic reasoning required. | NDCG@10 / Hit Rate (30), Cold-Start & Cross-Domain (25), Contextual Relevance (20), Paper (15), Reproducibility (10) |

**Critical rubric**: *"The solution paper is what we read first."* — paper quality is the primary talent signal.  
**Bonus**: Additional marks for Nigerian contextualisation — authentic Nigerian English/pidgin, cultural references, consumer behaviour patterns.

**Datasets**: Yelp, Amazon Reviews 2023, Goodreads (Amazon is fully integrated; Yelp + Goodreads partially).

## 2. Submission Freeze

Before final submission, the only new model work in scope is the fast `implicit` baseline pass over `data/processed/all_categories`: ALS, BPR, and item-item. LightGCN, SASRec, HSTU, PETER, PEPLER, NARRE, and trained Wide & Deep remain future or benchmark work unless a clean same-slice run is already complete.

The remaining final-submission path is:

1. Freeze scope.
2. Complete human eval from the CSV packs.
3. Run the optional `implicit` baseline.
4. Run final validation.
5. Finalize the 4-8 page paper.
6. Package the repo safely.
7. Demo-check API and UI.
8. Submit repo link, paper PDF, app/API instructions, architecture diagram, and eval summary.

---

## 3. Architecture Overview

```
POST /api/simulate-review          POST /api/recommend
POST /api/conversation/turn        POST /api/infer-cold-start
POST /api/transfer-cross-domain    POST /api/nigerian/context
          │                                │
          ▼                                ▼
┌──────────────────────────────────────────────────────────┐
│                   Orchestrator Layer                      │
│  ReviewSimulationAgent    RecommendationAgent             │
│  ┌─────────────────────┐ ┌─────────────────────────────┐ │
│  │ • UserSimulator     │ │ • RecommenderReasoner       │ │
│  │ • NigerianEngine    │ │ • NigerianEngine            │ │
│  │ • VoiceInjector     │ │ • ConversationManager       │ │
│  │ • Validator         │ │ • EvidenceCritic            │ │
│  └─────────────────────┘ └─────────────────────────────┘ │
├──────────────────────────────────────────────────────────┤
│                   Service Layer                           │
│  ┌──────────┐ ┌──────────┐ ┌──────────┐ ┌─────────────┐ │
│  │Profiling │ │Retrieval │ │ Ranking  │ │ Generation  │ │
│  │(user/item)│ │(11 srcs) │ │(16 feats)│ │ (LLM/tmpl)  │ │
│  └──────────┘ └──────────┘ └──────────┘ └─────────────┘ │
│  ┌──────────┐ ┌──────────┐ ┌──────────┐ ┌─────────────┐ │
│  │ Agentic  │ │ Nigerian │ │Conversat.│ │ Validation  │ │
│  │ Reasoner │ │ Context  │ │  State   │ │  (critic)   │ │
│  └──────────┘ └──────────┘ └──────────┘ └─────────────┘ │
├──────────────────────────────────────────────────────────┤
│                  Data & Infrastructure                    │
│  ┌──────────┐ ┌──────────┐ ┌──────────┐ ┌─────────────┐ │
│  │ FAISS    │ │ Feature  │ │ Model    │ │ Trace Store │ │
│  │ 188K vec │ │ Store    │ │ Registry │ │ (JSONL)     │ │
│  └──────────┘ └──────────┘ └──────────┘ └─────────────┘ │
└──────────────────────────────────────────────────────────┘
```

### Technology Stack

| Layer | Technology |
|---|---|
| API Framework | FastAPI + Uvicorn (Python 3.12) |
| Data Validation | Pydantic v2 |
| LLM Providers | OpenRouter (DeepSeek V4 Flash), DeepSeek Direct, OpenAI (stdlib HTTP, no SDKs) |
| Neural Embeddings | sentence-transformers (all-MiniLM-L6-v2, 384-dim) |
| Vector Search | FAISS (IndexFlatIP, 188K vectors) |
| Containerisation | Docker + docker-compose |
| CI | GitHub Actions |
| Testing | pytest (47 tests) |
| Linting | Ruff (0 errors) |
| Evaluation | Custom metrics: MAE, RMSE, HitRate@K, NDCG@K, Recall@K, ROUGE-L |

### API Endpoints (20)

| Method | Endpoint | Purpose |
|---|---|---|
| `GET` | `/api/health` | Health check |
| `POST` | `/api/profile-user` | Build user profile from persona + history |
| `POST` | `/api/simulate-review` | **Task A**: Predict rating + generate review |
| `POST` | `/api/recommend` | **Task B**: Rank + explain recommendations |
| `POST` | `/api/conversation/turn` | Multi-turn recommendation with state |
| `POST` | `/api/conversation/{id}/feedback` | Accept/reject recommendation feedback |
| `GET` | `/api/conversation/{id}` | Get conversation state |
| `GET` | `/api/conversations` | List all conversations |
| `POST` | `/api/infer-cold-start` | LLM cold-start preference inference |
| `POST` | `/api/transfer-cross-domain` | Cross-domain preference transfer |
| `POST` | `/api/nigerian/context` | Analyse Nigerian cultural context |
| `GET` | `/api/metrics` | Runtime observability metrics |
| `GET` | `/api/traces` | Recent agent trace records |
| `GET` | `/api/runtime/registry` | Model registry payload |
| `GET` | `/api/runtime/feature-store` | Feature store summary |
| `GET` | `/` | Redirect to `/ui/` |
| `GET` | `/docs` | OpenAPI docs |

---

## 4. What Has Been Built

### 3.1 Neural Embeddings + FAISS Vector Store

**Files**:
- `app/services/retrieval/neural_embeddings.py` (109 lines) — New
- `app/services/retrieval/vector_store.py` (227 lines, +155) — Modified
- `app/services/retrieval/embeddings.py` (75 lines, +17) — Modified
- `scripts/build_neural_index.py` (70 lines) — New

**What it does**:
- Replaces deterministic SHA-256 hash embeddings (which gave 0.0 vector recall) with `all-MiniLM-L6-v2` sentence-transformers (384-dim)
- Lazy-loads the model once (thread-safe singleton), the first encode call takes ~12s, subsequent calls <1ms for cached
- LRU cache holds up to 50K encoded texts to avoid re-encoding
- Graceful fallback: if `sentence-transformers` or `faiss` not available, falls back to the old hashing embeddings
- `FAISSVectorStore` uses `IndexFlatIP` (inner product = cosine similarity on L2-normalised vectors)
- `create_retriever(items, method="neural")` factory picks FAISS or LocalVectorRetriever
- Pre-built index: **188,236 vectors × 384 dimensions** (276MB on disk)

**Bug fixed**: `FAISSVectorStore.deserialize()` now loads a companion `neural_index_ids.json` file, maps FAISS integer positions back to item IDs, and can bind a cached index to the request's candidate items. Existing FAISS files can get the companion ID file with `scripts/build_neural_index.py --ids-only`.

### 3.2 LLM Agentic Reasoning Core

**Files**: `app/services/agentic/` (4 files, 1,222 lines)

| File | Lines | Key Class | Purpose |
|---|---|---|---|
| `reasoner.py` | 282 | `LLMReasoner` | Singleton reasoning with `reason()`, `reason_structured()`, `build_user_mental_model()`, `extract_behavioral_dimensions()` |
| `user_simulator.py` | 359 | `UserSimulator` | LLM-as-user-model: `simulate_review_decision()`, `generate_authentic_review()`, `simulate_rating()` |
| `recommender_agent.py` | 580 | `RecommenderReasoner` | LLM re-ranking: `reason_about_preferences()`, `rerank_candidates()`, `handle_cold_start()`, `handle_cross_domain()` |
| `__init__.py` | 1 | — | Package marker |

**Key design decisions**:
- All LLM methods have deterministic fallbacks — system works without API keys
- JSON extraction tolerates markdown fences, malformed responses
- Cold-start: LLM infers preferences from persona text alone
- Cross-domain: LLM maps preferences across categories (e.g., beauty → music)
- Re-ranking: LLM receives top candidates with signals and reasons about re-ordering

### 3.3 Nigerian Context Engine

**Files**: `app/services/nigerian/` (4 files, 1,236 lines)

| File | Lines | Key Content |
|---|---|---|
| `context.py` | 483 | `NigerianContextEngine`: 20 Nigerian cities with regional vibes, 6 authentic consumer behavioural patterns (price sensitivity, social proof, brand consciousness, quality-for-money, haggling culture, delivery awareness), marker detection, relevance scoring |
| `pidgin.py` | 353 | `NigerianVoiceInjector`: Pidgin/Naija expressions by sentiment, `nigerianize_review()` with configurable intensity, culturally-appropriate greetings, price sensitivity phrases |
| `personas.py` | 399 | 10 authentic Nigerian personas: Lagos professional, Aba entrepreneur, Kano trader, Ibadan student, Enugu educator, PH oil worker, Abuja civil servant, Calabar tourism, Jos creative, Onitsha market vendor |
| `__init__.py` | 1 | — |

**Nigerian-specific features**:
- Okrika (thrift/secondhand) shopping patterns
- Naira economy and exchange-rate awareness
- Quality scepticism (original vs fake)
- Delivery/logistics concerns
- Social proof from community recommendations
- Market vs mall vs online shopping contexts
- Regional food/cultural references (jollof, suya, amala, etc.)

### 3.4 Multi-Turn Conversation State

**Files**: `app/services/conversation/` (2 files, 160 lines)

| File | Lines | Key Content |
|---|---|---|
| `state.py` | 157 | `ConversationManager` (singleton): in-memory session store, `ConversationState` with turn history, context refinement, accept/reject feedback tracking |
| `__init__.py` | 3 | Re-exports |

**Features**:
- Creates and manages conversation sessions with UUID
- Tracks turn history, accepted/rejected recommendations
- Refines context across turns (adds new terms, removes contradictory ones)
- Auto-evicts oldest conversations when >1000 active
- API: create, get, get_or_create, delete, list_conversations

### 3.5 Orchestrator Integration

**Files modified**:
- `app/serving/orchestrators/review_simulation.py` (191 lines, +89)
- `app/serving/orchestrators/recommendation.py` (311 lines, +99)

**Task A changes** (`ReviewSimulationAgent`):
1. Profiles user and item (existing)
2. Runs `NigerianContextEngine.inject_nigerian_context()` to detect cultural signals
3. Runs `UserSimulator.simulate_review_decision()` for LLM-reasoned review decision
4. If LLM augmented, uses LLM-reasoned rating; otherwise falls back to trained model
5. Generates review using `UserSimulator.generate_authentic_review()` with mental model + decision context
6. Applies `NigerianVoiceInjector.nigerianize_review()` if relevance score > 0.25
7. Validates rating-review consistency (existing)
8. All steps produce trace records

**Task B changes** (`RecommendationAgent`):
1. Profiles user (existing)
2. Cold-start: uses `RecommenderReasoner.handle_cold_start()` if no history
3. Scores Nigerian relevance via `NigerianContextEngine.score_nigerian_relevance()`
4. Analyses preferences via `RecommenderReasoner.reason_about_preferences()` (LLM)
5. Retrieves candidates (existing multi-source + new neural_vector source)
6. Ranks candidates (existing 16-feature hybrid)
7. Re-ranks via `RecommenderReasoner.rerank_candidates()` (LLM)
8. Generates explanations (existing LLM/template)
9. Validates via evidence critic (existing)

### 3.6 Data Pipeline

**Files**:
- `scripts/download_yelp.py` (351 lines) — New: Downloads Yelp from HuggingFace, normalises to project JSONL format
- Dataset: Amazon Reviews 2023 (5 categories, 188K items, 1.07M reviews)
- Temporal split: 978K train / 93.5K Task A holdout / 93.5K Task B holdout

### 3.7 Evaluation Framework Updates

**Files modified**:
- `eval/eval_task_b.py` (793 lines, +39): Added `--retriever neural` flag that uses FAISS, new `neural_vector` source family, per-retriever metrics
- `eval/eval_task_a_generation.py` (612 lines, +207): Added `--agentic` flag for LLM-driven review generation, `--nigerian-voice-intensity` for pidgin injection, comparison deltas (agentic vs baseline)

### 3.8 Solution Paper

**File**: `paper/solution_paper.md` (1,299 lines, +1,340/-1)

**13 sections covering**:
1. Problem Framing (abstract, challenge landscape, anti-pattern analysis)
2. Related Work & Positioning (BPR, NCF, Wide&Deep, SASRec, LightGCN, MTMH, HSTU)
3. Architecture Deep-Dive (Mermaid diagram, data pipeline, profiling, retrieval, ranking)
4. Task A: Rating-First Review Simulation (multi-head rating, plan-then-write, validation)
5. Task B: Retrieval Before Ranking (11 sources, 16 features, cold-start, cross-domain)
6. Neural Embedding Integration (migration from hashing → FAISS + MiniLM)
7. Nigerian Contextualisation (behavioural patterns, pidgin, cultural signals)
8. Agentic Workflow Design (LLM reasoning infrastructure, multi-turn, tracing)
9. Experiments & Ablation Studies (source ablation, weight sensitivity, cold/warm slices)
10. Results & Analysis (per-task tables, slice analysis, failure analysis, baselines)
11. Reproducibility (docker, eval commands, provider agnosticism, CI/CD)
12. Limitations & Future Work (honest assessment, production roadmap)
13. Rubric Alignment Table (explicit scoring criterion mapping)

### 3.9 Quality Gates

| Gate | Status |
|---|---|
| **47 pytest tests** | All passing |
| **Ruff lint** | 0 errors |
| **FastAPI compilation** | 20 routes healthy |
| **All new module imports** | Clean |
| **FAISS index built** | 188,236 vectors, 276MB |

### 3.10 Schema Updates

**File**: `app/models/schemas.py` (283 lines, +66)

**New models added**:
- `ConversationTurnRequest` / `ConversationTurnResponse`
- `ColdStartInferenceResponse`
- `CrossDomainTransferRequest` / `CrossDomainTransferResponse`
- `NigerianContextResponse`
- `ConversationSummary`

### 3.11 Dependencies Updated

**File**: `pyproject.toml` (+3 lines)

```
sentence-transformers>=2.7.0
faiss-cpu>=1.8.0
numpy>=1.24.0
```

---

## 5. What Still Needs to Be Done

### 4.1 DONE: Fix FAISS Index Item Mapping

Implemented:

1. `scripts/build_neural_index.py` writes `neural_index_ids.json` beside every FAISS index and supports `--ids-only` for existing indexes.
2. `FAISSVectorStore.deserialize_with_ids()` maps FAISS positions to `Item` objects through durable item IDs.
3. `FAISSVectorStore.bind_items()` lets serving reuse a cached FAISS index while binding it to each request's candidate payload.
4. Recommendation serving disables neural FAISS when transformer weights are unavailable offline, falling back safely to local vector retrieval instead of crashing on dimension mismatch.
5. Companion ID maps were generated for both `data/processed/neural_index.faiss` and `data/processed/all_categories/neural_index.faiss`.

### 4.2 DONE: Run Bounded Evaluation Suite

**Task B with neural retriever**:
```bash
.venv/bin/python eval/eval_task_b.py \
  --processed-dir data/processed/all_categories \
  --max-examples 500 \
  --candidate-limit 1000 \
  --retriever neural \
  --output runs/eval/neural_task_b.json \
  --miss-output runs/eval/neural_task_b_misses.json
```

**Task A generation with agentic workflow**:
```bash
.venv/bin/python eval/eval_task_a_generation.py \
  --processed-dir data/processed/all_categories \
  --max-examples 200 \
  --agentic \
  --nigerian-voice-intensity 0.35 \
  --output runs/eval/agentic_task_a.json
```

**Comparison evaluation** (legacy vs neural):
```bash
# Legacy baseline
.venv/bin/python eval/eval_task_b.py --retriever legacy --max-examples 500 --output runs/eval/legacy_baseline.json
# Neural
.venv/bin/python eval/eval_task_b.py --retriever neural --max-examples 500 --output runs/eval/neural_improved.json
# Diff the results
```

Completed validation:

- `pytest -q`: 48 passed.
- `ruff check .`: passed.
- Task B sample legacy smoke: 3 examples, HitRate@10 `1.0`.
- Task B sample neural smoke: neural FAISS active on 11 items, `neural_vector` source recall@100 `1.0`.
- Task B all-categories legacy smoke: 50 examples, candidate-limit 100, HitRate@10 `0.06`, NDCG@10 `0.0471`, candidate Recall@100 `0.12`.
- Task B all-categories neural smoke: 5 examples, loaded `188,236`-vector FAISS index with companion IDs; `neural_vector` contributed candidates but had `0/5` held-out hits.
- Task A all-categories agentic generation smoke: 25 examples, validation consistency `1.0`, rating mention `1.0`, item mention `1.0`, sentiment alignment `1.0`.

### 4.3 Test Multi-Turn Conversation API

```bash
# Start the API
.venv/bin/uvicorn app.main:app --reload &

# Create a conversation
curl -X POST http://127.0.0.1:8000/api/conversation/turn \
  -H "Content-Type: application/json" \
  -d '{"user_persona":"Lagos-based student who loves beauty products", "context":"need affordable skincare for humid weather", "candidate_items":[...], "user_message":"What moisturizer should I buy?"}'

# Send feedback
curl -X POST "http://127.0.0.1:8000/api/conversation/{id}/feedback?item_id=XXX&accepted=true"

# Send follow-up turn
curl -X POST http://127.0.0.1:8000/api/conversation/turn \
  -d '{"conversation_id":"...","user_persona":"...","candidate_items":[...], "user_message":"Something cheaper?"}'
```

### 4.4 Build & Test Docker Container

```bash
docker build -t bluechip-agent .
docker run -p 8000:8000 bluechip-agent
# Verify health
curl http://127.0.0.1:8000/api/health
# Test all endpoints via /docs
```

**Dockerfile considerations**:
- Must include `sentence-transformers` and `faiss-cpu` in the image
- The FAISS index (276MB) should be included or built on container start
- Model weights for `all-MiniLM-L6-v2` (~90MB) need to be downloaded on first run or pre-baked

### 4.5 Polish Nigerian Context Demo

Generate example reviews and recommendations with visible Nigerian voice injection:

```bash
# Test Nigerian review generation
curl -X POST http://127.0.0.1:8000/api/simulate-review \
  -H "Content-Type: application/json" \
  -d '{
    "user_persona": "Aba-based entrepreneur who shops at Onitsha market, very price-conscious",
    "locale": "nigeria",
    "user_history": [...],
    "target_item": {...}
  }'

# Test Nigerian context analysis
curl -X POST http://127.0.0.1:8000/api/nigerian/context \
  -H "Content-Type: application/json" \
  -d '{"persona": "Lagos student buying beauty products from Jumia, price-sensitive", "locale": "nigeria"}'
```

### 4.6 Update Documentation Files

- **`README.md`**: Add sections for new neural embeddings, agentic workflows, Nigerian context, and conversation API
- **`.env.example`**: Document `TASK_B_NEURAL_INDEX` environment variable for custom FAISS index path
- **`docs/`**: Update architecture diagrams to include new agentic and Nigerian layers

### 4.7 Legendary Polish (Optional, if time permits)

- Embed the FAISS index path in `docker-compose.yml` as a mounted volume
- Add `ui/index.html` examples for Nigerian personas
- Create `docs/human_eval_nigerian.md` with judge-ready examples showing Nigerian voice injection
- Write a `scripts/run_all_evals.sh` that runs the full evaluation suite with both legacy and neural
- Generate the human-eval judge packs for Task B contextual relevance

### 4.8 Final Submission Checklist

- [x] FAISS index item mapping fixed (Section 4.1)
- [x] Bounded evaluation smokes run with legacy and neural retrievers (Section 4.2)
- [x] Comparison metrics recorded; neural is runtime-validated but not claimed as a quality lift
- [x] Multi-turn conversation API covered by routes and docs; live smoke covered core endpoints
- [x] Docker config validates; Docker build blocked locally by Docker Hub network reachability
- [x] Nigerian context demo path documented and exercised through Task A agentic smoke
- [x] Solution paper final polish — metric tables and limitations updated
- [ ] Code repository: final git review, no secrets check, submit
- [ ] Submit via submission form: container app + solution paper + code repo

---

## 6. File Inventory

### New Files (13)

| File | Lines | Purpose |
|---|---|---|
| `app/services/agentic/__init__.py` | 1 | Package marker |
| `app/services/agentic/reasoner.py` | 282 | LLM reasoning core |
| `app/services/agentic/user_simulator.py` | 359 | LLM-as-user-model for Task A |
| `app/services/agentic/recommender_agent.py` | 580 | LLM re-ranking for Task B |
| `app/services/nigerian/__init__.py` | 1 | Package marker |
| `app/services/nigerian/context.py` | 483 | Nigerian context engine |
| `app/services/nigerian/pidgin.py` | 353 | Pidgin/Naija voice injection |
| `app/services/nigerian/personas.py` | 399 | 10 Nigerian personas |
| `app/services/conversation/__init__.py` | 3 | Re-exports |
| `app/services/conversation/state.py` | 157 | Multi-turn state management |
| `app/services/retrieval/neural_embeddings.py` | 109 | Sentence-transformer + FAISS |
| `scripts/build_neural_index.py` | 70 | FAISS index builder CLI |
| `scripts/download_yelp.py` | 351 | Yelp data pipeline |

### Modified Files (12)

| File | Δ Lines | Changes |
|---|---|---|
| `app/serving/orchestrators/review_simulation.py` | +89 | Integrated UserSimulator, NigerianContextEngine, VoiceInjector |
| `app/serving/orchestrators/recommendation.py` | +99 | Integrated RecommenderReasoner, NigerianContextEngine, neural FAISS |
| `app/models/schemas.py` | +66 | 6 new request/response schemas |
| `app/api/routes.py` | +159 | 12 new API endpoints |
| `app/platform/model_registry.py` | +8 | Added `task_b_neural_index` artifact |
| `app/services/retrieval/candidates.py` | +19 | Added `neural_vector` source at priority 0.90 |
| `app/services/retrieval/embeddings.py` | +17 | Added `neural_available()` and `encode()` dispatch |
| `app/services/retrieval/vector_store.py` | +155 | Added `FAISSVectorStore` class |
| `eval/eval_task_b.py` | +39 | Added `--retriever neural` flag |
| `eval/eval_task_a_generation.py` | +207 | Added `--agentic` and `--nigerian-voice-intensity` flags |
| `paper/solution_paper.md` | +1,340 | Expanded from 153 to 1,299 lines (13 sections) |
| `pyproject.toml` | +3 | Added sentence-transformers, faiss-cpu, numpy |

### Data Artifacts

| File | Size | Description |
|---|---|---|
| `data/processed/all_categories/items.jsonl` | 188,236 items | Combined Amazon corpus items |
| `data/processed/all_categories/reviews.jsonl` | 1,071,963 reviews | Combined reviews |
| `data/processed/all_categories/neural_index.faiss` | 276MB | FAISS index — 188K vectors × 384-dim |

**Total new code**: ~2,091 lines added, ~8,555 lines across all new files.

---

## 7. Scoring Rubric Alignment

| Criterion | Weight | How Addressed | Status |
|---|---|---|---|
| **Task A: Review Text Quality** | 30 | plan-then-write generation + LLM agentic authentic review + Nigerian voice injection | Implemented, pending eval |
| **Task A: Rating Accuracy (RMSE)** | 15 | Multi-head rating (calibrated_profile, trained linear model, ensemble) + LLM rating reasoning | Implemented, RMSE benchmarks in eval |
| **Task A: Behavioural Fidelity** | 20 | LLM-as-user-model (UserSimulator), behavioural dimension extraction, voice style matching, validation pipeline | Implemented, pending human eval |
| **Task B: Ranking Quality** | 30 | 11-source retrieval + 16-feature hybrid ranking + neural FAISS retrieval + LLM re-ranking | Implemented, pending neural eval |
| **Task B: Cold-Start & Cross-Domain** | 25 | LLM cold-start inference, cross-domain preference transfer, exploration budgets for sparse users, evidence graph transitions | Implemented |
| **Contextual Relevance** | 20 | Nigerian context engine, multi-turn conversation, context refinement, human-eval judge packs | Implemented |
| **Solution Paper** | 15 | 13 sections, 1,299 lines, comprehensive architecture + experiments + rubric alignment | Complete |
| **Code Reproducibility** | 10 | Docker, sample data, 0-key fallback, 47 tests, ruff clean, CI pipeline | Complete |
| **Nigerian Context (bonus)** | — | NigerianContextEngine, NigerianVoiceInjector, 10 authentic personas, pidgin injection | Complete |

---

## 8. Immediate Next Actions (Priority Order)

1. Review final diff and secrets.
2. Retry `make docker-test` when Docker Hub access is available.
3. Submit via competition form.
