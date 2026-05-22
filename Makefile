.PHONY: run test test-api api-smoke api-smoke-live lint data index evidence-graph registry sqlite-feature-store human-eval-csv implicit-baselines eval eval-all eval-generation eval-generation-strict eval-evidence tune-task-a train-task-a train-task-a-rmse promote-task-a docker docker-build docker-test docker-smoke

PYTHON ?= python3
PROCESSED_DIR ?= data/processed
IMAGE ?= bluechip-user-intelligence-agent
SMOKE_BASE_URL ?= http://127.0.0.1:8000
LLM_PROVIDER ?= mock

run:
	uvicorn app.main:app --reload

data:
	$(PYTHON) scripts/build_splits.py --reviews data/sample/reviews.jsonl --items data/sample/items.jsonl --output-dir $(PROCESSED_DIR)

index:
	$(PYTHON) scripts/build_retrieval_index.py --train $(PROCESSED_DIR)/train.jsonl --items $(PROCESSED_DIR)/items.jsonl --output-dir $(PROCESSED_DIR)

evidence-graph:
	$(PYTHON) scripts/build_evidence_graph.py --train $(PROCESSED_DIR)/train.jsonl --items $(PROCESSED_DIR)/items.jsonl --output $(PROCESSED_DIR)/evidence_graph_retrieval.json

registry:
	$(PYTHON) scripts/build_model_registry.py --output $(PROCESSED_DIR)/model_registry.json

sqlite-feature-store:
	$(PYTHON) scripts/build_sqlite_feature_store.py --processed-dir $(PROCESSED_DIR) --output $(PROCESSED_DIR)/feature_store.sqlite

human-eval-csv:
	$(PYTHON) eval/export_human_eval_csv.py

implicit-baselines:
	$(PYTHON) eval/eval_implicit_baselines.py --processed-dir $(PROCESSED_DIR) --output runs/eval/implicit_baselines.json

test:
	pytest

test-api:
	$(PYTHON) -m pytest tests/test_api_contracts.py

api-smoke: test-api

api-smoke-live:
	$(PYTHON) tests/api_smoke.py --base-url $(SMOKE_BASE_URL)

lint:
	ruff check .

eval:
	$(PYTHON) scripts/build_splits.py --reviews data/sample/reviews.jsonl --items data/sample/items.jsonl --output-dir $(PROCESSED_DIR)
	$(PYTHON) scripts/build_retrieval_index.py --train $(PROCESSED_DIR)/train.jsonl --items $(PROCESSED_DIR)/items.jsonl --output-dir $(PROCESSED_DIR)
	$(PYTHON) eval/eval_task_a.py
	$(PYTHON) eval/eval_task_b.py

eval-all:
	PYTHON=$(PYTHON) bash scripts/run_all_evals.sh

eval-generation:
	LLM_PROVIDER=$(LLM_PROVIDER) $(PYTHON) eval/eval_task_a_generation.py

eval-generation-strict:
	LLM_PROVIDER=$(LLM_PROVIDER) $(PYTHON) eval/eval_task_a_generation.py --strict-provider

eval-evidence:
	$(PYTHON) eval/eval_evidence_intelligence.py

tune-task-a:
	$(PYTHON) eval/tune_task_a.py

train-task-a:
	$(PYTHON) eval/train_task_a_model.py

train-task-a-rmse:
	$(PYTHON) eval/train_task_a_model.py --output-model data/processed/task_a_model_rmse.json --selection-metric rmse --ensemble

promote-task-a:
	$(PYTHON) eval/promote_task_a.py

docker:
	docker compose up --build

docker-build:
	docker build -t $(IMAGE) .

docker-test:
	docker compose build api
	docker compose run --rm api python -m pytest tests/test_api_contracts.py

docker-smoke:
	docker compose up --build -d api
	$(PYTHON) tests/api_smoke.py --base-url $(SMOKE_BASE_URL)
