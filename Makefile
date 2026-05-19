.PHONY: run test lint data index eval eval-generation tune tune-task-a train-task-a train-task-a-rmse promote-task-a train-ranker promote-ranker docker

PYTHON ?= python3
PROCESSED_DIR ?= data/processed

run:
	uvicorn app.main:app --reload

data:
	$(PYTHON) scripts/build_splits.py --reviews data/sample/reviews.jsonl --items data/sample/items.jsonl --output-dir $(PROCESSED_DIR)

index:
	$(PYTHON) scripts/build_retrieval_index.py --train $(PROCESSED_DIR)/train.jsonl --items $(PROCESSED_DIR)/items.jsonl --output-dir $(PROCESSED_DIR)

test:
	pytest

lint:
	ruff check .

eval:
	$(PYTHON) scripts/build_splits.py --reviews data/sample/reviews.jsonl --items data/sample/items.jsonl --output-dir $(PROCESSED_DIR)
	$(PYTHON) scripts/build_retrieval_index.py --train $(PROCESSED_DIR)/train.jsonl --items $(PROCESSED_DIR)/items.jsonl --output-dir $(PROCESSED_DIR)
	$(PYTHON) eval/eval_task_a.py
	$(PYTHON) eval/eval_task_b.py

eval-generation:
	$(PYTHON) eval/eval_task_a_generation.py --strict-provider

tune:
	$(PYTHON) eval/tune_ranker.py

tune-task-a:
	$(PYTHON) eval/tune_task_a.py

train-task-a:
	$(PYTHON) eval/train_task_a_model.py

train-task-a-rmse:
	$(PYTHON) eval/train_task_a_model.py --output-model data/processed/task_a_model_rmse.json --selection-metric rmse --ensemble

promote-task-a:
	$(PYTHON) eval/promote_task_a.py

train-ranker:
	$(PYTHON) eval/train_ranker.py

promote-ranker:
	$(PYTHON) eval/promote_ranker.py

docker:
	docker compose up --build
