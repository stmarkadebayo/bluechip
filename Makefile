.PHONY: run test lint data index eval docker

PYTHON ?= python3

run:
	uvicorn app.main:app --reload

data:
	$(PYTHON) scripts/build_splits.py --reviews data/sample/reviews.jsonl --items data/sample/items.jsonl --output-dir data/processed

index:
	$(PYTHON) scripts/build_retrieval_index.py --train data/processed/train.jsonl --output-dir data/processed

test:
	pytest

lint:
	ruff check .

eval:
	$(PYTHON) scripts/build_splits.py --reviews data/sample/reviews.jsonl --items data/sample/items.jsonl --output-dir data/processed
	$(PYTHON) scripts/build_retrieval_index.py --train data/processed/train.jsonl --output-dir data/processed
	$(PYTHON) eval/eval_task_a.py
	$(PYTHON) eval/eval_task_b.py

docker:
	docker compose up --build
