#!/usr/bin/env bash
set -euo pipefail

PYTHON_BIN="${PYTHON:-python3}"
PROCESSED_DIR="${PROCESSED_DIR:-data/processed}"
MAX_EXAMPLES="${MAX_EXAMPLES:-0}"
GENERATION_MAX_EXAMPLES="${GENERATION_MAX_EXAMPLES:-25}"
EVIDENCE_MAX_EXAMPLES="${EVIDENCE_MAX_EXAMPLES:-25}"
LLM_PROVIDER="${LLM_PROVIDER:-mock}"
export LLM_PROVIDER

mkdir -p "$PROCESSED_DIR" runs/eval

echo "Building sample splits in $PROCESSED_DIR"
"$PYTHON_BIN" scripts/build_splits.py \
  --reviews data/sample/reviews.jsonl \
  --items data/sample/items.jsonl \
  --output-dir "$PROCESSED_DIR"

echo "Building retrieval and registry artifacts"
"$PYTHON_BIN" scripts/build_retrieval_index.py \
  --train "$PROCESSED_DIR/train.jsonl" \
  --items "$PROCESSED_DIR/items.jsonl" \
  --output-dir "$PROCESSED_DIR"
"$PYTHON_BIN" scripts/build_evidence_graph.py \
  --train "$PROCESSED_DIR/train.jsonl" \
  --items "$PROCESSED_DIR/items.jsonl" \
  --output "$PROCESSED_DIR/evidence_graph_retrieval.json"
"$PYTHON_BIN" scripts/build_model_registry.py \
  --output "$PROCESSED_DIR/model_registry.json"

echo "Running Task A rating eval"
"$PYTHON_BIN" eval/eval_task_a.py \
  --processed-dir "$PROCESSED_DIR" \
  --max-examples "$MAX_EXAMPLES"

echo "Running Task B recommendation eval"
"$PYTHON_BIN" eval/eval_task_b.py \
  --processed-dir "$PROCESSED_DIR" \
  --build-collaborative \
  --max-examples "$MAX_EXAMPLES"

echo "Running evidence intelligence eval"
"$PYTHON_BIN" eval/eval_evidence_intelligence.py \
  --processed-dir "$PROCESSED_DIR" \
  --max-examples "$EVIDENCE_MAX_EXAMPLES"

echo "Running Task A generation eval with LLM_PROVIDER=$LLM_PROVIDER"
"$PYTHON_BIN" eval/eval_task_a_generation.py \
  --processed-dir "$PROCESSED_DIR" \
  --max-examples "$GENERATION_MAX_EXAMPLES"

echo "All eval reports written under runs/eval"
