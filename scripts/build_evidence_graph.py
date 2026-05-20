from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.services.retrieval.evidence_graph import build_evidence_graph_index  # noqa: E402
from scripts.data_utils import read_jsonl, write_json  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description="Build aspect-aware evidence graph retrieval index.")
    parser.add_argument("--train", default="data/processed/train.jsonl")
    parser.add_argument("--items", default="data/processed/items.jsonl")
    parser.add_argument("--output", default="data/processed/evidence_graph_retrieval.json")
    parser.add_argument("--top-k", type=int, default=120)
    args = parser.parse_args()

    train = read_jsonl(Path(args.train))
    items = read_jsonl(Path(args.items))
    graph = build_evidence_graph_index(train=train, items=items, top_k=args.top_k)
    write_json(Path(args.output), graph)


if __name__ == "__main__":
    main()
