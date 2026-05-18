from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.services.retrieval.item_similarity import build_item_neighbors_from_reviews  # noqa: E402
from scripts.data_utils import read_jsonl, write_json  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description="Build local retrieval artifacts.")
    parser.add_argument("--train", default="data/processed/train.jsonl")
    parser.add_argument("--output-dir", default="data/processed")
    parser.add_argument("--top-k", type=int, default=20)
    args = parser.parse_args()

    train = read_jsonl(Path(args.train))
    item_neighbors = build_item_neighbors_from_reviews(train, top_k=args.top_k)
    write_json(
        Path(args.output_dir) / "item_neighbors.json",
        {
            "type": "positive_item_cooccurrence",
            "top_k": args.top_k,
            "items": item_neighbors,
        },
    )


if __name__ == "__main__":
    main()
