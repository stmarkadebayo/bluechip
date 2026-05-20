from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.services.retrieval.item_similarity import (  # noqa: E402
    build_collaborative_retrieval_index,
    build_item_neighbors_from_reviews,
    build_review_term_retrieval_index,
)
from app.services.retrieval.evidence_graph import build_evidence_graph_index  # noqa: E402
from scripts.data_utils import read_jsonl, write_json  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description="Build local retrieval artifacts.")
    parser.add_argument("--train", default="data/processed/train.jsonl")
    parser.add_argument("--output-dir", default="data/processed")
    parser.add_argument("--items", default="")
    parser.add_argument("--top-k", type=int, default=20)
    parser.add_argument("--max-users-per-item", type=int, default=500)
    parser.add_argument("--max-positive-items-per-user", type=int, default=50)
    parser.add_argument("--review-term-max-terms-per-item", type=int, default=18)
    parser.add_argument("--review-term-max-items-per-term", type=int, default=250)
    args = parser.parse_args()

    train = read_jsonl(Path(args.train))
    items = read_jsonl(Path(args.items)) if args.items else []
    item_neighbors = build_item_neighbors_from_reviews(train, top_k=args.top_k)
    output_dir = Path(args.output_dir)
    write_json(
        output_dir / "item_neighbors.json",
        {
            "type": "positive_item_cooccurrence",
            "top_k": args.top_k,
            "items": item_neighbors,
        },
    )
    collaborative = build_collaborative_retrieval_index(
        train,
        top_k=args.top_k,
        max_positive_items_per_user=args.max_positive_items_per_user,
        max_users_per_item=args.max_users_per_item,
    )
    write_json(output_dir / "collaborative_retrieval.json", collaborative)
    review_terms = build_review_term_retrieval_index(
        train,
        items=items,
        max_terms_per_item=args.review_term_max_terms_per_item,
        max_items_per_term=args.review_term_max_items_per_term,
    )
    write_json(output_dir / "review_term_retrieval.json", review_terms)
    evidence_graph = build_evidence_graph_index(train=train, items=items, top_k=args.top_k)
    write_json(output_dir / "evidence_graph_retrieval.json", evidence_graph)


if __name__ == "__main__":
    main()
