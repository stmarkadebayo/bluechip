from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.data_utils import group_by, read_jsonl, write_json, write_jsonl


def main() -> None:
    parser = argparse.ArgumentParser(description="Build temporal train/test splits for both tasks.")
    parser.add_argument("--reviews", default="data/sample/reviews.jsonl")
    parser.add_argument("--items", default="data/sample/items.jsonl")
    parser.add_argument("--output-dir", default="data/processed")
    parser.add_argument("--min-history", type=int, default=1)
    parser.add_argument("--limit", type=int, default=0, help="Optional max reviews for local experiments.")
    args = parser.parse_args()

    reviews = read_jsonl(Path(args.reviews), limit=args.limit)
    items = read_jsonl(Path(args.items))
    train, test_a, test_b = build_temporal_splits(reviews, min_history=args.min_history)
    output_dir = Path(args.output_dir)
    write_jsonl(output_dir / "train.jsonl", train)
    write_jsonl(output_dir / "test_task_a.jsonl", test_a)
    write_jsonl(output_dir / "test_task_b.jsonl", test_b)
    write_jsonl(output_dir / "items.jsonl", items)
    write_json(
        output_dir / "split_stats.json",
        {
            "reviews": len(reviews),
            "train": len(train),
            "test_task_a": len(test_a),
            "test_task_b": len(test_b),
            "items": len(items),
        },
    )


def build_temporal_splits(reviews: list[dict], min_history: int = 1) -> tuple[list[dict], list[dict], list[dict]]:
    train = []
    test = []
    by_user = group_by(reviews, "user_id")
    for _, user_reviews in by_user.items():
        ordered = sorted(user_reviews, key=lambda row: (int(row.get("timestamp") or 0), row["review_id"]))
        if len(ordered) <= min_history:
            train.extend(ordered)
            continue
        train.extend(ordered[:-1])
        test.append(ordered[-1])
    return train, test, test.copy()


if __name__ == "__main__":
    main()
