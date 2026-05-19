from __future__ import annotations

import argparse
from pathlib import Path

from scripts.data_utils import stream_jsonl, write_json, write_jsonl


def main() -> None:
    parser = argparse.ArgumentParser(description="Combine processed category artifacts.")
    parser.add_argument("--input-root", default="data/processed/categories")
    parser.add_argument("--output-dir", default="data/processed/all_categories")
    args = parser.parse_args()

    input_root = Path(args.input_root)
    output_dir = Path(args.output_dir)
    reviews = []
    items_by_id = {}
    stats = {}

    for category_dir in sorted(path for path in input_root.iterdir() if path.is_dir()):
        reviews_path = category_dir / "reviews.jsonl"
        items_path = category_dir / "items.jsonl"
        if not reviews_path.exists() or not items_path.exists():
            continue
        category_reviews = list(stream_jsonl(reviews_path))
        reviews.extend(category_reviews)
        for item in stream_jsonl(items_path):
            items_by_id.setdefault(item["item_id"], item)
        stats[category_dir.name] = {
            "reviews": len(category_reviews),
            "items": sum(1 for _ in stream_jsonl(items_path)),
        }

    write_jsonl(output_dir / "reviews.jsonl", reviews)
    write_jsonl(output_dir / "items.jsonl", items_by_id.values())
    write_json(
        output_dir / "dataset_stats.json",
        {
            "reviews": len(reviews),
            "items": len(items_by_id),
            "users": len({review["user_id"] for review in reviews}),
            "categories": stats,
        },
    )


if __name__ == "__main__":
    main()
