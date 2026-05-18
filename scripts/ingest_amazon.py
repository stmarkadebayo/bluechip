from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.data_utils import stable_review_id, stream_jsonl, write_json, write_jsonl


def main() -> None:
    parser = argparse.ArgumentParser(description="Normalize Amazon Reviews 2023 category files.")
    parser.add_argument("--reviews", required=True, help="Path to review JSONL or JSONL.gz file.")
    parser.add_argument("--metadata", help="Optional path to metadata JSONL or JSONL.gz file.")
    parser.add_argument("--category", default="amazon", help="Fallback category name.")
    parser.add_argument("--output-dir", default="data/processed", help="Output artifact directory.")
    parser.add_argument("--limit", type=int, default=0, help="Optional max reviews for local experiments.")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    reviews = []
    reviews_by_item: dict[str, list[dict]] = {}
    item_ids = set()

    for index, row in enumerate(stream_jsonl(Path(args.reviews))):
        if args.limit and index >= args.limit:
            break
        user_id = str(row.get("user_id") or row.get("reviewerID") or "")
        item_id = str(row.get("parent_asin") or row.get("asin") or row.get("item_id") or "")
        if not user_id or not item_id:
            continue
        timestamp = int(row.get("timestamp") or row.get("unixReviewTime") or 0)
        category = str(row.get("category") or args.category)
        item_name = str(row.get("title") or row.get("item_name") or item_id)
        review_text = row.get("text") or row.get("reviewText") or row.get("review") or ""
        rating = float(row.get("rating") or row.get("overall") or 0)
        if not review_text or rating < 1 or rating > 5:
            continue
        normalized = {
            "review_id": stable_review_id(user_id, item_id, timestamp, index),
            "user_id": user_id,
            "item_id": item_id,
            "item_name": item_name,
            "rating": rating,
            "review": review_text,
            "category": category,
            "timestamp": timestamp,
        }
        reviews.append(normalized)
        reviews_by_item.setdefault(item_id, []).append(normalized)
        item_ids.add(item_id)

    metadata = _load_metadata(Path(args.metadata), item_ids, args.category) if args.metadata else {}
    items = []
    for item_id in sorted(item_ids):
        item = metadata.get(item_id, {})
        item_reviews = reviews_by_item[item_id]
        item_name = item.get("name") or item_reviews[0]["item_name"]
        category = item.get("category") or item_reviews[0]["category"]
        for review in item_reviews:
            review["item_name"] = item_name
            review["category"] = category
        avg = sum(review["rating"] for review in item_reviews) / len(item_reviews)
        items.append(
            {
                "item_id": item_id,
                "name": item_name,
                "category": category,
                "metadata": {
                    **item.get("metadata", {}),
                    "rating_number": len(item_reviews),
                },
                "summary": item.get("summary") or _summarize_reviews(item_reviews),
                "average_rating": round(avg, 3),
            }
        )

    write_jsonl(output_dir / "reviews.jsonl", reviews)
    write_jsonl(output_dir / "items.jsonl", items)
    write_json(
        output_dir / "dataset_stats.json",
        {
            "reviews": len(reviews),
            "items": len(items),
            "users": len({review["user_id"] for review in reviews}),
            "category": args.category,
        },
    )


def _load_metadata(path: Path, item_ids: set[str], fallback_category: str) -> dict[str, dict]:
    metadata = {}
    for row in stream_jsonl(path):
        item_id = str(row.get("parent_asin") or row.get("asin") or row.get("item_id") or "")
        if not item_id or item_id not in item_ids:
            continue
        categories = row.get("categories") or row.get("category") or []
        category = categories[-1] if isinstance(categories, list) and categories else fallback_category
        description = row.get("description") or row.get("features") or ""
        if isinstance(description, list):
            description = " ".join(str(value) for value in description[:3])
        details = row.get("details") if isinstance(row.get("details"), dict) else {}
        metadata[item_id] = {
            "name": row.get("title") or item_id,
            "category": category,
            "summary": str(description)[:500],
            "metadata": {
                "price": row.get("price"),
                "rating_number": row.get("rating_number"),
                **details,
            },
        }
    return metadata


def _summarize_reviews(reviews: list[dict]) -> str:
    snippets = [review["review"].replace("\n", " ")[:120] for review in reviews[:3]]
    return " ".join(snippets)[:360]


if __name__ == "__main__":
    main()
