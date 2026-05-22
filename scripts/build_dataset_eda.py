from __future__ import annotations

import argparse
import math
from collections import Counter, defaultdict
from pathlib import Path
from statistics import mean

from scripts.data_utils import stream_jsonl, write_json


def main() -> None:
    parser = argparse.ArgumentParser(description="Build a compact dataset EDA report.")
    parser.add_argument("--processed-dir", default="data/processed/all_categories")
    parser.add_argument("--output-json", default="docs/evaluation/dataset_eda.json")
    parser.add_argument("--output-md", default="docs/evaluation/DATASET_EDA.md")
    args = parser.parse_args()

    processed_dir = Path(args.processed_dir)
    reviews_path = processed_dir / "reviews.jsonl"
    train_path = processed_dir / "train.jsonl"
    task_a_path = processed_dir / "test_task_a.jsonl"
    task_b_path = processed_dir / "test_task_b.jsonl"
    items_path = processed_dir / "items.jsonl"

    reviews_summary = _summarize_reviews(reviews_path)
    item_summary = _summarize_items(items_path)
    split_summary = {
        "train": _count_lines(train_path),
        "test_task_a": _count_lines(task_a_path),
        "test_task_b": _count_lines(task_b_path),
    }
    task_b_summary = _summarize_task_b(task_b_path, train_path)
    sparsity = _matrix_sparsity(
        users=reviews_summary["users"],
        items=item_summary["items"],
        interactions=reviews_summary["user_item_pairs"],
    )

    payload = {
        "processed_dir": str(processed_dir),
        "reviews": reviews_summary,
        "items": item_summary,
        "splits": split_summary,
        "task_b": task_b_summary,
        "matrix": sparsity,
        "notes": [
            "Rows with missing user_id, missing item_id, empty source review text, or invalid ratings are filtered during ingestion; residual missing-like normalized text is tracked in this report.",
            "Temporal splits keep each eligible user's latest review as the Task A and Task B holdout target.",
            "Task A and Task B currently share the same latest-review holdout rows because the brief evaluates different outputs over the same behavioral prediction setup.",
        ],
    }

    output_json = Path(args.output_json)
    output_md = Path(args.output_md)
    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_md.parent.mkdir(parents=True, exist_ok=True)
    write_json(output_json, payload)
    output_md.write_text(_markdown(payload), encoding="utf-8")
    print(f"Wrote {output_json}")
    print(f"Wrote {output_md}")


def _summarize_reviews(path: Path) -> dict:
    rows = 0
    users = set()
    items = set()
    user_item_pairs = set()
    rating_counts: Counter[str] = Counter()
    category_reviews: Counter[str] = Counter()
    user_counts: Counter[str] = Counter()
    item_counts: Counter[str] = Counter()
    timestamps = []
    missing_review = 0
    missing_category = 0

    for row in stream_jsonl(path):
        rows += 1
        user_id = str(row.get("user_id") or "")
        item_id = str(row.get("item_id") or "")
        category = str(row.get("category") or "unknown")
        rating = row.get("rating")
        review = str(row.get("review") or "")
        timestamp = row.get("timestamp")

        if user_id:
            users.add(user_id)
            user_counts[user_id] += 1
        if item_id:
            items.add(item_id)
            item_counts[item_id] += 1
        if user_id and item_id:
            user_item_pairs.add((user_id, item_id))
        if rating is not None:
            rating_counts[str(int(float(rating)))] += 1
        if not review.strip():
            missing_review += 1
        if not category or category == "unknown":
            missing_category += 1
        category_reviews[category] += 1
        if timestamp:
            timestamps.append(int(timestamp))

    user_lengths = list(user_counts.values())
    item_lengths = list(item_counts.values())
    return {
        "reviews": rows,
        "users": len(users),
        "items": len(items),
        "user_item_pairs": len(user_item_pairs),
        "rating_distribution": dict(sorted(rating_counts.items())),
        "top_categories_by_reviews": category_reviews.most_common(10),
        "reviews_per_user": _distribution(user_lengths),
        "reviews_per_item": _distribution(item_lengths),
        "sparse_users_1_2_reviews": sum(1 for count in user_lengths if count <= 2),
        "sparse_users_1_2_reviews_pct": _pct(
            sum(1 for count in user_lengths if count <= 2),
            len(user_lengths),
        ),
        "missing_review_text_rows": missing_review,
        "missing_category_rows": missing_category,
        "timestamp_min": min(timestamps) if timestamps else None,
        "timestamp_max": max(timestamps) if timestamps else None,
    }


def _summarize_items(path: Path) -> dict:
    rows = 0
    category_items: Counter[str] = Counter()
    missing_name = 0
    missing_summary = 0
    rating_numbers = []
    average_ratings = []
    for row in stream_jsonl(path):
        rows += 1
        category = str(row.get("category") or "unknown")
        category_items[category] += 1
        if not str(row.get("name") or "").strip():
            missing_name += 1
        if not str(row.get("summary") or "").strip():
            missing_summary += 1
        metadata = row.get("metadata") if isinstance(row.get("metadata"), dict) else {}
        rating_number = metadata.get("rating_number")
        if isinstance(rating_number, (int, float)):
            rating_numbers.append(float(rating_number))
        average_rating = row.get("average_rating")
        if isinstance(average_rating, (int, float)):
            average_ratings.append(float(average_rating))
    return {
        "items": rows,
        "category_distribution": dict(category_items),
        "top_categories_by_items": category_items.most_common(10),
        "missing_name_rows": missing_name,
        "missing_summary_rows": missing_summary,
        "metadata_rating_number": _distribution(rating_numbers),
        "average_rating": _distribution(average_ratings),
    }


def _summarize_task_b(task_b_path: Path, train_path: Path) -> dict:
    positive_categories_by_user: dict[str, set[str]] = defaultdict(set)
    history_count_by_user: Counter[str] = Counter()
    for row in stream_jsonl(train_path):
        user_id = str(row.get("user_id") or "")
        category = str(row.get("category") or "")
        rating = float(row.get("rating") or 0)
        if not user_id:
            continue
        history_count_by_user[user_id] += 1
        if category and rating >= 4:
            positive_categories_by_user[user_id].add(category)

    rows = 0
    cross_domain = 0
    sparse = 0
    medium = 0
    warm = 0
    category_counts: Counter[str] = Counter()
    for row in stream_jsonl(task_b_path):
        rows += 1
        user_id = str(row.get("user_id") or "")
        category = str(row.get("category") or "unknown")
        category_counts[category] += 1
        history_len = history_count_by_user[user_id]
        if history_len <= 2:
            sparse += 1
        elif history_len <= 7:
            medium += 1
        else:
            warm += 1
        positive_categories = positive_categories_by_user[user_id]
        if positive_categories and category not in positive_categories:
            cross_domain += 1

    return {
        "examples": rows,
        "sparse_history_1_2": sparse,
        "sparse_history_1_2_pct": _pct(sparse, rows),
        "medium_history_3_7": medium,
        "medium_history_3_7_pct": _pct(medium, rows),
        "warm_history_8_plus": warm,
        "warm_history_8_plus_pct": _pct(warm, rows),
        "cross_domain_examples": cross_domain,
        "cross_domain_pct": _pct(cross_domain, rows),
        "top_categories": category_counts.most_common(10),
    }


def _matrix_sparsity(users: int, items: int, interactions: int) -> dict:
    possible = users * items
    density = interactions / possible if possible else 0.0
    return {
        "users": users,
        "items": items,
        "observed_user_item_pairs": interactions,
        "possible_user_item_pairs": possible,
        "density": density,
        "sparsity": 1.0 - density,
    }


def _distribution(values: list[float | int]) -> dict:
    if not values:
        return {"min": 0, "p50": 0, "mean": 0, "p90": 0, "p95": 0, "p99": 0, "max": 0}
    ordered = sorted(values)
    return {
        "min": _round(ordered[0]),
        "p50": _round(_percentile(ordered, 0.50)),
        "mean": _round(mean(ordered)),
        "p90": _round(_percentile(ordered, 0.90)),
        "p95": _round(_percentile(ordered, 0.95)),
        "p99": _round(_percentile(ordered, 0.99)),
        "max": _round(ordered[-1]),
    }


def _percentile(values: list[float | int], percentile: float) -> float:
    if not values:
        return 0.0
    index = min(len(values) - 1, max(0, math.ceil(len(values) * percentile) - 1))
    return float(values[index])


def _count_lines(path: Path) -> int:
    with path.open("r", encoding="utf-8") as handle:
        return sum(1 for _ in handle)


def _pct(part: int, whole: int) -> float:
    return round(part / whole, 4) if whole else 0.0


def _round(value: float | int) -> float:
    return round(float(value), 4)


def _markdown(payload: dict) -> str:
    reviews = payload["reviews"]
    items = payload["items"]
    splits = payload["splits"]
    task_b = payload["task_b"]
    matrix = payload["matrix"]
    lines = [
        "# Dataset EDA",
        "",
        f"Processed directory: `{payload['processed_dir']}`",
        "",
        "## Corpus Summary",
        "",
        "| Metric | Value |",
        "| --- | ---: |",
        f"| Reviews | {reviews['reviews']:,} |",
        f"| Users | {reviews['users']:,} |",
        f"| Items | {items['items']:,} |",
        f"| Observed user-item pairs | {reviews['user_item_pairs']:,} |",
        f"| Matrix density | {matrix['density']:.8f} |",
        f"| Matrix sparsity | {matrix['sparsity']:.8f} |",
        f"| Missing review text rows after ingestion | {reviews['missing_review_text_rows']:,} |",
        f"| Missing item names | {items['missing_name_rows']:,} |",
        f"| Missing item summaries | {items['missing_summary_rows']:,} |",
        "",
        "## Splits",
        "",
        "| Split | Rows |",
        "| --- | ---: |",
        f"| Train | {splits['train']:,} |",
        f"| Task A holdout | {splits['test_task_a']:,} |",
        f"| Task B holdout | {splits['test_task_b']:,} |",
        "",
        "## Ratings",
        "",
        "| Rating | Reviews |",
        "| --- | ---: |",
    ]
    for rating, count in reviews["rating_distribution"].items():
        lines.append(f"| {rating} | {count:,} |")

    lines.extend(
        [
            "",
            "## Top Review Categories",
            "",
            "| Category | Reviews |",
            "| --- | ---: |",
        ]
    )
    for category, count in reviews["top_categories_by_reviews"]:
        lines.append(f"| {category} | {count:,} |")

    lines.extend(
        [
            "",
            "## Top Item Categories",
            "",
            "| Category | Items |",
            "| --- | ---: |",
        ]
    )
    for category, count in items["top_categories_by_items"]:
        lines.append(f"| {category} | {count:,} |")

    lines.extend(
        [
            "",
            "## Distribution Summaries",
            "",
            "| Distribution | Min | P50 | Mean | P90 | P95 | P99 | Max |",
            "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
            _dist_row("Reviews per user", reviews["reviews_per_user"]),
            _dist_row("Reviews per item", reviews["reviews_per_item"]),
            _dist_row("Metadata rating_number", items["metadata_rating_number"]),
            _dist_row("Item average_rating", items["average_rating"]),
            "",
            "## Task B Slice Shape",
            "",
            "| Slice | Examples | Share |",
            "| --- | ---: | ---: |",
            f"| Sparse history, 1-2 train reviews | {task_b['sparse_history_1_2']:,} | {task_b['sparse_history_1_2_pct']:.4f} |",
            f"| Medium history, 3-7 train reviews | {task_b['medium_history_3_7']:,} | {task_b['medium_history_3_7_pct']:.4f} |",
            f"| Warm history, 8+ train reviews | {task_b['warm_history_8_plus']:,} | {task_b['warm_history_8_plus_pct']:.4f} |",
            f"| Cross-domain examples | {task_b['cross_domain_examples']:,} | {task_b['cross_domain_pct']:.4f} |",
            "",
            "## Notes",
            "",
        ]
    )
    lines.extend(f"- {note}" for note in payload["notes"])
    return "\n".join(lines) + "\n"


def _dist_row(label: str, stats: dict) -> str:
    return (
        f"| {label} | {stats['min']} | {stats['p50']} | {stats['mean']} | "
        f"{stats['p90']} | {stats['p95']} | {stats['p99']} | {stats['max']} |"
    )


if __name__ == "__main__":
    main()
