from __future__ import annotations

import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.models.schemas import Item, UserHistoryItem  # noqa: E402
from scripts.build_splits import build_temporal_splits  # noqa: E402
from scripts.data_utils import read_jsonl, write_json  # noqa: E402


def load_eval_data(
    reviews_path: Path,
    items_path: Path,
    processed_dir: Path,
) -> tuple[list[dict], list[dict], list[dict], dict[str, Item]]:
    if (processed_dir / "train.jsonl").exists() and (processed_dir / "test_task_a.jsonl").exists():
        train = read_jsonl(processed_dir / "train.jsonl")
        test_a = read_jsonl(processed_dir / "test_task_a.jsonl")
        test_b = read_jsonl(processed_dir / "test_task_b.jsonl")
        items_rows = read_jsonl(processed_dir / "items.jsonl")
    else:
        reviews = read_jsonl(reviews_path)
        train, test_a, test_b = build_temporal_splits(reviews)
        items_rows = read_jsonl(items_path)

    items = {row["item_id"]: Item(**row) for row in items_rows}
    return train, test_a, test_b, items


def history_for_user(train: list[dict], user_id: str) -> list[UserHistoryItem]:
    return [
        UserHistoryItem(
            item_id=row["item_id"],
            item_name=row["item_name"],
            rating=row["rating"],
            review=row["review"],
            category=row.get("category"),
            timestamp=row.get("timestamp"),
        )
        for row in train
        if row["user_id"] == user_id
    ]


def histories_by_user(train: list[dict]) -> dict[str, list[UserHistoryItem]]:
    grouped: dict[str, list[UserHistoryItem]] = defaultdict(list)
    for row in train:
        grouped[row["user_id"]].append(
            UserHistoryItem(
                item_id=row["item_id"],
                item_name=row["item_name"],
                rating=row["rating"],
                review=row["review"],
                category=row.get("category"),
                timestamp=row.get("timestamp"),
            )
        )
    return grouped


def persona_from_history(history: list[UserHistoryItem]) -> str:
    if not history:
        return "A user with limited review history."
    categories = Counter(item.category for item in history if item.category)
    positive = [item.review for item in history if item.rating >= 4]
    negative = [item.review for item in history if item.rating <= 2]
    pieces = []
    if categories:
        pieces.append("User frequently reviews " + ", ".join(category for category, _ in categories.most_common(3)))
    if positive:
        pieces.append("They liked: " + " ".join(positive[:2]))
    if negative:
        pieces.append("They disliked: " + " ".join(negative[:2]))
    return " ".join(pieces)


def item_means(train: list[dict]) -> dict[str, float]:
    totals: dict[str, list[float]] = defaultdict(list)
    for row in train:
        totals[row["item_id"]].append(float(row["rating"]))
    return {item_id: sum(values) / len(values) for item_id, values in totals.items()}


def user_means(train: list[dict]) -> dict[str, float]:
    totals: dict[str, list[float]] = defaultdict(list)
    for row in train:
        totals[row["user_id"]].append(float(row["rating"]))
    return {user_id: sum(values) / len(values) for user_id, values in totals.items()}


def global_mean(train: list[dict]) -> float:
    if not train:
        return 3.5
    return sum(float(row["rating"]) for row in train) / len(train)


def popularity_ranking(train: list[dict], item_ids: list[str]) -> list[str]:
    counts = Counter(row["item_id"] for row in train if row["rating"] >= 4)
    return sorted(item_ids, key=lambda item_id: (counts[item_id], item_id), reverse=True)


def write_report(report_path: Path, payload: dict) -> None:
    write_json(report_path, payload)
    md_path = report_path.with_suffix(".md")
    lines = [f"# {payload['task']} Evaluation", "", f"Dataset: `{payload['dataset']}`", ""]
    lines.append("| Metric | Value |")
    lines.append("| --- | ---: |")
    for metric, value in payload["metrics"].items():
        lines.append(f"| {metric} | {value} |")
    if payload.get("notes"):
        lines.extend(["", "## Notes", ""])
        lines.extend(f"- {note}" for note in payload["notes"])
    md_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def print_report(payload: dict) -> None:
    print(json.dumps(payload, ensure_ascii=True, indent=2))
