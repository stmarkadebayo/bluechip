from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from eval.common import (
    global_mean,
    histories_by_user,
    item_means,
    load_eval_data,
    persona_from_history,
    print_report,
    user_means,
    write_report,
)
from eval.metrics import mae, rmse, rounded
from app.models.schemas import Item
from app.services.profiling.item_profile import build_item_profile
from app.services.profiling.user_profile import build_user_profile
from app.services.ranking.rating import predict_rating


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate Task A rating prediction baselines.")
    parser.add_argument("--reviews", default="data/sample/reviews.jsonl")
    parser.add_argument("--items", default="data/sample/items.jsonl")
    parser.add_argument("--processed-dir", default="data/processed")
    parser.add_argument("--output", default="runs/eval/task_a_report.json")
    parser.add_argument("--max-examples", type=int, default=0)
    args = parser.parse_args()

    train, test_a, _, items = load_eval_data(
        reviews_path=Path(args.reviews),
        items_path=Path(args.items),
        processed_dir=Path(args.processed_dir),
    )
    actual = [float(row["rating"]) for row in test_a]
    if args.max_examples:
        test_a = test_a[: args.max_examples]
        actual = actual[: args.max_examples]
    g_mean = global_mean(train)
    u_means = user_means(train)
    i_means = item_means(train)
    history_map = histories_by_user(train)

    predictions = {
        "global_mean": [g_mean for _ in test_a],
        "user_mean": [u_means.get(row["user_id"], g_mean) for row in test_a],
        "item_mean": [i_means.get(row["item_id"], g_mean) for row in test_a],
        "user_item_bias": [
            _user_item_bias(row, g_mean, u_means, i_means)
            for row in test_a
        ],
        "hybrid_profile": [
            _hybrid_prediction(row, history_map, items, g_mean)
            for row in test_a
        ],
    }

    metrics = {}
    for name, predicted in predictions.items():
        metrics[f"{name}_mae"] = rounded(mae(actual, predicted))
        metrics[f"{name}_rmse"] = rounded(rmse(actual, predicted))

    payload = {
        "task": "Task A",
        "dataset": str(Path(args.processed_dir) if Path(args.processed_dir).exists() else Path(args.reviews)),
        "examples": len(test_a),
        "metrics": metrics,
        "notes": [
            "Temporal split: earlier reviews are history; last review per eligible user is held out.",
            "Hybrid profile scorer uses user behavior, item quality, category affinity, and aspect overlap.",
        ],
    }
    write_report(Path(args.output), payload)
    print_report(payload)


def _user_item_bias(row: dict, g_mean: float, u_means: dict[str, float], i_means: dict[str, float]) -> float:
    user_bias = u_means.get(row["user_id"], g_mean) - g_mean
    item_bias = i_means.get(row["item_id"], g_mean) - g_mean
    return min(max(g_mean + user_bias + item_bias, 1), 5)


def _hybrid_prediction(
    row: dict,
    history_map: dict[str, list],
    items: dict[str, Item],
    g_mean: float,
) -> float:
    history = history_map.get(row["user_id"], [])
    persona = persona_from_history(history)
    user_profile = build_user_profile(persona=persona, history=history, locale=None)
    target = items.get(row["item_id"])
    if target is None:
        target = Item(
            item_id=row["item_id"],
            name=row["item_name"],
            category=row.get("category") or "unknown",
            metadata={},
            summary=row.get("review") or "",
            average_rating=g_mean,
        )
    item_profile = build_item_profile(target)
    return float(predict_rating(user_profile, item_profile).predicted_rating)


if __name__ == "__main__":
    main()
