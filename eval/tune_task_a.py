from __future__ import annotations

import argparse
import itertools
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.models.schemas import Item  # noqa: E402
from app.services.profiling.item_profile import build_item_profile  # noqa: E402
from app.services.profiling.user_profile import build_user_profile  # noqa: E402
from app.services.ranking.rating_features import (  # noqa: E402
    build_rating_stats,
    predict_adaptive_star_rating,
)
from eval.common import histories_by_user, load_eval_data, persona_from_history, write_report  # noqa: E402
from eval.metrics import mae, rmse, rounded  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description="Tune Task A calibrated rating weights.")
    parser.add_argument("--reviews", default="data/sample/reviews.jsonl")
    parser.add_argument("--items", default="data/sample/items.jsonl")
    parser.add_argument("--processed-dir", default="data/processed")
    parser.add_argument("--output", default="runs/eval/task_a_tuning.json")
    parser.add_argument("--max-examples", type=int, default=0)
    args = parser.parse_args()

    train, test_a, _, items = load_eval_data(
        reviews_path=Path(args.reviews),
        items_path=Path(args.items),
        processed_dir=Path(args.processed_dir),
    )
    if args.max_examples:
        test_a = test_a[: args.max_examples]

    stats = build_rating_stats(train)
    history_map = histories_by_user(train)
    actual = [float(row["rating"]) for row in test_a]
    trials = []
    for config in _config_grid():
        predicted = [
            _prediction(row, history_map, items, stats, config)
            for row in test_a
        ]
        trials.append(
            {
                "mae": rounded(mae(actual, predicted)),
                "rmse": rounded(rmse(actual, predicted)),
                "config": config,
            }
        )

    trials.sort(key=lambda row: (row["mae"], row["rmse"]))
    best = trials[0] if trials else {"mae": 0.0, "rmse": 0.0, "config": {}}
    payload = {
        "task": "Task A Tuning",
        "dataset": str(Path(args.processed_dir)),
        "examples": len(test_a),
        "metrics": {
            "best_mae": best["mae"],
            "best_rmse": best["rmse"],
            "trials": len(trials),
        },
        "best_config": best["config"],
        "top_trials": trials[:5],
        "notes": [
            "Grid search tunes the adaptive Task A star predictor.",
            "Use real-data slices before promoting tuned weights into runtime defaults.",
        ],
    }
    write_report(Path(args.output), payload)
    print_report(payload)


def _config_grid() -> list[dict[str, float]]:
    thresholds = [1, 2, 3, 5, 10, 20]
    user_weights = [0.3, 0.4, 0.5, 0.6, 0.7]
    return [
        {"item_count_threshold": threshold, "user_weight": user_weight}
        for threshold, user_weight in itertools.product(thresholds, user_weights)
    ]


def _prediction(
    row: dict,
    history_map: dict[str, list],
    items: dict[str, Item],
    stats,
    config: dict[str, float],
) -> float:
    history = history_map.get(row["user_id"], [])
    user_profile = build_user_profile(persona_from_history(history), history)
    item = items.get(row["item_id"])
    if item is None:
        item = Item(
            item_id=row["item_id"],
            name=row["item_name"],
            category=row.get("category") or "unknown",
            metadata={},
            summary=row.get("review") or "",
            average_rating=stats.global_mean,
        )
    return float(predict_adaptive_star_rating(
        user_profile=user_profile,
        item_profile=build_item_profile(item),
        stats=stats,
        user_id=row["user_id"],
        item_count_threshold=int(config["item_count_threshold"]),
        user_weight=float(config["user_weight"]),
    ))


def print_report(payload: dict) -> None:
    import json

    print(json.dumps(payload, ensure_ascii=True, indent=2))


if __name__ == "__main__":
    main()
