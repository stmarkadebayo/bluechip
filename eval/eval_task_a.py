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
from app.services.ranking.rating_features import build_rating_stats, predict_calibrated_rating
from app.services.ranking.rating_features import predict_adaptive_star_rating
from app.services.ranking.rating_features import predict_profile_heuristic_rating
from app.services.ranking.task_a_model import load_task_a_model


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate Task A rating prediction baselines.")
    parser.add_argument("--reviews", default="data/sample/reviews.jsonl")
    parser.add_argument("--items", default="data/sample/items.jsonl")
    parser.add_argument("--processed-dir", default="data/processed")
    parser.add_argument("--output", default="runs/eval/task_a_report.json")
    parser.add_argument("--max-examples", type=int, default=0)
    parser.add_argument("--model-path", default="", help="Optional trained Task A model artifact.")
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
    rating_stats = build_rating_stats(train)
    history_map = histories_by_user(train)
    trained_model = load_task_a_model(Path(args.model_path)) if args.model_path else None

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
        "calibrated_profile": [
            _calibrated_prediction(row, history_map, items, rating_stats)
            for row in test_a
        ],
        "adaptive_star": [
            _adaptive_star_prediction(row, history_map, items, rating_stats)
            for row in test_a
        ],
    }
    if trained_model is not None:
        predictions["trained_model"] = [
            _trained_model_prediction(row, history_map, items, rating_stats, trained_model)
            for row in test_a
        ]
        predictions["trained_model_raw"] = [
            _trained_model_raw_prediction(row, history_map, items, rating_stats, trained_model)
            for row in test_a
        ]
        predictions["trained_model_star"] = [
            round(value) for value in predictions["trained_model_raw"]
        ]

    metrics = {}
    for name, predicted in predictions.items():
        metrics[f"{name}_mae"] = rounded(mae(actual, predicted))
        metrics[f"{name}_rmse"] = rounded(rmse(actual, predicted))

    payload = {
        "task": "Task A",
        "dataset": str(Path(args.processed_dir) if Path(args.processed_dir).exists() else Path(args.reviews)),
        "examples": len(test_a),
        "metrics": metrics,
        "slices": _slice_metrics(
            test_a,
            actual,
            predictions.get("trained_model") or predictions["adaptive_star"],
            history_map,
        ),
        "notes": [
            "Temporal split: earlier reviews are history; last review per eligible user is held out.",
            "Hybrid profile scorer uses user behavior, item quality, category affinity, and aspect overlap.",
            "Calibrated profile blends shrinkage user, item, and category priors with profile/vector adjustments.",
            "Adaptive star uses profile heuristics for sparse items and user/item mean blending for items with enough evidence.",
            "Trained model metrics are included when --model-path points to a saved Task A model artifact; trained_model_raw reports the continuous regressor before ordinal calibration.",
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
    return float(predict_profile_heuristic_rating(user_profile, item_profile))


def _calibrated_prediction(
    row: dict,
    history_map: dict[str, list],
    items: dict[str, Item],
    rating_stats,
) -> float:
    history = history_map.get(row["user_id"], [])
    persona = persona_from_history(history)
    user_profile = build_user_profile(persona=persona, history=history, locale=None)
    item_profile = build_item_profile(_target_item(row, items, rating_stats.global_mean))
    return predict_calibrated_rating(
        user_profile=user_profile,
        item_profile=item_profile,
        stats=rating_stats,
        user_id=row["user_id"],
    )


def _adaptive_star_prediction(
    row: dict,
    history_map: dict[str, list],
    items: dict[str, Item],
    rating_stats,
) -> float:
    history = history_map.get(row["user_id"], [])
    user_profile = build_user_profile(persona_from_history(history), history)
    item_profile = build_item_profile(_target_item(row, items, rating_stats.global_mean))
    return float(
        predict_adaptive_star_rating(
            user_profile=user_profile,
            item_profile=item_profile,
            stats=rating_stats,
            user_id=row["user_id"],
        )
    )


def _trained_model_prediction(
    row: dict,
    history_map: dict[str, list],
    items: dict[str, Item],
    rating_stats,
    model,
) -> float:
    history = history_map.get(row["user_id"], [])
    user_profile = build_user_profile(persona_from_history(history), history)
    item_profile = build_item_profile(_target_item(row, items, rating_stats.global_mean))
    return model.predict(
        user_profile=user_profile,
        item_profile=item_profile,
        stats=rating_stats,
        user_id=row["user_id"],
    )


def _trained_model_raw_prediction(
    row: dict,
    history_map: dict[str, list],
    items: dict[str, Item],
    rating_stats,
    model,
) -> float:
    history = history_map.get(row["user_id"], [])
    user_profile = build_user_profile(persona_from_history(history), history)
    item_profile = build_item_profile(_target_item(row, items, rating_stats.global_mean))
    return model.predict_raw(
        user_profile=user_profile,
        item_profile=item_profile,
        stats=rating_stats,
        user_id=row["user_id"],
    )


def _target_item(row: dict, items: dict[str, Item], fallback_rating: float) -> Item:
    target = items.get(row["item_id"])
    if target is not None:
        return target
    return Item(
        item_id=row["item_id"],
        name=row["item_name"],
        category=row.get("category") or "unknown",
        metadata={},
        summary=row.get("review") or "",
        average_rating=fallback_rating,
    )


def _slice_metrics(
    rows: list[dict],
    actual: list[float],
    predicted: list[float],
    history_map: dict[str, list],
) -> dict[str, dict[str, float]]:
    slices: dict[str, list[int]] = {}
    for index, row in enumerate(rows):
        history_count = len(history_map.get(row["user_id"], []))
        slices.setdefault(_history_bucket(history_count), []).append(index)
        slices.setdefault(f"category:{row.get('category') or 'unknown'}", []).append(index)

    output = {}
    for name, indexes in sorted(slices.items()):
        if len(indexes) < 2:
            continue
        slice_actual = [actual[index] for index in indexes]
        slice_predicted = [predicted[index] for index in indexes]
        output[name] = {
            "examples": len(indexes),
            "mae": rounded(mae(slice_actual, slice_predicted)),
            "rmse": rounded(rmse(slice_actual, slice_predicted)),
        }
    return output


def _history_bucket(history_count: int) -> str:
    if history_count <= 0:
        return "history:cold"
    if history_count <= 2:
        return "history:light_1_2"
    if history_count <= 9:
        return "history:medium_3_9"
    return "history:warm_10_plus"


if __name__ == "__main__":
    main()
