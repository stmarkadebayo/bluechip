from __future__ import annotations

import argparse
import os
import sys
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.models.schemas import Item, UserHistoryItem  # noqa: E402
from app.services.profiling.item_profile import build_item_profile  # noqa: E402
from app.services.profiling.user_profile import build_user_profile  # noqa: E402
from app.services.ranking.rating_features import build_rating_stats  # noqa: E402
from app.services.ranking.rating_features import predict_adaptive_star_rating  # noqa: E402
from app.services.ranking.rating_features import predict_calibrated_rating  # noqa: E402
from app.services.ranking.rating_features import save_rating_stats  # noqa: E402
from app.services.ranking.task_a_model import (  # noqa: E402
    COMPACT_FEATURES,
    FULL_FEATURES,
    evaluate_model_rows,
    fit_linear_rating_model,
    fit_rmse_ensemble,
    model_feature_vector,
)
from eval.common import load_eval_data, persona_from_history, write_report  # noqa: E402
from eval.metrics import mae, rmse, rounded  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description="Train and select Task A rating models.")
    parser.add_argument("--reviews", default="data/sample/reviews.jsonl")
    parser.add_argument("--items", default="data/sample/items.jsonl")
    parser.add_argument("--processed-dir", default="data/processed")
    parser.add_argument("--output-model", default="data/processed/task_a_model.json")
    parser.add_argument(
        "--output-stats",
        default="",
        help="Optional precomputed RatingStats artifact for runtime serving.",
    )
    parser.add_argument("--output-report", default="runs/eval/task_a_model_training.json")
    parser.add_argument("--candidate-dir", default="", help="Optional directory for all candidate models.")
    parser.add_argument("--max-train-examples", type=int, default=0)
    parser.add_argument("--max-eval-examples", type=int, default=1000)
    parser.add_argument("--validation-fraction", type=float, default=0.15)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--learning-rate", type=float, default=0.025)
    parser.add_argument("--l2", type=float, default=0.001)
    parser.add_argument(
        "--workers",
        type=int,
        default=max(1, min((os.cpu_count() or 2) - 1, 8)),
        help="Parallel model-fit workers. Defaults to using most local CPU cores.",
    )
    parser.add_argument(
        "--selection-metric",
        choices=["mae", "rmse"],
        default="rmse",
        help="Metric used to select the final artifact on the train-validation split.",
    )
    parser.add_argument(
        "--ensemble",
        action="store_true",
        help="Fit a nonnegative weighted ensemble against validation RMSE after selecting a base model.",
    )
    args = parser.parse_args()

    train, test_a, _, items = load_eval_data(
        reviews_path=Path(args.reviews),
        items_path=Path(args.items),
        processed_dir=Path(args.processed_dir),
    )
    stats = build_rating_stats(train)
    stats_path = Path(args.output_stats) if args.output_stats else Path(args.output_model).with_name("task_a_rating_stats.json")
    save_rating_stats(stats, stats_path)
    rows = _training_rows(train, items, stats, args.max_train_examples)
    fit_rows, validation_rows = _split_training_rows(rows, args.validation_fraction)
    candidates = _train_candidates(
        fit_rows=fit_rows,
        validation_rows=validation_rows,
        epochs=args.epochs,
        learning_rate=args.learning_rate,
        l2=args.l2,
        workers=args.workers,
    )
    selected = min(
        candidates,
        key=lambda result: result["validation"][args.selection_metric],
    )
    model = selected["model"]
    selected_name = selected["name"]
    ensemble_metrics = None
    if args.ensemble:
        base_validation = evaluate_model_rows(model, validation_rows)
        ensemble = fit_rmse_ensemble(model, validation_rows)
        ensemble_validation = evaluate_model_rows(ensemble, validation_rows)
        ensemble_metrics = {
            "base_mae": rounded(base_validation["mae"]),
            "base_rmse": rounded(base_validation["rmse"]),
            "ensemble_mae": rounded(ensemble_validation["mae"]),
            "ensemble_rmse": rounded(ensemble_validation["rmse"]),
            "weights": ensemble.weights,
            "promoted": ensemble_validation["rmse"] <= base_validation["rmse"],
        }
        if ensemble_validation["rmse"] <= base_validation["rmse"]:
            model = ensemble
            selected_name = ensemble.metadata["name"]
    model.save(Path(args.output_model))
    final_validation = evaluate_model_rows(model, validation_rows)

    if args.candidate_dir:
        candidate_dir = Path(args.candidate_dir)
        for result in candidates:
            result["model"].save(candidate_dir / f"{result['name']}.json")

    eval_rows = _eval_rows(test_a[: args.max_eval_examples], train, items, stats, model)
    actual = [target for target, _ in eval_rows]
    predicted = [prediction for _, prediction in eval_rows]
    payload = {
        "task": "Task A Model Training",
        "dataset": str(Path(args.processed_dir)),
        "examples": len(rows),
        "fit_examples": len(fit_rows),
        "validation_examples": len(validation_rows),
        "selected_model": selected_name,
        "selection_metric": args.selection_metric,
        "ensemble": ensemble_metrics,
        "workers": args.workers,
        "metrics": {
            "eval_examples": len(eval_rows),
            "trained_model_mae": rounded(mae(actual, predicted)) if eval_rows else 0.0,
            "trained_model_rmse": rounded(rmse(actual, predicted)) if eval_rows else 0.0,
            "validation_mae": rounded(final_validation["mae"]),
            "validation_rmse": rounded(final_validation["rmse"]),
        },
        "candidate_metrics": {
            result["name"]: {
                "mae": rounded(result["validation"]["mae"]),
                "rmse": rounded(result["validation"]["rmse"]),
                "loss": result["model"].metadata["loss"],
                "feature_count": result["model"].metadata["feature_count"],
                "star_policy": result["model"].metadata.get(
                    "star_policy",
                    "none" if not result["model"].star_thresholds else "fixed",
                ),
            }
            for result in candidates
        },
        "model_path": str(Path(args.output_model)),
        "stats_path": str(stats_path),
        "candidate_dir": args.candidate_dir,
        "model_metadata": model.metadata,
        "notes": [
            "Training examples use leave-one-out user histories from the temporal training split.",
            "Feature engineering includes user/item/category shrinkage, recency, volatility, star shares, user-category affinity, semantic overlap, and reliability interactions.",
            "Candidate models cover compact/full feature sets, MSE, Huber, MAE subgradient losses, and calibrated ordinal-star variants.",
            "Candidate fitting runs in parallel when --workers is greater than 1.",
            "The saved artifact is selected by validation RMSE by default; use --selection-metric mae to optimize MAE instead.",
            "When --ensemble is enabled, the selected model is blended with rating baselines and promoted only if validation RMSE does not regress.",
        ],
    }
    write_report(Path(args.output_report), payload)
    print_report(payload)


def _train_candidates(
    fit_rows: list[tuple[dict[str, float], float]],
    validation_rows: list[tuple[dict[str, float], float]],
    epochs: int,
    learning_rate: float,
    l2: float,
    workers: int,
) -> list[dict]:
    configs = []
    for feature_name, features in [("compact", COMPACT_FEATURES), ("full", FULL_FEATURES)]:
        for loss, loss_learning_rate in [
            ("mse", learning_rate),
            ("huber", learning_rate),
            ("mae", learning_rate * 0.25),
        ]:
            configs.append((f"{feature_name}_{loss}", features, loss, loss_learning_rate, "none"))
            configs.append(
                (f"{feature_name}_{loss}_calibrated_star", features, loss, loss_learning_rate, "calibrated")
            )
            configs.append(
                (f"{feature_name}_{loss}_round_star", features, loss, loss_learning_rate, "round")
            )

    if workers <= 1 or len(configs) <= 1:
        return [
            _fit_candidate(config, fit_rows, validation_rows, epochs, l2)
            for config in configs
        ]

    results = {}
    try:
        with ProcessPoolExecutor(max_workers=min(workers, len(configs))) as executor:
            futures = {
                executor.submit(_fit_candidate, config, fit_rows, validation_rows, epochs, l2): config[0]
                for config in configs
            }
            for future in as_completed(futures):
                result = future.result()
                results[result["name"]] = result
    except PermissionError:
        with ThreadPoolExecutor(max_workers=min(workers, len(configs))) as executor:
            futures = {
                executor.submit(_fit_candidate, config, fit_rows, validation_rows, epochs, l2): config[0]
                for config in configs
            }
            for future in as_completed(futures):
                result = future.result()
                results[result["name"]] = result
    return [results[name] for name, *_ in configs]


def _fit_candidate(
    config: tuple[str, list[str], str, float, str],
    fit_rows: list[tuple[dict[str, float], float]],
    validation_rows: list[tuple[dict[str, float], float]],
    epochs: int,
    l2: float,
) -> dict:
    name, features, loss, loss_learning_rate, star_mode = config
    model = fit_linear_rating_model(
        fit_rows,
        learning_rate=loss_learning_rate,
        epochs=epochs,
        l2=l2,
        loss=loss,
        feature_names=features,
        name=name,
        validation_rows=validation_rows,
        calibrate_stars=star_mode == "calibrated",
        fixed_star_thresholds=[1.5, 2.5, 3.5, 4.5] if star_mode == "round" else None,
    )
    return {
        "name": name,
        "model": model,
        "validation": evaluate_model_rows(model, validation_rows),
    }


def _split_training_rows(
    rows: list[tuple[dict[str, float], float]],
    validation_fraction: float,
) -> tuple[list[tuple[dict[str, float], float]], list[tuple[dict[str, float], float]]]:
    if len(rows) < 10:
        return rows, rows
    validation_size = max(1, int(len(rows) * min(max(validation_fraction, 0.05), 0.40)))
    split_at = max(1, len(rows) - validation_size)
    return rows[:split_at], rows[split_at:]


def _training_rows(
    train: list[dict],
    items: dict[str, Item],
    stats,
    max_examples: int,
) -> list[tuple[dict[str, float], float]]:
    rows = []
    history_by_user: dict[str, list[UserHistoryItem]] = defaultdict(list)
    ordered = sorted(train, key=lambda row: (row["user_id"], int(row.get("timestamp") or 0), row["review_id"]))
    for row in ordered:
        history = history_by_user[row["user_id"]]
        if history:
            rows.append((_features_for_row(row, history, items, stats), float(row["rating"])))
            if max_examples and len(rows) >= max_examples:
                break
        history.append(
            UserHistoryItem(
                item_id=row["item_id"],
                item_name=row["item_name"],
                rating=row["rating"],
                review=row["review"],
                category=row.get("category"),
                timestamp=row.get("timestamp"),
            )
        )
    return rows


def _eval_rows(test: list[dict], train: list[dict], items: dict[str, Item], stats, model) -> list[tuple[float, float]]:
    histories: dict[str, list[UserHistoryItem]] = defaultdict(list)
    for row in train:
        histories[row["user_id"]].append(
            UserHistoryItem(
                item_id=row["item_id"],
                item_name=row["item_name"],
                rating=row["rating"],
                review=row["review"],
                category=row.get("category"),
                timestamp=row.get("timestamp"),
            )
        )
    output = []
    for row in test:
        history = histories.get(row["user_id"], [])
        if not history:
            continue
        user_profile = build_user_profile(persona_from_history(history), history)
        item_profile = build_item_profile(_target_item(row, items, stats.global_mean))
        output.append(
            (
                float(row["rating"]),
                model.predict(
                    user_profile=user_profile,
                    item_profile=item_profile,
                    stats=stats,
                    user_id=row["user_id"],
                ),
            )
        )
    return output


def _features_for_row(row: dict, history: list[UserHistoryItem], items: dict[str, Item], stats) -> dict[str, float]:
    user_profile = build_user_profile(persona_from_history(history), history)
    item_profile = build_item_profile(_target_item(row, items, stats.global_mean))
    features = model_feature_vector(
        user_profile=user_profile,
        item_profile=item_profile,
        stats=stats,
        user_id=row["user_id"],
    )
    features["_baseline_calibrated_profile"] = predict_calibrated_rating(
        user_profile=user_profile,
        item_profile=item_profile,
        stats=stats,
        user_id=row["user_id"],
    )
    features["_baseline_adaptive_star"] = float(
        predict_adaptive_star_rating(
            user_profile=user_profile,
            item_profile=item_profile,
            stats=stats,
            user_id=row["user_id"],
        )
    )
    return features


def _target_item(row: dict, items: dict[str, Item], fallback_rating: float) -> Item:
    item = items.get(row["item_id"])
    if item is not None:
        return item
    return Item(
        item_id=row["item_id"],
        name=row["item_name"],
        category=row.get("category") or "unknown",
        metadata={},
        summary=row.get("review") or "",
        average_rating=fallback_rating,
    )


def print_report(payload: dict) -> None:
    import json

    print(json.dumps(payload, ensure_ascii=True, indent=2))


if __name__ == "__main__":
    main()
