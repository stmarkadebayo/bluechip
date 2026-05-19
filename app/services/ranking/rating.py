from __future__ import annotations

import json
import os
from functools import lru_cache
from pathlib import Path

from app.models.schemas import ItemProfile, RatingPrediction, UserProfile
from app.services.ranking.rating_features import (
    build_rating_stats,
    clamp_rating,
    load_rating_stats,
    predict_adaptive_star_rating,
    predict_calibrated_rating,
    predict_profile_heuristic_rating,
    rating_features,
)
from app.services.ranking.task_a_model import load_task_a_model


def predict_rating(
    user_profile: UserProfile,
    item_profile: ItemProfile,
    user_id: str | None = None,
) -> RatingPrediction:
    features = rating_features(user_profile, item_profile)
    model, stats = _runtime_rating_assets()
    runtime_user_id = user_id or "runtime_user"
    model_name = None
    if stats is not None:
        serving_head = _runtime_serving_head()
        continuous = _predict_with_serving_head(
            serving_head=serving_head,
            user_profile=user_profile,
            item_profile=item_profile,
            stats=stats,
            user_id=runtime_user_id,
            model=model,
        )
        model_name = serving_head
        predicted = int(round(clamp_rating(continuous)))
        predicted_score = round(clamp_rating(continuous), 4)
    else:
        predicted = predict_profile_heuristic_rating(user_profile, item_profile)
        predicted_score = float(predicted)
        model_name = "profile_heuristic"
    confidence = min(
        max(
            0.25
            + user_profile.confidence * 0.35
            + features["preference_match"] * 0.25
            + features["vector_match"] * 0.10
            + (item_profile.quality_score * 0.15),
            0,
        ),
        0.95,
    )

    return RatingPrediction(
        predicted_rating=int(predicted),
        predicted_score=predicted_score,
        confidence=round(confidence, 2),
        model_name=model_name,
        user_signals=user_profile.signals,
        item_signals=item_profile.signals,
    )


def _predict_with_serving_head(
    serving_head: str,
    user_profile: UserProfile,
    item_profile: ItemProfile,
    stats,
    user_id: str,
    model,
) -> float:
    if serving_head == "calibrated_profile":
        return predict_calibrated_rating(
            user_profile=user_profile,
            item_profile=item_profile,
            stats=stats,
            user_id=user_id,
        )
    if serving_head == "adaptive_star":
        return float(
            predict_adaptive_star_rating(
                user_profile=user_profile,
                item_profile=item_profile,
                stats=stats,
                user_id=user_id,
            )
        )
    if serving_head in {"profile_heuristic", "hybrid_profile"}:
        return float(predict_profile_heuristic_rating(user_profile, item_profile))
    if model is None:
        return predict_calibrated_rating(
            user_profile=user_profile,
            item_profile=item_profile,
            stats=stats,
            user_id=user_id,
        )
    if serving_head == "trained_model_star":
        return float(
            round(
                model.predict_raw(
                    user_profile=user_profile,
                    item_profile=item_profile,
                    stats=stats,
                    user_id=user_id,
                )
            )
        )
    if serving_head == "trained_model_selected":
        return float(
            model.predict(
                user_profile=user_profile,
                item_profile=item_profile,
                stats=stats,
                user_id=user_id,
            )
        )
    return float(
        model.predict_raw(
            user_profile=user_profile,
            item_profile=item_profile,
            stats=stats,
            user_id=user_id,
        )
    )


@lru_cache(maxsize=1)
def _runtime_rating_assets() -> tuple[object | None, object | None]:
    configured_stats_path = (os.getenv("TASK_A_STATS_PATH") or "").strip()
    model_path = _first_existing_path(
        (os.getenv("TASK_A_MODEL_PATH") or "").strip(),
        "data/processed/all_categories/task_a_model_rmse.json",
        "data/processed/all_categories/task_a_model.json",
        "data/processed/task_a_model_rmse.json",
        "data/processed/task_a_model.json",
    )
    stats_path = _first_existing_path(
        configured_stats_path,
        "data/processed/all_categories/task_a_rating_stats.json",
        "data/processed/task_a_rating_stats.json",
    )
    if not model_path or not stats_path:
        return None, None
    model = load_task_a_model(model_path)
    if model is None:
        return None, None
    stats = load_rating_stats(stats_path)
    if stats is None and configured_stats_path and stats_path.suffix == ".jsonl":
        train_rows = _read_jsonl(stats_path)
        if not train_rows:
            return None, None
        stats = build_rating_stats(train_rows)
    if stats is None:
        return None, None
    return model, stats


@lru_cache(maxsize=1)
def _runtime_serving_head() -> str:
    configured = (os.getenv("TASK_A_SERVING_HEAD") or "").strip()
    if configured:
        return configured
    policy_path = _first_existing_path(
        (os.getenv("TASK_A_SERVING_POLICY") or "").strip(),
        "data/processed/all_categories/task_a_serving_policy.json",
        "data/processed/task_a_serving_policy.json",
    )
    if not policy_path:
        return "trained_model_raw"
    try:
        payload = json.loads(policy_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return "trained_model_raw"
    head = str(payload.get("serving_head") or "").strip()
    return head or "trained_model_raw"


def _first_existing_path(*candidates: str) -> Path | None:
    for candidate in candidates:
        if not candidate:
            continue
        path = Path(candidate)
        if path.exists():
            return path
    return None


def _read_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        return []
    rows = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows
