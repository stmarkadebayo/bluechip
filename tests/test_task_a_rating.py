from __future__ import annotations

import json

from app.models.schemas import Item, UserHistoryItem
from app.services.profiling.item_profile import build_item_profile
from app.services.profiling.user_profile import build_user_profile
from app.services.ranking import rating as rating_module
from app.services.ranking.rating import predict_rating
from app.services.ranking.rating_features import (
    build_rating_stats,
    predict_adaptive_star_rating,
    predict_calibrated_rating,
    shrinkage_mean,
)
from app.services.ranking.task_a_model import TaskARatingModel, fit_linear_rating_model, load_task_a_model
from app.services.ranking.task_a_model import evaluate_model_rows, fit_rmse_ensemble


def test_shrinkage_mean_moves_sparse_values_toward_prior() -> None:
    assert shrinkage_mean(mean=5.0, count=1, prior=3.0, prior_weight=3.0) == 3.5
    assert shrinkage_mean(mean=5.0, count=100, prior=3.0, prior_weight=3.0) > 4.9


def test_calibrated_rating_uses_user_item_and_category_priors() -> None:
    train = [
        {"user_id": "u1", "item_id": "i1", "category": "books", "rating": 5},
        {"user_id": "u1", "item_id": "i2", "category": "books", "rating": 4},
        {"user_id": "u2", "item_id": "i3", "category": "books", "rating": 5},
        {"user_id": "u3", "item_id": "i4", "category": "tools", "rating": 2},
    ]
    stats = build_rating_stats(train)
    user_profile = build_user_profile(
        persona="A careful reader who likes detailed useful books.",
        history=[
            UserHistoryItem(
                item_id="i1",
                item_name="Helpful Book",
                rating=5,
                review="Detailed and useful.",
                category="books",
            )
        ],
    )
    item_profile = build_item_profile(
        Item(
            item_id="i5",
            name="Another Useful Book",
            category="books",
            summary="Detailed useful book for careful readers.",
            average_rating=4.8,
            metadata={"review_count": 20},
        )
    )

    predicted = predict_calibrated_rating(user_profile, item_profile, stats=stats, user_id="u1")

    assert predicted > stats.global_mean


def test_dislike_match_lowers_calibrated_rating() -> None:
    user_profile = build_user_profile(
        persona="A diner who dislikes slow loud restaurants.",
        history=[
            UserHistoryItem(
                item_id="r1",
                item_name="Bad Spot",
                rating=1,
                review="Slow loud service.",
                category="restaurant",
            )
        ],
    )
    calm_item = build_item_profile(
        Item(
            item_id="r2",
            name="Calm Spot",
            category="restaurant",
            summary="Quiet calm service.",
            average_rating=4.2,
        )
    )
    loud_item = build_item_profile(
        Item(
            item_id="r3",
            name="Loud Spot",
            category="restaurant",
            summary="Slow loud crowded service.",
            average_rating=4.2,
        )
    )

    assert predict_calibrated_rating(user_profile, loud_item) < predict_calibrated_rating(
        user_profile, calm_item
    )


def test_adaptive_star_uses_profile_for_sparse_items_and_blend_for_known_items() -> None:
    stats = build_rating_stats(
        [
            {"user_id": "u1", "item_id": "known", "category": "books", "rating": 5},
            {"user_id": "u2", "item_id": "known", "category": "books", "rating": 5},
            {"user_id": "u3", "item_id": "known", "category": "books", "rating": 4},
        ]
    )
    user_profile = build_user_profile(
        persona="A generous reader who likes useful books.",
        history=[
            UserHistoryItem(
                item_id="old",
                item_name="Old Book",
                rating=5,
                review="Useful and clear.",
                category="books",
            )
        ],
    )
    known_item = build_item_profile(
        Item(
            item_id="known",
            name="Known Book",
            category="books",
            summary="Useful book.",
            average_rating=4.6,
        )
    )

    assert predict_adaptive_star_rating(user_profile, known_item, stats, "u1", 2) >= 4


def test_task_a_model_round_trips_artifact(tmp_path) -> None:
    model = fit_linear_rating_model(
        [
            ({"bias": 1.0, "user_prior": 5.0, "item_prior": 5.0}, 5.0),
            ({"bias": 1.0, "user_prior": 1.0, "item_prior": 1.0}, 1.0),
        ],
        epochs=2,
    )
    path = tmp_path / "task_a_model.json"
    model.save(path)

    loaded = load_task_a_model(path)

    assert loaded is not None
    assert loaded.features == model.features
    assert loaded.metadata["examples"] == 2


def test_rmse_ensemble_promotes_best_validation_component(tmp_path) -> None:
    model = fit_linear_rating_model(
        [
            ({"x": 0.0}, 3.0),
            ({"x": 1.0}, 3.0),
        ],
        feature_names=["x"],
        epochs=1,
        name="flat",
    )
    validation_rows = [
        ({"x": 0.0, "_baseline_adaptive_star": 1.0}, 1.0),
        ({"x": 1.0, "_baseline_adaptive_star": 5.0}, 5.0),
    ]

    ensemble = fit_rmse_ensemble(model, validation_rows, weight_step=0.5)

    assert ensemble.weights["adaptive_star"] == 1.0
    assert evaluate_model_rows(ensemble, validation_rows)["rmse"] == 0.0

    path = tmp_path / "task_a_ensemble.json"
    ensemble.save(path)
    loaded = load_task_a_model(path)

    assert loaded is not None
    assert loaded.metadata["name"] == "flat_rmse_ensemble"


def test_predict_rating_can_use_runtime_model_artifact(tmp_path, monkeypatch) -> None:
    model = TaskARatingModel(
        features=["user_prior"],
        weights=[2.0],
        target_mean=1.0,
        feature_means=[1.0],
        feature_scales=[1.0],
        metadata={"name": "runtime_test"},
    )
    model_path = tmp_path / "task_a_model.json"
    stats_path = tmp_path / "train.jsonl"
    model.save(model_path)
    stats_path.write_text(
        "\n".join(
            json.dumps(row)
            for row in [
                {"user_id": "u1", "item_id": "i1", "category": "books", "rating": 5},
                {"user_id": "u2", "item_id": "i2", "category": "books", "rating": 3},
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    monkeypatch.setenv("TASK_A_MODEL_PATH", str(model_path))
    monkeypatch.setenv("TASK_A_STATS_PATH", str(stats_path))
    rating_module._runtime_rating_assets.cache_clear()
    user_profile = build_user_profile(
        persona="A generous reader who likes useful books.",
        history=[
            UserHistoryItem(
                item_id="old",
                item_name="Old Book",
                rating=5,
                review="Useful and clear.",
                category="books",
            )
        ],
    )
    item_profile = build_item_profile(
        Item(
            item_id="new",
            name="Useful Book",
            category="books",
            summary="Useful book.",
            average_rating=4.5,
        )
    )

    prediction = predict_rating(user_profile, item_profile)

    assert prediction.predicted_rating >= 4
    rating_module._runtime_rating_assets.cache_clear()
