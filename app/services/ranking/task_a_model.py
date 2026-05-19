from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path

from app.models.schemas import ItemProfile, UserProfile
from app.services.ranking.rating_features import (
    DEFAULT_RATING_WEIGHTS,
    RatingStats,
    clamp_rating,
    predict_adaptive_star_rating,
    predict_calibrated_rating,
    rating_features,
)


COMPACT_FEATURES = [
    "user_prior",
    "item_prior",
    "category_prior",
    "preference_match",
    "dislike_match",
    "category_affinity",
    "vector_match",
    "log_user_count",
    "log_item_count",
    "log_category_count",
]

FULL_FEATURES = [
    *COMPACT_FEATURES,
    "user_recent_prior",
    "user_category_prior",
    "log_user_category_count",
    "user_std",
    "item_std",
    "category_std",
    "user_positive_share",
    "user_negative_share",
    "item_positive_share",
    "item_negative_share",
    "category_positive_share",
    "category_negative_share",
    "user_global_delta",
    "item_global_delta",
    "category_global_delta",
    "user_category_delta",
    "item_category_delta",
    "quality_score",
    "review_length_log",
    "rating_trend",
    "preference_x_affinity",
    "dislike_x_negative_item",
    "vector_x_affinity",
    "reliability_gap",
    "user_item_prior_gap",
]

MODEL_FEATURES = FULL_FEATURES


@dataclass(frozen=True)
class TaskARatingModel:
    features: list[str]
    weights: list[float]
    target_mean: float
    feature_means: list[float]
    feature_scales: list[float]
    metadata: dict
    star_thresholds: list[float] | None = None

    def predict(
        self,
        user_profile: UserProfile,
        item_profile: ItemProfile,
        stats: RatingStats,
        user_id: str,
    ) -> float:
        raw = self.predict_raw(
            user_profile=user_profile,
            item_profile=item_profile,
            stats=stats,
            user_id=user_id,
        )
        if self.star_thresholds:
            return float(apply_star_thresholds(raw, self.star_thresholds))
        return raw

    def predict_raw(
        self,
        user_profile: UserProfile,
        item_profile: ItemProfile,
        stats: RatingStats,
        user_id: str,
    ) -> float:
        values = model_feature_vector(
            user_profile=user_profile,
            item_profile=item_profile,
            stats=stats,
            user_id=user_id,
        )
        centered = [
            (values.get(name, 0.0) - mean) / scale
            for name, mean, scale in zip(self.features, self.feature_means, self.feature_scales)
        ]
        raw = self.target_mean + sum(weight * value for weight, value in zip(self.weights, centered))
        return clamp_rating(raw)

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self.to_dict(), ensure_ascii=True, indent=2) + "\n", encoding="utf-8")

    def to_dict(self) -> dict:
        return {
            "kind": "linear_rating_model",
            "features": self.features,
            "weights": self.weights,
            "target_mean": self.target_mean,
            "feature_means": self.feature_means,
            "feature_scales": self.feature_scales,
            "star_thresholds": self.star_thresholds,
            "metadata": self.metadata,
        }


@dataclass(frozen=True)
class TaskARatingEnsembleModel:
    weights: dict[str, float]
    model: TaskARatingModel
    metadata: dict
    adaptive_item_count_threshold: int = 10
    adaptive_user_weight: float = 0.40

    def predict(
        self,
        user_profile: UserProfile,
        item_profile: ItemProfile,
        stats: RatingStats,
        user_id: str,
    ) -> float:
        return self.predict_raw(
            user_profile=user_profile,
            item_profile=item_profile,
            stats=stats,
            user_id=user_id,
        )

    def predict_raw(
        self,
        user_profile: UserProfile,
        item_profile: ItemProfile,
        stats: RatingStats,
        user_id: str,
    ) -> float:
        raw = self.model.predict_raw(
            user_profile=user_profile,
            item_profile=item_profile,
            stats=stats,
            user_id=user_id,
        )
        components = {
            "trained_raw": raw,
            "trained_selected": self.model.predict(
                user_profile=user_profile,
                item_profile=item_profile,
                stats=stats,
                user_id=user_id,
            ),
            "trained_rounded": float(round(raw)),
            "calibrated_profile": predict_calibrated_rating(
                user_profile=user_profile,
                item_profile=item_profile,
                stats=stats,
                user_id=user_id,
            ),
            "adaptive_star": float(
                predict_adaptive_star_rating(
                    user_profile=user_profile,
                    item_profile=item_profile,
                    stats=stats,
                    user_id=user_id,
                    item_count_threshold=self.adaptive_item_count_threshold,
                    user_weight=self.adaptive_user_weight,
                )
            ),
        }
        return clamp_rating(
            sum(self.weights[name] * components[name] for name in self.weights if name in components)
        )

    def predict_from_features(self, features: dict[str, float]) -> float:
        components = ensemble_components_from_features(self.model, features)
        return clamp_rating(
            sum(self.weights[name] * components[name] for name in self.weights if name in components)
        )

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self.to_dict(), ensure_ascii=True, indent=2) + "\n", encoding="utf-8")

    def to_dict(self) -> dict:
        return {
            "kind": "rmse_weighted_ensemble",
            "weights": self.weights,
            "base_model": self.model.to_dict(),
            "adaptive_item_count_threshold": self.adaptive_item_count_threshold,
            "adaptive_user_weight": self.adaptive_user_weight,
            "metadata": self.metadata,
        }


def load_task_a_model(path: Path) -> TaskARatingModel | TaskARatingEnsembleModel | None:
    if not path.exists():
        return None
    data = json.loads(path.read_text(encoding="utf-8"))
    if data.get("kind") == "rmse_weighted_ensemble":
        return TaskARatingEnsembleModel(
            weights={key: float(value) for key, value in data["weights"].items()},
            model=_task_a_model_from_dict(data["base_model"]),
            adaptive_item_count_threshold=int(data.get("adaptive_item_count_threshold", 10)),
            adaptive_user_weight=float(data.get("adaptive_user_weight", 0.40)),
            metadata=dict(data.get("metadata") or {}),
        )
    return _task_a_model_from_dict(data)


def _task_a_model_from_dict(data: dict) -> TaskARatingModel:
    return TaskARatingModel(
        features=list(data["features"]),
        weights=[float(value) for value in data["weights"]],
        target_mean=float(data["target_mean"]),
        feature_means=[float(value) for value in data["feature_means"]],
        feature_scales=[float(value) for value in data["feature_scales"]],
        metadata=dict(data.get("metadata") or {}),
        star_thresholds=(
            [float(value) for value in data["star_thresholds"]]
            if data.get("star_thresholds")
            else None
        ),
    )


def model_feature_vector(
    user_profile: UserProfile,
    item_profile: ItemProfile,
    stats: RatingStats,
    user_id: str,
) -> dict[str, float]:
    features = rating_features(user_profile, item_profile, stats=stats, user_id=user_id)
    preference_x_affinity = features["preference_match"] * features["category_affinity"]
    dislike_x_negative_item = features["dislike_match"] * features["item_negative_share"]
    vector_x_affinity = features["vector_match"] * features["category_affinity"]
    reliability_gap = math.log1p(features["item_count"]) - math.log1p(features["user_count"])
    user_item_prior_gap = abs(features["user_prior"] - features["item_prior"])

    return {
        "user_prior": features["user_prior"],
        "item_prior": features["item_prior"],
        "category_prior": features["category_prior"],
        "user_recent_prior": features["user_recent_prior"],
        "user_category_prior": features["user_category_prior"],
        "preference_match": features["preference_match"],
        "dislike_match": features["dislike_match"],
        "category_affinity": features["category_affinity"],
        "vector_match": features["vector_match"],
        "log_user_count": math.log1p(features["user_count"]),
        "log_item_count": math.log1p(features["item_count"]),
        "log_category_count": math.log1p(features["category_count"]),
        "log_user_category_count": math.log1p(features["user_category_count"]),
        "user_std": features["user_std"],
        "item_std": features["item_std"],
        "category_std": features["category_std"],
        "user_positive_share": features["user_positive_share"],
        "user_negative_share": features["user_negative_share"],
        "item_positive_share": features["item_positive_share"],
        "item_negative_share": features["item_negative_share"],
        "category_positive_share": features["category_positive_share"],
        "category_negative_share": features["category_negative_share"],
        "user_global_delta": features["user_global_delta"],
        "item_global_delta": features["item_global_delta"],
        "category_global_delta": features["category_global_delta"],
        "user_category_delta": features["user_category_delta"],
        "item_category_delta": features["item_category_delta"],
        "quality_score": features["quality_score"],
        "review_length_log": math.log1p(features["review_length_mean"]),
        "rating_trend": features["rating_trend"],
        "preference_x_affinity": preference_x_affinity,
        "dislike_x_negative_item": dislike_x_negative_item,
        "vector_x_affinity": vector_x_affinity,
        "reliability_gap": reliability_gap,
        "user_item_prior_gap": user_item_prior_gap,
    }


def fit_linear_rating_model(
    rows: list[tuple[dict[str, float], float]],
    learning_rate: float = 0.03,
    epochs: int = 80,
    l2: float = 0.001,
    loss: str = "mse",
    feature_names: list[str] | None = None,
    name: str = "linear",
    validation_rows: list[tuple[dict[str, float], float]] | None = None,
    calibrate_stars: bool = False,
    fixed_star_thresholds: list[float] | None = None,
    huber_delta: float = 1.0,
) -> TaskARatingModel:
    feature_names = feature_names or MODEL_FEATURES
    if not rows:
        return TaskARatingModel(
            features=feature_names,
            weights=[0.0 for _ in feature_names],
            target_mean=3.5,
            feature_means=[0.0 for _ in feature_names],
            feature_scales=[1.0 for _ in feature_names],
            metadata={"examples": 0, "name": name, "loss": loss},
        )

    design = _prepare_design(rows, feature_names)
    weights = [0.0 for _ in feature_names]
    row_count = len(rows)
    step = learning_rate

    for epoch in range(epochs):
        gradients = [0.0 for _ in feature_names]
        for values, target in zip(design.matrix, design.centered_targets):
            prediction = sum(weight * value for weight, value in zip(weights, values))
            gradient_scale = _loss_gradient(prediction - target, loss, huber_delta)
            for index, value in enumerate(values):
                gradients[index] += gradient_scale * value
        epoch_step = step / (1.0 + 0.015 * epoch)
        for index in range(len(weights)):
            gradients[index] = gradients[index] / row_count + (l2 * weights[index])
            weights[index] -= epoch_step * gradients[index]

    base_model = TaskARatingModel(
        features=feature_names,
        weights=[round(weight, 8) for weight in weights],
        target_mean=round(design.target_mean, 8),
        feature_means=[round(value, 8) for value in design.feature_means],
        feature_scales=[round(value, 8) for value in design.feature_scales],
        metadata={
            "examples": len(rows),
            "epochs": epochs,
            "learning_rate": learning_rate,
            "l2": l2,
            "loss": loss,
            "name": name,
            "feature_count": len(feature_names),
        },
    )
    if fixed_star_thresholds:
        metadata = dict(base_model.metadata)
        metadata["star_thresholds"] = fixed_star_thresholds
        metadata["star_policy"] = "fixed"
        return TaskARatingModel(
            features=base_model.features,
            weights=base_model.weights,
            target_mean=base_model.target_mean,
            feature_means=base_model.feature_means,
            feature_scales=base_model.feature_scales,
            metadata=metadata,
            star_thresholds=fixed_star_thresholds,
        )

    if not calibrate_stars or not validation_rows:
        return base_model

    validation_scores = [predict_from_features(base_model, features) for features, _ in validation_rows]
    validation_targets = [target for _, target in validation_rows]
    thresholds = calibrate_star_thresholds(validation_scores, validation_targets)
    metadata = dict(base_model.metadata)
    metadata["star_thresholds"] = thresholds
    metadata["star_policy"] = "calibrated"
    metadata["calibration_examples"] = len(validation_rows)
    return TaskARatingModel(
        features=base_model.features,
        weights=base_model.weights,
        target_mean=base_model.target_mean,
        feature_means=base_model.feature_means,
        feature_scales=base_model.feature_scales,
        metadata=metadata,
        star_thresholds=thresholds,
    )


def predict_from_features(model: TaskARatingModel, features: dict[str, float]) -> float:
    centered = [
        (features.get(name, 0.0) - mean) / scale
        for name, mean, scale in zip(model.features, model.feature_means, model.feature_scales)
    ]
    raw = model.target_mean + sum(weight * value for weight, value in zip(model.weights, centered))
    return clamp_rating(raw)


def evaluate_model_rows(
    model: TaskARatingModel | TaskARatingEnsembleModel,
    rows: list[tuple[dict[str, float], float]],
) -> dict[str, float]:
    if not rows:
        return {"mae": 0.0, "rmse": 0.0}
    actual = [target for _, target in rows]
    predicted = [_predict_from_feature_row(model, features) for features, _ in rows]
    return {"mae": _mae(actual, predicted), "rmse": _rmse(actual, predicted)}


def fit_rmse_ensemble(
    model: TaskARatingModel,
    validation_rows: list[tuple[dict[str, float], float]],
    weight_step: float = 0.05,
    adaptive_item_count_threshold: int = 10,
    adaptive_user_weight: float = 0.40,
) -> TaskARatingEnsembleModel:
    """Fit a small nonnegative validation-RMSE ensemble.

    The ensemble intentionally stays simple and dependency-free: it blends the
    selected trained model, its rounded/star variants, and calibrated profile
    baselines using a coarse simplex grid.
    """

    component_names = _available_ensemble_components(model, validation_rows)
    if not component_names:
        return TaskARatingEnsembleModel(
            weights={"trained_raw": 1.0},
            model=model,
            adaptive_item_count_threshold=adaptive_item_count_threshold,
            adaptive_user_weight=adaptive_user_weight,
            metadata={
                "name": f"{model.metadata.get('name', 'model')}_rmse_ensemble",
                "component_count": 1,
                "validation_examples": len(validation_rows),
                "validation_mae": 0.0,
                "validation_rmse": 0.0,
            },
        )

    actual = [target for _, target in validation_rows]
    component_predictions = {
        name: [
            ensemble_components_from_features(model, features)[name]
            for features, _ in validation_rows
        ]
        for name in component_names
    }
    best_weights = {component_names[0]: 1.0}
    best_predicted = component_predictions[component_names[0]]
    best_rmse = _rmse(actual, best_predicted)
    best_mae = _mae(actual, best_predicted)

    units = max(1, int(round(1.0 / weight_step)))
    for allocation in _simplex_allocations(len(component_names), units):
        weights = {
            name: allocation[index] / units
            for index, name in enumerate(component_names)
            if allocation[index] > 0
        }
        predicted = [
            sum(weights[name] * component_predictions[name][row_index] for name in weights)
            for row_index in range(len(validation_rows))
        ]
        candidate_rmse = _rmse(actual, predicted)
        candidate_mae = _mae(actual, predicted)
        if candidate_rmse < best_rmse or (
            abs(candidate_rmse - best_rmse) <= 1e-12 and candidate_mae < best_mae
        ):
            best_weights = weights
            best_predicted = predicted
            best_rmse = candidate_rmse
            best_mae = candidate_mae

    metadata = {
        "name": f"{model.metadata.get('name', 'model')}_rmse_ensemble",
        "base_model": model.metadata.get("name", "model"),
        "component_count": len(best_weights),
        "validation_examples": len(validation_rows),
        "validation_mae": round(best_mae, 8),
        "validation_rmse": round(best_rmse, 8),
        "weight_step": weight_step,
        "candidate_components": component_names,
    }
    return TaskARatingEnsembleModel(
        weights={name: round(weight, 6) for name, weight in best_weights.items()},
        model=model,
        adaptive_item_count_threshold=adaptive_item_count_threshold,
        adaptive_user_weight=adaptive_user_weight,
        metadata=metadata,
    )


def ensemble_components_from_features(
    model: TaskARatingModel,
    features: dict[str, float],
) -> dict[str, float]:
    raw = predict_from_features(model, features)
    components = {
        "trained_raw": raw,
        "trained_selected": _predict_from_feature_row(model, features),
        "trained_rounded": float(round(raw)),
    }
    if _has_calibrated_profile_features(features):
        components["calibrated_profile"] = calibrated_profile_from_features(features)
    if "_baseline_adaptive_star" in features:
        components["adaptive_star"] = float(features["_baseline_adaptive_star"])
    return components


def calibrated_profile_from_features(features: dict[str, float]) -> float:
    weights = DEFAULT_RATING_WEIGHTS
    prior = (
        weights.user_prior * features["user_prior"]
        + weights.item_prior * features["item_prior"]
        + weights.category_prior * features["category_prior"]
    ) / (weights.user_prior + weights.item_prior + weights.category_prior)
    adjustment = weights.profile_adjustment * (
        weights.preference_match * features["preference_match"]
        + weights.vector_match * features["vector_match"]
        + weights.category_affinity * features["category_affinity"]
        - weights.dislike_penalty * features["dislike_match"]
    )
    return clamp_rating(prior + adjustment)


def _has_calibrated_profile_features(features: dict[str, float]) -> bool:
    return {
        "user_prior",
        "item_prior",
        "category_prior",
        "preference_match",
        "vector_match",
        "category_affinity",
        "dislike_match",
    } <= set(features)


def _predict_from_feature_row(
    model: TaskARatingModel | TaskARatingEnsembleModel,
    features: dict[str, float],
) -> float:
    if isinstance(model, TaskARatingEnsembleModel):
        return model.predict_from_features(features)
    raw = predict_from_features(model, features)
    if model.star_thresholds:
        return float(apply_star_thresholds(raw, model.star_thresholds))
    return raw


def _available_ensemble_components(
    model: TaskARatingModel,
    validation_rows: list[tuple[dict[str, float], float]],
) -> list[str]:
    if not validation_rows:
        return ["trained_raw"]
    sample_components = ensemble_components_from_features(model, validation_rows[0][0])
    return [name for name in sample_components if name in {"trained_raw", "trained_selected", "trained_rounded", "calibrated_profile", "adaptive_star"}]


def _simplex_allocations(parts: int, units: int) -> list[tuple[int, ...]]:
    if parts == 1:
        return [(units,)]
    allocations = []
    for value in range(units + 1):
        for tail in _simplex_allocations(parts - 1, units - value):
            allocations.append((value, *tail))
    return allocations


def calibrate_star_thresholds(
    scores: list[float],
    targets: list[float],
    max_bins: int = 300,
) -> list[float]:
    if len(scores) < 5:
        return [1.5, 2.5, 3.5, 4.5]
    pairs = sorted(zip(scores, targets), key=lambda pair: pair[0])
    buckets = _bucket_pairs(pairs, max_bins=max_bins)
    bucket_count = len(buckets)
    prefix_counts = [[0, 0, 0, 0, 0, 0]]
    for _, counts in buckets:
        previous = prefix_counts[-1]
        prefix_counts.append([previous[index] + counts[index] for index in range(6)])

    def segment_cost(start: int, end: int, star: int) -> float:
        counts = [
            prefix_counts[end][rating] - prefix_counts[start][rating]
            for rating in range(1, 6)
        ]
        return float(sum(count * abs(rating - star) for rating, count in enumerate(counts, start=1)))

    classes = 5
    infinity = float("inf")
    dp = [[infinity] * (bucket_count + 1) for _ in range(classes + 1)]
    back = [[0] * (bucket_count + 1) for _ in range(classes + 1)]
    dp[0][0] = 0.0

    for star in range(1, classes + 1):
        for end in range(star, bucket_count + 1):
            best_cost = infinity
            best_start = star - 1
            for start in range(star - 1, end):
                cost = dp[star - 1][start] + segment_cost(start, end, star)
                if cost < best_cost:
                    best_cost = cost
                    best_start = start
            dp[star][end] = best_cost
            back[star][end] = best_start

    boundaries = []
    end = bucket_count
    for star in range(classes, 1, -1):
        start = back[star][end]
        boundaries.append(start)
        end = start
    boundaries = list(reversed(boundaries))

    thresholds = []
    for boundary in boundaries:
        if boundary <= 0:
            thresholds.append(1.0)
        elif boundary >= bucket_count:
            thresholds.append(5.0)
        else:
            thresholds.append((buckets[boundary - 1][0] + buckets[boundary][0]) / 2)
    return _monotonic_thresholds(thresholds)


def apply_star_thresholds(score: float, thresholds: list[float]) -> int:
    for index, threshold in enumerate(thresholds, start=1):
        if score < threshold:
            return index
    return 5


@dataclass(frozen=True)
class _DesignMatrix:
    matrix: list[list[float]]
    centered_targets: list[float]
    target_mean: float
    feature_means: list[float]
    feature_scales: list[float]


def _prepare_design(rows: list[tuple[dict[str, float], float]], features: list[str]) -> _DesignMatrix:
    targets = [target for _, target in rows]
    target_mean = sum(targets) / len(targets)
    feature_means = []
    feature_scales = []
    for name in features:
        values = [row_features.get(name, 0.0) for row_features, _ in rows]
        mean = sum(values) / len(values)
        variance = sum((value - mean) ** 2 for value in values) / len(values)
        feature_means.append(mean)
        feature_scales.append(max(variance ** 0.5, 1e-6))

    matrix = [
        [
            (row_features.get(name, 0.0) - mean) / scale
            for name, mean, scale in zip(features, feature_means, feature_scales)
        ]
        for row_features, _ in rows
    ]
    return _DesignMatrix(
        matrix=matrix,
        centered_targets=[target - target_mean for target in targets],
        target_mean=target_mean,
        feature_means=feature_means,
        feature_scales=feature_scales,
    )


def _loss_gradient(error: float, loss: str, huber_delta: float) -> float:
    if loss == "mae":
        if error > 0:
            return 1.0
        if error < 0:
            return -1.0
        return 0.0
    if loss == "huber":
        if abs(error) <= huber_delta:
            return error
        return huber_delta if error > 0 else -huber_delta
    return error


def _bucket_pairs(
    pairs: list[tuple[float, float]],
    max_bins: int,
) -> list[tuple[float, list[int]]]:
    if len(pairs) <= max_bins:
        return [(score, _rating_counts([target])) for score, target in pairs]
    bucket_size = max(1, math.ceil(len(pairs) / max_bins))
    buckets = []
    for start in range(0, len(pairs), bucket_size):
        chunk = pairs[start : start + bucket_size]
        score = sum(pair[0] for pair in chunk) / len(chunk)
        buckets.append((score, _rating_counts([target for _, target in chunk])))
    return buckets


def _rating_counts(targets: list[float]) -> list[int]:
    counts = [0, 0, 0, 0, 0, 0]
    for target in targets:
        rating = min(max(int(round(target)), 1), 5)
        counts[rating] += 1
    return counts


def _monotonic_thresholds(thresholds: list[float]) -> list[float]:
    output = []
    previous = 1.0
    for threshold in thresholds:
        value = min(max(threshold, previous + 1e-4), 5.0)
        output.append(round(value, 6))
        previous = value
    return output


def _mae(actual: list[float], predicted: list[float]) -> float:
    return sum(abs(a - p) for a, p in zip(actual, predicted)) / len(actual)


def _rmse(actual: list[float], predicted: list[float]) -> float:
    return (sum((a - p) ** 2 for a, p in zip(actual, predicted)) / len(actual)) ** 0.5
