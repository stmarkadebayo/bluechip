from __future__ import annotations

import json
from collections import defaultdict
from dataclasses import asdict, dataclass, field
from pathlib import Path

from app.models.schemas import ItemProfile, UserProfile
from app.services.ranking.features import overlap
from app.services.retrieval.embeddings import cosine_similarity


@dataclass(frozen=True)
class RatingStats:
    global_mean: float = 3.5
    global_std: float = 0.0
    global_positive_share: float = 0.0
    global_negative_share: float = 0.0
    user_means: dict[str, float] = field(default_factory=dict)
    user_counts: dict[str, int] = field(default_factory=dict)
    user_stds: dict[str, float] = field(default_factory=dict)
    user_recent_means: dict[str, float] = field(default_factory=dict)
    user_positive_shares: dict[str, float] = field(default_factory=dict)
    user_negative_shares: dict[str, float] = field(default_factory=dict)
    item_means: dict[str, float] = field(default_factory=dict)
    item_counts: dict[str, int] = field(default_factory=dict)
    item_stds: dict[str, float] = field(default_factory=dict)
    item_positive_shares: dict[str, float] = field(default_factory=dict)
    item_negative_shares: dict[str, float] = field(default_factory=dict)
    category_means: dict[str, float] = field(default_factory=dict)
    category_counts: dict[str, int] = field(default_factory=dict)
    category_stds: dict[str, float] = field(default_factory=dict)
    category_positive_shares: dict[str, float] = field(default_factory=dict)
    category_negative_shares: dict[str, float] = field(default_factory=dict)
    user_category_means: dict[str, float] = field(default_factory=dict)
    user_category_counts: dict[str, int] = field(default_factory=dict)


@dataclass(frozen=True)
class RatingWeights:
    user_prior: float = 0.42
    item_prior: float = 0.38
    category_prior: float = 0.14
    profile_adjustment: float = 0.06
    preference_match: float = 0.28
    vector_match: float = 0.10
    category_affinity: float = 0.18
    dislike_penalty: float = 0.42


DEFAULT_RATING_WEIGHTS = RatingWeights()


def build_rating_stats(train: list[dict]) -> RatingStats:
    global_values = [float(row["rating"]) for row in train if row.get("rating")]
    global_mean = sum(global_values) / len(global_values) if global_values else 3.5
    user_values: dict[str, list[float]] = defaultdict(list)
    user_rows: dict[str, list[dict]] = defaultdict(list)
    item_values: dict[str, list[float]] = defaultdict(list)
    category_values: dict[str, list[float]] = defaultdict(list)
    user_category_values: dict[str, list[float]] = defaultdict(list)

    for row in train:
        rating = float(row["rating"])
        user_id = str(row["user_id"])
        item_id = str(row["item_id"])
        category = str(row.get("category") or "unknown")
        user_values[user_id].append(rating)
        user_rows[user_id].append(row)
        item_values[item_id].append(rating)
        user_category_values[_user_category_key(user_id, category)].append(rating)
        category = str(row.get("category") or "unknown")
        category_values[category].append(rating)

    return RatingStats(
        global_mean=global_mean,
        global_std=_std(global_values),
        global_positive_share=_share(global_values, lower_bound=4),
        global_negative_share=_share(global_values, upper_bound=2),
        user_means=_means(user_values),
        user_counts={key: len(values) for key, values in user_values.items()},
        user_stds=_stds(user_values),
        user_recent_means=_recent_means(user_rows),
        user_positive_shares=_shares(user_values, lower_bound=4),
        user_negative_shares=_shares(user_values, upper_bound=2),
        item_means=_means(item_values),
        item_counts={key: len(values) for key, values in item_values.items()},
        item_stds=_stds(item_values),
        item_positive_shares=_shares(item_values, lower_bound=4),
        item_negative_shares=_shares(item_values, upper_bound=2),
        category_means=_means(category_values),
        category_counts={key: len(values) for key, values in category_values.items()},
        category_stds=_stds(category_values),
        category_positive_shares=_shares(category_values, lower_bound=4),
        category_negative_shares=_shares(category_values, upper_bound=2),
        user_category_means=_means(user_category_values),
        user_category_counts={key: len(values) for key, values in user_category_values.items()},
    )


def save_rating_stats(stats: RatingStats, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "kind": "rating_stats",
        "stats": asdict(stats),
    }
    path.write_text(json.dumps(payload, ensure_ascii=True, indent=2) + "\n", encoding="utf-8")


def load_rating_stats(path: Path) -> RatingStats | None:
    if not path.exists():
        return None
    if path.suffix == ".jsonl":
        return None
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    data = payload.get("stats", payload)
    if not isinstance(data, dict):
        return None
    return RatingStats(
        global_mean=float(data.get("global_mean", 3.5)),
        global_std=float(data.get("global_std", 0.0)),
        global_positive_share=float(data.get("global_positive_share", 0.0)),
        global_negative_share=float(data.get("global_negative_share", 0.0)),
        user_means=_float_dict(data.get("user_means")),
        user_counts=_int_dict(data.get("user_counts")),
        user_stds=_float_dict(data.get("user_stds")),
        user_recent_means=_float_dict(data.get("user_recent_means")),
        user_positive_shares=_float_dict(data.get("user_positive_shares")),
        user_negative_shares=_float_dict(data.get("user_negative_shares")),
        item_means=_float_dict(data.get("item_means")),
        item_counts=_int_dict(data.get("item_counts")),
        item_stds=_float_dict(data.get("item_stds")),
        item_positive_shares=_float_dict(data.get("item_positive_shares")),
        item_negative_shares=_float_dict(data.get("item_negative_shares")),
        category_means=_float_dict(data.get("category_means")),
        category_counts=_int_dict(data.get("category_counts")),
        category_stds=_float_dict(data.get("category_stds")),
        category_positive_shares=_float_dict(data.get("category_positive_shares")),
        category_negative_shares=_float_dict(data.get("category_negative_shares")),
        user_category_means=_float_dict(data.get("user_category_means")),
        user_category_counts=_int_dict(data.get("user_category_counts")),
    )


def predict_calibrated_rating(
    user_profile: UserProfile,
    item_profile: ItemProfile,
    stats: RatingStats | None = None,
    user_id: str | None = None,
    weights: RatingWeights = DEFAULT_RATING_WEIGHTS,
) -> float:
    features = rating_features(user_profile, item_profile, stats=stats, user_id=user_id)
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


def predict_profile_heuristic_rating(
    user_profile: UserProfile,
    item_profile: ItemProfile,
) -> int:
    features = rating_features(user_profile, item_profile)
    item_rating_adjustment = 0.0
    if item_profile.average_rating is not None:
        item_rating_adjustment = (item_profile.average_rating - 3.5) * 0.25
    raw = (
        user_profile.average_rating
        + item_rating_adjustment
        + (0.45 * min(features["preference_match"], 0.75))
        + (0.15 * features["vector_match"])
        + (0.25 if item_profile.category in user_profile.preferred_categories else 0.0)
        + (0.50 * features["category_affinity"])
        - (0.65 * min(features["dislike_match"], 0.75))
    )
    return int(round(clamp_rating(raw)))


def predict_adaptive_star_rating(
    user_profile: UserProfile,
    item_profile: ItemProfile,
    stats: RatingStats,
    user_id: str,
    item_count_threshold: int = 10,
    user_weight: float = 0.40,
) -> int:
    item_count = stats.item_counts.get(item_profile.item_id, 0)
    if item_count < item_count_threshold:
        return predict_profile_heuristic_rating(user_profile, item_profile)

    user_mean = stats.user_means.get(user_id, user_profile.average_rating)
    item_mean = stats.item_means.get(item_profile.item_id, item_profile.average_rating or stats.global_mean)
    item_weight = 1.0 - user_weight
    return int(round(clamp_rating((user_weight * user_mean) + (item_weight * item_mean))))


def rating_features(
    user_profile: UserProfile,
    item_profile: ItemProfile,
    stats: RatingStats | None = None,
    user_id: str | None = None,
) -> dict[str, float]:
    global_mean = stats.global_mean if stats else 3.5
    user_count = user_profile.evidence_count
    user_mean = user_profile.average_rating
    if stats and user_id:
        user_mean = stats.user_means.get(user_id, user_mean)
        user_count = stats.user_counts.get(user_id, user_count)

    item_count = item_profile.popularity
    item_mean = item_profile.average_rating or global_mean
    if stats:
        item_mean = stats.item_means.get(item_profile.item_id, item_mean)
        item_count = stats.item_counts.get(item_profile.item_id, item_count)

    category_count = 0
    category_mean = global_mean
    category_std = stats.global_std if stats else 0.0
    category_positive_share = stats.global_positive_share if stats else 0.0
    category_negative_share = stats.global_negative_share if stats else 0.0
    if stats:
        category_mean = stats.category_means.get(item_profile.category, global_mean)
        category_count = stats.category_counts.get(item_profile.category, 0)
        category_std = stats.category_stds.get(item_profile.category, category_std)
        category_positive_share = stats.category_positive_shares.get(
            item_profile.category, category_positive_share
        )
        category_negative_share = stats.category_negative_shares.get(
            item_profile.category, category_negative_share
        )

    user_std = user_profile.rating_std
    user_recent_mean = user_profile.recent_average_rating
    user_positive_share = user_profile.positive_rating_share
    user_negative_share = user_profile.negative_rating_share
    if stats and user_id:
        user_std = stats.user_stds.get(user_id, user_std)
        user_recent_mean = stats.user_recent_means.get(user_id, user_recent_mean)
        user_positive_share = stats.user_positive_shares.get(user_id, user_positive_share)
        user_negative_share = stats.user_negative_shares.get(user_id, user_negative_share)

    item_std = stats.global_std if stats else 0.0
    item_positive_share = stats.global_positive_share if stats else 0.0
    item_negative_share = stats.global_negative_share if stats else 0.0
    if stats:
        item_std = stats.item_stds.get(item_profile.item_id, item_std)
        item_positive_share = stats.item_positive_shares.get(
            item_profile.item_id, item_positive_share
        )
        item_negative_share = stats.item_negative_shares.get(
            item_profile.item_id, item_negative_share
        )

    user_category_count = 0
    user_category_mean = user_mean
    if stats and user_id:
        user_category_key = _user_category_key(user_id, item_profile.category)
        user_category_mean = stats.user_category_means.get(user_category_key, user_mean)
        user_category_count = stats.user_category_counts.get(user_category_key, 0)

    preference_match = overlap(
        user_profile.preferred_terms + user_profile.positive_aspects,
        item_profile.terms + item_profile.positive_aspects,
    )
    dislike_match = overlap(
        user_profile.disliked_terms + user_profile.negative_aspects,
        item_profile.terms + item_profile.negative_aspects,
    )
    category_affinity = user_profile.category_affinity.get(item_profile.category, 0.0)
    if item_profile.category in user_profile.preferred_categories:
        category_affinity = max(category_affinity, 0.25)

    return {
        "global_mean": global_mean,
        "user_prior": shrinkage_mean(user_mean, user_count, global_mean, prior_weight=4.0),
        "user_recent_prior": shrinkage_mean(
            user_recent_mean, min(user_count, 3), global_mean, prior_weight=3.0
        ),
        "item_prior": shrinkage_mean(item_mean, item_count, global_mean, prior_weight=8.0),
        "category_prior": shrinkage_mean(category_mean, category_count, global_mean, prior_weight=100.0),
        "user_category_prior": shrinkage_mean(
            user_category_mean,
            user_category_count,
            category_mean,
            prior_weight=4.0,
        ),
        "preference_match": preference_match,
        "dislike_match": dislike_match,
        "category_affinity": category_affinity,
        "vector_match": max(cosine_similarity(user_profile.embedding, item_profile.embedding), 0.0),
        "user_count": float(user_count),
        "item_count": float(item_count),
        "category_count": float(category_count),
        "user_category_count": float(user_category_count),
        "user_std": user_std,
        "item_std": item_std,
        "category_std": category_std,
        "user_positive_share": user_positive_share,
        "user_negative_share": user_negative_share,
        "item_positive_share": item_positive_share,
        "item_negative_share": item_negative_share,
        "category_positive_share": category_positive_share,
        "category_negative_share": category_negative_share,
        "user_global_delta": user_mean - global_mean,
        "item_global_delta": item_mean - global_mean,
        "category_global_delta": category_mean - global_mean,
        "user_category_delta": user_category_mean - category_mean,
        "item_category_delta": item_mean - category_mean,
        "quality_score": item_profile.quality_score,
        "review_length_mean": user_profile.review_length_mean,
        "rating_trend": user_profile.rating_trend,
    }


def shrinkage_mean(mean: float, count: int, prior: float, prior_weight: float) -> float:
    if count <= 0:
        return prior
    return ((mean * count) + (prior * prior_weight)) / (count + prior_weight)


def clamp_rating(value: float) -> float:
    return min(max(value, 1.0), 5.0)


def _means(values_by_key: dict[str, list[float]]) -> dict[str, float]:
    return {
        key: sum(values) / len(values)
        for key, values in values_by_key.items()
        if values
    }


def _float_dict(raw: object) -> dict[str, float]:
    if not isinstance(raw, dict):
        return {}
    return {str(key): float(value) for key, value in raw.items()}


def _int_dict(raw: object) -> dict[str, int]:
    if not isinstance(raw, dict):
        return {}
    return {str(key): int(value) for key, value in raw.items()}


def _std(values: list[float]) -> float:
    if len(values) < 2:
        return 0.0
    mean = sum(values) / len(values)
    return (sum((value - mean) ** 2 for value in values) / len(values)) ** 0.5


def _stds(values_by_key: dict[str, list[float]]) -> dict[str, float]:
    return {key: _std(values) for key, values in values_by_key.items() if values}


def _share(
    values: list[float],
    lower_bound: float | None = None,
    upper_bound: float | None = None,
) -> float:
    if not values:
        return 0.0
    count = 0
    for value in values:
        if lower_bound is not None and value >= lower_bound:
            count += 1
        elif upper_bound is not None and value <= upper_bound:
            count += 1
    return count / len(values)


def _shares(
    values_by_key: dict[str, list[float]],
    lower_bound: float | None = None,
    upper_bound: float | None = None,
) -> dict[str, float]:
    return {
        key: _share(values, lower_bound=lower_bound, upper_bound=upper_bound)
        for key, values in values_by_key.items()
        if values
    }


def _recent_means(rows_by_user: dict[str, list[dict]]) -> dict[str, float]:
    output = {}
    for user_id, rows in rows_by_user.items():
        ordered = sorted(rows, key=lambda row: int(row.get("timestamp") or 0))
        recent = [float(row["rating"]) for row in ordered[-3:]]
        if recent:
            output[user_id] = sum(recent) / len(recent)
    return output


def _user_category_key(user_id: str, category: str) -> str:
    return f"{user_id}\t{category}"
