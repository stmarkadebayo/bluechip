from __future__ import annotations

from app.models.schemas import ItemProfile, RatingPrediction, UserProfile


def predict_rating(user_profile: UserProfile, item_profile: ItemProfile) -> RatingPrediction:
    preference_match = _overlap_score(
        user_profile.preferred_terms + user_profile.positive_aspects,
        item_profile.terms + item_profile.positive_aspects,
    )
    dislike_match = _overlap_score(
        user_profile.disliked_terms + user_profile.negative_aspects,
        item_profile.terms + item_profile.negative_aspects,
    )
    category_affinity = user_profile.category_affinity.get(item_profile.category, 0.0)
    category_bonus = 0.25 if item_profile.category in user_profile.preferred_categories else 0.0

    raw = (
        user_profile.average_rating
        + (item_profile.quality_score - 0.5)
        + preference_match
        + category_bonus
        + category_affinity
        - dislike_match
    )
    predicted = round(min(max(raw, 1), 5))
    confidence = min(
        max(
            0.25
            + user_profile.confidence * 0.35
            + preference_match * 0.25
            + (item_profile.quality_score * 0.15),
            0,
        ),
        0.95,
    )

    return RatingPrediction(
        predicted_rating=int(predicted),
        confidence=round(confidence, 2),
        user_signals=user_profile.signals,
        item_signals=item_profile.signals,
    )


def _overlap_score(left: list[str], right: list[str]) -> float:
    if not left or not right:
        return 0.0
    right_set = set(right)
    return min(sum(1 for term in left if term in right_set) / max(len(left), 1), 0.75)
