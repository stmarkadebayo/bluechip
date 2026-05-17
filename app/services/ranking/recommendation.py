from __future__ import annotations

import math
import re

from app.models.schemas import Item, RecommendationItem, UserProfile
from app.services.profiling.item_profile import build_item_profile


def rank_candidates(
    user_profile: UserProfile,
    context: str,
    candidate_items: list[Item],
    limit: int,
) -> list[RecommendationItem]:
    ranked = []
    context_terms = _terms(context)
    profiled_items = [(item, build_item_profile(item)) for item in candidate_items]
    max_popularity = max((profile.popularity for _, profile in profiled_items), default=0)
    personalization_weight = min(max((user_profile.confidence - 0.25) / 0.70, 0.05), 0.85)

    for item, profile in profiled_items:
        preference_match = _overlap(
            user_profile.preferred_terms + user_profile.positive_aspects,
            profile.terms + profile.positive_aspects,
        )
        matched_signals = _matched_terms(
            user_profile.preferred_terms + user_profile.positive_aspects,
            profile.terms + profile.positive_aspects,
        )
        dislike_match = _overlap(
            user_profile.disliked_terms + user_profile.negative_aspects,
            profile.terms + profile.negative_aspects,
        )
        context_match = _overlap(context_terms, profile.terms + profile.positive_aspects)
        item_quality = profile.quality_score
        popularity = _popularity_score(profile.popularity, max_popularity)
        category_match = _category_match(user_profile, profile.category)
        novelty = 0.65 if item.item_id not in _seen_item_ids(user_profile) else 0.1

        base_score = 0.85 * popularity + 0.10 * item_quality + 0.05 * novelty
        personalized_score = (
            0.35 * preference_match
            + 0.25 * context_match
            + 0.25 * category_match
            + 0.15 * item_quality
            - 0.25 * dislike_match
        )
        score = (
            (1 - personalization_weight) * base_score
            + personalization_weight * personalized_score
        )

        tradeoffs = "No major tradeoff detected from available metadata."
        if item.metadata.get("price") in {"high", "premium", "expensive"}:
            tradeoffs = "Price may be higher than the user's usual preference."
        elif profile.negative_aspects:
            tradeoffs = "Potential mismatch: " + ", ".join(profile.negative_aspects[:3])

        ranked.append(
            RecommendationItem(
                rank=0,
                item_id=item.item_id,
                name=item.name,
                score=round(min(max(score, 0), 1), 2),
                reason="",
                tradeoffs=tradeoffs,
                signals=profile.signals,
                matched_signals=matched_signals,
            )
        )

    ranked.sort(key=lambda item: item.score, reverse=True)
    limited = ranked[:limit]
    for index, item in enumerate(limited, start=1):
        item.rank = index
    return limited


def _terms(text: str) -> list[str]:
    return re.findall(r"[a-zA-Z][a-zA-Z0-9_-]{2,}", text.lower())


def _overlap(left: list[str], right: list[str]) -> float:
    if not left or not right:
        return 0.0
    right_set = set(right)
    return min(sum(1 for term in left if term in right_set) / max(len(left), 1), 1.0)


def _matched_terms(left: list[str], right: list[str], limit: int = 5) -> list[str]:
    right_set = set(right)
    seen = []
    for term in left:
        if term in right_set and term not in seen:
            seen.append(term)
        if len(seen) >= limit:
            break
    return seen


def _category_match(user_profile: UserProfile, category: str) -> float:
    if category in user_profile.preferred_categories:
        return 1.0
    return max(user_profile.category_affinity.get(category, 0.0), 0.0)


def _popularity_score(popularity: int, max_popularity: int) -> float:
    if popularity <= 0 or max_popularity <= 0:
        return 0.0
    return math.log1p(popularity) / math.log1p(max_popularity)


def _seen_item_ids(user_profile: UserProfile) -> set[str]:
    explicit = set(user_profile.seen_item_ids)
    legacy = {
        signal.removeprefix("seen item: ")
        for signal in user_profile.signals
        if signal.startswith("seen item: ")
    }
    return explicit | legacy
