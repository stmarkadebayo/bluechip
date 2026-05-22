from __future__ import annotations

import re
from dataclasses import dataclass, field, replace
from typing import Any

from app.models.schemas import Item, ItemProfile, RecommendationItem, UserProfile
from app.services.ranking.context_intents import (
    context_category_hint,
    context_intent_boost,
    context_intent_penalty,
    is_gift_category,
)
from app.services.ranking.features import matched_terms, ranker_features, weighted_score
from app.services.profiling.item_profile import build_item_profile


@dataclass(frozen=True)
class RecommendationWeights:
    preference: float = 0.24
    context: float = 0.16
    category: float = 0.18
    vector: float = 0.18
    quality: float = 0.14
    popularity: float = 0.18
    novelty: float = 0.06
    confidence: float = 0.04
    aspect: float = 0.16
    sequential: float = 0.12
    evidence_graph: float = 0.12
    nigerian_context: float = 0.04
    collaborative: float = 0.22
    retrieval: float = 0.04
    source_diversity: float = 0.0
    dislike_penalty: float = 0.30

    def as_feature_weights(self) -> dict[str, float]:
        return {
            "preference_match": self.preference,
            "context_match": self.context,
            "category_match": self.category,
            "vector_match": self.vector,
            "item_quality": self.quality,
            "popularity": self.popularity,
            "novelty": self.novelty,
            "confidence": self.confidence,
            "aspect_match": self.aspect,
            "sequential_match": self.sequential,
            "evidence_graph_match": self.evidence_graph,
            "nigerian_context_match": self.nigerian_context,
            "collaborative_match": self.collaborative,
            "retrieval_match": self.retrieval,
            "source_diversity": self.source_diversity,
            "dislike_match": -self.dislike_penalty,
        }


@dataclass(frozen=True)
class RecommendationPolicy:
    name: str = "balanced"
    weights: RecommendationWeights = field(default_factory=RecommendationWeights)
    personalization_floor: float = 0.02
    personalization_ceiling: float = 0.65
    base_popularity: float = 0.85
    base_quality: float = 0.10
    base_novelty: float = 0.05
    feedback_rejection_penalty: float = 0.45
    feedback_acceptance_boost: float = 0.06


def adaptive_recommendation_policy(
    user_profile: UserProfile,
    context: str,
    strategy: str = "history_aware",
    preference_analysis: Any | None = None,
) -> RecommendationPolicy:
    """Build a ranking policy that changes with evidence depth and intent.

    The default ranker intentionally remains deterministic, but the weights
    should not be one-size-fits-all. Sparse profiles need more context,
    semantic, and category evidence; rich profiles can trust collaborative and
    preference features more heavily.
    """
    policy = RecommendationPolicy()
    weights = policy.weights
    names = ["balanced"]
    context_terms = set(_terms(context))
    sparse_profile = user_profile.evidence_count <= 2 or strategy == "cold_start"

    if sparse_profile:
        names.append("cold_start")
        weights = replace(
            weights,
            preference=max(weights.preference, 0.30),
            context=max(weights.context, 0.28),
            category=max(weights.category, 0.26),
            vector=max(weights.vector, 0.24),
            aspect=max(weights.aspect, 0.22),
            retrieval=max(weights.retrieval, 0.10),
            source_diversity=max(weights.source_diversity, 0.08),
            popularity=min(weights.popularity, 0.12),
        )
        policy = replace(
            policy,
            personalization_floor=0.22,
            personalization_ceiling=0.72,
            base_popularity=0.52,
            base_quality=0.28,
            base_novelty=0.20,
        )

    if context_terms:
        names.append("contextual")
        weights = replace(
            weights,
            context=max(weights.context, 0.30),
            aspect=max(weights.aspect, 0.20),
            retrieval=max(weights.retrieval, 0.08),
        )

    price_tolerance = str(getattr(preference_analysis, "price_tolerance", "")).lower()
    quality_priority = str(getattr(preference_analysis, "quality_priority", "")).lower()
    exploration = str(getattr(preference_analysis, "exploration_openness", "")).lower()

    budget_terms = {"budget", "cheap", "affordable", "value", "price", "fees"}
    if price_tolerance == "high" or context_terms & budget_terms:
        names.append("budget_sensitive")
        weights = replace(
            weights,
            dislike_penalty=max(weights.dislike_penalty, 0.40),
            context=max(weights.context, 0.32),
            aspect=max(weights.aspect, 0.24),
            popularity=min(weights.popularity, 0.12),
        )

    if quality_priority in {"yes", "high", "quality"}:
        names.append("quality_first")
        weights = replace(
            weights,
            quality=max(weights.quality, 0.22),
            aspect=max(weights.aspect, 0.24),
            popularity=min(weights.popularity, 0.14),
        )

    if exploration == "high" or user_profile.rating_std > 1.0:
        names.append("exploratory")
        weights = replace(
            weights,
            novelty=max(weights.novelty, 0.12),
            source_diversity=max(weights.source_diversity, 0.10),
            vector=max(weights.vector, 0.24),
        )

    if user_profile.evidence_count >= 5:
        names.append("history_rich")
        weights = replace(
            weights,
            collaborative=max(weights.collaborative, 0.26),
            sequential=max(weights.sequential, 0.16),
            evidence_graph=max(weights.evidence_graph, 0.16),
        )

    return replace(policy, name="+".join(dict.fromkeys(names)), weights=weights)


def rank_candidates(
    user_profile: UserProfile,
    context: str,
    candidate_items: list[Item],
    limit: int,
    weights: RecommendationWeights | None = None,
    policy: RecommendationPolicy | None = None,
    candidate_sources: dict[str, list[str]] | None = None,
    candidate_source_scores: dict[str, dict[str, float]] | None = None,
    item_profile_cache: dict[str, ItemProfile] | None = None,
    accepted_item_ids: list[str] | None = None,
    rejected_item_ids: list[str] | None = None,
) -> list[RecommendationItem]:
    policy = policy or RecommendationPolicy()
    if weights is not None:
        policy = replace(policy, weights=weights)
    feature_weights = policy.weights.as_feature_weights()
    candidate_sources = candidate_sources or {}
    candidate_source_scores = candidate_source_scores or {}
    accepted = set(accepted_item_ids or [])
    rejected = set(rejected_item_ids or [])
    ranked: list[tuple[float, RecommendationItem]] = []
    context_terms = _terms(context)
    profiled_items = [
        (item, _item_profile(item, item_profile_cache))
        for item in candidate_items
    ]
    max_popularity = max((profile.popularity for _, profile in profiled_items), default=0)
    personalization_weight = min(
        max((user_profile.confidence - 0.45) / 0.80, policy.personalization_floor),
        policy.personalization_ceiling,
    )

    for item, profile in profiled_items:
        features = ranker_features(
            user_profile=user_profile,
            item_profile=profile,
            context_terms=context_terms,
            max_popularity=max_popularity,
            seen_item_ids=_seen_item_ids(user_profile),
            source_scores=candidate_source_scores.get(item.item_id, {}),
        )
        matched_signals = matched_terms(
            user_profile.preferred_terms + user_profile.positive_aspects,
            profile.terms + profile.positive_aspects,
        )

        base_score = (
            policy.base_popularity * features["popularity"]
            + policy.base_quality * features["item_quality"]
            + policy.base_novelty * features["novelty"]
        )
        personalized_score = weighted_score(features, feature_weights)
        context_penalty = _context_penalty(context_terms, item, profile.negative_aspects)
        context_category_penalty = _context_category_penalty(context_terms, item)
        context_category_boost = _context_category_boost(context_terms, item)
        context_intent_boost_value = context_intent_boost(context_terms, item)
        context_intent_penalty_value = context_intent_penalty(context_terms, item)
        score = (
            (1 - personalization_weight) * base_score
            + personalization_weight * personalized_score
            + context_category_boost
            + context_intent_boost_value
            - context_penalty
            - context_category_penalty
            - context_intent_penalty_value
        )
        feedback_acceptance_boost = policy.feedback_acceptance_boost if item.item_id in accepted else 0.0
        feedback_rejection_penalty = policy.feedback_rejection_penalty if item.item_id in rejected else 0.0
        score += feedback_acceptance_boost - feedback_rejection_penalty
        score_components = {name: round(value, 4) for name, value in features.items()}
        score_components.update(
            {
                "context_penalty": round(context_penalty, 4),
                "context_category_penalty": round(context_category_penalty, 4),
                "context_category_boost": round(context_category_boost, 4),
                "context_intent_boost": round(context_intent_boost_value, 4),
                "context_intent_penalty": round(context_intent_penalty_value, 4),
                "personalization_weight": round(personalization_weight, 4),
                "feedback_acceptance_boost": round(feedback_acceptance_boost, 4),
                "feedback_rejection_penalty": round(feedback_rejection_penalty, 4),
            }
        )

        tradeoffs = "No obvious drawback stands out from the available metadata."
        if item.metadata.get("price") in {"high", "premium", "expensive"}:
            tradeoffs = "The main watch-out is price; it may sit above this user's usual budget."
        elif profile.negative_aspects:
            tradeoffs = "Watch out for " + ", ".join(profile.negative_aspects[:3]) + "."

        display_score = round(min(max(score, 0), 1), 2)
        score_components["raw_score"] = round(score, 4)

        ranked.append(
            (
                score,
                RecommendationItem(
                    rank=0,
                    item_id=item.item_id,
                    name=item.name,
                    score=display_score,
                    reason="",
                    tradeoffs=tradeoffs,
                    signals=profile.signals,
                    matched_signals=matched_signals,
                    candidate_sources=candidate_sources.get(item.item_id, []),
                    retrieval_scores={
                        name: round(value, 4)
                        for name, value in candidate_source_scores.get(item.item_id, {}).items()
                    },
                    score_components=score_components,
                ),
            )
        )

    ranked.sort(key=lambda item: item[0], reverse=True)
    limited = [item for _, item in ranked[:limit]]
    for index, item in enumerate(limited, start=1):
        item.rank = index
    return limited


def _terms(text: str) -> list[str]:
    return re.findall(r"[a-zA-Z][a-zA-Z0-9_-]{2,}", text.lower())


def _context_penalty(context_terms: list[str], item: Item, negative_aspects: list[str]) -> float:
    terms = set(context_terms)
    penalty = 0.0
    price = str(item.metadata.get("price") or "").lower()
    if "expensive" in terms and price in {"high", "premium", "expensive"}:
        penalty += 0.25
    if terms & {"conversation", "conversation-friendly", "quiet", "calm"}:
        if set(negative_aspects) & {"loud", "noisy", "crowded", "busy"}:
            penalty += 0.25
    return min(penalty, 0.5)


def _context_category_penalty(context_terms: list[str], item: Item) -> float:
    hinted_category = context_category_hint(context_terms)
    if not hinted_category or item.category == hinted_category:
        return 0.0
    if hinted_category == "gift" and is_gift_category(item.category):
        return 0.0
    return 0.42


def _context_category_boost(context_terms: list[str], item: Item) -> float:
    hinted_category = context_category_hint(context_terms)
    if not hinted_category:
        return 0.0
    if item.category == hinted_category:
        return 0.14
    if hinted_category == "gift" and is_gift_category(item.category):
        return 0.14
    return 0.0


def _item_profile(item: Item, cache: dict[str, ItemProfile] | None) -> ItemProfile:
    if cache is None:
        return build_item_profile(item)
    cached = cache.get(item.item_id)
    if cached is not None:
        return cached
    profile = build_item_profile(item)
    cache[item.item_id] = profile
    return profile


def _seen_item_ids(user_profile: UserProfile) -> set[str]:
    explicit = set(user_profile.seen_item_ids)
    legacy = {
        signal.removeprefix("seen item: ")
        for signal in user_profile.signals
        if signal.startswith("seen item: ")
    }
    return explicit | legacy
