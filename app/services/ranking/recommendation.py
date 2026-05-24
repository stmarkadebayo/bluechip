from __future__ import annotations

import re
from dataclasses import dataclass

from app.models.schemas import Item, RecommendationItem, UserProfile
from app.services.ranking.features import matched_terms, ranker_features, weighted_score
from app.services.profiling.item_profile import build_item_profile


@dataclass(frozen=True)
class RecommendationWeights:
    preference: float = 0.24
    context: float = 0.16
    category: float = 0.18
    vector: float = 0.18
    quality: float = 0.14
    popularity: float = 0.16  # Reduced from 0.18 to reduce popularity bias
    novelty: float = 0.08  # Increased from 0.06 for sparse users
    confidence: float = 0.04
    aspect: float = 0.16
    sequential: float = 0.12
    evidence_graph: float = 0.12
    nigerian_context: float = 0.04
    collaborative: float = 0.22
    retrieval: float = 0.10  # Increased from 0.06 - retrieval quality matters!
    source_diversity: float = 0.06  # Increased from 0.04 - diversity is important
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


def rank_candidates(
    user_profile: UserProfile,
    context: str,
    candidate_items: list[Item],
    limit: int,
    weights: RecommendationWeights | None = None,
    candidate_sources: dict[str, list[str]] | None = None,
    candidate_source_scores: dict[str, dict[str, float]] | None = None,
) -> list[RecommendationItem]:
    weights = weights or RecommendationWeights()
    feature_weights = weights.as_feature_weights()
    candidate_sources = candidate_sources or {}
    candidate_source_scores = candidate_source_scores or {}
    ranked: list[tuple[float, RecommendationItem]] = []
    context_terms = _terms(context)
    profiled_items = [(item, build_item_profile(item)) for item in candidate_items]
    max_popularity = max((profile.popularity for _, profile in profiled_items), default=0)
    
    # IMPROVED: More aggressive personalization for sparse users
    # Sparse users should rely less on popularity and more on personalized signals
    base_confidence = user_profile.confidence
    user_history_size = len(getattr(user_profile, 'history', []))
    if user_history_size <= 2:
        # For very sparse users, boost personalization weight significantly
        personalization_weight = min(max((base_confidence - 0.35) / 0.70, 0.25), 0.85)
    else:
        personalization_weight = min(max((base_confidence - 0.45) / 0.80, 0.02), 0.65)

    for item, profile in profiled_items:
        sources = candidate_sources.get(item.item_id, [])
        source_scores = candidate_source_scores.get(item.item_id, {})
        
        features = ranker_features(
            user_profile=user_profile,
            item_profile=profile,
            context_terms=context_terms,
            max_popularity=max_popularity,
            seen_item_ids=_seen_item_ids(user_profile),
            source_scores=source_scores,
        )
        matched_signals = matched_terms(
            user_profile.preferred_terms + user_profile.positive_aspects,
            profile.terms + profile.positive_aspects,
        )

        # IMPROVED: Reduce popularity bias, increase personalization component
        # For candidates already in the pool, popularity is less important than fit
        base_score = (
            0.65 * features["popularity"]  # Was 0.85 - reduced to favor personalization
            + 0.20 * features["item_quality"]  # Increased from 0.10
            + 0.15 * features["novelty"]  # Increased from 0.05
        )
        personalized_score = weighted_score(features, feature_weights)
        context_penalty = _context_penalty(context_terms, item, profile.negative_aspects)
        context_category_penalty = _context_category_penalty(context_terms, item)
        context_category_boost = _context_category_boost(context_terms, item)
        
        # IMPROVED: Add retrieval source priority boost
        # Items from higher-quality retrieval sources should rank higher
        retrieval_priority_boost = _calculate_retrieval_boost(sources, source_scores)
        
        score = (
            (1 - personalization_weight) * base_score
            + personalization_weight * personalized_score
            + context_category_boost
            + retrieval_priority_boost  # NEW
            - context_penalty
            - context_category_penalty
        )
        score_components = {name: round(value, 4) for name, value in features.items()}
        score_components.update(
            {
                "context_penalty": round(context_penalty, 4),
                "context_category_penalty": round(context_category_penalty, 4),
                "context_category_boost": round(context_category_boost, 4),
                "retrieval_priority_boost": round(retrieval_priority_boost, 4),
                "personalization_weight": round(personalization_weight, 4),
            }
        )

        tradeoffs = "No major tradeoff detected from available metadata."
        if item.metadata.get("price") in {"high", "premium", "expensive"}:
            tradeoffs = "Price may be higher than the user's usual preference."
        elif profile.negative_aspects:
            tradeoffs = "Potential mismatch: " + ", ".join(profile.negative_aspects[:3])

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
    hinted_category = _context_category_hint(context_terms)
    if not hinted_category or item.category == hinted_category:
        return 0.0
    if hinted_category == "gift" and _is_gift_category(item.category):
        return 0.0
    return 0.42


def _context_category_boost(context_terms: list[str], item: Item) -> float:
    hinted_category = _context_category_hint(context_terms)
    if not hinted_category:
        return 0.0
    if item.category == hinted_category:
        return 0.14
    if hinted_category == "gift" and _is_gift_category(item.category):
        return 0.14
    return 0.0


def _context_category_hint(context_terms: list[str]) -> str | None:
    terms = set(context_terms)
    if terms & {
        "beauty",
        "hair",
        "makeup",
        "manicure",
        "nail",
        "skincare",
        "skin",
        "styling",
    }:
        return "All_Beauty"
    if terms & {"music", "playlist", "replay", "song", "songs"}:
        return "Digital_Music"
    if terms & {"gift", "gifting", "low-risk"}:
        return "gift"
    return None


def _is_gift_category(category: str) -> bool:
    return category in {"For Him", "For Her", "Gift Cards", "Restaurants", "Specialty Cards"}


def _seen_item_ids(user_profile: UserProfile) -> set[str]:
    explicit = set(user_profile.seen_item_ids)
    legacy = {
        signal.removeprefix("seen item: ")
        for signal in user_profile.signals
        if signal.startswith("seen item: ")
    }
    return explicit | legacy


def _calculate_retrieval_boost(sources: list[str], source_scores: dict[str, float]) -> float:
    """
    Boost items based on retrieval source quality.
    Items from high-quality retrieval sources get higher ranking boost.
    """
    if not sources or not source_scores:
        return 0.0
    
    # SOURCE_PRIORITIES from candidates.py
    SOURCE_PRIORITIES = {
        "neural_vector": 0.90,
        "beauty_review_term_profile": 0.87,
        "beauty_lexical_item_neighbor": 0.86,
        "category_aspect_graph": 0.855,
        "sequential_transition": 0.845,
        "beauty_aspect_profile": 0.84,
        "aspect_evidence_graph": 0.835,
        "category_transition": 0.825,
        "review_term_profile": 0.82,
        "lexical_item_neighbor": 0.81,
        "aspect_profile": 0.80,
        "beauty_taxonomy_aspect": 0.812,
        "beauty_taxonomy_window": 0.805,
        "beauty_sparse_tail": 0.79,
        "sparse_category_tail": 0.77,
        "category_affinity_popular": 0.83,
        "category_popular": 0.81,
        "bm25_profile": 0.78,
        "description_fallback": 0.76,
        "global_popular": 0.76,
        "vector_profile": 0.74,
        "graph_walk": 0.735,
        "user_neighbor": 0.72,
        "co_visitation": 0.70,
    }
    
    # Calculate boost: higher priority sources get higher boost
    # Diversity bonus: items from multiple sources get additional boost
    source_priorities = [SOURCE_PRIORITIES.get(source, 0.5) for source in sources]
    max_priority = max(source_priorities) if source_priorities else 0.5
    source_diversity = min(len(sources) / 3, 1.0)  # Bonus for items from multiple sources
    
    # Base boost from best retrieval source
    base_boost = (max_priority - 0.65) * 0.08 if max_priority > 0.65 else 0.0
    
    # Diversity bonus for items retrieved from multiple sources
    diversity_bonus = source_diversity * 0.03
    
    return round(base_boost + diversity_bonus, 4)

