from __future__ import annotations

from app.models.schemas import ItemProfile, UserProfile
from app.services.intelligence.aspects import aspect_overlap
from app.services.retrieval.embeddings import cosine_similarity


FEATURE_NAMES = [
    "preference_match",
    "context_match",
    "category_match",
    "vector_match",
    "item_quality",
    "popularity",
    "novelty",
    "confidence",
    "dislike_match",
    "aspect_match",
    "sequential_match",
    "evidence_graph_match",
    "nigerian_context_match",
    "collaborative_match",
    "retrieval_match",
    "source_diversity",
]


def ranker_features(
    user_profile: UserProfile,
    item_profile: ItemProfile,
    context_terms: list[str],
    max_popularity: int,
    seen_item_ids: set[str],
    source_scores: dict[str, float] | None = None,
) -> dict[str, float]:
    source_scores = source_scores or {}
    preference_match = overlap(
        user_profile.preferred_terms + user_profile.positive_aspects,
        item_profile.terms + item_profile.positive_aspects,
    )
    dislike_match = overlap(
        user_profile.disliked_terms + user_profile.negative_aspects,
        item_profile.terms + item_profile.negative_aspects,
    )
    context_match = overlap(context_terms, item_profile.terms + item_profile.positive_aspects)
    aspect_match = aspect_overlap(user_profile.aspect_scores, item_profile.aspect_scores)
    sequential_match = max(
        source_scores.get("sequential_transition", 0.0),
        source_scores.get("category_transition", 0.0),
    )
    evidence_graph_match = max(
        source_scores.get("aspect_evidence_graph", 0.0),
        source_scores.get("category_aspect_graph", 0.0),
        sequential_match,
    )
    nigerian_context_match = overlap(
        user_profile.nigerian_context,
        item_profile.nigerian_context + item_profile.terms,
    )
    collaborative_match = max(
        source_scores.get("co_visitation", 0.0),
        source_scores.get("user_neighbor", 0.0),
        source_scores.get("graph_walk", 0.0),
        sequential_match,
    )
    retrieval_match = max(source_scores.values(), default=0.0)
    return {
        "preference_match": preference_match,
        "context_match": context_match,
        "category_match": category_match(user_profile, item_profile.category),
        "vector_match": max(cosine_similarity(user_profile.embedding, item_profile.embedding), 0.0),
        "item_quality": item_profile.quality_score,
        "popularity": popularity_score(item_profile.popularity, max_popularity),
        "novelty": 0.65 if item_profile.item_id not in seen_item_ids else 0.1,
        "confidence": user_profile.confidence,
        "dislike_match": dislike_match,
        "aspect_match": aspect_match,
        "sequential_match": sequential_match,
        "evidence_graph_match": evidence_graph_match,
        "nigerian_context_match": nigerian_context_match,
        "collaborative_match": collaborative_match,
        "retrieval_match": retrieval_match,
        "source_diversity": min(len(source_scores) / 4, 1.0),
    }


def weighted_score(features: dict[str, float], weights: dict[str, float]) -> float:
    return sum(features.get(name, 0.0) * weights.get(name, 0.0) for name in FEATURE_NAMES)


def overlap(left: list[str], right: list[str]) -> float:
    if not left or not right:
        return 0.0
    right_set = set(right)
    return min(sum(1 for term in left if term in right_set) / max(len(left), 1), 1.0)


def matched_terms(left: list[str], right: list[str], limit: int = 5) -> list[str]:
    right_set = set(right)
    seen = []
    for term in left:
        if term in right_set and term not in seen:
            seen.append(term)
        if len(seen) >= limit:
            break
    return seen


def category_match(user_profile: UserProfile, category: str) -> float:
    if category in user_profile.preferred_categories:
        return 1.0
    return max(user_profile.category_affinity.get(category, 0.0), 0.0)


def popularity_score(popularity: int, max_popularity: int) -> float:
    import math

    if popularity <= 0 or max_popularity <= 0:
        return 0.0
    return math.log1p(popularity) / math.log1p(max_popularity)
