from __future__ import annotations

import os
import re
from collections import Counter

from app.models.schemas import UserHistoryItem, UserProfile
from app.services.intelligence.aspects import user_aspect_evidence
from app.services.profiling.profile_enhancer import ProfileEnhancer
from app.services.retrieval.embeddings import embedding_text, hashed_embedding

STOPWORDS = {
    "and",
    "the",
    "for",
    "with",
    "that",
    "this",
    "was",
    "were",
    "very",
    "really",
    "from",
    "they",
    "their",
    "would",
    "could",
    "about",
    "based",
    "did",
    "does",
    "student",
    "students",
    "like",
    "likes",
    "liked",
    "lagos-based",
    "not",
    "who",
    "good",
    "great",
    "bad",
    "okay",
    "item",
    "product",
    "place",
}

POSITIVE_ASPECT_HINTS = {
    "affordable",
    "beautiful",
    "calm",
    "clean",
    "comfortable",
    "durable",
    "easy",
    "fast",
    "fresh",
    "friendly",
    "helpful",
    "quiet",
    "reliable",
    "smooth",
    "spicy",
    "tasty",
    "value",
}

NEGATIVE_ASPECT_HINTS = {
    "broken",
    "crowded",
    "delay",
    "delayed",
    "dirty",
    "expensive",
    "late",
    "loud",
    "noisy",
    "poor",
    "rude",
    "slow",
    "stale",
    "weak",
}


def build_user_profile(
    persona: str,
    history: list[UserHistoryItem],
    locale: str | None = None,
    enhance_with_llm: bool | None = None,
) -> UserProfile:
    ratings = [item.rating for item in history]
    average_rating = sum(ratings) / len(ratings) if ratings else 3.5
    recent_average_rating = _recent_average_rating(history, fallback=average_rating)
    rating_std = _rating_std(ratings)
    positive_rating_share = _rating_share(ratings, lower_bound=4)
    negative_rating_share = _rating_share(ratings, upper_bound=2)
    rating_trend = _rating_trend(history)

    positive_text = " ".join(item.review for item in history if item.rating >= 4)
    negative_text = " ".join(item.review for item in history if item.rating <= 2)
    positive_item_text = " ".join(
        f"{item.item_name} {item.category or ''}" for item in history if item.rating >= 4
    )
    negative_item_text = " ".join(
        f"{item.item_name} {item.category or ''}" for item in history if item.rating <= 2
    )
    persona_terms = _extract_terms(persona)

    preferred_terms = _top_terms(
        f"{positive_text} {positive_item_text}",
        fallback=persona_terms,
    )
    disliked_terms = _top_terms(f"{negative_text} {negative_item_text}", fallback=[])
    preferred_categories = _preferred_categories(history)
    category_affinity = _category_affinity(history)
    aspect_evidence = user_aspect_evidence(persona, history)
    positive_aspects = _merge_terms(
        aspect_evidence.positive_aspects,
        _aspect_terms(
            f"{positive_text} {positive_item_text} {persona}",
            POSITIVE_ASPECT_HINTS,
        ),
    )
    negative_aspects = _merge_terms(
        aspect_evidence.negative_aspects,
        _aspect_terms(
            f"{negative_text} {negative_item_text} {persona}",
            NEGATIVE_ASPECT_HINTS,
        ),
    )
    recent_terms = _recent_terms(history)
    review_length_mean = _review_length_mean(history)
    confidence = _confidence(history, persona)
    embedding = hashed_embedding(
        embedding_text(
            persona,
            positive_text,
            positive_item_text,
            preferred_terms,
            preferred_categories,
            positive_aspects,
            recent_terms,
        )
    )

    strictness = "balanced"
    if average_rating < 3.2:
        strictness = "strict"
    elif average_rating > 4.1:
        strictness = "generous"

    voice_style = _infer_voice_style(history)
    seen_item_ids = _seen_item_ids(history)
    signals = _build_signals(
        average_rating=average_rating,
        strictness=strictness,
        preferred_terms=preferred_terms,
        disliked_terms=disliked_terms,
        preferred_categories=preferred_categories,
        positive_aspects=positive_aspects,
        negative_aspects=negative_aspects,
        confidence=confidence,
        locale=locale,
    )

    profile = UserProfile(
        locale=locale,
        average_rating=round(average_rating, 2),
        recent_average_rating=recent_average_rating,
        rating_std=rating_std,
        positive_rating_share=positive_rating_share,
        negative_rating_share=negative_rating_share,
        rating_trend=rating_trend,
        rating_strictness=strictness,
        seen_item_ids=seen_item_ids,
        preferred_terms=preferred_terms,
        disliked_terms=disliked_terms,
        preferred_categories=preferred_categories,
        category_affinity=category_affinity,
        positive_aspects=positive_aspects,
        negative_aspects=negative_aspects,
        aspect_scores=aspect_evidence.aspect_scores,
        nigerian_context=aspect_evidence.nigerian_context,
        recent_terms=recent_terms,
        review_length_mean=review_length_mean,
        embedding=embedding,
        evidence_count=len(history),
        confidence=confidence,
        voice_style=voice_style,
        signals=signals,
    )
    if _should_enhance(enhance_with_llm):
        return ProfileEnhancer().enhance(
            profile=profile,
            persona=persona,
            history=history,
            locale=locale,
        )
    return profile


def _should_enhance(value: bool | None) -> bool:
    if value is not None:
        return value
    return os.getenv("BLUECHIP_PROFILE_ENHANCER", "").lower() in {"1", "true", "yes"}


def _extract_terms(text: str) -> list[str]:
    tokens = re.findall(r"[a-zA-Z][a-zA-Z0-9_-]{2,}", text.lower())
    return [token for token in tokens if token not in STOPWORDS]


def _top_terms(text: str, fallback: list[str], limit: int = 8) -> list[str]:
    terms = _extract_terms(text)
    if not terms:
        return fallback[:limit]
    counts = Counter(terms)
    return [term for term, _ in counts.most_common(limit)]


def _preferred_categories(history: list[UserHistoryItem], limit: int = 5) -> list[str]:
    counts = Counter()
    for item in history:
        if item.category and item.rating >= 4:
            counts[item.category] += 1
    if not counts:
        counts = Counter(item.category for item in history if item.category)
    return [category for category, _ in counts.most_common(limit)]


def _category_affinity(history: list[UserHistoryItem]) -> dict[str, float]:
    totals: dict[str, list[float]] = {}
    for item in history:
        if not item.category:
            continue
        totals.setdefault(item.category, []).append(item.rating)
    return {
        category: round((sum(values) / len(values) - 3) / 2, 3)
        for category, values in totals.items()
        if values
    }


def _aspect_terms(text: str, hints: set[str], limit: int = 8) -> list[str]:
    terms = _extract_terms(text)
    hinted = [term for term in terms if term in hints]
    if hinted:
        return list(dict.fromkeys(hinted))[:limit]
    return []


def _merge_terms(*groups: list[str], limit: int = 8) -> list[str]:
    output = []
    for group in groups:
        for term in group:
            if term not in output:
                output.append(term)
            if len(output) >= limit:
                return output
    return output


def _recent_terms(history: list[UserHistoryItem], limit: int = 8) -> list[str]:
    if not history:
        return []
    ordered = sorted(history, key=lambda item: item.timestamp or 0)
    recent_text = " ".join(
        f"{item.item_name} {item.category or ''} {item.review}"
        for item in ordered[-3:]
        if item.rating >= 3
    )
    return _top_terms(recent_text, fallback=[], limit=limit)


def _review_length_mean(history: list[UserHistoryItem]) -> float:
    if not history:
        return 0.0
    return round(sum(len(item.review.split()) for item in history) / len(history), 2)


def _recent_average_rating(history: list[UserHistoryItem], fallback: float) -> float:
    if not history:
        return round(fallback, 2)
    ordered = sorted(history, key=lambda item: item.timestamp or 0)
    recent = [item.rating for item in ordered[-3:]]
    return round(sum(recent) / len(recent), 2)


def _rating_std(ratings: list[float]) -> float:
    if len(ratings) < 2:
        return 0.0
    mean = sum(ratings) / len(ratings)
    variance = sum((rating - mean) ** 2 for rating in ratings) / len(ratings)
    return round(variance ** 0.5, 3)


def _rating_share(
    ratings: list[float],
    lower_bound: float | None = None,
    upper_bound: float | None = None,
) -> float:
    if not ratings:
        return 0.0
    count = 0
    for rating in ratings:
        if lower_bound is not None and rating >= lower_bound:
            count += 1
        elif upper_bound is not None and rating <= upper_bound:
            count += 1
    return round(count / len(ratings), 3)


def _rating_trend(history: list[UserHistoryItem]) -> float:
    if len(history) < 4:
        return 0.0
    ordered = sorted(history, key=lambda item: item.timestamp or 0)
    midpoint = len(ordered) // 2
    early = [item.rating for item in ordered[:midpoint]]
    late = [item.rating for item in ordered[midpoint:]]
    return round((sum(late) / len(late)) - (sum(early) / len(early)), 3)


def _seen_item_ids(history: list[UserHistoryItem]) -> list[str]:
    seen = []
    for item in history:
        if item.item_id not in seen:
            seen.append(item.item_id)
    return seen


def _confidence(history: list[UserHistoryItem], persona: str) -> float:
    evidence = min(len(history) / 8, 0.75)
    persona_bonus = 0.15 if len(persona.split()) >= 8 else 0.05 if persona else 0.0
    rating_variety = 0.10 if len({round(item.rating) for item in history}) > 1 else 0.0
    return round(min(0.10 + evidence + persona_bonus + rating_variety, 0.95), 2)


def _infer_voice_style(history: list[UserHistoryItem]) -> str:
    if not history:
        return "concise and preference-focused"

    avg_words = sum(len(item.review.split()) for item in history) / len(history)
    if avg_words < 20:
        return "short, direct, and practical"
    if avg_words > 80:
        return "detailed and experience-driven"
    return "balanced, specific, and conversational"


def _build_signals(
    average_rating: float,
    strictness: str,
    preferred_terms: list[str],
    disliked_terms: list[str],
    preferred_categories: list[str],
    positive_aspects: list[str],
    negative_aspects: list[str],
    confidence: float,
    locale: str | None,
) -> list[str]:
    signals = [
        f"average rating tendency is {average_rating:.2f}",
        f"rating style is {strictness}",
        f"profile confidence is {confidence:.2f}",
    ]
    if preferred_terms:
        signals.append("positive preference terms: " + ", ".join(preferred_terms[:5]))
    if disliked_terms:
        signals.append("negative sensitivity terms: " + ", ".join(disliked_terms[:5]))
    if preferred_categories:
        signals.append("preferred categories: " + ", ".join(preferred_categories[:3]))
    if positive_aspects:
        signals.append("positive aspects: " + ", ".join(positive_aspects[:5]))
    if negative_aspects:
        signals.append("negative sensitivities: " + ", ".join(negative_aspects[:5]))
    if locale:
        signals.append(f"locale context: {locale}")
    return signals
