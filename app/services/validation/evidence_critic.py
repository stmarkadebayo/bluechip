from __future__ import annotations

import re

from app.models.schemas import ItemProfile, RecommendationItem, UserProfile


SENSITIVE_INFERENCE_TERMS = {
    "depressed",
    "pregnant",
    "religion",
    "religious",
    "tribe",
    "wealthy",
    "poor person",
    "political",
    "sick",
}

GENERIC_REVIEW_TERMS = {
    "again",
    "buying",
    "call",
    "carries",
    "enough",
    "fair",
    "feels",
    "first",
    "here",
    "lands",
    "main",
    "matters",
    "overall",
    "out",
    "price",
    "puts",
    "rate",
    "rating",
    "reason",
    "shopper",
    "stood",
    "supports",
    "taste",
    "value",
    "would",
}


def review_evidence_issues(
    review: str,
    user_profile: UserProfile,
    item_profile: ItemProfile,
) -> list[str]:
    issues = _sensitive_inference_issues(review)
    supported_terms = _supported_terms(user_profile, item_profile)
    claim_terms = _content_terms(review)
    unsupported = [
        term
        for term in claim_terms
        if term not in supported_terms
        and term not in _content_terms(item_profile.name)
        and term not in GENERIC_REVIEW_TERMS
    ]
    if len(unsupported) > max(12, len(claim_terms) * 0.55):
        issues.append("review contains too many terms unsupported by user or item evidence")
    if user_profile.nigerian_context and "nigerian" not in review.lower():
        issues.append("Nigerian context evidence exists but is not reflected")
    return issues


def recommendation_evidence_issues(
    reason: str,
    user_profile: UserProfile,
    recommendation: RecommendationItem,
    context: str,
) -> list[str]:
    issues = _sensitive_inference_issues(reason)
    supported = set(_content_terms(" ".join(user_profile.signals)))
    supported.update(_content_terms(" ".join(recommendation.signals)))
    supported.update(_content_terms(" ".join(recommendation.matched_signals)))
    supported.update(_content_terms(context))
    reason_terms = _content_terms(reason)
    unsupported = [
        term
        for term in reason_terms
        if term not in supported and term not in _content_terms(recommendation.name)
    ]
    if len(unsupported) > max(10, len(reason_terms) * 0.60):
        issues.append("recommendation explanation is weakly grounded in available evidence")
    return issues


def _sensitive_inference_issues(text: str) -> list[str]:
    lowered = text.lower()
    return [
        f"unsafe sensitive inference: {term}"
        for term in SENSITIVE_INFERENCE_TERMS
        if term in lowered
    ]


def _supported_terms(user_profile: UserProfile, item_profile: ItemProfile) -> set[str]:
    values = []
    values.extend(user_profile.preferred_terms)
    values.extend(user_profile.disliked_terms)
    values.extend(user_profile.positive_aspects)
    values.extend(user_profile.negative_aspects)
    values.extend(user_profile.nigerian_context)
    values.extend(user_profile.signals)
    values.extend(item_profile.terms)
    values.extend(item_profile.positive_aspects)
    values.extend(item_profile.negative_aspects)
    values.extend(item_profile.nigerian_context)
    values.extend(item_profile.signals)
    return set(_content_terms(" ".join(values)))


def _content_terms(text: str) -> list[str]:
    stopwords = {
        "and",
        "because",
        "for",
        "from",
        "has",
        "into",
        "its",
        "the",
        "that",
        "this",
        "with",
        "your",
    }
    return [
        token
        for token in re.findall(r"[a-zA-Z][a-zA-Z0-9_-]{2,}", text.lower())
        if token not in stopwords
    ]
