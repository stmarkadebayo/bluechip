from __future__ import annotations

import re
from collections import Counter
from dataclasses import dataclass
from typing import Iterable

from app.models.schemas import Item, UserHistoryItem
from app.services.retrieval.embeddings import embedding_text


ASPECT_KEYWORDS = {
    "price_value": {
        "affordable",
        "cheap",
        "cost",
        "discount",
        "expensive",
        "price",
        "pricey",
        "value",
        "worth",
    },
    "quality_reliability": {
        "broken",
        "durable",
        "flimsy",
        "quality",
        "reliable",
        "solid",
        "sturdy",
        "weak",
        "works",
    },
    "service_speed": {
        "delay",
        "delayed",
        "fast",
        "late",
        "quick",
        "service",
        "slow",
        "staff",
        "wait",
    },
    "ambience_context": {
        "busy",
        "calm",
        "comfortable",
        "conversation",
        "crowded",
        "loud",
        "noisy",
        "quiet",
    },
    "taste_food": {
        "fresh",
        "pepper",
        "spicy",
        "stale",
        "sweet",
        "taste",
        "tasty",
    },
    "beauty_fit": {
        "beauty",
        "cream",
        "face",
        "hair",
        "makeup",
        "moisturizer",
        "nail",
        "serum",
        "shampoo",
        "skin",
        "wig",
    },
    "music_media": {
        "album",
        "artist",
        "music",
        "playlist",
        "song",
        "songs",
        "sound",
    },
    "gift_fit": {
        "card",
        "gift",
        "present",
        "recipient",
        "voucher",
    },
}

NEGATIVE_TERMS = {
    "bad",
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
    "terrible",
    "weak",
}

NIGERIAN_CONTEXT_TERMS = {
    "affordable",
    "budget",
    "campus",
    "delivery",
    "lagos",
    "mainland",
    "naira",
    "nigeria",
    "nigerian",
    "pepper",
    "student",
    "students",
}

STOPWORDS = {
    "about",
    "after",
    "amazon",
    "and",
    "are",
    "based",
    "because",
    "but",
    "for",
    "from",
    "item",
    "items",
    "like",
    "liked",
    "product",
    "review",
    "that",
    "the",
    "this",
    "was",
    "with",
}


@dataclass(frozen=True)
class AspectEvidence:
    aspect_scores: dict[str, float]
    positive_aspects: list[str]
    negative_aspects: list[str]
    nigerian_context: list[str]
    evidence_terms: list[str]


def user_aspect_evidence(persona: str, history: list[UserHistoryItem]) -> AspectEvidence:
    weighted_texts: list[tuple[str, float]] = [(persona, 0.65)]
    ordered = sorted(history, key=lambda item: item.timestamp or 0)
    for index, item in enumerate(ordered):
        recency = 1.0 + (index / max(len(ordered), 1)) * 0.35
        sentiment = 1.0 if item.rating >= 4 else -0.85 if item.rating <= 2 else 0.35
        weight = recency * sentiment
        weighted_texts.append((f"{item.item_name} {item.category or ''} {item.review}", weight))
    return _aspect_evidence(weighted_texts)


def item_aspect_evidence(item: Item) -> AspectEvidence:
    metadata_text = " ".join(f"{key} {value}" for key, value in item.metadata.items())
    text = embedding_text(item.name, item.category, item.summary, metadata_text)
    return _aspect_evidence([(text, 1.0)])


def aspect_overlap(left: dict[str, float], right: dict[str, float]) -> float:
    if not left or not right:
        return 0.0
    numerator = 0.0
    denominator = 0.0
    for aspect, left_score in left.items():
        if left_score <= 0:
            continue
        denominator += left_score
        numerator += min(left_score, max(right.get(aspect, 0.0), 0.0))
    if denominator <= 0:
        return 0.0
    return round(min(numerator / denominator, 1.0), 4)


def nigerian_context_terms(*texts: str) -> list[str]:
    terms = _terms(" ".join(texts))
    return _ordered_unique([term for term in terms if term in NIGERIAN_CONTEXT_TERMS], limit=8)


def _aspect_evidence(weighted_texts: Iterable[tuple[str, float]]) -> AspectEvidence:
    aspect_raw: Counter[str] = Counter()
    positive_terms: Counter[str] = Counter()
    negative_terms: Counter[str] = Counter()
    nigerian_terms: Counter[str] = Counter()
    evidence_terms: Counter[str] = Counter()

    for text, weight in weighted_texts:
        tokens = _terms(text)
        token_set = set(tokens)
        polarity = 1.0 if weight >= 0 else -1.0
        magnitude = abs(weight)
        for aspect, keywords in ASPECT_KEYWORDS.items():
            matches = token_set & keywords
            if matches:
                aspect_raw[aspect] += polarity * magnitude * len(matches)
        for token in tokens:
            if token in STOPWORDS:
                continue
            evidence_terms[token] += magnitude
            if token in NIGERIAN_CONTEXT_TERMS:
                nigerian_terms[token] += magnitude
            if token in NEGATIVE_TERMS or weight < 0:
                if token not in STOPWORDS:
                    negative_terms[token] += magnitude
            elif weight > 0:
                positive_terms[token] += magnitude

    max_abs = max((abs(value) for value in aspect_raw.values()), default=1.0)
    aspect_scores = {
        aspect: round(max(min(value / max_abs, 1.0), -1.0), 4)
        for aspect, value in aspect_raw.items()
    }
    return AspectEvidence(
        aspect_scores=aspect_scores,
        positive_aspects=[term for term, _ in positive_terms.most_common(8)],
        negative_aspects=[term for term, _ in negative_terms.most_common(8)],
        nigerian_context=[term for term, _ in nigerian_terms.most_common(8)],
        evidence_terms=[term for term, _ in evidence_terms.most_common(16)],
    )


def _terms(text: str) -> list[str]:
    return [
        token
        for token in re.findall(r"[a-zA-Z][a-zA-Z0-9_-]{2,}", text.lower())
        if token not in STOPWORDS and not token.isdigit()
    ]


def _ordered_unique(values: list[str], limit: int) -> list[str]:
    output = []
    for value in values:
        if value not in output:
            output.append(value)
        if len(output) >= limit:
            break
    return output
