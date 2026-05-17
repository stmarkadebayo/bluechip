from __future__ import annotations

import re

from app.models.schemas import Item, ItemProfile

POSITIVE_HINTS = {
    "affordable",
    "calm",
    "clean",
    "comfortable",
    "durable",
    "fast",
    "fresh",
    "quiet",
    "reliable",
    "spicy",
    "tasty",
}

NEGATIVE_HINTS = {
    "busy",
    "crowded",
    "delay",
    "expensive",
    "loud",
    "noisy",
    "slow",
}


def build_item_profile(item: Item) -> ItemProfile:
    text = " ".join([item.name, item.category, item.summary, " ".join(map(str, item.metadata.values()))])
    terms = _extract_terms(text)
    quality_score = _quality_score(item)
    positive_aspects = [term for term in terms if term in POSITIVE_HINTS]
    negative_aspects = [term for term in terms if term in NEGATIVE_HINTS]
    popularity = int(item.metadata.get("rating_number") or item.metadata.get("review_count") or 0)

    signals = [f"category: {item.category}", f"quality score: {quality_score:.2f}"]
    if item.average_rating:
        signals.append(f"average rating: {item.average_rating:.1f}")
    if popularity:
        signals.append(f"review count: {popularity}")
    if item.summary:
        signals.append(item.summary[:160])

    return ItemProfile(
        item_id=item.item_id,
        name=item.name,
        category=item.category,
        quality_score=quality_score,
        terms=terms,
        positive_aspects=positive_aspects,
        negative_aspects=negative_aspects,
        average_rating=item.average_rating,
        popularity=popularity,
        signals=signals,
        metadata=item.metadata,
    )


def _quality_score(item: Item) -> float:
    if item.average_rating is None:
        return 0.65
    return min(max((item.average_rating - 1) / 4, 0), 1)


def _extract_terms(text: str, limit: int = 16) -> list[str]:
    tokens = re.findall(r"[a-zA-Z][a-zA-Z0-9_-]{2,}", text.lower())
    seen = []
    for token in tokens:
        if token not in seen:
            seen.append(token)
        if len(seen) >= limit:
            break
    return seen
