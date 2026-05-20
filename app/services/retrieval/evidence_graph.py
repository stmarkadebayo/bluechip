from __future__ import annotations

import math
from collections import Counter, defaultdict

from app.models.schemas import Item, UserHistoryItem, UserProfile
from app.services.intelligence.aspects import ASPECT_KEYWORDS, item_aspect_evidence, user_aspect_evidence
from app.services.retrieval.embeddings import terms


def build_evidence_graph_index(
    train: list[dict],
    items: list[dict] | list[Item],
    top_k: int = 120,
) -> dict:
    item_models = [_item_from_row(item) for item in items]
    item_by_id = {item.item_id: item for item in item_models}
    item_aspects: dict[str, dict[str, float]] = {}
    item_terms: dict[str, list[str]] = {}
    aspect_items: dict[str, Counter[str]] = defaultdict(Counter)
    category_aspect_items: dict[str, dict[str, Counter[str]]] = defaultdict(lambda: defaultdict(Counter))
    item_transitions: dict[str, Counter[str]] = defaultdict(Counter)
    category_transitions: dict[str, Counter[str]] = defaultdict(Counter)

    for item in item_models:
        evidence = item_aspect_evidence(item)
        item_aspects[item.item_id] = evidence.aspect_scores
        item_terms[item.item_id] = evidence.evidence_terms
        quality = (item.average_rating or 3.5) / 5
        popularity = int(item.metadata.get("rating_number") or item.metadata.get("review_count") or 0)
        base = 0.6 * quality + 0.4 * _popularity_score(popularity)
        for aspect, score in evidence.aspect_scores.items():
            if score <= 0:
                continue
            weighted = max(score, 0.05) + base
            aspect_items[aspect][item.item_id] += weighted
            category_aspect_items[item.category][aspect][item.item_id] += weighted

    by_user: dict[str, list[dict]] = defaultdict(list)
    for row in train:
        by_user[str(row.get("user_id") or "")].append(row)
        if float(row.get("rating") or 0) >= 4:
            item_id = str(row.get("item_id") or "")
            item = item_by_id.get(item_id)
            if item:
                review_evidence = user_aspect_evidence(
                    "",
                    [
                        UserHistoryItem(
                            item_id=item_id,
                            item_name=str(row.get("item_name") or item.name),
                            rating=float(row.get("rating") or 0),
                            review=str(row.get("review") or ""),
                            category=row.get("category") or item.category,
                            timestamp=row.get("timestamp"),
                        )
                    ],
                )
                for aspect, score in review_evidence.aspect_scores.items():
                    if score > 0:
                        aspect_items[aspect][item_id] += score
                        category_aspect_items[item.category][aspect][item_id] += score

    for rows in by_user.values():
        ordered = sorted(rows, key=lambda row: (int(row.get("timestamp") or 0), str(row.get("review_id") or "")))
        positives = [row for row in ordered if float(row.get("rating") or 0) >= 4]
        for left, right in zip(positives, positives[1:]):
            left_id = str(left.get("item_id") or "")
            right_id = str(right.get("item_id") or "")
            if left_id and right_id and left_id != right_id:
                item_transitions[left_id][right_id] += 1
            left_category = str(left.get("category") or "")
            if left_category and right_id:
                category_transitions[left_category][right_id] += 1

    return {
        "type": "evidence_graph",
        "top_k": top_k,
        "aspect_items": _top_counter_map(aspect_items, top_k),
        "category_aspect_items": {
            category: _top_counter_map(aspect_map, top_k)
            for category, aspect_map in category_aspect_items.items()
        },
        "item_transitions": _top_counter_map(item_transitions, top_k),
        "category_transitions": _top_counter_map(category_transitions, top_k),
        "item_aspects": item_aspects,
        "item_terms": item_terms,
    }


def candidate_ids_from_evidence_graph(
    user_profile: UserProfile,
    history: list[UserHistoryItem],
    context: str,
    evidence_graph: dict,
    limit: int,
) -> list[tuple[str, float, str]]:
    if not evidence_graph:
        return []
    scores: Counter[str] = Counter()
    source_scores: dict[str, dict[str, float]] = defaultdict(dict)
    positive_history = [item for item in history if item.rating >= 4]
    recent_positive = sorted(positive_history, key=lambda item: item.timestamp or 0)[-4:]

    for item in recent_positive:
        for target_id, score in _entries(evidence_graph.get("item_transitions", {}).get(item.item_id, [])):
            weighted = 0.92 * score
            scores[target_id] += weighted
            source_scores[target_id]["sequential_transition"] = max(
                source_scores[target_id].get("sequential_transition", 0.0),
                weighted,
            )
        if item.category:
            for target_id, score in _entries(evidence_graph.get("category_transitions", {}).get(item.category, [])):
                weighted = 0.58 * score
                scores[target_id] += weighted
                source_scores[target_id]["category_transition"] = max(
                    source_scores[target_id].get("category_transition", 0.0),
                    weighted,
                )

    aspect_weights = _active_aspect_weights(user_profile, context)
    for aspect, weight in aspect_weights.items():
        for target_id, score in _entries(evidence_graph.get("aspect_items", {}).get(aspect, [])):
            weighted = weight * score
            scores[target_id] += weighted
            source_scores[target_id]["aspect_evidence_graph"] = max(
                source_scores[target_id].get("aspect_evidence_graph", 0.0),
                weighted,
            )
        for category in user_profile.preferred_categories[:4]:
            category_map = (evidence_graph.get("category_aspect_items", {}).get(category) or {})
            for target_id, score in _entries(category_map.get(aspect, [])):
                weighted = 1.18 * weight * score
                scores[target_id] += weighted
                source_scores[target_id]["category_aspect_graph"] = max(
                    source_scores[target_id].get("category_aspect_graph", 0.0),
                    weighted,
                )

    history_ids = {item.item_id for item in history}
    max_score = max(scores.values(), default=1.0)
    ranked = []
    for item_id, raw_score in scores.items():
        if item_id in history_ids:
            continue
        normalized = min(raw_score / max_score, 1.0)
        source = max(source_scores[item_id], key=source_scores[item_id].get)
        ranked.append((item_id, round(normalized, 4), source))
    ranked.sort(key=lambda row: row[1], reverse=True)
    return ranked[:limit]


def _active_aspect_weights(user_profile: UserProfile, context: str) -> dict[str, float]:
    weights = {aspect: max(score, 0.0) for aspect, score in user_profile.aspect_scores.items() if score > 0}
    context_terms = set(terms(context))
    for aspect, keywords in ASPECT_KEYWORDS.items():
        if context_terms & keywords:
            weights[aspect] = max(weights.get(aspect, 0.0), 0.85)
    if not weights and user_profile.preferred_categories:
        category_text = " ".join(user_profile.preferred_categories).lower()
        if "beauty" in category_text:
            weights["beauty_fit"] = 0.75
        if "music" in category_text:
            weights["music_media"] = 0.75
        if "gift" in category_text:
            weights["gift_fit"] = 0.75
    return weights


def _item_from_row(row: dict | Item) -> Item:
    if isinstance(row, Item):
        return row
    return Item(**row)


def _top_counter_map(counter_map, top_k: int) -> dict[str, list[dict]]:
    output = {}
    for key, counter in counter_map.items():
        max_score = max(counter.values(), default=1.0)
        output[str(key)] = [
            {"item_id": str(item_id), "score": round(float(score) / max_score, 4)}
            for item_id, score in counter.most_common(top_k)
        ]
    return output


def _entries(rows: list[dict]) -> list[tuple[str, float]]:
    return [(str(row["item_id"]), float(row.get("score") or 0.0)) for row in rows]


def _popularity_score(value: int) -> float:
    if value <= 0:
        return 0.0
    return min(math.log1p(value) / math.log1p(1000), 1.0)
