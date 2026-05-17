from __future__ import annotations

import math
from collections import Counter, defaultdict

from app.models.schemas import UserHistoryItem


def build_item_neighbors_from_reviews(
    reviews: list[dict],
    top_k: int = 20,
    positive_threshold: float = 4.0,
    max_positive_items_per_user: int = 50,
) -> dict[str, list[dict]]:
    """Build item-item neighbors from positive user co-occurrence."""

    user_items: dict[str, dict[str, int]] = defaultdict(dict)
    co_counts: dict[str, Counter] = defaultdict(Counter)

    for row in reviews:
        if float(row.get("rating") or 0) < positive_threshold:
            continue
        user_id = row["user_id"]
        item_id = row["item_id"]
        timestamp = int(row.get("timestamp") or 0)
        user_items[user_id][item_id] = max(timestamp, user_items[user_id].get(item_id, 0))

    item_counts = Counter(item_id for items in user_items.values() for item_id in items)

    for items in user_items.values():
        bounded_items = [
            item_id
            for item_id, _ in sorted(
                items.items(),
                key=lambda row: (row[1], row[0]),
                reverse=True,
            )[:max_positive_items_per_user]
        ]
        for left in bounded_items:
            for right in bounded_items:
                if left != right:
                    co_counts[left][right] += 1

    neighbors = {}
    for item_id, counts in co_counts.items():
        scored = []
        for other_id, co_count in counts.items():
            denom = math.sqrt(item_counts[item_id] * item_counts[other_id])
            score = co_count / denom if denom else 0.0
            scored.append({"item_id": other_id, "score": round(score, 4)})
        scored.sort(key=lambda row: row["score"], reverse=True)
        neighbors[item_id] = scored[:top_k]
    return neighbors


def candidate_ids_from_history(
    history: list[UserHistoryItem],
    item_neighbors: dict[str, list[dict]],
    limit: int = 100,
) -> list[str]:
    scores = Counter()
    for item in history:
        if item.rating < 4:
            continue
        for neighbor in item_neighbors.get(item.item_id, []):
            scores[neighbor["item_id"]] += float(neighbor["score"])
    return [item_id for item_id, _ in scores.most_common(limit)]
