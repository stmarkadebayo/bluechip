from __future__ import annotations

from app.models.schemas import Item, UserHistoryItem, UserProfile
from app.services.retrieval.item_similarity import candidate_ids_from_history
from app.services.retrieval.text import BM25Retriever


def generate_candidates(
    user_profile: UserProfile,
    history: list[UserHistoryItem],
    items: list[Item],
    context: str,
    item_neighbors: dict[str, list[dict]] | None = None,
    bm25_retriever: BM25Retriever | None = None,
    limit: int = 100,
) -> list[Item]:
    by_id = {item.item_id: item for item in items}
    selected: list[Item] = []
    history_item_ids = {item.item_id for item in history}
    seen: set[str] = set(history_item_ids)

    if item_neighbors:
        neighbor_budget = max(1, int(limit * 0.35))
        for item_id in candidate_ids_from_history(history, item_neighbors, limit=neighbor_budget):
            item = by_id.get(item_id)
            if item and item_id not in seen:
                selected.append(item)
                seen.add(item_id)

    query = " ".join(user_profile.preferred_terms + user_profile.positive_aspects + [context])
    search_limit = min(len(items), limit + len(history_item_ids))
    bm25_target = max(1, int(limit * 0.70))
    retriever = bm25_retriever or BM25Retriever.from_items(items)
    for item in retriever.search(query, limit=search_limit):
        if item.item_id not in seen:
            selected.append(item)
            seen.add(item.item_id)
        if len(selected) >= bm25_target:
            break

    if len(selected) < limit:
        fallback = sorted(
            items,
            key=lambda item: (
                int(item.metadata.get("rating_number") or item.metadata.get("review_count") or 0),
                item.average_rating or 0,
            ),
            reverse=True,
        )
        for item in fallback:
            if item.item_id not in seen:
                selected.append(item)
                seen.add(item.item_id)
            if len(selected) >= limit:
                break

    return selected[:limit]
