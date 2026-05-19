from __future__ import annotations

import math
import re
from collections import Counter, defaultdict

from app.models.schemas import UserHistoryItem
from app.services.retrieval.embeddings import embedding_text, terms


REVIEW_TERM_STOPWORDS = {
    "about",
    "after",
    "again",
    "all",
    "also",
    "amazon",
    "are",
    "but",
    "because",
    "been",
    "brand",
    "can",
    "cant",
    "could",
    "does",
    "doesn",
    "for",
    "good",
    "great",
    "had",
    "has",
    "have",
    "her",
    "him",
    "his",
    "how",
    "into",
    "its",
    "item",
    "just",
    "like",
    "love",
    "much",
    "nice",
    "only",
    "order",
    "out",
    "product",
    "products",
    "really",
    "review",
    "reviews",
    "she",
    "that",
    "them",
    "then",
    "these",
    "they",
    "this",
    "use",
    "very",
    "was",
    "what",
    "when",
    "will",
    "with",
    "would",
    "you",
    "your",
}


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


def build_collaborative_retrieval_index(
    reviews: list[dict],
    top_k: int = 50,
    positive_threshold: float = 4.0,
    max_positive_items_per_user: int = 50,
    max_users_per_item: int = 500,
) -> dict:
    """Build compact collaborative retrieval artifacts from temporal train reviews.

    The artifact supports two retrieval paths:
    - item co-visitation: liked item -> nearby liked items
    - user-neighbor CF: current history -> similar users -> their liked items
    """

    user_items: dict[str, dict[str, dict]] = defaultdict(dict)
    item_users: dict[str, list[dict]] = defaultdict(list)

    for row in reviews:
        rating = float(row.get("rating") or 0)
        if rating < positive_threshold:
            continue
        user_id = str(row["user_id"])
        item_id = str(row["item_id"])
        record = {
            "item_id": item_id,
            "rating": rating,
            "timestamp": int(row.get("timestamp") or 0),
            "category": row.get("category") or "unknown",
        }
        existing = user_items[user_id].get(item_id)
        if existing is None or record["timestamp"] >= int(existing.get("timestamp") or 0):
            user_items[user_id][item_id] = record

    compact_users: dict[str, list[dict]] = {}
    for user_id, items in user_items.items():
        bounded = sorted(
            items.values(),
            key=lambda row: (float(row.get("rating") or 0), int(row.get("timestamp") or 0)),
            reverse=True,
        )[:max_positive_items_per_user]
        compact_users[user_id] = bounded
        for record in bounded:
            item_users[record["item_id"]].append(
                {
                    "user_id": user_id,
                    "rating": round(float(record.get("rating") or 0), 3),
                    "timestamp": int(record.get("timestamp") or 0),
                }
            )

    compact_item_users = {}
    for item_id, users in item_users.items():
        users.sort(key=lambda row: (row["rating"], row["timestamp"], row["user_id"]), reverse=True)
        compact_item_users[item_id] = users[:max_users_per_item]

    return {
        "type": "collaborative_retrieval",
        "positive_threshold": positive_threshold,
        "top_k": top_k,
        "item_neighbors": build_item_neighbors_from_reviews(
            reviews,
            top_k=top_k,
            positive_threshold=positive_threshold,
            max_positive_items_per_user=max_positive_items_per_user,
        ),
        "user_positive_items": compact_users,
        "item_positive_users": compact_item_users,
    }


def build_review_term_retrieval_index(
    reviews: list[dict],
    items: list[dict] | None = None,
    positive_threshold: float = 4.0,
    max_terms_per_item: int = 18,
    max_items_per_term: int = 250,
) -> dict:
    """Build item retrieval from item metadata plus positive review language.

    This artifact targets tail items with weak/no co-visitation by indexing
    terms reviewers actually used to describe items. It intentionally stores a
    compact inverted index and per-item top terms rather than dense vectors.
    """

    item_counters: dict[str, Counter] = defaultdict(Counter)
    item_categories: dict[str, str] = {}
    item_positive_counts: Counter = Counter()

    for item in items or []:
        item_id = str(item.get("item_id") or "")
        if not item_id:
            continue
        category = str(item.get("category") or "unknown")
        item_categories[item_id] = category
        metadata = item.get("metadata") if isinstance(item.get("metadata"), dict) else {}
        metadata_text = " ".join(f"{key} {value}" for key, value in metadata.items())
        text = embedding_text(
            item.get("name"),
            category,
            item.get("summary"),
            metadata_text,
        )
        item_counters[item_id].update(_weighted_terms(text, weight=1.25))

    for row in reviews:
        rating = float(row.get("rating") or 0)
        if rating < positive_threshold:
            continue
        item_id = str(row.get("item_id") or "")
        if not item_id:
            continue
        category = str(row.get("category") or item_categories.get(item_id) or "unknown")
        item_categories[item_id] = category
        item_positive_counts[item_id] += 1
        text = embedding_text(row.get("item_name"), category, row.get("review"))
        item_counters[item_id].update(_weighted_terms(text, weight=2.0))

    item_ids = sorted(item_counters)
    doc_freq = Counter()
    for item_id in item_ids:
        doc_freq.update(item_counters[item_id].keys())
    item_count = max(len(item_ids), 1)

    item_terms: dict[str, list[list[object]]] = {}
    term_items: dict[str, list[list[object]]] = defaultdict(list)
    for item_id in item_ids:
        scored_terms = []
        for term, count in item_counters[item_id].items():
            idf = math.log(1 + item_count / (1 + doc_freq[term]))
            score = (1 + math.log(float(count))) * idf
            scored_terms.append((term, score))
        scored_terms.sort(key=lambda row: row[1], reverse=True)
        top_terms = [
            [term, round(score, 4)]
            for term, score in scored_terms[:max_terms_per_item]
            if score > 0
        ]
        if not top_terms:
            continue
        item_terms[item_id] = top_terms
        popularity_boost = 1.0 + math.log1p(item_positive_counts[item_id]) * 0.04
        for term, score in top_terms:
            term_items[str(term)].append([item_id, round(float(score) * popularity_boost, 4)])

    compact_term_items = {}
    for term, postings in term_items.items():
        postings.sort(key=lambda row: (float(row[1]), str(row[0])), reverse=True)
        compact_term_items[term] = postings[:max_items_per_term]

    return {
        "type": "review_term_retrieval",
        "positive_threshold": positive_threshold,
        "max_terms_per_item": max_terms_per_item,
        "max_items_per_term": max_items_per_term,
        "item_terms": item_terms,
        "term_items": compact_term_items,
        "item_categories": item_categories,
        "item_positive_counts": dict(item_positive_counts),
    }


def candidate_ids_from_history(
    history: list[UserHistoryItem],
    item_neighbors: dict[str, list[dict]],
    limit: int = 100,
) -> list[str]:
    return [item_id for item_id, _ in scored_candidate_ids_from_history(history, item_neighbors, limit)]


def scored_candidate_ids_from_history(
    history: list[UserHistoryItem],
    item_neighbors: dict[str, list[dict]],
    limit: int = 100,
) -> list[tuple[str, float]]:
    scores = Counter()
    for item in history:
        if item.rating < 4:
            continue
        for neighbor in item_neighbors.get(item.item_id, []):
            scores[neighbor["item_id"]] += float(neighbor.get("score") or 0)
    return _normalized_counter(scores, limit)


def candidate_ids_from_user_neighbors(
    history: list[UserHistoryItem],
    collaborative_index: dict,
    limit: int = 100,
    max_neighbor_users: int = 200,
) -> list[tuple[str, float]]:
    item_users = collaborative_index.get("item_positive_users") or {}
    user_items = collaborative_index.get("user_positive_items") or {}
    seen = {item.item_id for item in history}
    liked_history = [item for item in history if item.rating >= 4]
    if not liked_history:
        return []

    user_scores = Counter()
    for item in liked_history:
        history_weight = max((float(item.rating) - 3.0) / 2.0, 0.1)
        for neighbor in item_users.get(item.item_id, []):
            rating_weight = max((float(neighbor.get("rating") or 0) - 3.0) / 2.0, 0.1)
            user_scores[str(neighbor["user_id"])] += history_weight * rating_weight

    item_scores = Counter()
    for user_id, user_score in user_scores.most_common(max_neighbor_users):
        for record in user_items.get(user_id, []):
            item_id = str(record["item_id"])
            if item_id in seen:
                continue
            rating_weight = max((float(record.get("rating") or 0) - 3.0) / 2.0, 0.1)
            item_scores[item_id] += user_score * rating_weight

    return _normalized_counter(item_scores, limit)


def candidate_ids_from_graph_walk(
    history: list[UserHistoryItem],
    collaborative_index: dict,
    limit: int = 100,
    max_direct_neighbors: int = 80,
    max_second_hop_neighbors: int = 12,
    max_neighbor_users: int = 180,
    max_user_items: int = 35,
) -> list[tuple[str, float]]:
    """Retrieve candidates from a bounded user-item graph walk.

    This is a dependency-free approximation of graph retrieval: liked history
    items vote for direct item neighbors, similar users' positives, and a small
    second hop from those positives. It complements direct co-visitation by
    reaching items that are not immediate neighbors of the seed item.
    """

    item_neighbors = collaborative_index.get("item_neighbors") or {}
    item_users = collaborative_index.get("item_positive_users") or {}
    user_items = collaborative_index.get("user_positive_items") or {}
    liked_history = [item for item in history if item.rating >= 4]
    if not liked_history:
        return []

    seen = {item.item_id for item in history}
    scores = Counter()
    seed_categories = {item.item_id: item.category for item in liked_history}

    for item in liked_history:
        history_weight = max((float(item.rating) - 3.0) / 2.0, 0.2)
        seed_category = item.category

        direct_neighbors = _postings_from_dicts(item_neighbors.get(item.item_id, []))
        for direct_id, direct_score in direct_neighbors[:max_direct_neighbors]:
            if direct_id not in seen:
                scores[direct_id] += history_weight * direct_score * 0.65
            for second_id, second_score in _postings_from_dicts(
                item_neighbors.get(direct_id, [])
            )[:max_second_hop_neighbors]:
                if second_id in seen or second_id == item.item_id:
                    continue
                scores[second_id] += history_weight * direct_score * second_score * 0.18

        user_scores = Counter()
        for neighbor_user in item_users.get(item.item_id, [])[:max_neighbor_users]:
            rating_weight = max((float(neighbor_user.get("rating") or 0) - 3.0) / 2.0, 0.1)
            user_scores[str(neighbor_user["user_id"])] += history_weight * rating_weight

        for user_id, user_score in user_scores.most_common(max_neighbor_users):
            for rank, record in enumerate(user_items.get(user_id, [])[:max_user_items]):
                candidate_id = str(record["item_id"])
                if candidate_id in seen:
                    continue
                rating_weight = max((float(record.get("rating") or 0) - 3.0) / 2.0, 0.1)
                category_boost = (
                    1.12
                    if seed_category and record.get("category") == seed_category
                    else 1.0
                )
                scores[candidate_id] += (
                    user_score
                    * rating_weight
                    * category_boost
                    * 0.34
                    / math.sqrt(rank + 1)
                )
                for expanded_id, expanded_score in _postings_from_dicts(
                    item_neighbors.get(candidate_id, [])
                )[:4]:
                    if expanded_id in seen or expanded_id in seed_categories:
                        continue
                    scores[expanded_id] += (
                        user_score
                        * rating_weight
                        * expanded_score
                        * category_boost
                        * 0.08
                    )

    return _normalized_counter(scores, limit)


def candidate_ids_from_review_terms(
    history: list[UserHistoryItem],
    review_term_index: dict,
    limit: int = 100,
    max_query_terms: int = 28,
    extra_terms: list[str] | None = None,
    preferred_categories: list[str] | None = None,
) -> list[tuple[str, float]]:
    """Retrieve items by matching history language to positive item-review terms."""

    term_items = review_term_index.get("term_items") or {}
    query_terms = _history_query_terms(
        history=history,
        review_term_index=review_term_index,
        max_query_terms=max_query_terms,
        extra_terms=extra_terms,
    )
    if not query_terms:
        return []
    item_categories = review_term_index.get("item_categories") or {}
    preferred_category_set = set(preferred_categories or [])
    seen = {item.item_id for item in history}
    scores = Counter()
    for term, query_weight in query_terms:
        for item_id, posting_score in _postings(term_items.get(term, [])):
            if item_id in seen:
                continue
            category_boost = 1.16 if item_categories.get(item_id) in preferred_category_set else 1.0
            scores[item_id] += query_weight * posting_score * category_boost
    return _normalized_counter(scores, limit)


def candidate_ids_from_lexical_item_neighbors(
    history: list[UserHistoryItem],
    review_term_index: dict,
    limit: int = 100,
    max_terms_per_history_item: int = 12,
) -> list[tuple[str, float]]:
    """Retrieve item lexical neighbors from precomputed per-item review terms."""

    item_terms = review_term_index.get("item_terms") or {}
    term_items = review_term_index.get("term_items") or {}
    seen = {item.item_id for item in history}
    scores = Counter()
    for item in history:
        if item.rating < 4:
            continue
        history_weight = max((float(item.rating) - 3.0) / 2.0, 0.2)
        for term, term_weight in _item_terms(item_terms.get(item.item_id, []))[:max_terms_per_history_item]:
            for neighbor_id, posting_score in _postings(term_items.get(term, [])):
                if neighbor_id in seen:
                    continue
                scores[neighbor_id] += history_weight * term_weight * posting_score
    return _normalized_counter(scores, limit)


def _normalized_counter(scores: Counter, limit: int) -> list[tuple[str, float]]:
    if not scores:
        return []
    max_score = max(scores.values()) or 1.0
    return [
        (item_id, round(min(float(score) / max_score, 1.0), 4))
        for item_id, score in scores.most_common(limit)
    ]


def _history_query_terms(
    history: list[UserHistoryItem],
    review_term_index: dict,
    max_query_terms: int,
    extra_terms: list[str] | None = None,
) -> list[tuple[str, float]]:
    item_terms = review_term_index.get("item_terms") or {}
    scores = Counter()
    for term in extra_terms or []:
        if term in REVIEW_TERM_STOPWORDS:
            continue
        scores[term] += 0.75
    for item in history:
        if item.rating < 3:
            continue
        rating_weight = max((float(item.rating) - 2.0) / 3.0, 0.2)
        for term in _review_terms(f"{item.item_name} {item.category or ''} {item.review}"):
            scores[term] += rating_weight
        for term, term_weight in _item_terms(item_terms.get(item.item_id, [])):
            scores[term] += rating_weight * term_weight
    return [
        (term, float(score))
        for term, score in scores.most_common(max_query_terms)
    ]


def _weighted_terms(text: str, weight: float) -> Counter:
    return Counter({term: count * weight for term, count in Counter(_review_terms(text)).items()})


def _review_terms(text: str) -> list[str]:
    base_terms = [
        token
        for token in terms(_normalize_units(text))
        if token not in REVIEW_TERM_STOPWORDS and not token.isdigit()
    ]
    output = []
    for token in base_terms:
        output.append(token)
    for left, right in zip(base_terms, base_terms[1:]):
        if left != right:
            output.append(f"{left}_{right}")
    return output


def _normalize_units(text: str) -> str:
    return re.sub(r"(\d+)\s*(oz|ml|inch|inches|count|pack)", r"\1_\2", text.lower())


def _item_terms(rows: list) -> list[tuple[str, float]]:
    output = []
    for row in rows:
        if isinstance(row, dict):
            term = str(row.get("term") or "")
            score = float(row.get("score") or 0.0)
        elif isinstance(row, (list, tuple)) and len(row) >= 2:
            term = str(row[0])
            score = float(row[1] or 0.0)
        else:
            continue
        if term and score > 0:
            output.append((term, score))
    return output


def _postings(rows: list) -> list[tuple[str, float]]:
    output = []
    for row in rows:
        if isinstance(row, dict):
            item_id = str(row.get("item_id") or "")
            score = float(row.get("score") or 0.0)
        elif isinstance(row, (list, tuple)) and len(row) >= 2:
            item_id = str(row[0])
            score = float(row[1] or 0.0)
        else:
            continue
        if item_id and score > 0:
            output.append((item_id, score))
    return output


def _postings_from_dicts(rows: list[dict]) -> list[tuple[str, float]]:
    output = []
    for row in rows:
        item_id = str(row.get("item_id") or "")
        score = float(row.get("score") or 0.0)
        if item_id and score > 0:
            output.append((item_id, score))
    return output
