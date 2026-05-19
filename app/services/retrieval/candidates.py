from __future__ import annotations

import math
from collections import Counter, defaultdict
from dataclasses import dataclass, field

from app.models.schemas import Item, UserHistoryItem, UserProfile
from app.services.retrieval.embeddings import embedding_text, terms
from app.services.retrieval.item_similarity import (
    candidate_ids_from_lexical_item_neighbors,
    candidate_ids_from_review_terms,
    candidate_ids_from_user_neighbors,
    scored_candidate_ids_from_history,
)
from app.services.retrieval.text import BM25Retriever
from app.services.retrieval.vector_store import LocalVectorRetriever


SOURCE_PRIORITIES = {
    "beauty_review_term_profile": 0.87,
    "beauty_lexical_item_neighbor": 0.86,
    "beauty_aspect_profile": 0.84,
    "review_term_profile": 0.82,
    "lexical_item_neighbor": 0.81,
    "aspect_profile": 0.80,
    "beauty_sparse_tail": 0.79,
    "beauty_taxonomy_aspect": 0.775,
    "sparse_category_tail": 0.77,
    "category_affinity_popular": 0.83,
    "category_popular": 0.81,
    "bm25_profile": 0.78,
    "global_popular": 0.76,
    "vector_profile": 0.74,
    "graph_walk": 0.735,
    "user_neighbor": 0.72,
    "co_visitation": 0.70,
}
EXPLORATION_SOURCES = {
    "aspect_profile",
    "beauty_aspect_profile",
    "beauty_sparse_tail",
    "beauty_taxonomy_aspect",
    "sparse_category_tail",
}

BEAUTY_CATEGORIES = {"All_Beauty"}
BEAUTY_ASPECT_GROUPS = {
    "hair_care": {
        "conditioner",
        "curl",
        "curly",
        "hair",
        "scalp",
        "shampoo",
        "spray",
        "texturizing",
        "wave",
    },
    "wigs_extensions": {
        "body_wave",
        "bundle",
        "bundles",
        "extension",
        "extensions",
        "hairpiece",
        "human_hair",
        "lace",
        "wig",
        "wigs",
    },
    "skin_care": {
        "anti",
        "anti-aging",
        "cleanser",
        "cream",
        "face",
        "facial",
        "hyaluronic",
        "laser",
        "lotion",
        "moisturizer",
        "serum",
        "skin",
        "sunscreen",
    },
    "sensitive_skin": {
        "calm",
        "fragrance",
        "fragrance_free",
        "gentle",
        "hypoallergenic",
        "sensitive",
        "sensitive_skin",
        "unscented",
    },
    "eye_face": {
        "collagen",
        "dark_circle",
        "eye",
        "eyebrow",
        "eyelash",
        "mask",
        "patch",
        "puffiness",
    },
    "nails": {
        "acrylic",
        "drill",
        "gel",
        "manicure",
        "nail",
        "nails",
        "polish",
        "rhinestone",
        "stencil",
    },
    "makeup_color": {
        "blush",
        "color",
        "concealer",
        "foundation",
        "gloss",
        "lip",
        "lipstick",
        "makeup",
        "powder",
        "shade",
    },
    "tools_accessories": {
        "barrette",
        "brush",
        "cleaner",
        "clip",
        "comb",
        "headband",
        "mirror",
        "rake",
        "scissor",
        "tool",
        "tools",
    },
    "natural_ingredients": {
        "aloe",
        "butter",
        "coconut",
        "natural",
        "oil",
        "organic",
        "shea",
        "vegan",
        "vitamin",
    },
}
ASPECT_STOPWORDS = {
    "all_beauty",
    "amazon",
    "brand",
    "count",
    "description",
    "dimensions",
    "discontinued",
    "item",
    "items",
    "manufacturer",
    "number",
    "package",
    "product",
    "products",
    "rating",
    "review",
    "reviews",
    "unknown",
    "with",
    "without",
}


@dataclass
class CandidatePool:
    items: list[Item]
    sources: dict[str, list[str]] = field(default_factory=dict)
    source_scores: dict[str, dict[str, float]] = field(default_factory=dict)

    def source_counts(self) -> dict[str, int]:
        counts: dict[str, int] = {}
        for source_names in self.sources.values():
            for source in source_names:
                counts[source] = counts.get(source, 0) + 1
        return dict(sorted(counts.items()))


@dataclass
class CandidateCatalog:
    global_popular: list[Item]
    by_category_popular: dict[str, list[Item]]
    by_category_token_index: dict[str, dict[str, list[Item]]]
    item_terms: dict[str, set[str]]

    @classmethod
    def from_items(cls, items: list[Item]) -> "CandidateCatalog":
        global_popular = sorted(items, key=_popularity_quality_key, reverse=True)
        by_category: dict[str, list[Item]] = {}
        token_index: dict[str, dict[str, list[Item]]] = defaultdict(lambda: defaultdict(list))
        item_terms: dict[str, set[str]] = {}
        for item in items:
            by_category.setdefault(item.category, []).append(item)
            item_terms[item.item_id] = _item_terms(item)
            for token in item_terms[item.item_id]:
                token_index[item.category][token].append(item)
        for category, category_items in by_category.items():
            by_category[category] = sorted(category_items, key=_popularity_quality_key, reverse=True)
        token_index_payload = {
            category: dict(token_items)
            for category, token_items in token_index.items()
        }
        return cls(
            global_popular=global_popular,
            by_category_popular=by_category,
            by_category_token_index=token_index_payload,
            item_terms=item_terms,
        )


def generate_candidates(
    user_profile: UserProfile,
    history: list[UserHistoryItem],
    items: list[Item],
    context: str,
    item_neighbors: dict[str, list[dict]] | None = None,
    bm25_retriever: BM25Retriever | None = None,
    vector_retriever: LocalVectorRetriever | None = None,
    catalog: CandidateCatalog | None = None,
    limit: int = 100,
) -> list[Item]:
    return generate_candidate_pool(
        user_profile=user_profile,
        history=history,
        items=items,
        context=context,
        item_neighbors=item_neighbors,
        bm25_retriever=bm25_retriever,
        vector_retriever=vector_retriever,
        catalog=catalog,
        limit=limit,
    ).items


def generate_candidate_pool(
    user_profile: UserProfile,
    history: list[UserHistoryItem],
    items: list[Item],
    context: str,
    collaborative_index: dict | None = None,
    item_neighbors: dict[str, list[dict]] | None = None,
    bm25_retriever: BM25Retriever | None = None,
    vector_retriever: LocalVectorRetriever | None = None,
    catalog: CandidateCatalog | None = None,
    limit: int = 100,
) -> CandidatePool:
    catalog = catalog or CandidateCatalog.from_items(items)
    by_id = {item.item_id: item for item in items}
    selected: list[Item] = []
    history_item_ids = {item.item_id for item in history}
    seen: set[str] = set(history_item_ids)
    sources: dict[str, list[str]] = {}
    source_scores: dict[str, dict[str, float]] = {}

    def add_candidate(item: Item, source: str, score: float = 0.0) -> None:
        if item.item_id in history_item_ids:
            return
        if item.item_id not in sources:
            sources[item.item_id] = []
            source_scores[item.item_id] = {}
        if source not in sources[item.item_id]:
            sources[item.item_id].append(source)
        source_scores[item.item_id][source] = max(source_scores[item.item_id].get(source, 0.0), score)
        if item.item_id not in seen:
            selected.append(item)
            seen.add(item.item_id)

    if collaborative_index and not item_neighbors:
        item_neighbors = collaborative_index.get("item_neighbors")

    query_terms = _query_terms(user_profile, history, context)
    query = " ".join(query_terms)
    review_term_index = (
        collaborative_index.get("review_term_retrieval") if collaborative_index else None
    )

    if item_neighbors:
        neighbor_budget = max(1, int(limit * 0.35))
        for item_id, score in scored_candidate_ids_from_history(
            history,
            item_neighbors,
            limit=neighbor_budget,
        ):
            item = by_id.get(item_id)
            if item:
                add_candidate(item, "co_visitation", score)

    if collaborative_index:
        user_neighbor_budget = max(1, int(limit * 0.35))
        for item_id, score in candidate_ids_from_user_neighbors(
            history,
            collaborative_index,
            limit=user_neighbor_budget,
        ):
            item = by_id.get(item_id)
            if item:
                add_candidate(item, "user_neighbor", score)

    if review_term_index:
        review_term_budget = max(1, int(limit * 0.38))
        for item_id, score in candidate_ids_from_review_terms(
            history=history,
            review_term_index=review_term_index,
            limit=review_term_budget,
            extra_terms=query_terms,
            preferred_categories=user_profile.preferred_categories,
        ):
            item = by_id.get(item_id)
            if item:
                source = (
                    "beauty_review_term_profile"
                    if item.category in BEAUTY_CATEGORIES
                    else "review_term_profile"
                )
                add_candidate(item, source, score)

        lexical_budget = max(1, int(limit * 0.30))
        for item_id, score in candidate_ids_from_lexical_item_neighbors(
            history,
            review_term_index,
            limit=lexical_budget,
        ):
            item = by_id.get(item_id)
            if item:
                source = (
                    "beauty_lexical_item_neighbor"
                    if item.category in BEAUTY_CATEGORIES
                    else "lexical_item_neighbor"
                )
                add_candidate(item, source, score)

    search_limit = min(len(items), limit + len(history_item_ids))
    bm25_target = max(1, int(limit * 0.35))
    retriever = bm25_retriever or BM25Retriever.from_items(items)
    bm25_added = 0
    for item, score in retriever.search_with_scores(query, limit=search_limit):
        had_source = "bm25_profile" in sources.get(item.item_id, [])
        add_candidate(item, "bm25_profile", score)
        if item.item_id not in history_item_ids and not had_source:
            bm25_added += 1
        if bm25_added >= bm25_target:
            break

    vector_target = max(bm25_target, int(limit * 0.50))
    vectors = vector_retriever or LocalVectorRetriever(items)
    vector_added = 0
    for item, score in vectors.search_with_scores(query or context, limit=search_limit):
        had_source = "vector_profile" in sources.get(item.item_id, [])
        add_candidate(item, "vector_profile", score)
        if item.item_id not in history_item_ids and not had_source:
            vector_added += 1
        if vector_added >= vector_target:
            break

    aspect_budget = max(1, int(limit * (0.26 if len(history) <= 2 else 0.16)))
    aspect_added = 0
    for item, score, source in _aspect_profile_candidates(
        user_profile=user_profile,
        catalog=catalog,
        query_terms=query_terms,
        limit=limit,
    ):
        had_source = source in sources.get(item.item_id, [])
        add_candidate(item, source, score)
        if item.item_id not in history_item_ids and not had_source:
            aspect_added += 1
        if aspect_added >= aspect_budget:
            break

    if len(history) <= 2:
        tail_budget = max(1, int(limit * 0.08))
        tail_added = 0
        for item, score, source in _sparse_category_tail_candidates(user_profile, catalog, limit):
            had_source = source in sources.get(item.item_id, [])
            add_candidate(item, source, score)
            if item.item_id not in history_item_ids and not had_source:
                tail_added += 1
            if tail_added >= tail_budget:
                break

    category_affinity_budget = max(1, int(limit * (0.42 if len(history) <= 2 else 0.30)))
    category_affinity_added = 0
    for item, score in _category_affinity_candidates(user_profile, catalog, limit):
        had_source = "category_affinity_popular" in sources.get(item.item_id, [])
        add_candidate(item, "category_affinity_popular", score)
        if item.item_id not in history_item_ids and not had_source:
            category_affinity_added += 1
        if category_affinity_added >= category_affinity_budget:
            break

    category_target = max(vector_target, int(limit * 0.60))
    if len(selected) < category_target and user_profile.preferred_categories:
        category_budget = max(1, category_target - len(selected))
        category_candidates = []
        for category in user_profile.preferred_categories:
            category_candidates.extend(catalog.by_category_popular.get(category, [])[: limit * 2])
        category_added = 0
        for item in category_candidates:
            had_source = "category_popular" in sources.get(item.item_id, [])
            add_candidate(item, "category_popular", _quality_popularity_score(item))
            if item.item_id not in history_item_ids and not had_source:
                category_added += 1
            if category_added >= category_budget:
                break

    fallback = catalog.global_popular
    fallback_budget = min(len(fallback), max(limit, int(limit * 0.50)))
    for index, item in enumerate(fallback[:fallback_budget]):
        rank_score = 1.0 - (index / max(fallback_budget, 1))
        add_candidate(item, "global_popular", max(rank_score, _quality_popularity_score(item)))

    limited = _select_balanced_candidates(selected, sources, source_scores, limit)
    limited_ids = {item.item_id for item in limited}
    return CandidatePool(
        items=limited,
        sources={item_id: values for item_id, values in sources.items() if item_id in limited_ids},
        source_scores={
            item_id: values for item_id, values in source_scores.items() if item_id in limited_ids
        },
    )


def _quality_popularity_score(item: Item) -> float:
    popularity = int(item.metadata.get("rating_number") or item.metadata.get("review_count") or 0)
    quality = (item.average_rating or 3.5) / 5
    if popularity <= 0:
        return round(min(max(quality, 0.0), 1.0), 4)
    return round(min(0.70 * quality + 0.30, 1.0), 4)


def _category_affinity_candidates(
    user_profile: UserProfile,
    catalog: CandidateCatalog,
    limit: int,
) -> list[tuple[Item, float]]:
    category_weights = _category_weights(user_profile)
    if not category_weights:
        return []

    eligible = []
    per_category_limit = max(limit * 4, 200)
    for category in category_weights:
        eligible.extend(catalog.by_category_popular.get(category, [])[:per_category_limit])
    max_popularity = max(
        (int(item.metadata.get("rating_number") or item.metadata.get("review_count") or 0) for item in eligible),
        default=0,
    )
    scored = []
    for item in eligible:
        popularity = int(item.metadata.get("rating_number") or item.metadata.get("review_count") or 0)
        popularity_score = (
            math.log1p(popularity) / math.log1p(max_popularity)
            if popularity > 0 and max_popularity > 0
            else 0.0
        )
        quality = (item.average_rating or 3.5) / 5
        category_weight = category_weights[item.category]
        score = (0.50 * category_weight) + (0.35 * popularity_score) + (0.15 * quality)
        scored.append((item, round(min(max(score, 0.0), 1.0), 4)))
    scored.sort(key=lambda row: row[1], reverse=True)
    return scored


def _category_weights(user_profile: UserProfile) -> dict[str, float]:
    weights: dict[str, float] = {}
    for index, category in enumerate(user_profile.preferred_categories):
        weights[category] = max(weights.get(category, 0.0), max(1.0 - (index * 0.08), 0.65))
    for category, affinity in user_profile.category_affinity.items():
        if affinity > 0:
            weights[category] = max(weights.get(category, 0.0), min(0.65 + affinity, 1.0))
    return weights


def _query_terms(
    user_profile: UserProfile,
    history: list[UserHistoryItem],
    context: str,
) -> list[str]:
    text = embedding_text(
        user_profile.preferred_terms,
        user_profile.positive_aspects,
        user_profile.recent_terms,
        user_profile.preferred_categories,
        context,
        [
            f"{item.item_name} {item.category or ''} {item.review}"
            for item in history
            if item.rating >= 3
        ],
    )
    output = []
    for token in terms(text):
        if token in ASPECT_STOPWORDS or token in output:
            continue
        output.append(token)
        if len(output) >= 32:
            break
    return output


def _aspect_profile_candidates(
    user_profile: UserProfile,
    catalog: CandidateCatalog,
    query_terms: list[str],
    limit: int,
) -> list[tuple[Item, float, str]]:
    category_weights = _category_weights(user_profile)
    if not category_weights and _looks_like_beauty_query(query_terms):
        category_weights = {"All_Beauty": 0.85}
    if not category_weights or not query_terms:
        return []

    candidate_scores: Counter[str] = Counter()
    candidate_items: dict[str, Item] = {}
    per_token_limit = max(250, min(limit, 1500))
    useful_terms = [token for token in query_terms if token not in ASPECT_STOPWORDS][:24]
    for category, category_weight in category_weights.items():
        token_index = catalog.by_category_token_index.get(category, {})
        if not token_index:
            continue
        for token in useful_terms:
            postings = sorted(
                token_index.get(token, []),
                key=_popularity_quality_key,
                reverse=True,
            )[:per_token_limit]
            token_boost = _beauty_token_boost(token) if category in BEAUTY_CATEGORIES else 1.0
            for rank, item in enumerate(postings):
                candidate_items[item.item_id] = item
                candidate_scores[item.item_id] += category_weight * token_boost / math.sqrt(rank + 1)

    if not candidate_scores:
        return []

    max_raw = max(candidate_scores.values(), default=1.0)
    scored = []
    for item_id, raw_score in candidate_scores.items():
        item = candidate_items[item_id]
        item_term_set = catalog.item_terms.get(item_id, set())
        overlap_score = len(set(useful_terms) & item_term_set) / max(len(set(useful_terms)), 1)
        quality = (item.average_rating or 3.5) / 5
        popularity = int(item.metadata.get("rating_number") or item.metadata.get("review_count") or 0)
        popularity_score = min(math.log1p(popularity) / math.log1p(500), 1.0) if popularity > 0 else 0.0
        score = (
            0.50 * min(raw_score / max_raw, 1.0)
            + 0.24 * overlap_score
            + 0.16 * quality
            + 0.10 * popularity_score
        )
        source = "beauty_aspect_profile" if item.category in BEAUTY_CATEGORIES else "aspect_profile"
        scored.append((item, round(min(max(score, 0.0), 1.0), 4), source))
    scored.sort(key=lambda row: (row[1], _popularity_quality_key(row[0])), reverse=True)
    return scored


def _sparse_category_tail_candidates(
    user_profile: UserProfile,
    catalog: CandidateCatalog,
    limit: int,
) -> list[tuple[Item, float, str]]:
    category_weights = _category_weights(user_profile)
    if not category_weights:
        return []
    output = []
    for category, category_weight in category_weights.items():
        category_items = catalog.by_category_popular.get(category, [])
        if not category_items:
            continue
        window = category_items[: max(limit * 12, 2000)]
        budget = max(1, int(limit * 0.12))
        stride = max(1, len(window) // budget)
        for rank, item in enumerate(window[::stride][:budget]):
            quality = (item.average_rating or 3.5) / 5
            score = (0.60 * category_weight) + (0.25 * quality) + (0.15 * (1.0 - rank / budget))
            source = "beauty_sparse_tail" if item.category in BEAUTY_CATEGORIES else "sparse_category_tail"
            output.append((item, round(min(max(score, 0.0), 1.0), 4), source))
    output.sort(key=lambda row: (row[1], _popularity_quality_key(row[0])), reverse=True)
    return output


def _beauty_taxonomy_candidates(
    user_profile: UserProfile,
    catalog: CandidateCatalog,
    query_terms: list[str],
    history: list[UserHistoryItem],
    limit: int,
) -> list[tuple[Item, float]]:
    if not _should_use_beauty_taxonomy(user_profile, query_terms, history):
        return []

    token_index = catalog.by_category_token_index.get("All_Beauty", {})
    if not token_index:
        return []

    evidence_terms = set(query_terms)
    for item in history:
        if item.rating >= 3:
            evidence_terms.update(
                token for token in terms(embedding_text(item.item_name, item.category, item.review))
            )
    evidence_terms.update(_phrase_terms(evidence_terms))

    active_groups = _active_beauty_groups(evidence_terms)
    if not active_groups and "All_Beauty" in user_profile.preferred_categories:
        active_groups = {
            "hair_care": 0.35,
            "skin_care": 0.35,
            "tools_accessories": 0.25,
            "natural_ingredients": 0.20,
        }
    if not active_groups:
        return []

    candidate_scores: Counter[str] = Counter()
    candidate_items: dict[str, Item] = {}
    per_token_limit = max(350, min(limit * 2, 1800))
    for group_name, group_weight in active_groups.items():
        group_terms = BEAUTY_ASPECT_GROUPS[group_name]
        for token in group_terms:
            postings = sorted(
                token_index.get(token, []),
                key=_popularity_quality_key,
                reverse=True,
            )[:per_token_limit]
            if not postings:
                continue
            token_weight = 1.35 if token in evidence_terms else 0.55
            token_weight *= _beauty_token_boost(token)
            for rank, item in enumerate(postings):
                candidate_items[item.item_id] = item
                candidate_scores[item.item_id] += (
                    group_weight * token_weight / math.sqrt(rank + 1)
                )

    if not candidate_scores:
        return []

    max_raw = max(candidate_scores.values(), default=1.0)
    scored = []
    for item_id, raw_score in candidate_scores.items():
        item = candidate_items[item_id]
        item_terms = catalog.item_terms.get(item_id, set())
        overlap = len(item_terms & evidence_terms) / max(len(evidence_terms), 1)
        quality = (item.average_rating or 3.5) / 5
        popularity = int(item.metadata.get("rating_number") or item.metadata.get("review_count") or 0)
        popularity_score = min(math.log1p(popularity) / math.log1p(1000), 1.0) if popularity else 0.0
        score = (
            0.52 * min(raw_score / max_raw, 1.0)
            + 0.20 * min(overlap, 1.0)
            + 0.16 * popularity_score
            + 0.12 * quality
        )
        scored.append((item, round(min(max(score, 0.0), 1.0), 4)))
    scored.sort(key=lambda row: (row[1], _popularity_quality_key(row[0])), reverse=True)
    return scored


def _should_use_beauty_taxonomy(
    user_profile: UserProfile,
    query_terms: list[str],
    history: list[UserHistoryItem],
) -> bool:
    if "All_Beauty" in user_profile.preferred_categories:
        return True
    if _looks_like_beauty_query(query_terms):
        return True
    return any(item.category in BEAUTY_CATEGORIES and item.rating >= 3 for item in history)


def _active_beauty_groups(evidence_terms: set[str]) -> dict[str, float]:
    active = {}
    for group_name, group_terms in BEAUTY_ASPECT_GROUPS.items():
        matches = evidence_terms & group_terms
        if matches:
            active[group_name] = min(0.45 + (0.18 * len(matches)), 1.25)
    return active


def _item_terms(item: Item) -> set[str]:
    metadata_text = " ".join(f"{key} {value}" for key, value in item.metadata.items())
    text = embedding_text(item.name, item.category, item.summary, metadata_text)
    return {
        token
        for token in terms(text)
        if token not in ASPECT_STOPWORDS and not token.isdigit()
    }


def _phrase_terms(tokens: set[str] | list[str]) -> set[str]:
    ordered = list(tokens)
    return {
        f"{left}_{right}"
        for left, right in zip(ordered, ordered[1:])
        if left != right
    }


def _looks_like_beauty_query(query_terms: list[str]) -> bool:
    beauty_terms = {
        "beauty",
        "body",
        "cream",
        "face",
        "hair",
        "lotion",
        "makeup",
        "moisturizer",
        "nail",
        "serum",
        "shampoo",
        "skin",
        "spray",
    }
    return bool(set(query_terms) & beauty_terms)


def _beauty_token_boost(token: str) -> float:
    strong_terms = {
        "anti-aging",
        "body",
        "cream",
        "face",
        "hair",
        "human",
        "laser",
        "lotion",
        "makeup",
        "moisturizer",
        "nail",
        "natural",
        "serum",
        "shampoo",
        "skin",
        "spray",
        "wave",
        "wig",
    }
    return 1.35 if token in strong_terms else 1.0


def _popularity_quality_key(item: Item) -> tuple[int, float, str]:
    return (
        int(item.metadata.get("rating_number") or item.metadata.get("review_count") or 0),
        item.average_rating or 0.0,
        item.item_id,
    )


def _select_balanced_candidates(
    selected: list[Item],
    sources: dict[str, list[str]],
    source_scores: dict[str, dict[str, float]],
    limit: int,
) -> list[Item]:
    order = {item.item_id: index for index, item in enumerate(selected)}

    def priority(item: Item) -> tuple[float, int, float, int]:
        scores = source_scores.get(item.item_id, {})
        source_score = max(
            (
                SOURCE_PRIORITIES.get(source, 0.50) + (0.18 * score)
                for source, score in scores.items()
            ),
            default=0.0,
        )
        diversity_bonus = min(len(scores), 4) * 0.025
        popularity = int(item.metadata.get("rating_number") or item.metadata.get("review_count") or 0)
        return (
            round(source_score + diversity_bonus, 6),
            popularity,
            item.average_rating or 0.0,
            -order.get(item.item_id, 0),
        )

    ranked = sorted(selected, key=priority, reverse=True)
    exploration_limit = max(10, int(limit * 0.08)) if limit >= 50 else max(1, int(limit * 0.10))
    protected_limit = max(limit - exploration_limit, 0)
    protected = [
        item
        for item in ranked
        if set(sources.get(item.item_id, [])) - EXPLORATION_SOURCES
    ]
    protected_source_ids = {item.item_id for item in protected}
    exploration = [
        item
        for item in ranked
        if item.item_id not in protected_source_ids
        and set(sources.get(item.item_id, [])) & EXPLORATION_SOURCES
    ]
    limited = protected[:protected_limit]
    limited_ids = {item.item_id for item in limited}
    for item in exploration:
        if len(limited) >= limit:
            break
        if item.item_id not in limited_ids:
            limited.append(item)
            limited_ids.add(item.item_id)
    for item in ranked:
        if len(limited) >= limit:
            break
        if item.item_id not in limited_ids:
            limited.append(item)
            limited_ids.add(item.item_id)
    return limited
