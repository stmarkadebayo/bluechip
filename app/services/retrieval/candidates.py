from __future__ import annotations

import heapq
import math
from collections import Counter, defaultdict
from dataclasses import dataclass, field

from app.models.schemas import Item, UserHistoryItem, UserProfile
from app.services.retrieval.embeddings import embedding_text, terms
from app.services.retrieval.evidence_graph import candidate_ids_from_evidence_graph
from app.services.retrieval.item_similarity import (
    candidate_ids_from_implicit_item_neighbors,
    candidate_ids_from_lexical_item_neighbors,
    candidate_ids_from_review_terms,
    candidate_ids_from_user_neighbors,
    scored_candidate_ids_from_history,
)
from app.services.retrieval.source_registry import candidate_selection_score
from app.services.retrieval.text import BM25Retriever
from app.services.retrieval.vector_store import FAISSVectorStore, LocalVectorRetriever


EXPLORATION_SOURCES = {
    "aspect_profile",
    "beauty_aspect_profile",
    "beauty_sparse_tail",
    "beauty_taxonomy_aspect",
    "beauty_taxonomy_window",
    "sparse_category_tail",
}
BEAUTY_EXPLORATION_SOURCES = {
    "beauty_aspect_profile",
    "beauty_sparse_tail",
    "beauty_taxonomy_aspect",
    "beauty_taxonomy_window",
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
    "bath_body": {
        "bath",
        "body",
        "butter",
        "exfoliating",
        "lotion",
        "salt",
        "scrub",
        "soap",
        "wash",
    },
    "fragrance_body": {
        "body",
        "cologne",
        "deodorant",
        "fragrance",
        "mist",
        "perfume",
        "scent",
        "scented",
    },
    "men_grooming": {
        "beard",
        "razor",
        "shave",
        "shaving",
        "trimmer",
    },
}
ASPECT_STOPWORDS = {
    "all_beauty",
    "about",
    "after",
    "amazon",
    "are",
    "been",
    "brand",
    "but",
    "can",
    "count",
    "could",
    "description",
    "dimensions",
    "discontinued",
    "does",
    "doesn",
    "for",
    "from",
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
    "items",
    "just",
    "like",
    "much",
    "need",
    "manufacturer",
    "number",
    "only",
    "order",
    "out",
    "package",
    "product",
    "products",
    "rating",
    "really",
    "review",
    "reviews",
    "she",
    "that",
    "them",
    "then",
    "these",
    "they",
    "unknown",
    "use",
    "very",
    "was",
    "what",
    "when",
    "will",
    "with",
    "without",
    "would",
    "you",
    "your",
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
    neural_retriever: FAISSVectorStore | None = None,
    catalog: CandidateCatalog | None = None,
    disabled_sources: set[str] | None = None,
    excluded_item_ids: set[str] | None = None,
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
        neural_retriever=neural_retriever,
        catalog=catalog,
        disabled_sources=disabled_sources,
        excluded_item_ids=excluded_item_ids,
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
    neural_retriever: FAISSVectorStore | None = None,
    catalog: CandidateCatalog | None = None,
    disabled_sources: set[str] | None = None,
    excluded_item_ids: set[str] | None = None,
    limit: int = 100,
) -> CandidatePool:
    catalog = catalog or CandidateCatalog.from_items(items)
    disabled_sources = disabled_sources or set()
    by_id = {item.item_id: item for item in items}
    selected: list[Item] = []
    history_item_ids = {item.item_id for item in history}
    excluded_item_ids = excluded_item_ids or set()
    seen: set[str] = set(history_item_ids)
    sources: dict[str, list[str]] = {}
    source_scores: dict[str, dict[str, float]] = {}

    def add_candidate(item: Item, source: str, score: float = 0.0) -> None:
        if source in disabled_sources:
            return
        if item.item_id in history_item_ids or item.item_id in excluded_item_ids:
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
    evidence_graph_index = (
        collaborative_index.get("evidence_graph_retrieval") if collaborative_index else None
    )
    implicit_item_neighbors = (
        collaborative_index.get("implicit_item_neighbors") if collaborative_index else None
    )

    if item_neighbors and "co_visitation" not in disabled_sources:
        neighbor_budget = max(1, int(limit * 0.35))
        for item_id, score in scored_candidate_ids_from_history(
            history,
            item_neighbors,
            limit=neighbor_budget,
        ):
            item = by_id.get(item_id)
            if item:
                add_candidate(item, "co_visitation", score)

    if collaborative_index and "user_neighbor" not in disabled_sources:
        user_neighbor_budget = max(1, int(limit * 0.35))
        for item_id, score in candidate_ids_from_user_neighbors(
            history,
            collaborative_index,
            limit=user_neighbor_budget,
        ):
            item = by_id.get(item_id)
            if item:
                add_candidate(item, "user_neighbor", score)

    if implicit_item_neighbors and "implicit_item_item" not in disabled_sources:
        implicit_budget = max(1, int(limit * 0.42))
        for item_id, score in candidate_ids_from_implicit_item_neighbors(
            history,
            implicit_item_neighbors,
            limit=implicit_budget,
        ):
            item = by_id.get(item_id)
            if item:
                add_candidate(item, "implicit_item_item", score)

    if review_term_index:
        review_term_budget = max(1, int(limit * 0.38))
        if not {"beauty_review_term_profile", "review_term_profile"} <= disabled_sources:
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
        if not {"beauty_lexical_item_neighbor", "lexical_item_neighbor"} <= disabled_sources:
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

    if evidence_graph_index and not {
        "aspect_evidence_graph",
        "category_aspect_graph",
        "sequential_transition",
        "category_transition",
    } <= disabled_sources:
        graph_budget = max(1, int(limit * (0.48 if len(history) <= 2 else 0.36)))
        graph_added = 0
        for item_id, score, source in candidate_ids_from_evidence_graph(
            user_profile=user_profile,
            history=history,
            context=context,
            evidence_graph=evidence_graph_index,
            limit=max(limit, graph_budget * 2),
        ):
            item = by_id.get(item_id)
            if item:
                had_source = source in sources.get(item.item_id, [])
                add_candidate(item, source, score)
                if item.item_id not in history_item_ids and not had_source:
                    graph_added += 1
            if graph_added >= graph_budget:
                break

    search_limit = min(len(items), limit + len(history_item_ids))
    bm25_target = max(1, int(limit * 0.35))
    if "bm25_profile" not in disabled_sources:
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
    if "vector_profile" not in disabled_sources and vector_retriever is not None:
        vector_added = 0
        for item, score in vector_retriever.search_with_scores(query or context, limit=search_limit):
            had_source = "vector_profile" in sources.get(item.item_id, [])
            add_candidate(item, "vector_profile", score)
            if item.item_id not in history_item_ids and not had_source:
                vector_added += 1
            if vector_added >= vector_target:
                break

    if neural_retriever is not None and neural_retriever._built and "neural_vector" not in disabled_sources:
        neural_target = max(vector_target, int(limit * 0.65))
        neural_added = 0
        for item, score in neural_retriever.search_with_scores(
            query or context, limit=min(search_limit, neural_retriever.index.ntotal)
        ):
            had_source = "neural_vector" in sources.get(item.item_id, [])
            add_candidate(item, "neural_vector", score)
            if item.item_id not in history_item_ids and not had_source:
                neural_added += 1
            if neural_added >= neural_target:
                break

    use_beauty_exploration = _should_use_beauty_taxonomy(user_profile, query_terms, history)
    aspect_share = 0.26 if len(history) <= 2 else 0.16
    if use_beauty_exploration:
        aspect_share = max(aspect_share, 0.30 if len(history) <= 2 else 0.22)
    aspect_budget = max(1, int(limit * aspect_share))
    aspect_added = 0
    if not {"aspect_profile", "beauty_aspect_profile"} <= disabled_sources:
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

    taxonomy_budget = _beauty_taxonomy_budget(limit=limit, history_size=len(history))
    taxonomy_added = 0
    if "beauty_taxonomy_aspect" not in disabled_sources:
        for item, score in _beauty_taxonomy_candidates(
            user_profile=user_profile,
            catalog=catalog,
            query_terms=query_terms,
            history=history,
            limit=limit,
        ):
            had_source = "beauty_taxonomy_aspect" in sources.get(item.item_id, [])
            add_candidate(item, "beauty_taxonomy_aspect", score)
            if item.item_id not in history_item_ids and not had_source:
                taxonomy_added += 1
            if taxonomy_added >= taxonomy_budget:
                break

    taxonomy_window_budget = _beauty_taxonomy_window_budget(
        limit=limit,
        history_size=len(history),
    )
    taxonomy_window_added = 0
    if "beauty_taxonomy_window" not in disabled_sources:
        for item, score in _beauty_taxonomy_window_candidates(
            user_profile=user_profile,
            catalog=catalog,
            query_terms=query_terms,
            history=history,
            limit=limit,
        ):
            had_source = "beauty_taxonomy_window" in sources.get(item.item_id, [])
            add_candidate(item, "beauty_taxonomy_window", score)
            if item.item_id not in history_item_ids and not had_source:
                taxonomy_window_added += 1
            if taxonomy_window_added >= taxonomy_window_budget:
                break

    if len(history) <= 2 and not {"sparse_category_tail", "beauty_sparse_tail"} <= disabled_sources:
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
    if "category_affinity_popular" not in disabled_sources:
        for item, score in _category_affinity_candidates(user_profile, catalog, limit):
            had_source = "category_affinity_popular" in sources.get(item.item_id, [])
            add_candidate(item, "category_affinity_popular", score)
            if item.item_id not in history_item_ids and not had_source:
                category_affinity_added += 1
            if category_affinity_added >= category_affinity_budget:
                break

    category_target = max(vector_target, int(limit * 0.60))
    if (
        "category_popular" not in disabled_sources
        and len(selected) < category_target
        and user_profile.preferred_categories
    ):
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
    fallback_budget = _global_popular_budget(
        limit=limit,
        history_size=len(history),
        query_terms=query_terms,
        selected_count=len(selected),
        catalog_size=len(fallback),
    )
    if "global_popular" not in disabled_sources:
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


def _global_popular_budget(
    limit: int,
    history_size: int,
    query_terms: list[str],
    selected_count: int,
    catalog_size: int,
) -> int:
    if catalog_size <= 0:
        return 0
    sparse_profile = history_size <= 2
    if sparse_profile:
        return min(catalog_size, max(limit * 2, int(limit * 0.50)))
    if selected_count < limit:
        return min(catalog_size, max(limit, limit - selected_count))
    context_heavy = len([term for term in query_terms if term not in ASPECT_STOPWORDS]) >= 5
    share = 0.20 if context_heavy else 0.35
    return min(catalog_size, max(1, int(limit * share)))


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


def _beauty_taxonomy_budget(limit: int, history_size: int) -> int:
    share = 0.16 if history_size <= 2 else 0.12
    return min(180, max(6, int(limit * share)))


def _beauty_taxonomy_window_budget(limit: int, history_size: int) -> int:
    share = 0.12 if history_size <= 2 else 0.08
    return min(140, max(8, int(limit * share)))


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

    evidence_terms = _beauty_evidence_terms(query_terms, history)

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


def _beauty_taxonomy_window_candidates(
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

    evidence_terms = _beauty_evidence_terms(query_terms, history)
    active_groups = _active_beauty_groups(evidence_terms)
    if not active_groups and "All_Beauty" in user_profile.preferred_categories:
        active_groups = {
            "hair_care": 0.28,
            "skin_care": 0.28,
            "makeup_color": 0.22,
            "tools_accessories": 0.20,
            "bath_body": 0.18,
            "natural_ingredients": 0.18,
        }
    if not active_groups:
        return []

    windows = _beauty_taxonomy_windows(limit)
    candidate_scores: Counter[str] = Counter()
    candidate_items: dict[str, Item] = {}
    candidate_groups: dict[str, set[str]] = defaultdict(set)
    candidate_windows: dict[str, set[int]] = defaultdict(set)
    max_depth = max(stop for _, stop, _ in windows)
    per_window_take = max(4, min(18, int(limit * 0.018) or 4))

    for group_name, group_weight in active_groups.items():
        group_terms = BEAUTY_ASPECT_GROUPS[group_name]
        group_tokens = _beauty_group_window_tokens(
            group_terms=group_terms,
            evidence_terms=evidence_terms,
            query_terms=query_terms,
        )
        for token in group_tokens:
            postings = _top_popularity_items(token_index.get(token, []), max_depth)
            if not postings:
                continue
            token_weight = 1.25 if token in evidence_terms else 0.62
            token_weight *= _beauty_token_boost(token)
            for window_index, (start, stop, window_weight) in enumerate(windows):
                if start >= len(postings):
                    continue
                window_postings = _sample_window_items(postings, start, stop, per_window_take)
                for local_rank, item in enumerate(window_postings):
                    candidate_items[item.item_id] = item
                    candidate_groups[item.item_id].add(group_name)
                    candidate_windows[item.item_id].add(window_index)
                    candidate_scores[item.item_id] += (
                        group_weight * token_weight * window_weight / math.sqrt(local_rank + 1)
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
        group_diversity = min(len(candidate_groups[item_id]) / 3, 1.0)
        window_diversity = min(len(candidate_windows[item_id]) / 2, 1.0)
        score = (
            0.42 * min(raw_score / max_raw, 1.0)
            + 0.18 * min(overlap, 1.0)
            + 0.14 * quality
            + 0.10 * popularity_score
            + 0.10 * group_diversity
            + 0.06 * window_diversity
        )
        scored.append((item, round(min(max(score, 0.0), 1.0), 4)))
    scored.sort(key=lambda row: (row[1], _popularity_quality_key(row[0])), reverse=True)
    return scored


def _beauty_evidence_terms(
    query_terms: list[str],
    history: list[UserHistoryItem],
) -> set[str]:
    evidence_terms = set(query_terms)
    for item in history:
        if item.rating >= 3:
            evidence_terms.update(
                token for token in terms(embedding_text(item.item_name, item.category, item.review))
            )
    evidence_terms.update(_phrase_terms(evidence_terms))
    return evidence_terms


def _beauty_group_window_tokens(
    group_terms: set[str],
    evidence_terms: set[str],
    query_terms: list[str],
) -> list[str]:
    tokens: list[str] = []
    for token in query_terms:
        if token in group_terms and token not in tokens:
            tokens.append(token)
    for token in sorted(evidence_terms & group_terms):
        if token not in tokens:
            tokens.append(token)
    fallback_terms = sorted(
        group_terms,
        key=lambda token: (_beauty_token_boost(token), token in ASPECT_STOPWORDS, token),
        reverse=True,
    )
    for token in fallback_terms:
        if token not in tokens:
            tokens.append(token)
        if len(tokens) >= 8:
            break
    return tokens[:8]


def _beauty_taxonomy_windows(limit: int) -> tuple[tuple[int, int, float], ...]:
    shallow_stop = max(35, int(limit * 0.06))
    mid_stop = max(120, int(limit * 0.22))
    deep_stop = max(450, int(limit * 1.20))
    tail_stop = max(900, int(limit * 5.00))
    return (
        (0, shallow_stop, 1.00),
        (shallow_stop, mid_stop, 0.88),
        (mid_stop, deep_stop, 0.66),
        (deep_stop, tail_stop, 0.48),
    )


def _top_popularity_items(items: list[Item], limit: int) -> list[Item]:
    if len(items) <= limit:
        return sorted(items, key=_popularity_quality_key, reverse=True)
    return heapq.nlargest(limit, items, key=_popularity_quality_key)


def _sample_window_items(items: list[Item], start: int, stop: int, limit: int) -> list[Item]:
    window = items[start:stop]
    if len(window) <= limit:
        return window
    if limit <= 1:
        return window[:1]
    step = (len(window) - 1) / (limit - 1)
    sampled = []
    used_offsets = set()
    for index in range(limit):
        offset = round(index * step)
        if offset in used_offsets:
            continue
        sampled.append(window[offset])
        used_offsets.add(offset)
    return sampled


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
    return CandidateMixer(
        selected=selected,
        sources=sources,
        source_scores=source_scores,
        limit=limit,
    ).mix()


class CandidateMixer:
    def __init__(
        self,
        selected: list[Item],
        sources: dict[str, list[str]],
        source_scores: dict[str, dict[str, float]],
        limit: int,
    ) -> None:
        self.selected = selected
        self.sources = sources
        self.source_scores = source_scores
        self.limit = limit
        self.order = {item.item_id: index for index, item in enumerate(selected)}

    def mix(self) -> list[Item]:
        ranked = sorted(self.selected, key=self._priority, reverse=True)
        user_neighbor_floor = self._floor(ranked, "user_neighbor", 0.05)
        implicit_item_floor = self._floor(ranked, "implicit_item_item", 0.08)
        foundation = self._popularity_floor()
        exploration_limit = self._exploration_limit(ranked)
        protected_limit = max(self.limit - exploration_limit, 0)
        protected, exploration = self._split_protected_and_exploration(ranked)

        limited: list[Item] = []
        limited_ids: set[str] = set()
        self._append_unique(limited, limited_ids, user_neighbor_floor, self.limit)
        self._append_unique(limited, limited_ids, implicit_item_floor, self.limit)
        self._append_unique(limited, limited_ids, foundation, self.limit)
        self._append_unique(limited, limited_ids, protected, protected_limit)
        self._append_unique(limited, limited_ids, exploration, self.limit)
        self._append_unique(limited, limited_ids, ranked, self.limit)
        return limited

    def _priority(self, item: Item) -> tuple[float, int, float, int]:
        scores = self.source_scores.get(item.item_id, {})
        source_score = max(
            (
                candidate_selection_score(source, score)
                for source, score in scores.items()
            ),
            default=0.0,
        )
        diversity_bonus = min(len(scores), 4) * 0.025
        popularity = int(
            item.metadata.get("rating_number") or item.metadata.get("review_count") or 0
        )
        return (
            round(source_score + diversity_bonus, 6),
            popularity,
            item.average_rating or 0.0,
            -self.order.get(item.item_id, 0),
        )

    def _floor(self, ranked: list[Item], source: str, share: float) -> list[Item]:
        if self.limit < 100:
            return []
        return _source_floor_candidates(
            ranked=ranked,
            sources=self.sources,
            source_scores=self.source_scores,
            source=source,
            limit=max(1, int(self.limit * share)),
        )

    def _popularity_floor(self) -> list[Item]:
        popularity_floor = (
            max(20, int(self.limit * 0.06))
            if self.limit >= 100
            else max(1, int(self.limit * 0.10))
        )
        return sorted(
            (
                item
                for item in self.selected
                if "global_popular" in self.sources.get(item.item_id, [])
            ),
            key=_popularity_quality_key,
            reverse=True,
        )[:popularity_floor]

    def _exploration_limit(self, ranked: list[Item]) -> int:
        exploration_limit = (
            max(10, int(self.limit * 0.08))
            if self.limit >= 50
            else max(1, int(self.limit * 0.10))
        )
        has_beauty_exploration = any(
            set(self.sources.get(item.item_id, [])) & BEAUTY_EXPLORATION_SOURCES
            for item in ranked
        )
        if not has_beauty_exploration:
            return exploration_limit
        beauty_exploration_limit = (
            min(180, max(12, int(self.limit * 0.16)))
            if self.limit >= 50
            else max(1, int(self.limit * 0.16))
        )
        return min(self.limit, max(exploration_limit, beauty_exploration_limit))

    def _split_protected_and_exploration(
        self,
        ranked: list[Item],
    ) -> tuple[list[Item], list[Item]]:
        protected = [
            item
            for item in ranked
            if set(self.sources.get(item.item_id, [])) - EXPLORATION_SOURCES
        ]
        protected_source_ids = {item.item_id for item in protected}
        exploration = [
            item
            for item in ranked
            if item.item_id not in protected_source_ids
            and set(self.sources.get(item.item_id, [])) & EXPLORATION_SOURCES
        ]
        return protected, exploration

    def _append_unique(
        self,
        output: list[Item],
        output_ids: set[str],
        candidates: list[Item],
        limit: int,
    ) -> None:
        for item in candidates:
            if len(output) >= limit:
                break
            if item.item_id not in output_ids:
                output.append(item)
                output_ids.add(item.item_id)


def _source_floor_candidates(
    ranked: list[Item],
    sources: dict[str, list[str]],
    source_scores: dict[str, dict[str, float]],
    source: str,
    limit: int,
) -> list[Item]:
    if limit <= 0:
        return []
    candidates = [
        item
        for item in ranked
        if source in sources.get(item.item_id, [])
    ]
    candidates.sort(
        key=lambda item: (
            source_scores.get(item.item_id, {}).get(source, 0.0),
            _popularity_quality_key(item),
        ),
        reverse=True,
    )
    return candidates[:limit]
