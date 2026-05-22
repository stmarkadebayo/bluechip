from __future__ import annotations

from dataclasses import dataclass


SOURCE_FAMILY_ORDER = (
    "collaborative_co_engagement",
    "lexical_review_term",
    "semantic_vector",
    "neural_semantic_vector",
    "aspect_evidence",
    "popularity_fallback",
    "other",
)

SOURCE_FAMILY_LABELS = {
    "collaborative_co_engagement": "collaborative/co-engagement",
    "lexical_review_term": "lexical/review-term",
    "semantic_vector": "semantic/vector",
    "neural_semantic_vector": "neural/semantic-vector",
    "aspect_evidence": "aspect/evidence",
    "popularity_fallback": "popularity/fallback",
    "other": "other",
}

SOURCE_FAMILY_CONFIDENCE = {
    "collaborative_co_engagement": 1.00,
    "aspect_evidence": 0.86,
    "lexical_review_term": 0.68,
    "semantic_vector": 0.45,
    "neural_semantic_vector": 0.45,
    "popularity_fallback": 0.38,
    "other": 0.30,
}


@dataclass(frozen=True)
class RetrievalSourceSpec:
    name: str
    family: str
    priority: float
    default_disabled: bool = False
    no_context_disabled: bool = False


RETRIEVAL_SOURCE_SPECS: tuple[RetrievalSourceSpec, ...] = (
    RetrievalSourceSpec("neural_vector", "neural_semantic_vector", 0.90, default_disabled=True),
    RetrievalSourceSpec("beauty_review_term_profile", "lexical_review_term", 0.87),
    RetrievalSourceSpec("implicit_item_item", "collaborative_co_engagement", 0.865),
    RetrievalSourceSpec("beauty_lexical_item_neighbor", "lexical_review_term", 0.86),
    RetrievalSourceSpec("category_aspect_graph", "aspect_evidence", 0.855),
    RetrievalSourceSpec("sequential_transition", "collaborative_co_engagement", 0.845),
    RetrievalSourceSpec("beauty_aspect_profile", "aspect_evidence", 0.84),
    RetrievalSourceSpec("aspect_evidence_graph", "aspect_evidence", 0.835),
    RetrievalSourceSpec("category_transition", "collaborative_co_engagement", 0.825),
    RetrievalSourceSpec("review_term_profile", "lexical_review_term", 0.82),
    RetrievalSourceSpec("lexical_item_neighbor", "lexical_review_term", 0.81),
    RetrievalSourceSpec("beauty_taxonomy_aspect", "aspect_evidence", 0.812),
    RetrievalSourceSpec("beauty_taxonomy_window", "aspect_evidence", 0.805),
    RetrievalSourceSpec("aspect_profile", "aspect_evidence", 0.80),
    RetrievalSourceSpec(
        "beauty_sparse_tail",
        "popularity_fallback",
        0.79,
        default_disabled=True,
    ),
    RetrievalSourceSpec(
        "sparse_category_tail",
        "popularity_fallback",
        0.77,
        default_disabled=True,
    ),
    RetrievalSourceSpec("category_affinity_popular", "popularity_fallback", 0.83),
    RetrievalSourceSpec("category_popular", "popularity_fallback", 0.81),
    RetrievalSourceSpec("bm25_profile", "lexical_review_term", 0.78, no_context_disabled=True),
    RetrievalSourceSpec("global_popular", "popularity_fallback", 0.76),
    RetrievalSourceSpec("vector_profile", "semantic_vector", 0.74, default_disabled=True),
    RetrievalSourceSpec("user_neighbor", "collaborative_co_engagement", 0.72),
    RetrievalSourceSpec("co_visitation", "collaborative_co_engagement", 0.70),
)

RETRIEVAL_SOURCE_BY_NAME = {spec.name: spec for spec in RETRIEVAL_SOURCE_SPECS}
SOURCE_PRIORITIES = {
    spec.name: spec.priority
    for spec in RETRIEVAL_SOURCE_SPECS
}
SOURCES_BY_FAMILY = {
    family: {
        spec.name
        for spec in RETRIEVAL_SOURCE_SPECS
        if spec.family == family
    }
    for family in SOURCE_FAMILY_ORDER
}


def retrieval_source_family(source: str) -> str:
    spec = RETRIEVAL_SOURCE_BY_NAME.get(source)
    return spec.family if spec else "other"


def default_disabled_retrieval_sources(context: str) -> set[str]:
    has_context = bool(context.strip())
    return {
        spec.name
        for spec in RETRIEVAL_SOURCE_SPECS
        if spec.default_disabled or (spec.no_context_disabled and not has_context)
    }


def calibrated_source_score(source: str, raw_score: float) -> float:
    family = retrieval_source_family(source)
    confidence = SOURCE_FAMILY_CONFIDENCE.get(family, SOURCE_FAMILY_CONFIDENCE["other"])
    return min(max(raw_score, 0.0), 1.0) * confidence


def candidate_selection_score(source: str, raw_score: float) -> float:
    return SOURCE_PRIORITIES.get(source, 0.50) + (0.12 * calibrated_source_score(source, raw_score))
