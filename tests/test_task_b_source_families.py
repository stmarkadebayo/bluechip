from __future__ import annotations

from app.models.schemas import Item
from app.services.retrieval.candidates import CandidatePool
from eval.eval_task_b import (
    _source_diagnostics,
    _source_family,
    _source_family_diagnostics,
)


def test_task_b_source_family_diagnostics_group_counts_and_recall() -> None:
    pools = [
        CandidatePool(
            items=[
                _item("collab_hit"),
                _item("lexical_candidate"),
                _item("vector_candidate"),
                _item("evidence_candidate"),
                _item("popular_candidate"),
            ],
            sources={
                "collab_hit": ["co_visitation", "user_neighbor"],
                "lexical_candidate": ["review_term_profile", "bm25_profile"],
                "vector_candidate": ["vector_profile"],
                "evidence_candidate": ["aspect_evidence_graph"],
                "popular_candidate": ["global_popular"],
            },
        ),
        CandidatePool(
            items=[_item("other_hit"), _item("tail_candidate")],
            sources={
                "other_hit": ["custom_retriever"],
                "tail_candidate": ["beauty_sparse_tail"],
            },
        ),
    ]

    diagnostics = _source_family_diagnostics(
        pools=pools,
        positives=["collab_hit", "other_hit"],
        k=5,
    )

    assert diagnostics["collaborative_co_engagement"]["count"] == 2
    assert diagnostics["collaborative_co_engagement"]["hits@5"] == 1
    assert diagnostics["collaborative_co_engagement"]["misses@5"] == 1
    assert diagnostics["collaborative_co_engagement"]["candidate_recall@5"] == 0.5
    assert diagnostics["lexical_review_term"]["count"] == 2
    assert diagnostics["semantic_vector"]["count"] == 1
    assert diagnostics["aspect_evidence"]["count"] == 1
    assert diagnostics["popularity_fallback"]["count"] == 2
    assert diagnostics["other"]["count"] == 1
    assert diagnostics["other"]["candidate_recall@5"] == 0.5
    assert _source_family("custom_retriever") == "other"


def test_task_b_source_diagnostics_report_per_source_hits_and_misses() -> None:
    pools = [
        CandidatePool(
            items=[_item("bm25_hit"), _item("vector_candidate")],
            sources={
                "bm25_hit": ["bm25_profile"],
                "vector_candidate": ["vector_profile"],
            },
        ),
        CandidatePool(
            items=[_item("other_candidate")],
            sources={"other_candidate": ["bm25_profile"]},
        ),
    ]

    diagnostics = _source_diagnostics(
        pools=pools,
        positives=["bm25_hit", "missing_positive"],
        k=2,
    )

    assert diagnostics["bm25_profile"]["family"] == "lexical_review_term"
    assert diagnostics["bm25_profile"]["count"] == 2
    assert diagnostics["bm25_profile"]["hits@2"] == 1
    assert diagnostics["bm25_profile"]["misses@2"] == 1
    assert diagnostics["bm25_profile"]["candidate_recall@2"] == 0.5
    assert diagnostics["vector_profile"]["candidate_recall@2"] == 0.0


def _item(item_id: str) -> Item:
    return Item(item_id=item_id, name=item_id, category="test")
