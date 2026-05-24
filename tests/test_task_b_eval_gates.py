from __future__ import annotations

from app.models.schemas import UserHistoryItem
from eval.eval_task_b import (
    _filter_task_b_targets,
    _shard_eval_rows,
    _slice_candidate_metrics,
    _target_rating_distribution,
    _task_b_promotion_gate,
)


def test_task_b_target_mode_filters_positive_recommendations() -> None:
    rows = [
        {"item_id": "liked", "rating": 5},
        {"item_id": "neutral", "rating": 3},
        {"item_id": "missing_rating"},
    ]

    assert [row["item_id"] for row in _filter_task_b_targets(rows, "all_interactions")] == [
        "liked",
        "neutral",
        "missing_rating",
    ]
    assert [
        row["item_id"] for row in _filter_task_b_targets(rows, "positive_recommendation")
    ] == ["liked"]

    distribution = _target_rating_distribution(rows)

    assert distribution["total"] == 3
    assert distribution["rating_4_5"]["count"] == 1
    assert distribution["rating_1_3"]["count"] == 2


def test_task_b_sharding_partitions_rows_without_overlap() -> None:
    rows = [{"item_id": f"item-{index}"} for index in range(10)]

    shards = [_shard_eval_rows(rows, 3, shard_index) for shard_index in range(3)]
    flattened = [row["item_id"] for shard in shards for row in shard]

    assert [row["item_id"] for row in shards[0]] == ["item-0", "item-3", "item-6", "item-9"]
    assert sorted(flattened) == [row["item_id"] for row in rows]
    assert len(flattened) == len(set(flattened))


def test_task_b_slice_metrics_include_gate_slices_when_available() -> None:
    test_b = [
        {"user_id": "u1", "item_id": "hair_target", "category": "All_Beauty"},
        {"user_id": "u2", "item_id": "music_target", "category": "Digital_Music"},
        {"user_id": "u3", "item_id": "gift_target", "category": "For Her"},
    ]
    history_map = {
        "u1": [
            UserHistoryItem(
                item_id="skin_seen",
                item_name="Skin Cream",
                rating=5,
                review="Gentle face cream.",
                category="All_Beauty",
            )
        ],
        "u2": [
            UserHistoryItem(
                item_id="book_seen",
                item_name="Quiet Book",
                rating=5,
                review="Useful reference.",
                category="Books",
            )
        ],
        "u3": [],
    }
    contexts = [
        "Need a hair styling product under budget.",
        "",
        "Need a low-risk gift option.",
    ]

    slices = _slice_candidate_metrics(
        test_b=test_b,
        history_map=history_map,
        contexts=contexts,
        positives=["hair_target", "music_target", "gift_target"],
        candidate_ids=[["hair_target"], ["music_target"], ["gift_target"]],
        candidate_k=10,
    )

    assert slices["all"]["examples"] == 3
    assert slices["sparse_history_1_2"]["examples"] == 3
    assert slices["cross_domain"]["examples"] == 1
    assert slices["cold_start"]["examples"] == 1
    assert slices["context_heavy"]["examples"] == 2
    assert slices["intent_heavy"]["examples"] == 2
    assert slices["all"]["hybrid_candidate_recall@10"] == 1.0


def test_task_b_promotion_gate_reports_pass_and_reject_decisions() -> None:
    passing = _task_b_promotion_gate(
        metrics={
            "hybrid_candidate_recall@50": 0.14,
            "hybrid_candidate_recall@100": 0.19,
            "hybrid_candidate_recall@1000": 0.35,
            "base_candidate_recall@1000": 0.30,
            "hybrid_ranker_hit_rate@10": 0.11,
            "hybrid_ranker_ndcg@10": 0.08,
            "filtered_popularity_hit_rate@10": 0.09,
            "filtered_popularity_ndcg@10": 0.07,
        },
        slices={
            "sparse_history_1_2": {
                "examples": 20,
                "hybrid_candidate_recall@1000": 0.37,
            },
            "cross_domain": {
                "examples": 5,
                "hybrid_candidate_recall@1000": 0.55,
            },
            "context_heavy": {
                "examples": 5,
                "hybrid_candidate_recall@1000": 0.35,
            },
        },
        examples=25,
        k=10,
        candidate_k=1000,
    )

    assert passing["decision"] == "pass"
    assert passing["promotion_ready"] is True
    assert passing["slice_availability"]["cold_start"]["available"] is False

    rejecting = _task_b_promotion_gate(
        metrics={
            "hybrid_candidate_recall@50": 0.14,
            "hybrid_candidate_recall@100": 0.19,
            "hybrid_candidate_recall@1000": 0.35,
            "hybrid_ranker_hit_rate@10": 0.11,
            "hybrid_ranker_ndcg@10": 0.08,
        },
        slices={
            "sparse_history_1_2": {
                "examples": 20,
                "hybrid_candidate_recall@1000": 0.37,
            },
        },
        examples=25,
        k=10,
        candidate_k=1000,
    )

    assert rejecting["decision"] == "reject"
    assert {
        (check["scope"], check["reason"])
        for check in rejecting["failed_checks"]
    } == {("cross_domain", "required slice unavailable")}
