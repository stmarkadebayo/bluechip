from __future__ import annotations

from app.agents.recommendation_agent import RecommendationAgent
from app.models.schemas import Item, RecommendationRequest, UserHistoryItem
from scripts.build_splits import build_temporal_splits


def test_recommendation_agent_filters_seen_items() -> None:
    request = RecommendationRequest(
        user_persona="A practical listener who likes calm acoustic music.",
        user_history=[
            UserHistoryItem(
                item_id="song_seen",
                item_name="Quiet Guitar",
                rating=5,
                review="Calm acoustic guitar and a very relaxing listen.",
                category="music",
            )
        ],
        candidate_items=[
            Item(
                item_id="song_seen",
                name="Quiet Guitar",
                category="music",
                summary="Calm acoustic guitar and relaxing production.",
                average_rating=4.8,
            ),
            Item(
                item_id="song_new",
                name="Soft Piano",
                category="music",
                summary="Calm instrumental piano for focused listening.",
                average_rating=4.7,
            ),
        ],
        limit=2,
    )

    response = RecommendationAgent().run(request)

    assert [item.item_id for item in response.recommendations] == ["song_new"]


def test_temporal_split_holds_out_latest_review() -> None:
    reviews = [
        {
            "review_id": "r1",
            "user_id": "u1",
            "item_id": "a",
            "item_name": "A",
            "rating": 4,
            "review": "first",
            "timestamp": 10,
        },
        {
            "review_id": "r2",
            "user_id": "u1",
            "item_id": "b",
            "item_name": "B",
            "rating": 5,
            "review": "second",
            "timestamp": 20,
        },
    ]

    train, test_a, test_b = build_temporal_splits(reviews, min_history=1)

    assert [row["review_id"] for row in train] == ["r1"]
    assert [row["review_id"] for row in test_a] == ["r2"]
    assert test_b == test_a
