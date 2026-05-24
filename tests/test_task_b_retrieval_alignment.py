from __future__ import annotations

from app.services.retrieval.item_similarity import build_item_neighbors_from_reviews


def test_rating_weighted_item_neighbors_include_low_rated_next_interactions() -> None:
    reviews = [
        _review("u1", "liked", 5, 1),
        _review("u1", "low_rated_next", 2, 2),
        _review("u2", "liked", 5, 3),
        _review("u2", "other_positive", 5, 4),
    ]

    positive_only = build_item_neighbors_from_reviews(
        reviews,
        top_k=10,
        positive_threshold=4.0,
    )
    all_weighted = build_item_neighbors_from_reviews(
        reviews,
        top_k=10,
        positive_threshold=0.0,
        rating_weighted=True,
    )

    assert "low_rated_next" not in {
        row["item_id"] for row in positive_only.get("liked", [])
    }
    assert "low_rated_next" in {
        row["item_id"] for row in all_weighted.get("liked", [])
    }


def _review(user_id: str, item_id: str, rating: int, timestamp: int) -> dict:
    return {
        "user_id": user_id,
        "item_id": item_id,
        "rating": rating,
        "timestamp": timestamp,
        "review_id": f"{user_id}-{item_id}",
    }
