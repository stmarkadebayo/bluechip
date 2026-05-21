from __future__ import annotations

import json
from pathlib import Path

from app.models.schemas import Item, UserHistoryItem
from app.services.generation.review_plan import build_review_plan, fallback_review_from_plan
from app.services.intelligence.aspects import aspect_overlap, item_aspect_evidence, user_aspect_evidence
from app.services.profiling.item_profile import build_item_profile
from app.services.profiling.user_profile import build_user_profile
from app.services.retrieval.evidence_graph import (
    build_evidence_graph_index,
    candidate_ids_from_evidence_graph,
)


def test_aspect_evidence_extracts_user_and_item_fit() -> None:
    history = [
        UserHistoryItem(
            item_id="a",
            item_name="Calm Grill",
            rating=5,
            review="Affordable, quiet, fast service with spicy food.",
            category="restaurant",
        )
    ]
    user = user_aspect_evidence("A Lagos student on a budget.", history)
    item = item_aspect_evidence(
        Item(
            item_id="b",
            name="Pepper House",
            category="restaurant",
            metadata={"price": "affordable", "service": "fast"},
            summary="Spicy food in a calm place.",
            average_rating=4.4,
        )
    )

    assert user.aspect_scores["price_value"] > 0
    assert item.aspect_scores["service_speed"] > 0
    assert aspect_overlap(user.aspect_scores, item.aspect_scores) > 0
    assert "lagos" in user.nigerian_context


def test_evidence_graph_returns_aspect_and_transition_candidates() -> None:
    train = [
        {
            "review_id": "r1",
            "user_id": "u1",
            "item_id": "a",
            "item_name": "Budget Cream",
            "rating": 5,
            "review": "Affordable gentle skin cream.",
            "category": "All_Beauty",
            "timestamp": 1,
        },
        {
            "review_id": "r2",
            "user_id": "u1",
            "item_id": "b",
            "item_name": "Gentle Serum",
            "rating": 5,
            "review": "Gentle reliable serum for skin.",
            "category": "All_Beauty",
            "timestamp": 2,
        },
    ]
    items = [
        {
            "item_id": "a",
            "name": "Budget Cream",
            "category": "All_Beauty",
            "metadata": {"review_count": 5},
            "summary": "Affordable gentle skin cream.",
            "average_rating": 4.3,
        },
        {
            "item_id": "b",
            "name": "Gentle Serum",
            "category": "All_Beauty",
            "metadata": {"review_count": 4},
            "summary": "Gentle reliable serum for skin.",
            "average_rating": 4.5,
        },
    ]
    graph = build_evidence_graph_index(train=train, items=items, top_k=10)
    history = [
        UserHistoryItem(
            item_id="a",
            item_name="Budget Cream",
            rating=5,
            review="Affordable gentle skin cream.",
            category="All_Beauty",
            timestamp=1,
        )
    ]
    profile = build_user_profile("A budget beauty shopper with sensitive skin.", history)

    candidates = candidate_ids_from_evidence_graph(profile, history, "skin serum", graph, limit=5)

    assert candidates
    assert candidates[0][0] == "b"
    assert candidates[0][2] in {"sequential_transition", "category_aspect_graph", "aspect_evidence_graph"}


def test_review_plan_fallback_is_grounded() -> None:
    history = [
        UserHistoryItem(
            item_id="a",
            item_name="Quiet Bowl",
            rating=5,
            review="Quiet affordable food and fast service.",
            category="restaurant",
        )
    ]
    user_profile = build_user_profile("A Nigerian student who values affordable quiet places.", history, locale="Nigeria")
    item_profile = build_item_profile(
        Item(
            item_id="b",
            name="Mainland Grill",
            category="restaurant",
            metadata={"price": "affordable", "ambience": "quiet"},
            summary="Quiet affordable grill with fast service.",
            average_rating=4.4,
        )
    )

    plan = build_review_plan(user_profile, item_profile, predicted_rating=4)
    review = fallback_review_from_plan(item_profile, plan)

    assert "Mainland Grill 4 out of 5" in review
    assert "Nigerian shopper" in review
    assert plan.aspect_scores


def test_empty_collaborative_index_still_attaches_evidence_graph(
    tmp_path: Path,
    monkeypatch,
) -> None:
    from app.serving.orchestrators import recommendation as recommendation_orchestrator

    collaborative_path = tmp_path / "collaborative_retrieval.json"
    evidence_path = tmp_path / "evidence_graph_retrieval.json"
    collaborative_path.write_text(
        json.dumps(
            {
                "type": "collaborative_retrieval",
                "item_neighbors": {},
                "user_positive_items": {},
                "item_positive_users": {},
            }
        ),
        encoding="utf-8",
    )
    evidence_path.write_text(
        json.dumps(
            {
                "type": "evidence_graph",
                "aspect_items": {},
                "category_aspect_items": {},
                "item_transitions": {},
                "category_transitions": {},
            }
        ),
        encoding="utf-8",
    )
    monkeypatch.setenv("TASK_B_RETRIEVAL_INDEX", str(collaborative_path))
    recommendation_orchestrator._load_collaborative_index.cache_clear()

    payload = recommendation_orchestrator._load_collaborative_index()

    recommendation_orchestrator._load_collaborative_index.cache_clear()
    assert payload
    assert payload["evidence_graph_retrieval"]["type"] == "evidence_graph"
