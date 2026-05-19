from __future__ import annotations

from fastapi.testclient import TestClient

from app.main import app


client = TestClient(app)


def test_health_and_metrics_contracts() -> None:
    health = client.get("/api/health")
    assert health.status_code == 200
    assert health.json() == {"status": "ok"}

    metrics = client.get("/api/metrics")
    assert metrics.status_code == 200
    payload = metrics.json()
    assert {"requests", "by_endpoint", "average_latency_ms"} <= set(payload)


def test_simulate_review_contract_includes_trace() -> None:
    response = client.post(
        "/api/simulate-review",
        json={
            "user_persona": "A practical diner who likes quiet affordable restaurants.",
            "user_history": [
                {
                    "item_id": "a",
                    "item_name": "Quiet Bowl",
                    "rating": 5,
                    "review": "Quiet, affordable, and fast.",
                    "category": "restaurant",
                }
            ],
            "target_item": {
                "item_id": "b",
                "name": "Calm Grill",
                "category": "restaurant",
                "metadata": {"ambience": "quiet", "review_count": 10},
                "summary": "A quiet grill with fast service.",
                "average_rating": 4.5,
            },
            "locale": "Nigeria",
        },
    )

    assert response.status_code == 200
    payload = response.json()
    assert payload["trace_id"]
    assert 1 <= payload["predicted_rating"] <= 5
    assert payload["agent_trace"][-1]["step"] == "validate"


def test_recommend_contract_exposes_score_components() -> None:
    response = client.post(
        "/api/recommend",
        json={
            "user_persona": "A practical diner who likes quiet affordable restaurants.",
            "user_history": [],
            "context": "Dinner where conversation is easy.",
            "candidate_items": [
                {
                    "item_id": "b",
                    "name": "Calm Grill",
                    "category": "restaurant",
                    "metadata": {"ambience": "quiet", "review_count": 10},
                    "summary": "A quiet grill with fast service.",
                    "average_rating": 4.5,
                }
            ],
            "limit": 1,
        },
    )

    assert response.status_code == 200
    payload = response.json()
    assert payload["trace_id"]
    assert payload["recommendations"][0]["score_components"]["vector_match"] >= 0
    assert payload["candidate_diagnostics"]["source_counts"]
    assert "candidate_sources" in payload["recommendations"][0]
