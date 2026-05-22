from __future__ import annotations

from fastapi.testclient import TestClient

from app.main import app
from app.serving.orchestrators import recommendation as recommendation_orchestrator
from app.services.generation import generator


client = TestClient(app)


def test_health_and_metrics_contracts() -> None:
    health = client.get("/api/health")
    assert health.status_code == 200
    assert health.json() == {"status": "ok"}

    metrics = client.get("/api/metrics")
    assert metrics.status_code == 200
    payload = metrics.json()
    assert {"requests", "by_endpoint", "average_latency_ms"} <= set(payload)
    assert {
        "by_generation_provider",
        "validation_failure_rate",
        "fallback_rate",
        "retrieval_source_counts",
        "model_version_counts",
        "index_version_counts",
    } <= set(payload)

    registry = client.get("/api/runtime/registry")
    assert registry.status_code == 200
    assert registry.json()["type"] == "bluechip_model_registry"

    feature_store = client.get("/api/runtime/feature-store")
    assert feature_store.status_code == 200
    assert {"root", "version", "available", "artifacts", "counts"} <= set(feature_store.json())


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

    traces = client.get("/api/traces").json()
    trace = next(row for row in traces if row["trace_id"] == payload["trace_id"])
    assert trace["model_versions"]["task_a_serving_head"]
    assert trace["validation_status"] in {"ok", "warning"}


def test_generation_fallback_is_recorded_in_trace(monkeypatch) -> None:
    class FailingProvider:
        def generate(self, instructions: str, prompt: str) -> str:
            del instructions, prompt
            raise RuntimeError("provider unavailable")

    monkeypatch.setattr(generator, "get_generation_provider", lambda: FailingProvider())

    response = client.post(
        "/api/simulate-review",
        json={
            "user_persona": "A strict buyer who dislikes noisy products.",
            "user_history": [],
            "target_item": {
                "item_id": "fan-1",
                "name": "Quiet Fan",
                "category": "appliance",
                "summary": "Compact fan with quiet mode.",
                "average_rating": 4.0,
            },
        },
    )

    assert response.status_code == 200
    payload = response.json()
    traces = client.get("/api/traces").json()
    trace = next(row for row in traces if row["trace_id"] == payload["trace_id"])
    assert trace["fallback_reason"] == "provider unavailable"


def test_recommend_contract_exposes_score_components(monkeypatch) -> None:
    monkeypatch.setattr(recommendation_orchestrator, "_load_neural_index", lambda: None)

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

    traces = client.get("/api/traces").json()
    trace = next(row for row in traces if row["trace_id"] == payload["trace_id"])
    assert trace["model_versions"]["task_b_ranker"]
    assert trace["index_versions"]
    assert trace["retrieval_source_counts"]


def test_ui_demo_surface_exposes_submission_flow() -> None:
    response = client.get("/ui/")

    assert response.status_code == 200
    html = response.text
    assert "Bluechip Demo Console" in html
    assert "Task B Recommend" in html
    assert "Task A Review" in html
    assert "Cold-start Dinner" in html
    assert "Cross-domain Gift" in html
    assert "Candidate Sources" in html
    assert "score components" in html
