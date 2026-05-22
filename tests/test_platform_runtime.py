from __future__ import annotations

import json
import sqlite3

from fastapi import FastAPI
from fastapi.testclient import TestClient

from app.main import app
from app.middleware.rate_limit import RateLimitMiddleware
from app.platform.feature_store import LocalFeatureStore, SQLiteFeatureStore
from app.platform.model_registry import LocalModelRegistry
from app.models.schemas import AgentTraceStep
from app.services.conversation.state import ConversationManager
from app.services.profiling.user_profile import build_user_profile
from app.stores.trace_store import SQLiteTraceStore


def test_local_feature_store_loads_point_features(tmp_path) -> None:
    root = tmp_path / "features"
    root.mkdir()
    (root / "items.jsonl").write_text(
        json.dumps(
            {
                "item_id": "item_1",
                "name": "Practical Blender",
                "category": "kitchen",
                "metadata": {"review_count": 3},
                "summary": "Reliable daily blender.",
                "average_rating": 4.2,
            }
        )
        + "\n",
        encoding="utf-8",
    )
    (root / "train.jsonl").write_text(
        json.dumps(
            {
                "review_id": "r1",
                "user_id": "user_1",
                "item_id": "item_1",
                "item_name": "Practical Blender",
                "rating": 5,
                "review": "Reliable and simple.",
                "category": "kitchen",
                "timestamp": 1,
            }
        )
        + "\n",
        encoding="utf-8",
    )
    (root / "split_stats.json").write_text(
        json.dumps({"items": 1, "train": 1, "test_task_a": 0, "test_task_b": 0}),
        encoding="utf-8",
    )

    store = LocalFeatureStore(root)

    assert store.get_item("item_1").name == "Practical Blender"
    assert store.get_user_history("user_1")[0].rating == 5
    assert store.summary().counts["items"] == 1
    assert store.summary().available


def test_sqlite_feature_store_loads_indexed_point_features(tmp_path) -> None:
    db_path = tmp_path / "feature_store.sqlite"
    with sqlite3.connect(db_path) as conn:
        conn.executescript(
            """
            CREATE TABLE items (
                item_id TEXT PRIMARY KEY,
                category TEXT NOT NULL DEFAULT '',
                payload TEXT NOT NULL
            );
            CREATE TABLE reviews (
                review_id TEXT PRIMARY KEY,
                user_id TEXT NOT NULL,
                item_id TEXT NOT NULL,
                rating REAL NOT NULL,
                category TEXT,
                timestamp INTEGER NOT NULL DEFAULT 0,
                payload TEXT NOT NULL
            );
            """
        )
        conn.execute(
            "INSERT INTO items(item_id, category, payload) VALUES (?, ?, ?)",
            (
                "item_1",
                "kitchen",
                json.dumps(
                    {
                        "item_id": "item_1",
                        "name": "Practical Blender",
                        "category": "kitchen",
                        "summary": "Reliable daily blender.",
                        "average_rating": 4.2,
                    }
                ),
            ),
        )
        conn.execute(
            (
                "INSERT INTO reviews"
                "(review_id, user_id, item_id, rating, category, timestamp, payload) "
                "VALUES (?, ?, ?, ?, ?, ?, ?)"
            ),
            (
                "r1",
                "user_1",
                "item_1",
                5,
                "kitchen",
                1,
                json.dumps(
                    {
                        "review_id": "r1",
                        "user_id": "user_1",
                        "item_id": "item_1",
                        "item_name": "Practical Blender",
                        "rating": 5,
                        "review": "Reliable and simple.",
                        "category": "kitchen",
                        "timestamp": 1,
                    }
                ),
            ),
        )

    store = SQLiteFeatureStore(db_path)

    assert store.get_item("item_1").name == "Practical Blender"
    assert store.get_user_history("user_1")[0].review == "Reliable and simple."
    assert store.summary().root.startswith("sqlite://")
    assert store.summary().counts == {"items": 1, "train": 1}
    assert store.summary().available


def test_model_registry_uses_registered_artifact(tmp_path) -> None:
    artifact = tmp_path / "task_a_model.json"
    artifact.write_text('{"kind":"linear_rating_model"}\n', encoding="utf-8")
    registry_path = tmp_path / "model_registry.json"
    registry_path.write_text(
        json.dumps({"artifacts": {"task_a_model": {"path": str(artifact)}}}),
        encoding="utf-8",
    )

    registry = LocalModelRegistry(registry_path)
    record = registry.record("task_a_model")

    assert record.exists
    assert record.source == "registered"
    assert record.path == str(artifact)


def test_monolith_runtime_endpoints_are_deployable() -> None:
    client = TestClient(app)

    assert client.get("/api/health").json() == {"status": "ok"}
    assert client.get("/api/runtime/registry").status_code == 200
    assert client.get("/api/runtime/feature-store").status_code == 200


def test_sqlite_trace_store_persists_records(tmp_path) -> None:
    store = SQLiteTraceStore(tmp_path / "runtime.sqlite")
    record = store.append(
        endpoint="simulate-review",
        latency_ms=12.34,
        steps=[AgentTraceStep(step="test", status="ok")],
        generation_provider="mock",
        estimated_generation_tokens=7,
        validation_status="ok",
    )

    reloaded = SQLiteTraceStore(tmp_path / "runtime.sqlite")
    recent = reloaded.recent()

    assert recent[0].trace_id == record.trace_id
    assert reloaded.metrics().requests == 1
    assert reloaded.metrics().by_endpoint == {"simulate-review": 1}


def test_conversation_manager_persists_feedback_and_turns(tmp_path, monkeypatch) -> None:
    db_path = tmp_path / "runtime.sqlite"
    monkeypatch.setenv("BLUECHIP_CONVERSATION_SQLITE_PATH", str(db_path))
    ConversationManager._instance = None
    profile = build_user_profile(persona="A practical diner.", history=[])

    manager = ConversationManager()
    state = manager.create(user_profile=profile, context="quiet dinner")
    state.add_turn("Need dinner", "Try Calm Grill")
    state.record_feedback("loud-place", accepted=False)
    conversation_id = state.conversation_id

    ConversationManager._instance = None
    reloaded = ConversationManager()
    loaded = reloaded.get(conversation_id)

    assert loaded is not None
    assert loaded.turn_count == 1
    assert loaded.rejected_recommendations == ["loud-place"]
    assert reloaded.list_conversations()[0]["conversation_id"] == conversation_id


def test_rate_limit_middleware_blocks_excess_api_requests() -> None:
    limited = FastAPI()
    limited.add_middleware(RateLimitMiddleware, requests_per_window=2, window_seconds=60)

    @limited.get("/api/ping")
    async def ping() -> dict:
        return {"ok": True}

    client = TestClient(limited)

    assert client.get("/api/ping").status_code == 200
    assert client.get("/api/ping").status_code == 200
    blocked = client.get("/api/ping")

    assert blocked.status_code == 429
    assert blocked.json()["detail"] == "rate limit exceeded"
    assert blocked.headers["Retry-After"]
