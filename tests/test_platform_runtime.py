from __future__ import annotations

import json

from fastapi.testclient import TestClient

from app.main import app
from app.platform.feature_store import LocalFeatureStore
from app.platform.model_registry import LocalModelRegistry


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
