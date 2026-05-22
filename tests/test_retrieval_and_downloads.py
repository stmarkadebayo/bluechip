from __future__ import annotations

import json
from pathlib import Path

import pytest

from app.models.schemas import Item
from app.services.retrieval.vector_store import FAISSVectorStore, LocalVectorRetriever
from scripts.download_amazon_hf import DownloadTarget, status_for


def test_vector_retriever_prefers_semantic_overlap() -> None:
    items = [
        Item(
            item_id="quiet",
            name="Quiet Dinner",
            category="restaurant",
            summary="Calm affordable dinner with quiet seating.",
            average_rating=4.6,
        ),
        Item(
            item_id="loud",
            name="Party Dinner",
            category="restaurant",
            summary="Loud premium nightlife and crowded seating.",
            average_rating=4.7,
        ),
    ]

    result = LocalVectorRetriever(items).search("quiet affordable calm place", limit=2)

    assert result[0].item_id == "quiet"


def test_download_status_detects_missing_partial_and_complete(tmp_path: Path) -> None:
    target = DownloadTarget(
        category="Example",
        kind="reviews",
        url="https://example.invalid/file.jsonl",
        path=tmp_path / "file.jsonl",
        expected_bytes=4,
    )

    assert status_for(target) == "missing"
    target.path.write_bytes(b"ab")
    assert status_for(target) == "partial:2/4"
    target.path.write_bytes(b"abcd")
    assert status_for(target) == "complete"


def test_faiss_deserialize_uses_companion_item_ids(tmp_path: Path) -> None:
    faiss = pytest.importorskip("faiss")
    np = pytest.importorskip("numpy")

    index_path = tmp_path / "neural_index.faiss"
    ids_path = tmp_path / "neural_index_ids.json"
    index = faiss.IndexFlatIP(2)
    index.add(np.array([[1.0, 0.0], [0.0, 1.0]], dtype=np.float32))
    faiss.write_index(index, str(index_path))
    ids_path.write_text(json.dumps(["target", "other"]), encoding="utf-8")

    target = Item(item_id="target", name="Target", category="books")
    other = Item(item_id="other", name="Other", category="books")

    store = FAISSVectorStore.deserialize_with_ids(
        path=str(index_path),
        items_by_id={"other": other, "target": target},
        ids_path=str(ids_path),
    )

    assert store._built
    assert store._item_map[0].item_id == "target"
    assert store._item_map[1].item_id == "other"

    request_bound = store.bind_items([other])

    assert 0 not in request_bound._item_map
    assert request_bound._item_map[1].item_id == "other"
