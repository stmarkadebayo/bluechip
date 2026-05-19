from __future__ import annotations

from pathlib import Path

from app.models.schemas import Item
from app.services.retrieval.vector_store import LocalVectorRetriever
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
