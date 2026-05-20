from __future__ import annotations

import json
import os
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

from app.models.schemas import Item, UserHistoryItem
from app.platform.artifacts import artifact_version, first_existing_path


@dataclass(frozen=True)
class FeatureStoreSummary:
    root: str
    version: str
    available: bool
    artifacts: dict[str, str]
    counts: dict[str, int]


class LocalFeatureStore:
    """Local persistent feature store over processed Bluechip artifacts.

    This is the single-process equivalent of a production feature store. It
    resolves processed datasets, exposes point lookups, and reports versions so
    serving traces can identify the data snapshot behind a response.
    """

    def __init__(self, root: str | Path | None = None) -> None:
        self.root = Path(root or os.getenv("BLUECHIP_FEATURE_STORE_ROOT") or _default_root())

    def summary(self) -> FeatureStoreSummary:
        artifacts = {
            name: artifact_version(path)
            for name, path in self._artifact_paths().items()
            if path.exists()
        }
        counts = self._counts()
        return FeatureStoreSummary(
            root=str(self.root),
            version=self.version(),
            available=bool(artifacts),
            artifacts=artifacts,
            counts=counts,
        )

    def version(self) -> str:
        paths = [path for path in self._artifact_paths().values() if path.exists()]
        if not paths:
            return f"{self.root}:missing"
        newest = max(int(path.stat().st_mtime) for path in paths)
        total_size = sum(path.stat().st_size for path in paths)
        return f"{self.root}@{newest}:{total_size}"

    def get_item(self, item_id: str) -> Item | None:
        payload = self.items().get(item_id)
        if payload is None:
            return None
        return Item(**payload)

    def get_user_history(self, user_id: str, limit: int = 0) -> list[UserHistoryItem]:
        rows = self.user_histories().get(user_id, [])
        if limit > 0:
            rows = rows[-limit:]
        return [
            UserHistoryItem(
                item_id=str(row["item_id"]),
                item_name=str(row.get("item_name") or row["item_id"]),
                rating=float(row["rating"]),
                review=str(row.get("review") or ""),
                category=row.get("category"),
                timestamp=row.get("timestamp"),
            )
            for row in rows
        ]

    @lru_cache(maxsize=1)
    def items(self) -> dict[str, dict]:
        return {
            str(row["item_id"]): row
            for row in _read_jsonl(self.root / "items.jsonl")
            if row.get("item_id")
        }

    @lru_cache(maxsize=1)
    def user_histories(self) -> dict[str, list[dict]]:
        histories: dict[str, list[dict]] = {}
        for row in _read_jsonl(self.root / "train.jsonl"):
            user_id = str(row.get("user_id") or "")
            if not user_id:
                continue
            histories.setdefault(user_id, []).append(row)
        for rows in histories.values():
            rows.sort(key=lambda row: (int(row.get("timestamp") or 0), str(row.get("review_id") or "")))
        return histories

    def _artifact_paths(self) -> dict[str, Path]:
        return {
            "items": self.root / "items.jsonl",
            "train": self.root / "train.jsonl",
            "test_task_a": self.root / "test_task_a.jsonl",
            "test_task_b": self.root / "test_task_b.jsonl",
            "split_stats": self.root / "split_stats.json",
            "dataset_stats": self.root / "dataset_stats.json",
            "rating_stats": self.root / "task_a_rating_stats.json",
        }

    def _counts(self) -> dict[str, int]:
        counts: dict[str, int] = {}
        for path in (self.root / "split_stats.json", self.root / "dataset_stats.json"):
            if not path.exists():
                continue
            try:
                payload = json.loads(path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError):
                continue
            for key in ("reviews", "train", "test_task_a", "test_task_b", "items"):
                if isinstance(payload.get(key), int):
                    counts[key] = int(payload[key])
        for name, path in self._artifact_paths().items():
            if path.suffix == ".jsonl" and path.exists() and name not in counts:
                counts[name] = _line_count(path)
        return counts


def get_feature_store() -> LocalFeatureStore:
    return LocalFeatureStore()


def _default_root() -> str:
    path = first_existing_path(
        "data/processed/all_categories/items.jsonl",
        "data/processed/items.jsonl",
    )
    return str(path.parent) if path else "data/processed"


def _read_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        return []
    rows = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def _line_count(path: Path) -> int:
    with path.open("r", encoding="utf-8") as handle:
        return sum(1 for line in handle if line.strip())
