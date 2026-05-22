from __future__ import annotations

import json
import os
import sqlite3
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


class SQLiteFeatureStore:
    """SQLite-backed feature store for reproducible local serving.

    Tables use JSON payload columns so the store can preserve the same source
    rows as the JSONL feature store while adding indexed point lookups.
    """

    def __init__(self, sqlite_path: str | Path) -> None:
        self.sqlite_path = Path(sqlite_path)
        self.root = self.sqlite_path.parent

    def summary(self) -> FeatureStoreSummary:
        counts = self._counts()
        exists = self.sqlite_path.exists()
        return FeatureStoreSummary(
            root=f"sqlite://{self.sqlite_path}",
            version=self.version(),
            available=exists and bool(counts),
            artifacts={"sqlite": artifact_version(self.sqlite_path)} if exists else {},
            counts=counts,
        )

    def version(self) -> str:
        if not self.sqlite_path.exists():
            return f"sqlite://{self.sqlite_path}:missing"
        stat = self.sqlite_path.stat()
        return f"sqlite://{self.sqlite_path}@{int(stat.st_mtime)}:{stat.st_size}"

    def get_item(self, item_id: str) -> Item | None:
        row = self._fetch_payload("SELECT payload FROM items WHERE item_id = ?", (item_id,))
        return Item(**row) if row else None

    def get_user_history(self, user_id: str, limit: int = 0) -> list[UserHistoryItem]:
        sql = (
            "SELECT payload FROM reviews WHERE user_id = ? "
            "ORDER BY timestamp ASC, review_id ASC"
        )
        params: tuple[object, ...] = (user_id,)
        if limit > 0:
            sql = (
                "SELECT payload FROM ("
                "SELECT payload, timestamp, review_id FROM reviews WHERE user_id = ? "
                "ORDER BY timestamp DESC, review_id DESC LIMIT ?"
                ") ORDER BY timestamp ASC, review_id ASC"
            )
            params = (user_id, limit)
        rows = self._fetch_payloads(sql, params)
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
            for row in self._fetch_payloads("SELECT payload FROM items", ())
            if row.get("item_id")
        }

    @lru_cache(maxsize=1)
    def user_histories(self) -> dict[str, list[dict]]:
        histories: dict[str, list[dict]] = {}
        rows = self._fetch_payloads(
            "SELECT payload FROM reviews ORDER BY user_id ASC, timestamp ASC, review_id ASC",
            (),
        )
        for row in rows:
            user_id = str(row.get("user_id") or "")
            if user_id:
                histories.setdefault(user_id, []).append(row)
        return histories

    def _counts(self) -> dict[str, int]:
        if not self.sqlite_path.exists():
            return {}
        counts = {}
        with self._connect() as conn:
            for table, key in (("items", "items"), ("reviews", "train")):
                if _sqlite_table_exists(conn, table):
                    counts[key] = int(conn.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0])
        return counts

    def _fetch_payload(self, sql: str, params: tuple[object, ...]) -> dict | None:
        rows = self._fetch_payloads(sql, params)
        return rows[0] if rows else None

    def _fetch_payloads(self, sql: str, params: tuple[object, ...]) -> list[dict]:
        if not self.sqlite_path.exists():
            return []
        with self._connect() as conn:
            try:
                records = conn.execute(sql, params).fetchall()
            except sqlite3.Error:
                return []
        rows = []
        for record in records:
            try:
                rows.append(json.loads(record["payload"]))
            except (KeyError, TypeError, json.JSONDecodeError):
                continue
        return rows

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.sqlite_path)
        conn.row_factory = sqlite3.Row
        return conn


def get_feature_store() -> LocalFeatureStore | SQLiteFeatureStore:
    sqlite_path = os.getenv("BLUECHIP_FEATURE_STORE_SQLITE")
    if sqlite_path:
        return SQLiteFeatureStore(sqlite_path)
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


def _sqlite_table_exists(conn: sqlite3.Connection, table: str) -> bool:
    row = conn.execute(
        "SELECT 1 FROM sqlite_master WHERE type = 'table' AND name = ?",
        (table,),
    ).fetchone()
    return row is not None
