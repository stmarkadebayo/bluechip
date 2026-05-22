from __future__ import annotations

import argparse
import json
import sqlite3
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.data_utils import read_jsonl  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description="Build a SQLite Bluechip feature store.")
    parser.add_argument("--processed-dir", default="data/processed")
    parser.add_argument("--output", default="data/processed/feature_store.sqlite")
    args = parser.parse_args()

    processed_dir = Path(args.processed_dir)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)

    with sqlite3.connect(output) as conn:
        _init_schema(conn)
        _load_items(conn, processed_dir / "items.jsonl")
        _load_reviews(conn, processed_dir / "train.jsonl")
        conn.execute("PRAGMA optimize")

    print(f"SQLite feature store written to {output}")


def _init_schema(conn: sqlite3.Connection) -> None:
    conn.executescript(
        """
        DROP TABLE IF EXISTS items;
        DROP TABLE IF EXISTS reviews;

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

        CREATE INDEX idx_reviews_user_time ON reviews(user_id, timestamp, review_id);
        CREATE INDEX idx_reviews_item ON reviews(item_id);
        CREATE INDEX idx_reviews_category ON reviews(category);
        """
    )


def _load_items(conn: sqlite3.Connection, path: Path) -> None:
    rows = [
        (
            str(row["item_id"]),
            str(row.get("category") or ""),
            json.dumps(row, ensure_ascii=True, separators=(",", ":")),
        )
        for row in read_jsonl(path)
        if row.get("item_id")
    ]
    conn.executemany(
        "INSERT OR REPLACE INTO items(item_id, category, payload) VALUES (?, ?, ?)",
        rows,
    )


def _load_reviews(conn: sqlite3.Connection, path: Path) -> None:
    rows = []
    for index, row in enumerate(read_jsonl(path)):
        review_id = str(row.get("review_id") or f"review_{index}")
        user_id = str(row.get("user_id") or "")
        item_id = str(row.get("item_id") or "")
        if not user_id or not item_id:
            continue
        rows.append(
            (
                review_id,
                user_id,
                item_id,
                float(row.get("rating") or 0),
                row.get("category"),
                int(row.get("timestamp") or 0),
                json.dumps(row, ensure_ascii=True, separators=(",", ":")),
            )
        )
    conn.executemany(
        (
            "INSERT OR REPLACE INTO reviews"
            "(review_id, user_id, item_id, rating, category, timestamp, payload) "
            "VALUES (?, ?, ?, ?, ?, ?, ?)"
        ),
        rows,
    )


if __name__ == "__main__":
    main()
