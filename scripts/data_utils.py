from __future__ import annotations

import gzip
import json
from collections import defaultdict
from pathlib import Path
from typing import Iterable, Iterator


def open_text(path: Path):
    if path.suffix == ".gz":
        return gzip.open(path, "rt", encoding="utf-8")
    return path.open("r", encoding="utf-8")


def read_jsonl(path: Path, limit: int = 0) -> list[dict]:
    if not path.exists():
        return []
    rows = []
    with open_text(path) as handle:
        for line in handle:
            if not line.strip():
                continue
            rows.append(json.loads(line))
            if limit and len(rows) >= limit:
                break
    return rows


def stream_jsonl(path: Path, limit: int = 0) -> Iterator[dict]:
    count = 0
    with open_text(path) as handle:
        for line in handle:
            if line.strip():
                yield json.loads(line)
                count += 1
                if limit and count >= limit:
                    break


def write_jsonl(path: Path, rows: Iterable[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=True) + "\n")


def write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=True, indent=2) + "\n", encoding="utf-8")


def load_items(path: Path) -> dict[str, dict]:
    return {row["item_id"]: row for row in read_jsonl(path)}


def group_by(rows: Iterable[dict], key: str) -> dict[str, list[dict]]:
    grouped: dict[str, list[dict]] = defaultdict(list)
    for row in rows:
        grouped[str(row[key])].append(row)
    return grouped


def stable_review_id(user_id: str, item_id: str, timestamp: int, index: int) -> str:
    return f"{user_id}:{item_id}:{timestamp}:{index}"
