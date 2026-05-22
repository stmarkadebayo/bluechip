from __future__ import annotations

import argparse
import json
import sqlite3
import sys
import time
from collections import Counter
from pathlib import Path

import numpy as np
from scipy.sparse import csr_matrix

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.data_utils import read_jsonl  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build a lazy SQLite item-item cosine index from implicit feedback."
    )
    parser.add_argument("--processed-dir", default="data/processed")
    parser.add_argument("--train", default="")
    parser.add_argument("--items", default="")
    parser.add_argument("--output", default="")
    parser.add_argument("--neighbors", type=int, default=100)
    parser.add_argument("--positive-threshold", type=float, default=4.0)
    parser.add_argument("--min-item-positive-count", type=int, default=1)
    parser.add_argument("--threads", type=int, default=0)
    parser.add_argument("--progress-every", type=int, default=10000)
    args = parser.parse_args()

    try:
        from implicit.nearest_neighbours import CosineRecommender
    except ImportError as exc:  # pragma: no cover - optional dependency guard
        raise SystemExit(
            "Missing optional dependency 'implicit'. Install with: "
            "python -m pip install -e '.[baselines]'"
        ) from exc

    processed_dir = Path(args.processed_dir)
    train_path = Path(args.train) if args.train else processed_dir / "train.jsonl"
    items_path = Path(args.items) if args.items else processed_dir / "items.jsonl"
    output_path = (
        Path(args.output)
        if args.output
        else processed_dir / "implicit_item_neighbors.sqlite"
    )

    started = time.perf_counter()
    train = read_jsonl(train_path)
    item_ids = [str(row["item_id"]) for row in read_jsonl(items_path)]
    user_items, item_to_index, index_to_item = _build_user_item_matrix(train, item_ids)
    positive_counts = Counter(
        str(row["item_id"])
        for row in train
        if float(row.get("rating") or 0) >= args.positive_threshold
    )

    model = CosineRecommender(K=args.neighbors + 1, num_threads=args.threads)
    model.fit(user_items, show_progress=True)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    if output_path.exists():
        output_path.unlink()
    connection = sqlite3.connect(output_path)
    try:
        _initialise_database(connection)
        source_count, edge_count = _write_neighbors(
            connection=connection,
            similarity=model.similarity.tocsr(),
            item_to_index=item_to_index,
            index_to_item=index_to_item,
            positive_counts=positive_counts,
            neighbors=args.neighbors,
            min_item_positive_count=args.min_item_positive_count,
            progress_every=args.progress_every,
        )
        metadata = {
            "type": "implicit_item_item_neighbors",
            "model": "implicit.nearest_neighbours.CosineRecommender",
            "created_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "config": {
                "neighbors": args.neighbors,
                "positive_threshold": args.positive_threshold,
                "min_item_positive_count": args.min_item_positive_count,
                "threads": args.threads,
            },
            "data": {
                "train_interactions": len(train),
                "users": user_items.shape[0],
                "items": user_items.shape[1],
                "matrix_nonzeros": int(user_items.nnz),
                "source_items": source_count,
                "neighbor_edges": edge_count,
                "build_seconds": round(time.perf_counter() - started, 4),
            },
        }
        _write_metadata(connection, metadata)
        connection.commit()
    finally:
        connection.close()

    print(json.dumps(metadata, ensure_ascii=True, indent=2))


def _build_user_item_matrix(
    train: list[dict],
    item_ids: list[str],
) -> tuple[csr_matrix, dict[str, int], list[str]]:
    user_ids = sorted({str(row["user_id"]) for row in train})
    user_to_index = {user_id: index for index, user_id in enumerate(user_ids)}
    item_to_index = {item_id: index for index, item_id in enumerate(item_ids)}

    rows: list[int] = []
    cols: list[int] = []
    values: list[float] = []
    for row in train:
        item_id = str(row["item_id"])
        if item_id not in item_to_index:
            continue
        rows.append(user_to_index[str(row["user_id"])])
        cols.append(item_to_index[item_id])
        values.append(max(float(row.get("rating") or 1.0), 0.1))

    matrix = csr_matrix(
        (
            np.asarray(values, dtype=np.float32),
            (np.asarray(rows, dtype=np.int32), np.asarray(cols, dtype=np.int32)),
        ),
        shape=(len(user_ids), len(item_ids)),
        dtype=np.float32,
    )
    matrix.sum_duplicates()
    return matrix.tocsr(), item_to_index, item_ids


def _initialise_database(connection: sqlite3.Connection) -> None:
    connection.execute("PRAGMA journal_mode = WAL")
    connection.execute("PRAGMA synchronous = NORMAL")
    connection.execute(
        """
        CREATE TABLE item_neighbors (
            item_id TEXT NOT NULL,
            neighbor_id TEXT NOT NULL,
            score REAL NOT NULL,
            rank INTEGER NOT NULL,
            PRIMARY KEY (item_id, rank)
        )
        """
    )
    connection.execute(
        """
        CREATE TABLE metadata (
            key TEXT PRIMARY KEY,
            value TEXT NOT NULL
        )
        """
    )
    connection.execute("CREATE INDEX idx_item_neighbors_item ON item_neighbors(item_id)")


def _write_neighbors(
    *,
    connection: sqlite3.Connection,
    similarity: csr_matrix,
    item_to_index: dict[str, int],
    index_to_item: list[str],
    positive_counts: Counter,
    neighbors: int,
    min_item_positive_count: int,
    progress_every: int,
) -> tuple[int, int]:
    batch: list[tuple[str, str, float, int]] = []
    source_count = 0
    edge_count = 0
    eligible_items = [
        item_id
        for item_id, count in positive_counts.items()
        if count >= min_item_positive_count and item_id in item_to_index
    ]
    for offset, item_id in enumerate(eligible_items, start=1):
        item_index = item_to_index[item_id]
        row = similarity.getrow(item_index)
        entries = []
        for neighbor_index, raw_score in zip(row.indices, row.data):
            neighbor_index = int(neighbor_index)
            if neighbor_index == item_index:
                continue
            score = float(raw_score)
            if score <= 0:
                continue
            entries.append((index_to_item[neighbor_index], score))
        if not entries:
            continue
        entries.sort(key=lambda entry: (entry[1], entry[0]), reverse=True)
        source_count += 1
        for rank, (neighbor_id, score) in enumerate(entries[:neighbors], start=1):
            batch.append((item_id, neighbor_id, round(score, 6), rank))
        if len(batch) >= 100_000:
            connection.executemany(
                "INSERT INTO item_neighbors(item_id, neighbor_id, score, rank) VALUES (?, ?, ?, ?)",
                batch,
            )
            edge_count += len(batch)
            batch = []
            connection.commit()
        if progress_every and offset % progress_every == 0:
            print(f"exported {offset}/{len(eligible_items)} source items", file=sys.stderr)
    if batch:
        connection.executemany(
            "INSERT INTO item_neighbors(item_id, neighbor_id, score, rank) VALUES (?, ?, ?, ?)",
            batch,
        )
        edge_count += len(batch)
    return source_count, edge_count


def _write_metadata(connection: sqlite3.Connection, metadata: dict) -> None:
    rows = [(key, json.dumps(value, ensure_ascii=True)) for key, value in metadata.items()]
    connection.executemany("INSERT INTO metadata(key, value) VALUES (?, ?)", rows)


if __name__ == "__main__":
    main()
