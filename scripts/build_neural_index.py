from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.models.schemas import Item  # noqa: E402
from app.services.retrieval.vector_store import FAISSVectorStore, companion_ids_path  # noqa: E402
from scripts.data_utils import read_jsonl  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description="Build FAISS neural embedding index.")
    parser.add_argument(
        "--items",
        default="data/processed/items.jsonl",
        help="Path to items JSONL file",
    )
    parser.add_argument(
        "--output",
        default="data/processed/neural_index.faiss",
        help="Path to serialize the FAISS index",
    )
    parser.add_argument(
        "--ids-output",
        default="",
        help="Path to write FAISS index-position item ids (default: <output stem>_ids.json)",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=0,
        help="Limit items to index (0 = all)",
    )
    parser.add_argument(
        "--ids-only",
        action="store_true",
        help="Only write the companion item-id file for an existing index.",
    )
    args = parser.parse_args()

    items_path = Path(args.items)
    if not items_path.exists():
        alt_path = Path("data/processed/all_categories/items.jsonl")
        if alt_path.exists():
            items_path = alt_path
        else:
            print(f"Items file not found: {args.items}")
            sys.exit(1)

    t0 = time.perf_counter()
    raw_items = read_jsonl(items_path, limit=args.limit)
    items = [Item(**row) for row in raw_items]
    print(f"Loaded {len(items)} items from {items_path}")

    output_path = Path(args.output)
    ids_output = args.ids_output or companion_ids_path(str(output_path))
    if args.ids_only:
        Path(ids_output).parent.mkdir(parents=True, exist_ok=True)
        Path(ids_output).write_text(
            json.dumps([item.item_id for item in items]),
            encoding="utf-8",
        )
        print(f"Item ids saved to: {ids_output}")
        return

    t1 = time.perf_counter()
    store = FAISSVectorStore(items)
    build_time = time.perf_counter() - t1
    total_time = time.perf_counter() - t0

    if store.index is None:
        print("FAISS index was not built (neural embeddings or FAISS not available)")
        print("Falling back to hashed embeddings - no FAISS index to serialize")
        sys.exit(0)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    store.serialize(str(output_path), ids_path=ids_output)

    print(f"Index size: {store.index.ntotal} vectors x {store.index.d} dimensions")
    print(f"Build time: {build_time:.2f}s")
    print(f"Total time: {total_time:.2f}s")
    print(f"Index saved to: {output_path}")
    print(f"Item ids saved to: {ids_output}")


if __name__ == "__main__":
    main()
