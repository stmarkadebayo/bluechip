from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.models.schemas import Item  # noqa: E402
from app.services.profiling.user_profile import build_user_profile  # noqa: E402
from app.services.retrieval.candidates import CandidateCatalog, generate_candidate_pool  # noqa: E402
from app.services.retrieval.evidence_graph import build_evidence_graph_index  # noqa: E402
from app.services.retrieval.text import BM25Retriever  # noqa: E402
from app.services.retrieval.vector_store import LocalVectorRetriever  # noqa: E402
from eval.common import histories_by_user, load_eval_data, persona_from_history, print_report, write_report  # noqa: E402
from eval.metrics import hit_rate_at_k, recall_at_k, rounded  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate the aspect-aware evidence intelligence layer.")
    parser.add_argument("--reviews", default="data/sample/reviews.jsonl")
    parser.add_argument("--items", default="data/sample/items.jsonl")
    parser.add_argument("--processed-dir", default="data/processed")
    parser.add_argument("--output", default="runs/eval/evidence_intelligence_report.json")
    parser.add_argument("--candidate-limit", type=int, default=200)
    parser.add_argument("--max-examples", type=int, default=25)
    args = parser.parse_args()

    train, _, test_b, items = load_eval_data(
        reviews_path=Path(args.reviews),
        items_path=Path(args.items),
        processed_dir=Path(args.processed_dir),
    )
    if args.max_examples:
        test_b = test_b[: args.max_examples]
    item_list = _items_with_train_popularity(items, train)
    histories = histories_by_user(train)
    graph = build_evidence_graph_index(
        train=train,
        items=[item.model_dump() for item in item_list],
        top_k=max(args.candidate_limit, 50),
    )
    collaborative_index = {"evidence_graph_retrieval": graph}
    catalog = CandidateCatalog.from_items(item_list)
    bm25 = BM25Retriever.from_items(item_list)
    vector = LocalVectorRetriever(item_list)

    positives = [row["item_id"] for row in test_b]
    candidate_rankings = []
    source_counts = {}
    aspect_coverage = []
    nigerian_context_examples = 0
    for row in test_b:
        history = histories.get(row["user_id"], [])
        user_profile = build_user_profile(persona_from_history(history), history, locale=None)
        pool = generate_candidate_pool(
            user_profile=user_profile,
            history=history,
            items=item_list,
            context=row.get("review") or "",
            collaborative_index=collaborative_index,
            bm25_retriever=bm25,
            vector_retriever=vector,
            catalog=catalog,
            limit=min(args.candidate_limit, len(item_list)),
        )
        candidate_rankings.append([item.item_id for item in pool.items])
        for source, count in pool.source_counts().items():
            source_counts[source] = source_counts.get(source, 0) + count
        aspect_coverage.append(1.0 if user_profile.aspect_scores else 0.0)
        if user_profile.nigerian_context:
            nigerian_context_examples += 1

    metrics = {
        f"evidence_candidate_recall@{args.candidate_limit}": rounded(
            recall_at_k(candidate_rankings, positives, args.candidate_limit)
        ),
        "evidence_candidate_hit_rate@10": rounded(hit_rate_at_k(candidate_rankings, positives, 10)),
        "user_aspect_coverage": rounded(sum(aspect_coverage) / len(aspect_coverage))
        if aspect_coverage
        else 0.0,
        "nigerian_context_example_share": rounded(nigerian_context_examples / len(test_b))
        if test_b
        else 0.0,
    }
    payload = {
        "task": "Evidence Intelligence",
        "dataset": str(Path(args.processed_dir)),
        "examples": len(test_b),
        "metrics": metrics,
        "retrieval_sources": dict(sorted(source_counts.items())),
        "notes": [
            "Evidence intelligence evaluates aspect-aware graph retrieval as a candidate source.",
            "The graph blends aspect->item, category-aspect->item, item transition, and category transition paths.",
            "This report is a smoke/evidence-layer report; Task B promotion still uses eval_task_b.py.",
        ],
    }
    write_report(Path(args.output), payload)
    print_report(payload)


def _items_with_train_popularity(items: dict[str, Item], train: list[dict]) -> list[Item]:
    counts: dict[str, int] = {}
    for row in train:
        if float(row.get("rating") or 0) >= 4:
            counts[row["item_id"]] = counts.get(row["item_id"], 0) + 1
    enriched = []
    for item in items.values():
        metadata = dict(item.metadata)
        metadata["review_count"] = counts.get(item.item_id, 0)
        metadata["rating_number"] = counts.get(item.item_id, 0)
        enriched.append(item.model_copy(update={"metadata": metadata}))
    return enriched


if __name__ == "__main__":
    main()
