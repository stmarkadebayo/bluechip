from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from eval.common import (
    histories_by_user,
    load_eval_data,
    persona_from_history,
    popularity_ranking,
    print_report,
    write_report,
)
from eval.metrics import hit_rate_at_k, ndcg_at_k, recall_at_k, rounded
from app.models.schemas import Item
from app.services.profiling.user_profile import build_user_profile
from app.services.ranking.recommendation import rank_candidates
from app.services.retrieval.candidates import generate_candidates
from app.services.retrieval.text import BM25Retriever


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate Task B recommendation baselines.")
    parser.add_argument("--reviews", default="data/sample/reviews.jsonl")
    parser.add_argument("--items", default="data/sample/items.jsonl")
    parser.add_argument("--processed-dir", default="data/processed")
    parser.add_argument("--output", default="runs/eval/task_b_report.json")
    parser.add_argument("--k", type=int, default=10)
    parser.add_argument("--candidate-limit", type=int, default=200)
    parser.add_argument("--max-examples", type=int, default=0)
    args = parser.parse_args()

    train, _, test_b, items = load_eval_data(
        reviews_path=Path(args.reviews),
        items_path=Path(args.items),
        processed_dir=Path(args.processed_dir),
    )
    item_list = list(items.values())
    if args.max_examples:
        test_b = test_b[: args.max_examples]
    positives = [row["item_id"] for row in test_b]
    history_map = histories_by_user(train)
    retriever = BM25Retriever.from_items(item_list)
    popularity = _popularity_rank(train, item_list)

    rankings = {
        "popularity": [popularity for _ in test_b],
        "bm25_profile": [
            _bm25_rank(row, history_map, retriever, args.candidate_limit) for row in test_b
        ],
        "hybrid_ranker": [
            _hybrid_rank(row, history_map, retriever, item_list, args.k, args.candidate_limit)
            for row in test_b
        ],
    }

    metrics = {}
    for name, ranked_ids in rankings.items():
        metrics[f"{name}_hit_rate@{args.k}"] = rounded(hit_rate_at_k(ranked_ids, positives, args.k))
        metrics[f"{name}_recall@{args.k}"] = rounded(recall_at_k(ranked_ids, positives, args.k))
        metrics[f"{name}_ndcg@{args.k}"] = rounded(ndcg_at_k(ranked_ids, positives, args.k))

    payload = {
        "task": "Task B",
        "dataset": str(Path(args.processed_dir) if Path(args.processed_dir).exists() else Path(args.reviews)),
        "examples": len(test_b),
        "metrics": metrics,
        "notes": [
            "Positive item is the held-out next review for each eligible user.",
            "BM25 retrieves from user profile text against item metadata and summaries.",
            "Hybrid ranker uses preference, context, quality, novelty, and profile confidence.",
        ],
    }
    write_report(Path(args.output), payload)
    print_report(payload)


def _popularity_rank(train: list[dict], items: list[Item]) -> list[str]:
    return popularity_ranking(train, [item.item_id for item in items])


def _bm25_rank(
    row: dict,
    history_map: dict[str, list],
    retriever: BM25Retriever,
    candidate_limit: int,
) -> list[str]:
    history = history_map.get(row["user_id"], [])
    query = persona_from_history(history)
    return [item.item_id for item in retriever.search(query, limit=candidate_limit)]


def _hybrid_rank(
    row: dict,
    history_map: dict[str, list],
    retriever: BM25Retriever,
    items: list[Item],
    limit: int,
    candidate_limit: int,
) -> list[str]:
    history = history_map.get(row["user_id"], [])
    persona = persona_from_history(history)
    user_profile = build_user_profile(persona=persona, history=history, locale=None)
    retrieved = generate_candidates(
        user_profile=user_profile,
        history=history,
        items=items,
        context="",
        bm25_retriever=retriever,
        limit=min(candidate_limit, len(items)),
    )
    ranked = rank_candidates(user_profile, context="", candidate_items=retrieved, limit=max(limit, len(retrieved)))
    return [item.item_id for item in ranked]


if __name__ == "__main__":
    main()
