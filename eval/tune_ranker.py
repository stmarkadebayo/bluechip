from __future__ import annotations

import argparse
import itertools
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.models.schemas import Item  # noqa: E402
from app.services.profiling.user_profile import build_user_profile  # noqa: E402
from app.services.ranking.recommendation import RecommendationWeights, rank_candidates  # noqa: E402
from app.services.retrieval.candidates import generate_candidates  # noqa: E402
from app.services.retrieval.text import BM25Retriever  # noqa: E402
from app.services.retrieval.vector_store import LocalVectorRetriever  # noqa: E402
from eval.common import histories_by_user, load_eval_data, persona_from_history, write_report  # noqa: E402
from eval.metrics import ndcg_at_k, rounded  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description="Small offline grid search for ranker weights.")
    parser.add_argument("--reviews", default="data/sample/reviews.jsonl")
    parser.add_argument("--items", default="data/sample/items.jsonl")
    parser.add_argument("--processed-dir", default="data/processed")
    parser.add_argument("--output", default="runs/eval/ranker_tuning.json")
    parser.add_argument("--k", type=int, default=10)
    parser.add_argument("--candidate-limit", type=int, default=200)
    args = parser.parse_args()

    train, _, test_b, items = load_eval_data(
        reviews_path=Path(args.reviews),
        items_path=Path(args.items),
        processed_dir=Path(args.processed_dir),
    )
    item_list = list(items.values())
    history_map = histories_by_user(train)
    bm25 = BM25Retriever.from_items(item_list)
    vectors = LocalVectorRetriever(item_list)
    positives = [row["item_id"] for row in test_b]

    results = []
    for weights in _weight_grid():
        ranked_ids = [
            _rank(row, history_map, bm25, vectors, item_list, weights, args.k, args.candidate_limit)
            for row in test_b
        ]
        score = rounded(ndcg_at_k(ranked_ids, positives, args.k))
        results.append({"ndcg@10": score, "weights": weights.__dict__})

    results.sort(key=lambda row: row["ndcg@10"], reverse=True)
    payload = {
        "task": "Ranker Tuning",
        "dataset": str(Path(args.processed_dir)),
        "examples": len(test_b),
        "metrics": {
            "best_ndcg@10": results[0]["ndcg@10"] if results else 0.0,
            "trials": len(results),
        },
        "best_weights": results[0]["weights"] if results else {},
        "top_trials": results[:5],
        "notes": [
            "Grid search is intentionally small so it can run in CI on sample data.",
            "Use the same command against real processed data before freezing demo weights.",
        ],
    }
    write_report(Path(args.output), payload)
    print_report(payload)


def _weight_grid() -> list[RecommendationWeights]:
    grids = {
        "preference": [0.20, 0.26],
        "context": [0.10, 0.16],
        "category": [0.14, 0.20],
        "vector": [0.14, 0.22],
        "quality": [0.10, 0.16],
    }
    weights = []
    keys = list(grids)
    for values in itertools.product(*(grids[key] for key in keys)):
        params = dict(zip(keys, values))
        weights.append(RecommendationWeights(**params))
    return weights


def _rank(
    row: dict,
    history_map: dict[str, list],
    bm25: BM25Retriever,
    vectors: LocalVectorRetriever,
    items: list[Item],
    weights: RecommendationWeights,
    limit: int,
    candidate_limit: int,
) -> list[str]:
    history = history_map.get(row["user_id"], [])
    user_profile = build_user_profile(persona_from_history(history), history)
    candidates = generate_candidates(
        user_profile=user_profile,
        history=history,
        items=items,
        context="",
        bm25_retriever=bm25,
        vector_retriever=vectors,
        limit=min(candidate_limit, len(items)),
    )
    ranked = rank_candidates(
        user_profile=user_profile,
        context="",
        candidate_items=candidates,
        limit=max(limit, len(candidates)),
        weights=weights,
    )
    return [item.item_id for item in ranked]


def print_report(payload: dict) -> None:
    import json

    print(json.dumps(payload, ensure_ascii=True, indent=2))


if __name__ == "__main__":
    main()
