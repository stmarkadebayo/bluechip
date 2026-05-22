from __future__ import annotations

import argparse
import json
import sys
import time
from collections import Counter
from pathlib import Path

import numpy as np
from scipy.sparse import csr_matrix

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from eval.common import histories_by_user, load_eval_data, write_report  # noqa: E402
from eval.metrics import hit_rate_at_k, ndcg_at_k, recall_at_k, rounded  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Train/evaluate implicit ALS, BPR, and item-item Task B baselines."
    )
    parser.add_argument("--reviews", default="data/sample/reviews.jsonl")
    parser.add_argument("--items", default="data/sample/items.jsonl")
    parser.add_argument("--processed-dir", default="data/processed")
    parser.add_argument("--output", default="runs/eval/implicit_baselines.json")
    parser.add_argument("--models", default="als,bpr,item_item")
    parser.add_argument("--k", type=int, default=10)
    parser.add_argument("--candidate-limit", type=int, default=1000)
    parser.add_argument("--max-examples", type=int, default=0)
    parser.add_argument("--als-factors", type=int, default=64)
    parser.add_argument("--als-iterations", type=int, default=10)
    parser.add_argument("--bpr-factors", type=int, default=64)
    parser.add_argument("--bpr-iterations", type=int, default=10)
    parser.add_argument("--item-neighbors", type=int, default=100)
    parser.add_argument("--threads", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    try:
        from implicit.als import AlternatingLeastSquares
        from implicit.bpr import BayesianPersonalizedRanking
        from implicit.nearest_neighbours import CosineRecommender
    except ImportError as exc:  # pragma: no cover - dependency guard
        raise SystemExit(
            "Missing optional dependency 'implicit'. Install with: "
            "python -m pip install implicit"
        ) from exc

    train, _, test_b, items = load_eval_data(
        reviews_path=Path(args.reviews),
        items_path=Path(args.items),
        processed_dir=Path(args.processed_dir),
    )
    if args.max_examples:
        test_b = test_b[: args.max_examples]

    matrix_start = time.perf_counter()
    user_items, user_to_index, item_to_index, index_to_item = _build_user_item_matrix(
        train=train,
        item_ids=sorted(items),
    )
    matrix_seconds = time.perf_counter() - matrix_start

    history_map = histories_by_user(train)
    positives = [row["item_id"] for row in test_b]
    popularity = _popularity_rank(train, index_to_item)
    model_names = [name.strip() for name in args.models.split(",") if name.strip()]

    report = {
        "task": "Task B implicit baseline",
        "dataset": str(Path(args.processed_dir)),
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "config": {
            "models": model_names,
            "k": args.k,
            "candidate_limit": args.candidate_limit,
            "max_examples": args.max_examples,
            "als_factors": args.als_factors,
            "als_iterations": args.als_iterations,
            "bpr_factors": args.bpr_factors,
            "bpr_iterations": args.bpr_iterations,
            "item_neighbors": args.item_neighbors,
            "threads": args.threads,
            "seed": args.seed,
        },
        "data": {
            "train_interactions": len(train),
            "test_examples": len(test_b),
            "users": user_items.shape[0],
            "items": user_items.shape[1],
            "matrix_nonzeros": int(user_items.nnz),
            "matrix_seconds": round(matrix_seconds, 4),
        },
        "metrics": {},
        "slices": {},
        "notes": [
            "All training reviews are treated as implicit observed interactions weighted by rating.",
            "Recommendations filter items already observed in the user's training history.",
            "This is a conventional collaborative-filtering baseline, not a replacement for contextual ranking.",
        ],
    }

    if "als" in model_names:
        model = AlternatingLeastSquares(
            factors=args.als_factors,
            iterations=args.als_iterations,
            random_state=args.seed,
            num_threads=args.threads,
        )
        _fit_and_record(
            name="implicit_als",
            model=model,
            user_items=user_items,
            test_b=test_b,
            history_map=history_map,
            user_to_index=user_to_index,
            index_to_item=index_to_item,
            positives=positives,
            popularity=popularity,
            k=args.k,
            candidate_limit=args.candidate_limit,
            report=report,
        )

    if "bpr" in model_names:
        model = BayesianPersonalizedRanking(
            factors=args.bpr_factors,
            iterations=args.bpr_iterations,
            random_state=args.seed,
            num_threads=args.threads,
        )
        _fit_and_record(
            name="implicit_bpr",
            model=model,
            user_items=user_items,
            test_b=test_b,
            history_map=history_map,
            user_to_index=user_to_index,
            index_to_item=index_to_item,
            positives=positives,
            popularity=popularity,
            k=args.k,
            candidate_limit=args.candidate_limit,
            report=report,
        )

    if "item_item" in model_names:
        model = CosineRecommender(K=args.item_neighbors)
        _fit_and_record(
            name="implicit_item_item",
            model=model,
            user_items=user_items,
            test_b=test_b,
            history_map=history_map,
            user_to_index=user_to_index,
            index_to_item=index_to_item,
            positives=positives,
            popularity=popularity,
            k=args.k,
            candidate_limit=args.candidate_limit,
            report=report,
        )

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(report, ensure_ascii=True, indent=2) + "\n", encoding="utf-8")
    write_report(output_path, report)
    print(json.dumps(report, ensure_ascii=True, indent=2))


def _build_user_item_matrix(
    train: list[dict],
    item_ids: list[str],
) -> tuple[csr_matrix, dict[str, int], dict[str, int], list[str]]:
    user_ids = sorted({row["user_id"] for row in train})
    user_to_index = {user_id: index for index, user_id in enumerate(user_ids)}
    item_to_index = {item_id: index for index, item_id in enumerate(item_ids)}

    rows: list[int] = []
    cols: list[int] = []
    values: list[float] = []
    for row in train:
        item_id = row["item_id"]
        if item_id not in item_to_index:
            continue
        rows.append(user_to_index[row["user_id"]])
        cols.append(item_to_index[item_id])
        values.append(max(float(row.get("rating", 1.0)), 0.1))

    matrix = csr_matrix(
        (
            np.asarray(values, dtype=np.float32),
            (np.asarray(rows, dtype=np.int32), np.asarray(cols, dtype=np.int32)),
        ),
        shape=(len(user_ids), len(item_ids)),
        dtype=np.float32,
    )
    matrix.sum_duplicates()
    return matrix.tocsr(), user_to_index, item_to_index, item_ids


def _fit_and_record(
    *,
    name: str,
    model,
    user_items: csr_matrix,
    test_b: list[dict],
    history_map: dict[str, list],
    user_to_index: dict[str, int],
    index_to_item: list[str],
    positives: list[str],
    popularity: list[str],
    k: int,
    candidate_limit: int,
    report: dict,
    fit_matrix: csr_matrix | None = None,
) -> None:
    fit_start = time.perf_counter()
    model.fit(fit_matrix if fit_matrix is not None else user_items, show_progress=True)
    fit_seconds = time.perf_counter() - fit_start

    recommend_start = time.perf_counter()
    ranked_ids = _recommend_rankings(
        model=model,
        test_b=test_b,
        user_items=user_items,
        user_to_index=user_to_index,
        index_to_item=index_to_item,
        popularity=popularity,
        candidate_limit=candidate_limit,
    )
    recommend_seconds = time.perf_counter() - recommend_start

    model_metrics = _ranking_metrics(ranked_ids, positives, k, candidate_limit)
    model_metrics["fit_seconds"] = round(fit_seconds, 4)
    model_metrics["recommend_seconds"] = round(recommend_seconds, 4)
    report["metrics"].update({f"{name}_{metric}": value for metric, value in model_metrics.items()})
    report["slices"][name] = _slice_metrics(
        test_b=test_b,
        history_map=history_map,
        positives=positives,
        ranked_ids=ranked_ids,
        k=k,
        candidate_limit=candidate_limit,
    )


def _recommend_rankings(
    *,
    model,
    test_b: list[dict],
    user_items: csr_matrix,
    user_to_index: dict[str, int],
    index_to_item: list[str],
    popularity: list[str],
    candidate_limit: int,
    batch_size: int = 512,
) -> list[list[str]]:
    row_users = [user_to_index.get(row["user_id"]) for row in test_b]
    known_users = sorted({index for index in row_users if index is not None})
    recommendations_by_user: dict[int, list[str]] = {}

    for start in range(0, len(known_users), batch_size):
        batch = np.asarray(known_users[start : start + batch_size], dtype=np.int32)
        item_indices, _ = model.recommend(
            batch,
            user_items[batch],
            N=candidate_limit,
            filter_already_liked_items=True,
        )
        for user_index, recommended in zip(batch.tolist(), item_indices):
            recommendations_by_user[user_index] = [
                index_to_item[int(item_index)]
                for item_index in recommended
                if 0 <= int(item_index) < len(index_to_item)
            ]

    rankings = []
    for maybe_user_index in row_users:
        if maybe_user_index is None:
            rankings.append(popularity[:candidate_limit])
            continue
        ranking = recommendations_by_user.get(maybe_user_index, [])
        rankings.append(_dedupe_and_fill(ranking, popularity, candidate_limit))
    return rankings


def _dedupe_and_fill(ranking: list[str], popularity: list[str], limit: int) -> list[str]:
    seen = set()
    filled = []
    for item_id in ranking + popularity:
        if item_id in seen:
            continue
        filled.append(item_id)
        seen.add(item_id)
        if len(filled) >= limit:
            break
    return filled


def _ranking_metrics(
    ranked_ids: list[list[str]],
    positives: list[str],
    k: int,
    candidate_limit: int,
) -> dict[str, float]:
    metrics = {
        f"hit_rate@{k}": rounded(hit_rate_at_k(ranked_ids, positives, k)),
        f"recall@{k}": rounded(recall_at_k(ranked_ids, positives, k)),
        f"ndcg@{k}": rounded(ndcg_at_k(ranked_ids, positives, k)),
    }
    for recall_k in sorted({50, 100, candidate_limit}):
        if recall_k > 0:
            metrics[f"recall@{recall_k}"] = rounded(recall_at_k(ranked_ids, positives, recall_k))
    return metrics


def _slice_metrics(
    *,
    test_b: list[dict],
    history_map: dict[str, list],
    positives: list[str],
    ranked_ids: list[list[str]],
    k: int,
    candidate_limit: int,
) -> dict[str, dict[str, float]]:
    slice_indices = {
        "sparse_history_1_2": [],
        "medium_history_3_7": [],
        "warm_history_8_plus": [],
        "cross_domain": [],
    }
    for index, row in enumerate(test_b):
        history = history_map.get(row["user_id"], [])
        history_len = len(history)
        if history_len <= 2:
            slice_indices["sparse_history_1_2"].append(index)
        elif history_len <= 7:
            slice_indices["medium_history_3_7"].append(index)
        else:
            slice_indices["warm_history_8_plus"].append(index)
        if _is_cross_domain(row, history):
            slice_indices["cross_domain"].append(index)

    return {
        name: _metrics_for_indices(
            indices=indices,
            positives=positives,
            ranked_ids=ranked_ids,
            k=k,
            candidate_limit=candidate_limit,
        )
        for name, indices in slice_indices.items()
        if indices
    }


def _metrics_for_indices(
    *,
    indices: list[int],
    positives: list[str],
    ranked_ids: list[list[str]],
    k: int,
    candidate_limit: int,
) -> dict[str, float]:
    sliced_rankings = [ranked_ids[index] for index in indices]
    sliced_positives = [positives[index] for index in indices]
    metrics = _ranking_metrics(sliced_rankings, sliced_positives, k, candidate_limit)
    return {"examples": len(indices), **metrics}


def _is_cross_domain(row: dict, history: list) -> bool:
    target_category = row.get("category")
    if not target_category:
        return False
    positive_categories = {item.category for item in history if item.category and item.rating >= 4}
    return bool(positive_categories) and target_category not in positive_categories


def _popularity_rank(train: list[dict], index_to_item: list[str]) -> list[str]:
    counts = Counter(row["item_id"] for row in train)
    return sorted(index_to_item, key=lambda item_id: (counts[item_id], item_id), reverse=True)


if __name__ == "__main__":
    main()
