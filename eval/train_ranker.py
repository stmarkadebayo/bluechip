from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.models.schemas import Item, UserHistoryItem  # noqa: E402
from app.services.profiling.item_profile import build_item_profile  # noqa: E402
from app.services.profiling.user_profile import build_user_profile  # noqa: E402
from app.services.ranking.features import FEATURE_NAMES, ranker_features, weighted_score  # noqa: E402
from app.services.ranking.recommendation import rank_candidates  # noqa: E402
from app.services.retrieval.candidates import (  # noqa: E402
    CandidateCatalog,
    CandidatePool,
    generate_candidate_pool,
)
from app.services.retrieval.item_similarity import build_collaborative_retrieval_index  # noqa: E402
from app.services.retrieval.text import BM25Retriever  # noqa: E402
from app.services.retrieval.vector_store import LocalVectorRetriever  # noqa: E402
from eval.common import histories_by_user, load_eval_data, persona_from_history, write_report  # noqa: E402
from eval.metrics import hit_rate_at_k, ndcg_at_k, rounded  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description="Train a candidate-aware pairwise linear ranker.")
    parser.add_argument("--reviews", default="data/sample/reviews.jsonl")
    parser.add_argument("--items", default="data/sample/items.jsonl")
    parser.add_argument("--processed-dir", default="data/processed")
    parser.add_argument("--collaborative-index", default="")
    parser.add_argument("--output", default="runs/eval/learned_ranker.json")
    parser.add_argument("--epochs", type=int, default=12)
    parser.add_argument("--learning-rate", type=float, default=0.08)
    parser.add_argument("--k", type=int, default=10)
    parser.add_argument("--candidate-limit", type=int, default=200)
    parser.add_argument("--max-examples", type=int, default=0)
    parser.add_argument("--max-negatives", type=int, default=50)
    parser.add_argument("--validation-fraction", type=float, default=0.5)
    args = parser.parse_args()

    train, _, test_b, items = load_eval_data(
        reviews_path=Path(args.reviews),
        items_path=Path(args.items),
        processed_dir=Path(args.processed_dir),
    )
    if args.max_examples:
        test_b = test_b[: args.max_examples]
    train_rows, eval_rows = _split_rows(test_b, args.validation_fraction)
    item_list = _items_with_train_popularity(items, train)
    item_by_id = {item.item_id: item for item in item_list}
    history_map = histories_by_user(train)
    bm25 = BM25Retriever.from_items(item_list)
    vectors = LocalVectorRetriever(item_list)
    catalog = CandidateCatalog.from_items(item_list)
    collaborative_index = _load_collaborative_index(args, train)
    weights = _initial_weights()
    profile_cache = _ProfileCache()

    training_examples = _training_examples(
        rows=train_rows,
        history_map=history_map,
        item_by_id=item_by_id,
        item_list=item_list,
        bm25=bm25,
        vectors=vectors,
        catalog=catalog,
        collaborative_index=collaborative_index,
        candidate_limit=args.candidate_limit,
        profile_cache=profile_cache,
    )

    for _ in range(args.epochs):
        for example in training_examples:
            _train_example(
                example=example,
                weights=weights,
                learning_rate=args.learning_rate,
                max_negatives=args.max_negatives,
                profile_cache=profile_cache,
            )

    rankings = [
        _rank(
            row=row,
            history_map=history_map,
            item_by_id=item_by_id,
            item_list=item_list,
            bm25=bm25,
            vectors=vectors,
            catalog=catalog,
            collaborative_index=collaborative_index,
            candidate_limit=args.candidate_limit,
            weights=weights,
        )
        for row in eval_rows
    ]
    current_hybrid_rankings = [
        _rank(
            row=row,
            history_map=history_map,
            item_by_id=item_by_id,
            item_list=item_list,
            bm25=bm25,
            vectors=vectors,
            catalog=catalog,
            collaborative_index=collaborative_index,
            candidate_limit=args.candidate_limit,
            weights=None,
        )
        for row in eval_rows
    ]
    positives = [row["item_id"] for row in eval_rows]
    payload = {
        "task": "Learned Ranker",
        "dataset": str(Path(args.processed_dir)),
        "examples": len(eval_rows),
        "train_rows": len(train_rows),
        "training_examples": len(training_examples),
        "split_strategy": "test_b_train_validation_split",
        "metrics": {
            f"learned_ranker_hit_rate@{args.k}": rounded(hit_rate_at_k(rankings, positives, args.k)),
            f"learned_ranker_ndcg@{args.k}": rounded(ndcg_at_k(rankings, positives, args.k)),
            f"current_hybrid_hit_rate@{args.k}": rounded(
                hit_rate_at_k(current_hybrid_rankings, positives, args.k)
            ),
            f"current_hybrid_ndcg@{args.k}": rounded(
                ndcg_at_k(current_hybrid_rankings, positives, args.k)
            ),
            "epochs": args.epochs,
            "candidate_limit": args.candidate_limit,
        },
        "weights": {name: round(value, 5) for name, value in weights.items()},
        "notes": [
            "Candidate-aware pairwise training only learns from examples where retrieval surfaced the held-out item.",
            "Use promote_ranker.py before wiring weights into runtime via TASK_B_RANKER_WEIGHTS.",
        ],
    }
    write_report(Path(args.output), payload)
    print(json.dumps(payload, ensure_ascii=True, indent=2))


def _split_rows(rows: list[dict], validation_fraction: float) -> tuple[list[dict], list[dict]]:
    if len(rows) < 4:
        return rows, rows
    fraction = min(max(validation_fraction, 0.20), 0.80)
    validation_size = max(1, int(len(rows) * fraction))
    split_at = max(1, len(rows) - validation_size)
    return rows[:split_at], rows[split_at:]


class _ProfileCache:
    def __init__(self) -> None:
        self.item_profiles = {}

    def item_profile(self, item: Item):
        if item.item_id not in self.item_profiles:
            self.item_profiles[item.item_id] = build_item_profile(item)
        return self.item_profiles[item.item_id]


def _training_examples(
    rows: list[dict],
    history_map: dict[str, list[UserHistoryItem]],
    item_by_id: dict[str, Item],
    item_list: list[Item],
    bm25: BM25Retriever,
    vectors: LocalVectorRetriever,
    catalog: CandidateCatalog,
    collaborative_index: dict | None,
    candidate_limit: int,
    profile_cache: _ProfileCache,
) -> list[dict]:
    examples = []
    for row in rows:
        positive = item_by_id.get(row["item_id"])
        if not positive:
            continue
        history = history_map.get(row["user_id"], [])
        user_profile = build_user_profile(persona_from_history(history), history)
        pool = _candidate_pool(
            user_profile=user_profile,
            history=history,
            item_list=item_list,
            bm25=bm25,
            vectors=vectors,
            catalog=catalog,
            collaborative_index=collaborative_index,
            candidate_limit=candidate_limit,
        )
        pool_ids = {item.item_id for item in pool.items}
        if positive.item_id not in pool_ids:
            continue
        pool_profiles = [profile_cache.item_profile(item) for item in pool.items]
        max_popularity = max((profile.popularity for profile in pool_profiles), default=0)
        examples.append(
            {
                "user_profile": user_profile,
                "positive": positive,
                "pool": pool,
                "max_popularity": max_popularity,
            }
        )
    return examples


def _train_example(
    example: dict,
    weights: dict[str, float],
    learning_rate: float,
    max_negatives: int,
    profile_cache: _ProfileCache,
) -> None:
    user_profile = example["user_profile"]
    positive = example["positive"]
    pool = example["pool"]
    max_popularity = example["max_popularity"]
    positive_features = _features(
        user_profile=user_profile,
        item=positive,
        pool=pool,
        max_popularity=max_popularity,
        profile_cache=profile_cache,
    )
    negatives = [item for item in pool.items if item.item_id != positive.item_id][:max_negatives]
    for negative in negatives:
        negative_features = _features(
            user_profile=user_profile,
            item=negative,
            pool=pool,
            max_popularity=max_popularity,
            profile_cache=profile_cache,
        )
        if weighted_score(positive_features, weights) <= weighted_score(negative_features, weights):
            for name in FEATURE_NAMES:
                weights[name] += learning_rate * (
                    positive_features.get(name, 0.0) - negative_features.get(name, 0.0)
                )


def _features(
    user_profile,
    item: Item,
    pool: CandidatePool,
    max_popularity: int,
    profile_cache: _ProfileCache,
) -> dict[str, float]:
    return ranker_features(
        user_profile=user_profile,
        item_profile=profile_cache.item_profile(item),
        context_terms=[],
        max_popularity=max_popularity,
        seen_item_ids=set(user_profile.seen_item_ids),
        source_scores=pool.source_scores.get(item.item_id, {}),
    )


def _rank(
    row: dict,
    history_map: dict[str, list[UserHistoryItem]],
    item_by_id: dict[str, Item],
    item_list: list[Item],
    bm25: BM25Retriever,
    vectors: LocalVectorRetriever,
    catalog: CandidateCatalog,
    collaborative_index: dict | None,
    candidate_limit: int,
    weights: dict[str, float] | None,
) -> list[str]:
    history = history_map.get(row["user_id"], [])
    user_profile = build_user_profile(persona_from_history(history), history)
    pool = _candidate_pool(
        user_profile=user_profile,
        history=history,
        item_list=item_list,
        bm25=bm25,
        vectors=vectors,
        catalog=catalog,
        collaborative_index=collaborative_index,
        candidate_limit=candidate_limit,
    )
    ranked = rank_candidates(
        user_profile=user_profile,
        context="",
        candidate_items=pool.items,
        limit=max(10, len(pool.items)),
        weights=weights,
        candidate_sources=pool.sources,
        candidate_source_scores=pool.source_scores,
    )
    if row["item_id"] not in {item.item_id for item in pool.items}:
        # The ranker cannot recover a retrieval miss, but keeping the positive
        # absent preserves honest HitRate/NDCG.
        item_by_id.get(row["item_id"])
    return [item.item_id for item in ranked]


def _candidate_pool(
    user_profile,
    history: list[UserHistoryItem],
    item_list: list[Item],
    bm25: BM25Retriever,
    vectors: LocalVectorRetriever,
    catalog: CandidateCatalog,
    collaborative_index: dict | None,
    candidate_limit: int,
) -> CandidatePool:
    return generate_candidate_pool(
        user_profile=user_profile,
        history=history,
        items=item_list,
        context="",
        collaborative_index=collaborative_index,
        bm25_retriever=bm25,
        vector_retriever=vectors,
        catalog=catalog,
        limit=min(candidate_limit, len(item_list)),
    )


def _items_with_train_popularity(items: dict[str, Item], train: list[dict]) -> list[Item]:
    counts = Counter(row["item_id"] for row in train if row["rating"] >= 4)
    enriched = []
    for item in items.values():
        metadata = dict(item.metadata)
        metadata["review_count"] = counts[item.item_id]
        metadata["rating_number"] = counts[item.item_id]
        enriched.append(item.model_copy(update={"metadata": metadata}))
    return enriched


def _load_collaborative_index(args: argparse.Namespace, train: list[dict]) -> dict | None:
    configured = Path(args.collaborative_index) if args.collaborative_index else None
    candidates = [
        configured,
        Path(args.processed_dir) / "collaborative_retrieval.json",
        Path(args.processed_dir) / "item_neighbors.json",
    ]
    for path in candidates:
        if not path or not path.exists():
            continue
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        if payload.get("item_neighbors"):
            return _attach_review_term_index(payload, path)
        if payload.get("items"):
            return _attach_review_term_index(
                {"type": "legacy_item_neighbors", "item_neighbors": payload["items"]},
                path,
            )
    if len(train) <= 100_000:
        return build_collaborative_retrieval_index(train, top_k=50)
    return None


def _attach_review_term_index(payload: dict, source_path: Path) -> dict:
    review_term_path = source_path.parent / "review_term_retrieval.json"
    if not review_term_path.exists():
        return payload
    try:
        review_term_payload = json.loads(review_term_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return payload
    if review_term_payload.get("term_items"):
        payload = dict(payload)
        payload["review_term_retrieval"] = review_term_payload
    return payload


def _initial_weights() -> dict[str, float]:
    return {
        "preference_match": 0.20,
        "context_match": 0.10,
        "category_match": 0.14,
        "vector_match": 0.14,
        "item_quality": 0.16,
        "popularity": 0.06,
        "novelty": 0.04,
        "confidence": 0.02,
        "collaborative_match": 0.10,
        "retrieval_match": 0.04,
        "source_diversity": 0.02,
        "dislike_match": -0.25,
    }


if __name__ == "__main__":
    main()
