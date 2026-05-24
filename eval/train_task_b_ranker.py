from __future__ import annotations

import argparse
import json
import math
import multiprocessing as mp
import sys
from dataclasses import dataclass
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.models.schemas import Item  # noqa: E402
from app.services.profiling.user_profile import build_user_profile  # noqa: E402
from app.services.ranking.learned_task_b import (  # noqa: E402
    DEFAULT_FEATURE_NAMES,
    TaskBRankerArtifact,
    feature_vector,
    sigmoid,
)
from app.services.ranking.recommendation import rank_candidates  # noqa: E402
from eval.common import persona_from_history, print_report, write_report  # noqa: E402
from eval.eval_task_b import (  # noqa: E402
    EvalDatasetBuilder,
    _candidate_pool,
)
from eval.metrics import hit_rate_at_k, ndcg_at_k, recall_at_k, rounded  # noqa: E402


@dataclass(frozen=True)
class TrainingExample:
    features: dict[str, float]
    label: int


@dataclass(frozen=True)
class RankedFeatureItem:
    item_id: str
    features: dict[str, float]


@dataclass(frozen=True)
class RankingGroup:
    positive_item_id: str
    items: list[RankedFeatureItem]


GROUP_CACHE_SCHEMA_VERSION = 1
_WORKER_ARGS = None
_WORKER_DATASET = None
_WORKER_ITEM_PROFILE_CACHE = None


def main() -> None:
    parser = argparse.ArgumentParser(description="Train/evaluate a small Task B linear ranker.")
    parser.add_argument("--reviews", default="data/sample/reviews.jsonl")
    parser.add_argument("--items", default="data/sample/items.jsonl")
    parser.add_argument("--processed-dir", default="data/processed")
    parser.add_argument("--collaborative-index", default="")
    parser.add_argument("--output", default="runs/eval/task_b_ranker_report.json")
    parser.add_argument("--artifact-output", default="runs/models/task_b_linear_ranker.json")
    parser.add_argument("--k", type=int, default=10)
    parser.add_argument("--candidate-limit", type=int, default=200)
    parser.add_argument(
        "--rank-depth",
        type=int,
        default=0,
        help=(
            "Number of retrieved candidates to score before training. "
            "0 preserves full-pool ranking; positives outside the depth are appended for learning."
        ),
    )
    parser.add_argument("--max-examples", type=int, default=0)
    parser.add_argument("--sample-strategy", choices=["first", "stride"], default="stride")
    parser.add_argument(
        "--target-mode",
        choices=["all_interactions", "positive_recommendation"],
        default="all_interactions",
        help=(
            "Which held-out Task B rows to train on: all next interactions, or only "
            "rating >= 4 positive recommendation targets."
        ),
    )
    parser.add_argument("--context-mode", choices=["none", "synthetic"], default="none")
    parser.add_argument("--build-collaborative", action="store_true")
    parser.add_argument("--disabled-sources", default="")
    parser.add_argument("--retriever", choices=["legacy", "neural"], default="legacy")
    parser.add_argument("--train-fraction", type=float, default=0.7)
    parser.add_argument("--epochs", type=int, default=60)
    parser.add_argument("--learning-rate", type=float, default=0.08)
    parser.add_argument("--l2", type=float, default=0.0005)
    parser.add_argument("--negative-limit", type=int, default=30)
    parser.add_argument(
        "--loss",
        choices=["pointwise", "pairwise"],
        default="pointwise",
        help="Train the default pointwise logistic ranker or a pairwise hard-negative ranker.",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=1,
        help=(
            "Parallel worker processes for ranked group building. "
            "Uses fork copy-on-write when available to avoid reloading full artifacts."
        ),
    )
    parser.add_argument(
        "--group-cache",
        default="",
        help="Optional JSONL cache for expensive Task B ranked feature groups.",
    )
    parser.add_argument(
        "--rebuild-group-cache",
        action="store_true",
        help="Ignore an existing --group-cache and rebuild it.",
    )
    parser.add_argument(
        "--progress-every",
        type=int,
        default=1000,
        help="Print progress to stderr every N Task B rows; set 0 to disable.",
    )
    args = parser.parse_args()

    groups = _load_group_cache(args)
    if groups is None:
        dataset = EvalDatasetBuilder(args).build()
        print(
            "Task B ranker training dataset: "
            f"{len(dataset.train)} train rows, {len(dataset.test_b)} Task B rows, "
            f"{len(dataset.item_list)} items",
            file=sys.stderr,
            flush=True,
        )
        groups = _build_groups(args, dataset)
        _write_group_cache(args, dataset, groups)
    split_index = max(1, min(len(groups), int(len(groups) * args.train_fraction)))
    train_groups = groups[:split_index]
    eval_groups = groups[split_index:] or groups[:]
    examples = _examples_from_groups(train_groups, args.negative_limit)
    training_metadata = {
        "examples": len(examples),
        "ranking_groups": len(train_groups),
        "eval_groups": len(eval_groups),
        "candidate_limit": args.candidate_limit,
        "rank_depth": args.rank_depth,
        "negative_limit": args.negative_limit,
        "epochs": args.epochs,
        "learning_rate": args.learning_rate,
        "l2": args.l2,
        "loss": args.loss,
        "target_mode": args.target_mode,
        "group_cache": args.group_cache,
    }
    if args.loss == "pairwise":
        artifact = _train_pairwise(
            groups=train_groups,
            feature_names=DEFAULT_FEATURE_NAMES,
            negative_limit=args.negative_limit,
            epochs=args.epochs,
            learning_rate=args.learning_rate,
            l2=args.l2,
            training=training_metadata,
        )
    else:
        artifact = _train(
            examples=examples,
            feature_names=DEFAULT_FEATURE_NAMES,
            epochs=args.epochs,
            learning_rate=args.learning_rate,
            l2=args.l2,
            training=training_metadata,
        )
    artifact.write_json(args.artifact_output)
    payload = _report_payload(
        args=args,
        artifact=artifact,
        groups=groups,
        train_groups=train_groups,
        eval_groups=eval_groups,
        examples=examples,
    )
    write_report(Path(args.output), payload)
    print_report(payload)


def _build_groups(args: argparse.Namespace, dataset) -> list[RankingGroup]:
    if args.workers > 1:
        return _build_groups_parallel(args, dataset)

    groups = []
    item_profile_cache = {}
    for index, _row in enumerate(dataset.test_b):
        if args.progress_every and (index + 1) % args.progress_every == 0:
            print(
                "Task B ranker group progress: "
                f"{index + 1}/{len(dataset.test_b)} rows, {len(groups)} candidate-hit groups",
                file=sys.stderr,
                flush=True,
            )
        group = _build_group_for_index(args, dataset, index, item_profile_cache)
        if group is not None:
            groups.append(group)
    return groups


def _build_groups_parallel(args: argparse.Namespace, dataset) -> list[RankingGroup]:
    if "fork" not in mp.get_all_start_methods():
        print(
            "Fork multiprocessing unavailable; falling back to serial group building.",
            file=sys.stderr,
            flush=True,
        )
        serial_args = argparse.Namespace(**vars(args))
        serial_args.workers = 1
        return _build_groups(serial_args, dataset)

    global _WORKER_ARGS, _WORKER_DATASET
    _WORKER_ARGS = args
    _WORKER_DATASET = dataset
    groups: list[RankingGroup] = []
    ctx = mp.get_context("fork")
    chunksize = max(10, min(100, len(dataset.test_b) // max(args.workers * 20, 1)))
    with ctx.Pool(processes=args.workers, initializer=_init_worker) as pool:
        for index, group in enumerate(
            pool.imap(_build_group_for_worker_index, range(len(dataset.test_b)), chunksize),
            start=1,
        ):
            if group is not None:
                groups.append(group)
            if args.progress_every and index % args.progress_every == 0:
                print(
                    "Task B ranker group progress: "
                    f"{index}/{len(dataset.test_b)} rows, {len(groups)} candidate-hit groups",
                    file=sys.stderr,
                    flush=True,
                )
    return groups


def _init_worker() -> None:
    global _WORKER_ITEM_PROFILE_CACHE
    _WORKER_ITEM_PROFILE_CACHE = {}


def _build_group_for_worker_index(index: int) -> RankingGroup | None:
    if _WORKER_ARGS is None or _WORKER_DATASET is None:
        raise RuntimeError("Task B ranker worker was not initialized.")
    cache = _WORKER_ITEM_PROFILE_CACHE
    if cache is None:
        cache = {}
    return _build_group_for_index(
        args=_WORKER_ARGS,
        dataset=_WORKER_DATASET,
        index=index,
        item_profile_cache=cache,
    )


def _build_group_for_index(
    args: argparse.Namespace,
    dataset,
    index: int,
    item_profile_cache: dict,
) -> RankingGroup | None:
    row = dataset.test_b[index]
    contexts = getattr(dataset, "contexts", [""] * len(dataset.test_b))
    history = dataset.history_map.get(row["user_id"], [])
    context = contexts[index] if index < len(contexts) else ""
    pool = _candidate_pool(
        row=row,
        history=history,
        context=context,
        retriever=dataset.retriever,
        vector_retriever=dataset.vector_retriever,
        catalog=dataset.catalog,
        item_list=dataset.item_list,
        candidate_limit=args.candidate_limit,
        collaborative_index=dataset.collaborative_index,
        disabled_sources=dataset.disabled_sources,
        neural_retriever=dataset.neural_retriever,
    )
    if row["item_id"] not in {item.item_id for item in pool.items}:
        return None
    candidate_items = _training_rank_items(
        pool_items=pool.items,
        positive_item_id=row["item_id"],
        rank_depth=args.rank_depth,
    )
    persona = persona_from_history(history)
    user_profile = build_user_profile(persona=persona, history=history, locale=None)
    ranked = rank_candidates(
        user_profile=user_profile,
        context=context,
        candidate_items=candidate_items,
        limit=len(candidate_items),
        candidate_sources=pool.sources,
        candidate_source_scores=pool.source_scores,
        item_profile_cache=item_profile_cache,
    )
    return RankingGroup(
        positive_item_id=row["item_id"],
        items=[
            RankedFeatureItem(
                item_id=item.item_id,
                features=_non_zero_features(feature_vector(item, DEFAULT_FEATURE_NAMES)),
            )
            for item in ranked
        ],
    )


def _training_rank_items(
    pool_items: list[Item],
    positive_item_id: str,
    rank_depth: int,
) -> list[Item]:
    if rank_depth <= 0:
        return pool_items
    selected = list(pool_items[: max(rank_depth, 1)])
    if positive_item_id in {item.item_id for item in selected}:
        return selected
    positive = next((item for item in pool_items if item.item_id == positive_item_id), None)
    if positive is not None:
        selected.append(positive)
    return selected


def _examples_from_groups(
    groups: list[RankingGroup],
    negative_limit: int,
) -> list[TrainingExample]:
    examples = []
    for group in groups:
        positive = [item for item in group.items if item.item_id == group.positive_item_id]
        negatives = [item for item in group.items if item.item_id != group.positive_item_id]
        selected = positive + negatives[:max(negative_limit, 0)]
        for item in selected:
            examples.append(
                TrainingExample(
                    features=item.features,
                    label=1 if item.item_id == group.positive_item_id else 0,
                )
            )
    return examples


def _train(
    examples: list[TrainingExample],
    feature_names: tuple[str, ...],
    epochs: int,
    learning_rate: float,
    l2: float,
    training: dict,
) -> TaskBRankerArtifact:
    weights = {name: 0.0 for name in feature_names}
    positive_count = sum(example.label for example in examples)
    negative_count = max(len(examples) - positive_count, 1)
    intercept = _logit(positive_count / max(positive_count + negative_count, 1))
    for _ in range(max(epochs, 0)):
        for example in examples:
            prediction = sigmoid(
                intercept
                + sum(weights[name] * example.features.get(name, 0.0) for name in feature_names)
            )
            error = prediction - example.label
            intercept -= learning_rate * error
            for name in feature_names:
                value = example.features.get(name, 0.0)
                if value:
                    weights[name] -= learning_rate * ((error * value) + (l2 * weights[name]))
    non_zero = {
        name: weight
        for name, weight in weights.items()
        if abs(weight) >= 1e-9
    }
    return TaskBRankerArtifact(
        feature_names=feature_names,
        weights=non_zero,
        intercept=intercept,
        training=training,
    )


def _train_pairwise(
    groups: list[RankingGroup],
    feature_names: tuple[str, ...],
    negative_limit: int,
    epochs: int,
    learning_rate: float,
    l2: float,
    training: dict,
) -> TaskBRankerArtifact:
    weights = {name: 0.0 for name in feature_names}
    intercept = 0.0
    for _ in range(max(epochs, 0)):
        for group in groups:
            positive = next(
                (item for item in group.items if item.item_id == group.positive_item_id),
                None,
            )
            if positive is None:
                continue
            negatives = [
                item for item in group.items if item.item_id != group.positive_item_id
            ][:max(negative_limit, 0)]
            positive_score = _score_raw_features(positive.features, weights, intercept, feature_names)
            for negative in negatives:
                negative_score = _score_raw_features(
                    negative.features,
                    weights,
                    intercept,
                    feature_names,
                )
                error = sigmoid(negative_score - positive_score)
                for name in feature_names:
                    delta = positive.features.get(name, 0.0) - negative.features.get(name, 0.0)
                    if delta:
                        weights[name] += learning_rate * ((error * delta) - (l2 * weights[name]))
    non_zero = {
        name: weight
        for name, weight in weights.items()
        if abs(weight) >= 1e-9
    }
    return TaskBRankerArtifact(
        feature_names=feature_names,
        weights=non_zero,
        intercept=intercept,
        model_name="task_b_pairwise_ranker",
        training=training,
    )


def _score_raw_features(
    features: dict[str, float],
    weights: dict[str, float],
    intercept: float,
    feature_names: tuple[str, ...],
) -> float:
    return intercept + sum(weights[name] * features.get(name, 0.0) for name in feature_names)


def _report_payload(
    args: argparse.Namespace,
    artifact: TaskBRankerArtifact,
    groups: list[RankingGroup],
    train_groups: list[RankingGroup],
    eval_groups: list[RankingGroup],
    examples: list[TrainingExample],
) -> dict:
    train_metrics = _metrics(train_groups, artifact, args.k)
    eval_metrics = _metrics(eval_groups, artifact, args.k)
    return {
        "task": "Task B learned ranker",
        "dataset": str(Path(args.processed_dir)),
        "examples": len(examples),
        "metrics": {
            **{f"train_{name}": value for name, value in train_metrics.items()},
            **{f"eval_{name}": value for name, value in eval_metrics.items()},
        },
        "slices": {},
        "retrieval_sources": {},
        "miss_analysis": {
            "candidate_hit_groups": len(groups),
            "train_groups": len(train_groups),
            "eval_groups": len(eval_groups),
            "candidate_limit": args.candidate_limit,
            "rank_depth": args.rank_depth,
            "group_cache": args.group_cache,
        },
        "artifact": str(Path(args.artifact_output)),
        "notes": [
            "Training uses only rows where hybrid retrieval surfaced the held-out Task B item.",
            (
                "The artifact is a deterministic pairwise linear reranker over "
                "RecommendationItem score_components and retrieval_scores."
                if artifact.model_name == "task_b_pairwise_ranker"
                else "The artifact is a deterministic logistic linear model over "
                "RecommendationItem score_components and retrieval_scores."
            ),
            "Pass --learned-ranker-artifact to eval_task_b.py to compare the artifact against the main hybrid ranker.",
        ],
    }


def _metrics(groups: list[RankingGroup], artifact: TaskBRankerArtifact, k: int) -> dict[str, float]:
    positives = [group.positive_item_id for group in groups]
    baseline_rankings = [[item.item_id for item in group.items] for group in groups]
    learned_rankings = [
        [item.item_id for item in _rerank_feature_items(group.items, artifact)]
        for group in groups
    ]
    return {
        f"baseline_hit_rate@{k}": rounded(hit_rate_at_k(baseline_rankings, positives, k)),
        f"baseline_recall@{k}": rounded(recall_at_k(baseline_rankings, positives, k)),
        f"baseline_ndcg@{k}": rounded(ndcg_at_k(baseline_rankings, positives, k)),
        f"learned_hit_rate@{k}": rounded(hit_rate_at_k(learned_rankings, positives, k)),
        f"learned_recall@{k}": rounded(recall_at_k(learned_rankings, positives, k)),
        f"learned_ndcg@{k}": rounded(ndcg_at_k(learned_rankings, positives, k)),
    }


def _load_group_cache(args: argparse.Namespace) -> list[RankingGroup] | None:
    path = Path(args.group_cache) if args.group_cache else None
    if path is None or args.rebuild_group_cache or not path.exists():
        return None
    signature = _group_cache_signature(args)
    with path.open("r", encoding="utf-8") as handle:
        header_line = handle.readline()
        if not header_line:
            return None
        header = json.loads(header_line)
        if (
            header.get("type") != "task_b_ranker_group_cache"
            or header.get("schema_version") != GROUP_CACHE_SCHEMA_VERSION
            or header.get("signature") != signature
        ):
            raise ValueError(f"Task B group cache config mismatch: {path}")
        groups = []
        for line in handle:
            if not line.strip():
                continue
            row = json.loads(line)
            groups.append(
                RankingGroup(
                    positive_item_id=str(row["positive_item_id"]),
                    items=[
                        RankedFeatureItem(
                            item_id=str(item["item_id"]),
                            features={
                                str(name): float(value)
                                for name, value in (item.get("features") or {}).items()
                            },
                        )
                        for item in row.get("items", [])
                    ],
                )
            )
    print(
        f"Loaded {len(groups)} Task B ranked groups from cache {path}",
        file=sys.stderr,
        flush=True,
    )
    return groups


def _write_group_cache(
    args: argparse.Namespace,
    dataset,
    groups: list[RankingGroup],
) -> None:
    if not args.group_cache:
        return
    path = Path(args.group_cache)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp_path = path.with_name(path.name + ".tmp")
    header = {
        "type": "task_b_ranker_group_cache",
        "schema_version": GROUP_CACHE_SCHEMA_VERSION,
        "signature": _group_cache_signature(args),
        "metadata": _group_cache_metadata(dataset),
    }
    with temp_path.open("w", encoding="utf-8") as handle:
        handle.write(json.dumps(header, ensure_ascii=True, separators=(",", ":")) + "\n")
        for group in groups:
            handle.write(
                json.dumps(
                    {
                        "positive_item_id": group.positive_item_id,
                        "items": [
                            {"item_id": item.item_id, "features": item.features}
                            for item in group.items
                        ],
                    },
                    ensure_ascii=True,
                    separators=(",", ":"),
                )
                + "\n"
            )
    temp_path.replace(path)
    print(
        f"Wrote {len(groups)} Task B ranked groups to cache {path}",
        file=sys.stderr,
        flush=True,
    )


def _group_cache_signature(args: argparse.Namespace) -> dict:
    return {
        "processed_dir": str(Path(args.processed_dir)),
        "reviews": str(Path(args.reviews)),
        "items": str(Path(args.items)),
        "collaborative_index": str(args.collaborative_index or ""),
        "candidate_limit": int(args.candidate_limit),
        "rank_depth": int(args.rank_depth),
        "max_examples": int(args.max_examples),
        "sample_strategy": str(args.sample_strategy),
        "context_mode": str(args.context_mode),
        "retriever": str(args.retriever),
        "disabled_sources": sorted(
            source.strip()
            for source in str(args.disabled_sources or "").split(",")
            if source.strip()
        ),
        "target_mode": str(args.target_mode),
    }


def _group_cache_metadata(dataset) -> dict:
    return {
        "train_rows": len(dataset.train),
        "task_b_rows": len(dataset.test_b),
        "items_count": len(dataset.item_list),
    }


def _non_zero_features(features: dict[str, float]) -> dict[str, float]:
    return {
        name: round(float(value), 6)
        for name, value in features.items()
        if abs(float(value)) > 1e-12
    }


def _rerank_feature_items(
    items: list[RankedFeatureItem],
    artifact: TaskBRankerArtifact,
) -> list[RankedFeatureItem]:
    scored = [
        (
            _score_features(item.features, artifact),
            float(item.features.get("component:raw_score", 0.0)),
            -index,
            item,
        )
        for index, item in enumerate(items)
    ]
    scored.sort(key=lambda row: (row[0], row[1], row[2]), reverse=True)
    return [item for _, _, _, item in scored]


def _score_features(features: dict[str, float], artifact: TaskBRankerArtifact) -> float:
    return artifact.intercept + sum(
        artifact.weights.get(name, 0.0) * features.get(name, 0.0)
        for name in artifact.feature_names
    )


def _logit(probability: float) -> float:
    clipped = min(max(probability, 1e-6), 1 - 1e-6)
    return math.log(clipped / (1 - clipped))


if __name__ == "__main__":
    main()
