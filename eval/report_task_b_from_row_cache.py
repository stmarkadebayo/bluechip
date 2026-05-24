from __future__ import annotations

import argparse
import json
import sys
import math
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from eval.common import histories_by_user, load_eval_data, print_report, write_report  # noqa: E402
from eval.eval_task_b import (  # noqa: E402
    EVAL_ROW_CACHE_SCHEMA_VERSION,
    _context_for_eval,
    _filter_task_b_targets,
    _sample_eval_rows,
    _shard_eval_rows,
    _task_b_promotion_gate,
)
from app.services.retrieval.source_registry import (  # noqa: E402
    SOURCE_FAMILY_LABELS,
    SOURCE_FAMILY_ORDER,
    retrieval_source_family,
)
from eval.metrics import rounded  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description="Build Task B reports from eval row caches.")
    parser.add_argument("--processed-dir", required=True)
    parser.add_argument("--row-caches", nargs="+", required=True)
    parser.add_argument(
        "--target-mode",
        choices=["all_interactions", "positive_recommendation"],
        default="positive_recommendation",
    )
    parser.add_argument("--output", required=True)
    parser.add_argument("--miss-output", default="")
    parser.add_argument("--max-misses", type=int, default=500)
    parser.add_argument("--candidate-limit", type=int, default=0)
    parser.add_argument("--k", type=int, default=10)
    args = parser.parse_args()

    payload, miss_report = report_from_caches(args)
    if args.miss_output:
        Path(args.miss_output).parent.mkdir(parents=True, exist_ok=True)
        Path(args.miss_output).write_text(
            json.dumps(miss_report, ensure_ascii=True, indent=2) + "\n",
            encoding="utf-8",
        )
    write_report(Path(args.output), payload)
    print_report(payload)


def report_from_caches(args: argparse.Namespace) -> tuple[dict, dict]:
    train, _, test_b, items = load_eval_data(
        reviews_path=Path(""),
        items_path=Path(""),
        processed_dir=Path(args.processed_dir),
    )
    history_map = histories_by_user(train)
    stats = _StreamingStats(candidate_limit=args.candidate_limit or 1000, k=args.k)
    source_signature = None
    for cache_path in args.row_caches:
        cache_path = Path(cache_path)
        header = _cache_header(cache_path)
        pool_payloads = _iter_cache_pool_payloads(cache_path)
        signature = header.get("signature") or {}
        if source_signature is None:
            source_signature = signature
        source_rows = _cache_source_rows(test_b, signature)
        seen_rows = 0
        context_mode = str(signature.get("context_mode") or "none")
        for row, pool_payload in zip(source_rows, pool_payloads, strict=True):
            seen_rows += 1
            if args.target_mode == "positive_recommendation" and float(row.get("rating") or 0) < 4:
                continue
            context = _context_for_eval(
                row,
                history_map.get(row["user_id"], []),
                context_mode,
            )
            stats.add(row, context, pool_payload, history_map, items)
        if seen_rows != len(source_rows):
            raise ValueError(
                f"cache row count mismatch for {cache_path}: "
                f"{seen_rows} cache rows, {len(source_rows)} source rows"
            )

    source_signature = source_signature or {}
    candidate_limit = int(args.candidate_limit or source_signature.get("candidate_limit") or 1000)
    metrics = stats.metrics(candidate_limit)
    slices = stats.slices(candidate_limit)
    miss_report = stats.miss_report(candidate_limit, args.max_misses)
    payload = {
        "task": "Task B",
        "dataset": str(Path(args.processed_dir)),
        "examples": stats.examples,
        "retriever": source_signature.get("retriever", "legacy"),
        "context_mode": source_signature.get("context_mode", "none"),
        "target_mode": args.target_mode,
        "derived_from_row_cache": {
            "source_target_mode": source_signature.get("target_mode", ""),
            "row_caches": args.row_caches,
        },
        "target_rating_distribution": stats.rating_distribution(),
        "rank_depth": source_signature.get("rank_depth", 0),
        "learned_ranker_active": bool(source_signature.get("learned_ranker_artifact")),
        "neural_retriever_active": source_signature.get("retriever") == "neural",
        "disabled_sources": sorted(source_signature.get("disabled_sources") or []),
        "metrics": metrics,
        "slices": slices,
        "promotion_gate": _task_b_promotion_gate(
            metrics=metrics,
            slices=slices,
            examples=stats.examples,
            k=args.k,
            candidate_k=candidate_limit,
        ),
        "retrieval_sources": dict(sorted(stats.source_counts.items())),
        "retrieval_source_diagnostics": stats.source_diagnostics(),
        "retrieval_source_families": stats.source_family_diagnostics(),
        "miss_analysis": miss_report["summary"],
        "notes": [
            "Candidate recall measures whether retrieval surfaced the held-out item before ranking.",
            "This report was derived from Task B eval row caches and did not rerun retrieval.",
            (
                "Target mode is positive_recommendation, so Task B rows with rating < 4 "
                "were excluded from this report."
                if args.target_mode == "positive_recommendation"
                else "Target mode is all_interactions, so every held-out next review is a target."
            ),
        ],
    }
    return payload, miss_report


class _StreamingStats:
    def __init__(self, candidate_limit: int, k: int) -> None:
        self.candidate_limit = candidate_limit
        self.k = k
        self.examples = 0
        self.rating_counts: Counter[str] = Counter()
        self.hit_counts: Counter[int] = Counter()
        self.ndcg_at_k = 0.0
        self.slice_examples: Counter[str] = Counter()
        self.slice_hits: Counter[str] = Counter()
        self.source_counts: Counter[str] = Counter()
        self.source_hits: Counter[str] = Counter()
        self.family_counts: Counter[str] = Counter()
        self.family_hits: Counter[str] = Counter()
        self.family_source_counts = {family: Counter() for family in SOURCE_FAMILY_ORDER}
        self.miss_cause_counts: Counter[str] = Counter()
        self.miss_category_counts: Counter[str] = Counter()
        self.miss_history_buckets: Counter[str] = Counter()
        self.misses: list[dict] = []

    def add(
        self,
        row: dict,
        context: str,
        pool_payload: dict,
        history_map: dict,
        items: dict,
    ) -> None:
        positive = str(row["item_id"])
        item_ids = [str(item_id) for item_id in pool_payload.get("item_ids", [])]
        sources = {
            str(item_id): [str(source) for source in values]
            for item_id, values in (pool_payload.get("sources") or {}).items()
        }
        self.examples += 1
        rating = float(row.get("rating") or 0)
        self.rating_counts["rating_4_5" if rating >= 4 else "rating_1_3"] += 1

        rank = _rank_position(item_ids, positive)
        for recall_k in (self.k, 50, 100, self.candidate_limit):
            if rank is not None and rank <= recall_k:
                self.hit_counts[recall_k] += 1
        if rank is not None and rank <= self.k:
            self.ndcg_at_k += 1.0 / math.log2(rank + 1)

        history = history_map.get(row["user_id"], [])
        for slice_name in _slice_names(row, history, context):
            self.slice_examples[slice_name] += 1
            if rank is not None and rank <= self.candidate_limit:
                self.slice_hits[slice_name] += 1

        for source_names in sources.values():
            self.source_counts.update(source_names)
            for source in source_names:
                family = retrieval_source_family(source)
                self.family_counts[family] += 1
                self.family_source_counts[family][source] += 1

        if rank is not None and rank <= self.candidate_limit:
            positive_sources = sources.get(positive, [])
            self.source_hits.update(positive_sources)
            positive_families = {retrieval_source_family(source) for source in positive_sources}
            self.family_hits.update(positive_families)
        else:
            self._add_miss(row, history, items, rank)

    def metrics(self, candidate_limit: int) -> dict[str, float]:
        metrics = {
            f"hybrid_candidate_recall_hit_rate@{self.k}": rounded(
                self.hit_counts[self.k] / max(self.examples, 1)
            ),
            f"hybrid_candidate_recall_recall@{self.k}": rounded(
                self.hit_counts[self.k] / max(self.examples, 1)
            ),
            f"hybrid_candidate_recall_ndcg@{self.k}": rounded(
                self.ndcg_at_k / max(self.examples, 1)
            ),
        }
        for recall_k in (50, 100, candidate_limit):
            metrics[f"hybrid_candidate_recall@{recall_k}"] = rounded(
                self.hit_counts[recall_k] / max(self.examples, 1)
            )
        return metrics

    def slices(self, candidate_limit: int) -> dict:
        return {
            name: {
                "examples": examples,
                f"hybrid_candidate_recall@{candidate_limit}": rounded(
                    self.slice_hits[name] / max(examples, 1)
                ),
            }
            for name, examples in sorted(self.slice_examples.items())
            if examples
        }

    def rating_distribution(self) -> dict:
        total = max(self.examples, 1)
        return {
            "total": self.examples,
            **{
                bucket: {"count": count, "share": rounded(count / total)}
                for bucket, count in sorted(self.rating_counts.items())
            },
        }

    def source_diagnostics(self) -> dict:
        return {
            source: {
                "count": count,
                f"hits@{self.candidate_limit}": self.source_hits[source],
                f"misses@{self.candidate_limit}": max(
                    self.examples - self.source_hits[source],
                    0,
                ),
                f"candidate_recall@{self.candidate_limit}": rounded(
                    self.source_hits[source] / max(self.examples, 1)
                ),
                "family": retrieval_source_family(source),
            }
            for source, count in sorted(self.source_counts.items())
        }

    def source_family_diagnostics(self) -> dict:
        payload = {}
        for family in SOURCE_FAMILY_ORDER:
            hits = self.family_hits[family]
            payload[family] = {
                "label": SOURCE_FAMILY_LABELS[family],
                "count": self.family_counts[family],
                f"hits@{self.candidate_limit}": hits,
                f"misses@{self.candidate_limit}": max(self.examples - hits, 0),
                f"candidate_recall@{self.candidate_limit}": rounded(
                    hits / max(self.examples, 1)
                ),
                "sources": dict(sorted(self.family_source_counts[family].items())),
            }
        return payload

    def miss_report(self, candidate_limit: int, max_misses: int) -> dict:
        return {
            "summary": {
                "candidate_limit": candidate_limit,
                "candidate_misses": sum(self.miss_category_counts.values()),
                "miss_cause_counts": dict(sorted(self.miss_cause_counts.items())),
                "miss_category_counts": dict(self.miss_category_counts.most_common(10)),
                "miss_history_buckets": dict(sorted(self.miss_history_buckets.items())),
            },
            "misses": self.misses[:max_misses],
        }

    def _add_miss(self, row: dict, history: list, items: dict, rank: int | None) -> None:
        positive = str(row["item_id"])
        target = items.get(positive)
        target_category = (target.category if target else row.get("category")) or "unknown"
        bucket = _history_bucket(len(history))
        self.miss_cause_counts["hybrid_retrieval_miss"] += 1
        self.miss_category_counts[target_category] += 1
        self.miss_history_buckets[bucket] += 1
        if len(self.misses) < 500:
            self.misses.append(
                {
                    "user_id": row["user_id"],
                    "target_item_id": positive,
                    "target_name": target.name if target else row.get("item_name"),
                    "target_category": target_category,
                    "history_length": len(history),
                    "history_bucket": bucket,
                    "hybrid_candidate_rank": rank,
                    "likely_causes": ["hybrid_retrieval_miss"],
                }
            )


def _cache_header(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as handle:
        header = json.loads(handle.readline())
        if (
            header.get("type") != "task_b_eval_row_cache"
            or header.get("schema_version") != EVAL_ROW_CACHE_SCHEMA_VERSION
        ):
            raise ValueError(f"unsupported Task B row cache: {path}")
        return header


def _iter_cache_pool_payloads(path: Path):
    with path.open("r", encoding="utf-8") as handle:
        handle.readline()
        for line in handle:
            if line.strip():
                row = json.loads(line)
                yield row.get("hybrid_pool") or {}


def _rank_position(item_ids: list[str], positive: str) -> int | None:
    try:
        return item_ids.index(positive) + 1
    except ValueError:
        return None


def _slice_names(row: dict, history: list, context: str) -> list[str]:
    names = ["all"]
    history_len = len(history)
    if history_len <= 2:
        names.append("sparse_history_1_2")
    elif history_len <= 7:
        names.append("medium_history_3_7")
    else:
        names.append("warm_history_8_plus")
    if history_len == 0:
        names.append("cold_start")
    if _is_cross_domain(row, history):
        names.append("cross_domain")
    if context:
        names.extend(["context_heavy", "intent_heavy"])
    return names


def _is_cross_domain(row: dict, history: list) -> bool:
    target_category = row.get("category")
    if not target_category:
        return False
    positive_categories = {
        item.category
        for item in history
        if item.category and item.rating >= 4
    }
    return bool(positive_categories) and target_category not in positive_categories


def _history_bucket(history_len: int) -> str:
    if history_len <= 2:
        return "sparse_1_2"
    if history_len <= 7:
        return "medium_3_7"
    return "warm_8_plus"


def _cache_source_rows(test_b: list[dict], signature: dict) -> list[dict]:
    rows = _filter_task_b_targets(test_b, str(signature.get("target_mode") or "all_interactions"))
    max_examples = int(signature.get("max_examples") or 0)
    if max_examples:
        rows = _sample_eval_rows(
            rows,
            max_examples,
            str(signature.get("sample_strategy") or "first"),
        )
    return _shard_eval_rows(
        rows,
        int(signature.get("shard_count") or 1),
        int(signature.get("shard_index") or 0),
    )


if __name__ == "__main__":
    main()
