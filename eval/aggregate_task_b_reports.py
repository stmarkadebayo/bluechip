from __future__ import annotations

import argparse
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from eval.common import print_report, write_report  # noqa: E402
from eval.eval_task_b import _task_b_promotion_gate  # noqa: E402
from eval.metrics import rounded  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description="Aggregate sharded Task B eval reports.")
    parser.add_argument("--inputs", nargs="+", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    reports = [json.loads(Path(path).read_text(encoding="utf-8")) for path in args.inputs]
    payload = aggregate_reports(reports, args.inputs)
    write_report(Path(args.output), payload)
    print_report(payload)


def aggregate_reports(reports: list[dict[str, Any]], input_paths: list[str] | None = None) -> dict:
    if not reports:
        raise ValueError("at least one report is required")
    first = reports[0]
    examples = sum(int(report.get("examples") or 0) for report in reports)
    metrics = _weighted_metric_map(reports, "metrics")
    slices = _aggregate_slices(reports)
    candidate_k = _candidate_k(first, metrics)
    ranker_k = int((first.get("promotion_gate") or {}).get("ranker_gate_k") or 10)
    ranker_prefix = str(
        (first.get("promotion_gate") or {}).get("ranker_metric_prefix") or "hybrid_ranker"
    )
    return {
        "task": first.get("task", "Task B"),
        "dataset": first.get("dataset", ""),
        "examples": examples,
        "retriever": first.get("retriever", "legacy"),
        "context_mode": first.get("context_mode", "none"),
        "target_mode": first.get("target_mode", "all_interactions"),
        "aggregation": {
            "type": "sharded_task_b_eval",
            "shards": len(reports),
            "inputs": input_paths or [],
        },
        "target_rating_distribution": _aggregate_rating_distribution(reports),
        "rank_depth": first.get("rank_depth", 0),
        "learned_ranker_active": any(
            bool(report.get("learned_ranker_active")) for report in reports
        ),
        "neural_retriever_active": any(
            bool(report.get("neural_retriever_active")) for report in reports
        ),
        "disabled_sources": sorted(
            {
                source
                for report in reports
                for source in report.get("disabled_sources", [])
            }
        ),
        "metrics": metrics,
        "slices": slices,
        "promotion_gate": _task_b_promotion_gate(
            metrics=metrics,
            slices=slices,
            examples=examples,
            k=ranker_k,
            candidate_k=candidate_k,
            ranker_metric_prefix=ranker_prefix,
        ),
        "retrieval_sources": dict(_sum_counter_field(reports, "retrieval_sources")),
        "retrieval_source_diagnostics": _aggregate_source_diagnostics(
            reports,
            "retrieval_source_diagnostics",
        ),
        "retrieval_source_families": _aggregate_source_diagnostics(
            reports,
            "retrieval_source_families",
        ),
        "miss_analysis": _aggregate_miss_analysis(reports),
        "notes": _aggregate_notes(reports),
    }


def _weighted_metric_map(reports: list[dict[str, Any]], field: str) -> dict[str, float]:
    totals: dict[str, float] = defaultdict(float)
    weights: dict[str, int] = defaultdict(int)
    for report in reports:
        examples = int(report.get("examples") or 0)
        for metric, value in (report.get(field) or {}).items():
            if isinstance(value, int | float):
                totals[metric] += float(value) * examples
                weights[metric] += examples
    return {
        metric: rounded(total / weights[metric])
        for metric, total in sorted(totals.items())
        if weights[metric]
    }


def _aggregate_slices(reports: list[dict[str, Any]]) -> dict[str, dict[str, float | int]]:
    totals: dict[str, dict[str, float]] = defaultdict(lambda: defaultdict(float))
    weights: dict[str, dict[str, int]] = defaultdict(lambda: defaultdict(int))
    examples_by_slice: Counter[str] = Counter()
    for report in reports:
        for slice_name, slice_payload in (report.get("slices") or {}).items():
            slice_examples = int(slice_payload.get("examples") or 0)
            examples_by_slice[slice_name] += slice_examples
            for metric, value in slice_payload.items():
                if metric == "examples" or not isinstance(value, int | float):
                    continue
                totals[slice_name][metric] += float(value) * slice_examples
                weights[slice_name][metric] += slice_examples
    aggregated = {}
    for slice_name in sorted(examples_by_slice):
        aggregated[slice_name] = {"examples": examples_by_slice[slice_name]}
        for metric, total in sorted(totals[slice_name].items()):
            weight = weights[slice_name][metric]
            if weight:
                aggregated[slice_name][metric] = rounded(total / weight)
    return aggregated


def _aggregate_rating_distribution(reports: list[dict[str, Any]]) -> dict:
    buckets: Counter[str] = Counter()
    for report in reports:
        for bucket, payload in (report.get("target_rating_distribution") or {}).items():
            if bucket == "total":
                continue
            buckets[bucket] += int((payload or {}).get("count") or 0)
    total = sum(buckets.values())
    return {
        "total": total,
        **{
            bucket: {"count": count, "share": rounded(count / max(total, 1))}
            for bucket, count in sorted(buckets.items())
        },
    }


def _aggregate_source_diagnostics(reports: list[dict[str, Any]], field: str) -> dict:
    diagnostics: dict[str, dict[str, Any]] = {}
    nested_sources: dict[str, Counter[str]] = defaultdict(Counter)
    for report in reports:
        for source, payload in (report.get(field) or {}).items():
            target = diagnostics.setdefault(source, {})
            for key, value in payload.items():
                if key == "sources" and isinstance(value, dict):
                    nested_sources[source].update(
                        {name: int(count) for name, count in value.items()}
                    )
                elif key.startswith("candidate_recall@"):
                    continue
                elif isinstance(value, int | float):
                    target[key] = target.get(key, 0) + value
                elif key not in target:
                    target[key] = value
    for source, payload in diagnostics.items():
        if nested_sources[source]:
            payload["sources"] = dict(nested_sources[source])
        for key in list(payload):
            if not key.startswith("hits@"):
                continue
            suffix = key.removeprefix("hits@")
            misses = payload.get(f"misses@{suffix}")
            if isinstance(misses, int | float):
                total = float(payload[key]) + float(misses)
                payload[f"candidate_recall@{suffix}"] = rounded(float(payload[key]) / max(total, 1))
    return dict(sorted(diagnostics.items()))


def _aggregate_miss_analysis(reports: list[dict[str, Any]]) -> dict:
    aggregate: dict[str, Any] = {}
    counters: dict[str, Counter[str]] = defaultdict(Counter)
    for report in reports:
        for key, value in (report.get("miss_analysis") or {}).items():
            if isinstance(value, dict):
                counters[key].update({name: int(count) for name, count in value.items()})
            elif key == "candidate_limit":
                aggregate[key] = value
            elif isinstance(value, int | float):
                aggregate[key] = aggregate.get(key, 0) + value
            elif key not in aggregate:
                aggregate[key] = value
    for key, counter in counters.items():
        aggregate[key] = dict(counter)
    return aggregate


def _sum_counter_field(reports: list[dict[str, Any]], field: str) -> Counter[str]:
    counter: Counter[str] = Counter()
    for report in reports:
        counter.update({key: int(value) for key, value in (report.get(field) or {}).items()})
    return counter


def _candidate_k(report: dict[str, Any], metrics: dict[str, float]) -> int:
    gate = report.get("promotion_gate") or {}
    if gate.get("candidate_recall_gate_k"):
        return int(gate["candidate_recall_gate_k"])
    candidate_ks = []
    for metric in metrics:
        if metric.startswith("hybrid_candidate_recall@"):
            candidate_ks.append(int(metric.rsplit("@", 1)[1]))
    return max(candidate_ks or [1000])


def _aggregate_notes(reports: list[dict[str, Any]]) -> list[str]:
    notes = []
    seen = set()
    for report in reports:
        for note in report.get("notes") or []:
            if note in seen or note.startswith("Shard "):
                continue
            notes.append(note)
            seen.add(note)
    notes.append(f"Aggregated from {len(reports)} deterministic Task B eval shards.")
    return notes


if __name__ == "__main__":
    main()
