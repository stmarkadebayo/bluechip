from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.services.ranking.features import FEATURE_NAMES  # noqa: E402
from scripts.data_utils import write_json  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description="Gate Task B learned-ranker promotion.")
    parser.add_argument("--task-b-report", default="runs/eval/task_b_report.json")
    parser.add_argument("--learned-ranker-report", default="runs/eval/learned_ranker.json")
    parser.add_argument("--output", default="runs/eval/task_b_ranker_promotion.json")
    parser.add_argument("--weights-output", default="data/processed/task_b_ranker_weights.json")
    parser.add_argument("--k", type=int, default=10)
    parser.add_argument("--candidate-limit", type=int, default=200)
    parser.add_argument("--min-rank-gain", type=float, default=0.0001)
    parser.add_argument("--min-candidate-recall-delta", type=float, default=0.0)
    args = parser.parse_args()

    task_b = _read_json(Path(args.task_b_report))
    learned = _read_json(Path(args.learned_ranker_report))
    metrics = task_b.get("metrics", {})
    learned_metrics = learned.get("metrics", {})
    checks = _checks(task_b, learned, metrics, learned_metrics, args.k, args.candidate_limit, args)
    passed = all(check["passed"] for check in checks)
    weights = _validated_weights(learned.get("weights", {}))

    stale_weights_removed = False
    weights_output_path = Path(args.weights_output)
    if not (passed and bool(weights)) and weights_output_path.exists():
        weights_output_path.unlink()
        stale_weights_removed = True

    payload = {
        "task": "Task B Ranker Promotion Gate",
        "passed": passed and bool(weights),
        "checks": checks,
        "weights_output": args.weights_output if passed and weights else None,
        "stale_weights_removed": stale_weights_removed,
        "notes": [
            "Promotion requires retrieval not to regress before ranking quality is considered.",
            "Runtime only uses promoted weights when TASK_B_RANKER_WEIGHTS points to the output artifact.",
        ],
    }
    if payload["passed"]:
        write_json(
            weights_output_path,
            {
                "type": "task_b_ranker_weights",
                "source_report": args.learned_ranker_report,
                "weights": weights,
            },
        )
    write_json(Path(args.output), payload)
    print(json.dumps(payload, ensure_ascii=True, indent=2))


def _read_json(path: Path) -> dict:
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return {}


def _checks(
    task_b: dict,
    learned: dict,
    metrics: dict,
    learned_metrics: dict,
    k: int,
    candidate_limit: int,
    args: argparse.Namespace,
) -> list[dict]:
    filtered_hit = float(metrics.get(f"filtered_popularity_hit_rate@{k}", 0.0))
    filtered_ndcg = float(metrics.get(f"filtered_popularity_ndcg@{k}", 0.0))
    hybrid_hit = float(metrics.get(f"hybrid_ranker_hit_rate@{k}", 0.0))
    hybrid_ndcg = float(metrics.get(f"hybrid_ranker_ndcg@{k}", 0.0))
    learned_current_hybrid_ndcg = float(learned_metrics.get(f"current_hybrid_ndcg@{k}", hybrid_ndcg))
    base_recall = float(metrics.get(f"base_candidate_recall@{candidate_limit}", 0.0))
    hybrid_recall = float(metrics.get(f"hybrid_candidate_recall@{candidate_limit}", 0.0))
    learned_ndcg = float(learned_metrics.get(f"learned_ranker_ndcg@{k}", 0.0))
    return [
        {
            "name": "dataset_match",
            "observed": learned.get("dataset", ""),
            "required": task_b.get("dataset", ""),
            "passed": learned.get("dataset") == task_b.get("dataset"),
        },
        {
            "name": "learned_ranker_uses_holdout_split",
            "observed": learned.get("split_strategy", ""),
            "required": "test_b_train_validation_split",
            "passed": learned.get("split_strategy") == "test_b_train_validation_split",
        },
        _check(
            "candidate_recall_not_regressed",
            hybrid_recall,
            base_recall + args.min_candidate_recall_delta,
            hybrid_recall >= base_recall + args.min_candidate_recall_delta,
        ),
        _check(
            "hybrid_hit_rate_beats_filtered_popularity",
            hybrid_hit,
            filtered_hit + args.min_rank_gain,
            hybrid_hit >= filtered_hit + args.min_rank_gain,
        ),
        _check(
            "hybrid_ndcg_beats_filtered_popularity",
            hybrid_ndcg,
            filtered_ndcg + args.min_rank_gain,
            hybrid_ndcg >= filtered_ndcg + args.min_rank_gain,
        ),
        _check(
            "learned_ranker_not_worse_than_same_slice_hybrid_ndcg",
            learned_ndcg,
            learned_current_hybrid_ndcg,
            learned_ndcg >= learned_current_hybrid_ndcg,
        ),
    ]


def _check(name: str, observed: float, required: float, passed: bool) -> dict:
    return {
        "name": name,
        "observed": round(observed, 6),
        "required": round(required, 6),
        "passed": passed,
    }


def _validated_weights(raw: dict) -> dict[str, float]:
    if not isinstance(raw, dict):
        return {}
    return {
        name: round(float(raw[name]), 6)
        for name in FEATURE_NAMES
        if name in raw and isinstance(raw[name], (int, float))
    }


if __name__ == "__main__":
    main()
