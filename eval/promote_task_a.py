from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.data_utils import write_json  # noqa: E402


HEAD_METRICS = {
    "calibrated_profile": "calibrated_profile_rmse",
    "adaptive_star": "adaptive_star_rmse",
    "trained_model_raw": "trained_model_raw_rmse",
    "trained_model_selected": "trained_model_rmse",
    "trained_model_star": "trained_model_star_rmse",
    "profile_heuristic": "hybrid_profile_rmse",
}


def main() -> None:
    parser = argparse.ArgumentParser(description="Promote the lowest-RMSE Task A serving head.")
    parser.add_argument("--task-a-report", default="runs/eval/task_a_report.json")
    parser.add_argument("--output", default="runs/eval/task_a_serving_promotion.json")
    parser.add_argument("--policy-output", default="data/processed/task_a_serving_policy.json")
    parser.add_argument("--metric", choices=["rmse", "mae"], default="rmse")
    args = parser.parse_args()

    report = _read_json(Path(args.task_a_report))
    metrics = report.get("metrics", {})
    candidates = _candidate_heads(metrics, args.metric)
    selected = min(candidates, key=lambda row: row["value"]) if candidates else None
    payload = {
        "task": "Task A Serving Promotion",
        "dataset": report.get("dataset", ""),
        "metric": args.metric,
        "passed": bool(selected),
        "selected": selected,
        "candidates": candidates,
        "policy_output": args.policy_output if selected else None,
        "notes": [
            "This gate promotes the lowest error head from a single Task A eval report.",
            "Runtime reads the policy through TASK_A_SERVING_POLICY or the default processed-data path.",
        ],
    }
    if selected:
        write_json(
            Path(args.policy_output),
            {
                "type": "task_a_serving_policy",
                "source_report": args.task_a_report,
                "metric": args.metric,
                "serving_head": selected["head"],
                "metric_name": selected["metric_name"],
                "metric_value": selected["value"],
            },
        )
    write_json(Path(args.output), payload)
    print(json.dumps(payload, ensure_ascii=True, indent=2))


def _candidate_heads(metrics: dict, metric: str) -> list[dict]:
    suffix = f"_{metric}"
    candidates = []
    for head, rmse_metric in HEAD_METRICS.items():
        metric_name = rmse_metric if metric == "rmse" else rmse_metric.removesuffix("_rmse") + suffix
        if metric_name not in metrics:
            continue
        candidates.append(
            {
                "head": head,
                "metric_name": metric_name,
                "value": round(float(metrics[metric_name]), 6),
            }
        )
    return sorted(candidates, key=lambda row: row["value"])


def _read_json(path: Path) -> dict:
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return {}


if __name__ == "__main__":
    main()
