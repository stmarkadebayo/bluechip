from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path


DEFAULT_LEAN_DISABLED = (
    "vector_profile",
    "bm25_profile",
    "beauty_sparse_tail",
    "sparse_category_tail",
)

ABLATIONS = {
    "lean": (),
    "no_implicit_item_item": ("implicit_item_item",),
    "no_co_visitation": ("co_visitation",),
    "no_user_neighbor": ("user_neighbor",),
    "no_collaborative": (
        "co_visitation",
        "implicit_item_item",
        "user_neighbor",
        "sequential_transition",
        "category_transition",
    ),
    "no_review_terms": (
        "review_term_profile",
        "beauty_review_term_profile",
        "lexical_item_neighbor",
        "beauty_lexical_item_neighbor",
    ),
    "no_evidence_graph": (
        "aspect_evidence_graph",
        "category_aspect_graph",
        "sequential_transition",
        "category_transition",
    ),
    "no_aspect_taxonomy": (
        "aspect_profile",
        "beauty_aspect_profile",
        "beauty_taxonomy_aspect",
        "beauty_taxonomy_window",
    ),
    "no_popularity_fallback": (
        "category_affinity_popular",
        "category_popular",
        "global_popular",
    ),
}

TABLE_METRICS = (
    "hybrid_ranker_hit_rate@10",
    "hybrid_ranker_ndcg@10",
    "hybrid_candidate_recall@50",
    "hybrid_candidate_recall@100",
)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run Task B source ablations.")
    parser.add_argument("--processed-dir", default="data/processed/all_categories")
    parser.add_argument("--output-dir", default="runs/eval/task_b_source_ablation")
    parser.add_argument("--max-examples", type=int, default=100)
    parser.add_argument("--candidate-limit", type=int, default=1000)
    parser.add_argument("--sample-strategy", choices=["first", "stride"], default="stride")
    parser.add_argument(
        "--base-disabled-sources",
        default=",".join(DEFAULT_LEAN_DISABLED),
        help="Comma-separated sources disabled for every ablation run.",
    )
    parser.add_argument(
        "--candidate-recall-only",
        action="store_true",
        help="Use the faster candidate-recall gate instead of final ranking metrics.",
    )
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    base_disabled = _split_sources(args.base_disabled_sources)
    runs = []

    for name, extra_disabled in ABLATIONS.items():
        disabled = tuple(sorted(set(base_disabled) | set(extra_disabled)))
        report_path = output_dir / f"{name}.json"
        command = [
            sys.executable,
            "eval/eval_task_b.py",
            "--processed-dir",
            args.processed_dir,
            "--hybrid-only",
            "--sample-strategy",
            args.sample_strategy,
            "--max-examples",
            str(args.max_examples),
            "--candidate-limit",
            str(args.candidate_limit),
            "--disabled-sources",
            ",".join(disabled),
            "--output",
            str(report_path),
        ]
        if args.candidate_recall_only:
            command.append("--candidate-recall-only")
        subprocess.run(command, check=True)
        payload = json.loads(report_path.read_text(encoding="utf-8"))
        runs.append(
            {
                "name": name,
                "disabled_sources": list(disabled),
                "examples": payload.get("examples"),
                "metrics": payload.get("metrics", {}),
                "slices": payload.get("slices", {}),
                "retrieval_sources": payload.get("retrieval_sources", {}),
            }
        )

    summary = {
        "task": "Task B source ablation",
        "processed_dir": args.processed_dir,
        "max_examples": args.max_examples,
        "candidate_limit": args.candidate_limit,
        "sample_strategy": args.sample_strategy,
        "base_disabled_sources": list(base_disabled),
        "candidate_recall_only": args.candidate_recall_only,
        "runs": runs,
    }
    (output_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=True, indent=2) + "\n",
        encoding="utf-8",
    )
    (output_dir / "summary.md").write_text(_summary_markdown(summary), encoding="utf-8")
    print(f"Wrote ablation summary to {output_dir / 'summary.json'}")


def _split_sources(value: str) -> tuple[str, ...]:
    return tuple(source.strip() for source in value.split(",") if source.strip())


def _summary_markdown(summary: dict) -> str:
    lines = [
        "# Task B Source Ablation",
        "",
        f"- Processed dir: `{summary['processed_dir']}`",
        f"- Examples: `{summary['max_examples']}`",
        f"- Candidate limit: `{summary['candidate_limit']}`",
        f"- Sample strategy: `{summary['sample_strategy']}`",
        f"- Candidate recall only: `{summary['candidate_recall_only']}`",
        f"- Base disabled sources: `{', '.join(summary['base_disabled_sources'])}`",
        "",
        "| Run | Disabled Delta | "
        + " | ".join(metric.replace("hybrid_", "") for metric in TABLE_METRICS)
        + " |",
        "| --- | --- | " + " | ".join("---" for _ in TABLE_METRICS) + " |",
    ]
    base_disabled = set(summary["base_disabled_sources"])
    for run in summary["runs"]:
        disabled_delta = sorted(set(run["disabled_sources"]) - base_disabled)
        metrics = run["metrics"]
        metric_cells = [str(metrics.get(metric, "")) for metric in TABLE_METRICS]
        lines.append(
            "| "
            + run["name"]
            + " | `"
            + (", ".join(disabled_delta) if disabled_delta else "none")
            + "` | "
            + " | ".join(metric_cells)
            + " |"
        )
    lines.append("")
    return "\n".join(lines)


if __name__ == "__main__":
    main()
