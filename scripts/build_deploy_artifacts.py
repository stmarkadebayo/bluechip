from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.services.ranking.rating_features import build_rating_stats, save_rating_stats  # noqa: E402
from scripts.data_utils import read_jsonl, write_json  # noqa: E402


DEPLOY_FILES = (
    "items.jsonl",
    "train.jsonl",
    "test_task_a.jsonl",
    "test_task_b.jsonl",
    "split_stats.json",
    "collaborative_retrieval.json",
    "review_term_retrieval.json",
    "evidence_graph_retrieval.json",
    "item_neighbors.json",
    "neural_index.faiss",
    "neural_index_ids.json",
    "feature_store.sqlite",
    "task_a_model.json",
)


def main() -> None:
    parser = argparse.ArgumentParser(description="Build the small artifact bundle used by Docker deploys.")
    parser.add_argument("--source-dir", default="data/processed")
    parser.add_argument("--output-dir", default="data/deploy/processed")
    parser.add_argument("--runtime-dir", default="")
    args = parser.parse_args()

    source_dir = Path(args.source_dir)
    output_dir = Path(args.output_dir)
    runtime_dir = Path(args.runtime_dir) if args.runtime_dir else output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    copied = _copy_known_files(source_dir, output_dir)
    train_rows = read_jsonl(output_dir / "train.jsonl")
    if train_rows:
        save_rating_stats(build_rating_stats(train_rows), output_dir / "task_a_rating_stats.json")
        copied.append("task_a_rating_stats.json")

    write_json(
        output_dir / "task_a_serving_policy.json",
        {
            "type": "task_a_serving_policy",
            "source_report": "deploy_lean_artifacts",
            "metric": "demo_stability",
            "serving_head": "profile_heuristic",
            "metric_name": "deploy_safe_heuristic",
            "metric_value": 0.0,
        },
    )
    copied.append("task_a_serving_policy.json")

    write_json(output_dir / "model_registry.json", _registry_payload(output_dir, runtime_dir))
    copied.append("model_registry.json")

    print(f"Deploy artifacts ready in {output_dir}")
    for name in sorted(set(copied)):
        path = output_dir / name
        print(f"- {name} ({path.stat().st_size} bytes)")


def _copy_known_files(source_dir: Path, output_dir: Path) -> list[str]:
    copied = []
    missing = []
    for name in DEPLOY_FILES:
        source = source_dir / name
        if not source.exists():
            missing.append(name)
            continue
        target = output_dir / name
        shutil.copy2(source, target)
        copied.append(name)
    if missing:
        print("Skipped missing optional deploy files: " + ", ".join(missing))
    return copied


def _registry_payload(output_dir: Path, runtime_dir: Path) -> dict:
    return {
        "type": "bluechip_model_registry",
        "registry_path": str(runtime_dir / "model_registry.json"),
        "feature_store": {
            "root": str(runtime_dir),
            "available": True,
            "counts": {},
        },
        "artifacts": {
            "task_a_model": _artifact(output_dir, runtime_dir, "task_a_model.json", "model"),
            "task_a_rating_stats": _artifact(
                output_dir,
                runtime_dir,
                "task_a_rating_stats.json",
                "feature_stats",
            ),
            "task_a_serving_policy": _artifact(
                output_dir,
                runtime_dir,
                "task_a_serving_policy.json",
                "policy",
            ),
            "task_b_retrieval_index": _artifact(
                output_dir,
                runtime_dir,
                "collaborative_retrieval.json",
                "index",
            ),
            "task_b_review_term_index": _artifact(
                output_dir,
                runtime_dir,
                "review_term_retrieval.json",
                "index",
            ),
            "task_b_implicit_item_index": _artifact(
                output_dir,
                runtime_dir,
                "item_neighbors.json",
                "index",
            ),
            "task_b_evidence_graph_index": _artifact(
                output_dir,
                runtime_dir,
                "evidence_graph_retrieval.json",
                "index",
            ),
            "task_b_neural_index": _artifact(output_dir, runtime_dir, "neural_index.faiss", "index"),
        },
    }


def _artifact(output_dir: Path, runtime_dir: Path, filename: str, kind: str) -> dict:
    source_path = output_dir / filename
    runtime_path = runtime_dir / filename
    exists = source_path.exists()
    version = (
        f"{runtime_path}@{int(source_path.stat().st_mtime)}:{source_path.stat().st_size}"
        if exists
        else "missing"
    )
    return {
        "kind": kind,
        "path": str(runtime_path),
        "version": version,
        "exists": exists,
        "source": "deploy_bundle" if exists else "missing",
    }


if __name__ == "__main__":
    main()
