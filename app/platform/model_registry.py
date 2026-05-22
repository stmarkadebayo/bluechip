from __future__ import annotations

import json
import os
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

from app.platform.artifacts import artifact_version, first_existing_path
from app.platform.feature_store import get_feature_store


@dataclass(frozen=True)
class ArtifactRecord:
    name: str
    kind: str
    path: str
    version: str
    exists: bool
    source: str


class LocalModelRegistry:
    """Persistent local registry for model and index artifacts.

    A registry file may override discovered paths, but discovery keeps local and
    CI workflows working without requiring a generated registry artifact.
    """

    def __init__(self, registry_path: str | Path | None = None) -> None:
        self.registry_path = Path(
            registry_path
            or os.getenv("BLUECHIP_MODEL_REGISTRY")
            or "data/processed/model_registry.json"
        )

    def resolve_path(self, name: str) -> Path | None:
        registered = self._registered_path(name)
        if registered and registered.exists():
            return registered
        candidates = _artifact_candidates(name)
        return first_existing_path(*candidates)

    def record(self, name: str) -> ArtifactRecord:
        path = self.resolve_path(name)
        kind = _artifact_kind(name)
        if not path:
            return ArtifactRecord(
                name=name,
                kind=kind,
                path="",
                version="missing",
                exists=False,
                source="missing",
            )
        return ArtifactRecord(
            name=name,
            kind=kind,
            path=str(path),
            version=artifact_version(path),
            exists=True,
            source="registered" if self._registered_path(name) == path else "discovered",
        )

    def records(self) -> dict[str, ArtifactRecord]:
        return {name: self.record(name) for name in ARTIFACT_NAMES}

    def versions(self, *names: str) -> dict[str, str]:
        target_names = names or tuple(ARTIFACT_NAMES)
        return {
            name: record.version
            for name in target_names
            if (record := self.record(name)).exists
        }

    def payload(self) -> dict:
        feature_store = get_feature_store().summary()
        return {
            "type": "bluechip_model_registry",
            "registry_path": str(self.registry_path),
            "feature_store": {
                "root": feature_store.root,
                "version": feature_store.version,
                "available": feature_store.available,
                "counts": feature_store.counts,
            },
            "artifacts": {
                name: {
                    "kind": record.kind,
                    "path": record.path,
                    "version": record.version,
                    "exists": record.exists,
                    "source": record.source,
                }
                for name, record in self.records().items()
            },
        }

    def _registered_path(self, name: str) -> Path | None:
        payload = self._registry_payload()
        artifact = (payload.get("artifacts") or {}).get(name) or {}
        path = str(artifact.get("path") or "").strip()
        return Path(path) if path else None

    @lru_cache(maxsize=1)
    def _registry_payload(self) -> dict:
        if not self.registry_path.exists():
            return {}
        try:
            return json.loads(self.registry_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return {}


ARTIFACT_NAMES = (
    "task_a_model",
    "task_a_rating_stats",
    "task_a_serving_policy",
    "task_b_retrieval_index",
    "task_b_review_term_index",
    "task_b_evidence_graph_index",
    "task_b_neural_index",
)


def get_model_registry() -> LocalModelRegistry:
    return LocalModelRegistry()


def _artifact_candidates(name: str) -> tuple[str | Path | None, ...]:
    feature_root = get_feature_store().root
    if name == "task_a_model":
        return (
            os.getenv("TASK_A_MODEL_PATH"),
            feature_root / "task_a_model_rmse.json",
            feature_root / "task_a_model.json",
            "data/processed/all_categories/task_a_model_rmse.json",
            "data/processed/all_categories/task_a_model.json",
            "data/processed/task_a_model_rmse.json",
            "data/processed/task_a_model.json",
        )
    if name == "task_a_rating_stats":
        return (
            os.getenv("TASK_A_STATS_PATH"),
            feature_root / "task_a_rating_stats.json",
            "data/processed/all_categories/task_a_rating_stats.json",
            "data/processed/task_a_rating_stats.json",
        )
    if name == "task_a_serving_policy":
        return (
            os.getenv("TASK_A_SERVING_POLICY"),
            feature_root / "task_a_serving_policy.json",
            "data/processed/all_categories/task_a_serving_policy.json",
            "data/processed/task_a_serving_policy.json",
        )
    if name == "task_b_retrieval_index":
        return (
            os.getenv("TASK_B_RETRIEVAL_INDEX"),
            feature_root / "collaborative_retrieval.json",
            feature_root / "item_neighbors.json",
            "data/processed/all_categories/collaborative_retrieval.json",
            "data/processed/collaborative_retrieval.json",
            "data/processed/all_categories/item_neighbors.json",
            "data/processed/item_neighbors.json",
        )
    if name == "task_b_review_term_index":
        retrieval_index = get_model_registry().resolve_path("task_b_retrieval_index")
        adjacent = retrieval_index.parent / "review_term_retrieval.json" if retrieval_index else None
        return (
            os.getenv("TASK_B_REVIEW_TERM_INDEX"),
            adjacent,
            feature_root / "review_term_retrieval.json",
            "data/processed/all_categories/review_term_retrieval.json",
            "data/processed/review_term_retrieval.json",
        )
    if name == "task_b_evidence_graph_index":
        retrieval_index = get_model_registry().resolve_path("task_b_retrieval_index")
        adjacent = retrieval_index.parent / "evidence_graph_retrieval.json" if retrieval_index else None
        return (
            os.getenv("TASK_B_EVIDENCE_GRAPH_INDEX"),
            adjacent,
            feature_root / "evidence_graph_retrieval.json",
            "data/processed/all_categories/evidence_graph_retrieval.json",
            "data/processed/evidence_graph_retrieval.json",
        )
    if name == "task_b_neural_index":
        return (
            os.getenv("TASK_B_NEURAL_INDEX"),
            feature_root / "neural_index.faiss",
            "data/processed/all_categories/neural_index.faiss",
            "data/processed/neural_index.faiss",
        )
    return ()


def _artifact_kind(name: str) -> str:
    if "index" in name:
        return "index"
    if "policy" in name:
        return "policy"
    if "stats" in name:
        return "feature_stats"
    return "model"
