from __future__ import annotations

import json
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable

from app.models.schemas import RecommendationItem
from app.services.ranking.features import FEATURE_NAMES
from app.services.retrieval.source_registry import (
    RETRIEVAL_SOURCE_SPECS,
    SOURCE_FAMILY_ORDER,
    retrieval_source_family,
)


ARTIFACT_SCHEMA_VERSION = 1
DEFAULT_FEATURE_NAMES = (
    "component:raw_score",
    *(f"component:{name}" for name in FEATURE_NAMES),
    "component:context_penalty",
    "component:context_category_penalty",
    "component:context_category_boost",
    "component:context_intent_boost",
    "component:context_intent_penalty",
    "component:personalization_weight",
    "component:feedback_acceptance_boost",
    "component:feedback_rejection_penalty",
    "meta:candidate_source_count",
    "meta:retrieval_score_max",
    *(f"retrieval:{spec.name}" for spec in RETRIEVAL_SOURCE_SPECS),
    *(f"source_family:{family}" for family in SOURCE_FAMILY_ORDER),
)


@dataclass(frozen=True)
class TaskBRankerArtifact:
    """Small deterministic linear artifact for optional Task B reranking."""

    feature_names: tuple[str, ...] = DEFAULT_FEATURE_NAMES
    weights: dict[str, float] = field(default_factory=dict)
    intercept: float = 0.0
    model_name: str = "task_b_linear_ranker"
    schema_version: int = ARTIFACT_SCHEMA_VERSION
    training: dict = field(default_factory=dict)

    @classmethod
    def from_json(cls, path: str | Path) -> "TaskBRankerArtifact":
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
        if int(payload.get("schema_version", 0)) != ARTIFACT_SCHEMA_VERSION:
            raise ValueError(
                "Unsupported Task B ranker artifact schema "
                f"{payload.get('schema_version')!r}"
            )
        feature_names = tuple(str(name) for name in payload.get("feature_names", []))
        weights = {
            str(name): float(value)
            for name, value in (payload.get("weights") or {}).items()
        }
        return cls(
            feature_names=feature_names or DEFAULT_FEATURE_NAMES,
            weights=weights,
            intercept=float(payload.get("intercept", 0.0)),
            model_name=str(payload.get("model_name") or "task_b_linear_ranker"),
            schema_version=ARTIFACT_SCHEMA_VERSION,
            training=dict(payload.get("training") or {}),
        )

    def to_dict(self) -> dict:
        return {
            "schema_version": self.schema_version,
            "model_name": self.model_name,
            "feature_names": list(self.feature_names),
            "weights": {
                name: round(float(self.weights.get(name, 0.0)), 10)
                for name in self.feature_names
            },
            "intercept": round(float(self.intercept), 10),
            "training": self.training,
        }

    def write_json(self, path: str | Path) -> None:
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(
            json.dumps(self.to_dict(), ensure_ascii=True, indent=2) + "\n",
            encoding="utf-8",
        )


class TaskBLinearRanker:
    def __init__(self, artifact: TaskBRankerArtifact) -> None:
        self.artifact = artifact

    @classmethod
    def from_json(cls, path: str | Path) -> "TaskBLinearRanker":
        return cls(TaskBRankerArtifact.from_json(path))

    def score(self, item: RecommendationItem) -> float:
        vector = feature_vector(item, self.artifact.feature_names)
        return self.artifact.intercept + sum(
            self.artifact.weights.get(name, 0.0) * value
            for name, value in vector.items()
        )

    def probability(self, item: RecommendationItem) -> float:
        return sigmoid(self.score(item))

    def rerank(
        self,
        items: Iterable[RecommendationItem],
        limit: int | None = None,
    ) -> list[RecommendationItem]:
        scored: list[tuple[float, float, int, RecommendationItem]] = []
        for index, item in enumerate(items):
            logit = self.score(item)
            original_score = float(item.score_components.get("raw_score", item.score))
            copy = item.model_copy(deep=True)
            copy.score_components["learned_ranker_logit"] = round(logit, 6)
            copy.score_components["learned_ranker_score"] = round(sigmoid(logit), 6)
            scored.append((logit, original_score, -index, copy))
        scored.sort(key=lambda row: (row[0], row[1], row[2]), reverse=True)
        selected = [item for _, _, _, item in scored[:limit]]
        for rank, item in enumerate(selected, start=1):
            item.rank = rank
        return selected


def feature_vector(
    item: RecommendationItem,
    feature_names: Iterable[str] = DEFAULT_FEATURE_NAMES,
) -> dict[str, float]:
    retrieval_scores = item.retrieval_scores or {}
    score_components = item.score_components or {}
    families: dict[str, float] = {}
    for source, raw_value in retrieval_scores.items():
        family = retrieval_source_family(source)
        families[family] = max(families.get(family, 0.0), _bounded(raw_value))

    vector = {}
    for name in feature_names:
        if name.startswith("component:"):
            vector[name] = _bounded(score_components.get(name.removeprefix("component:"), 0.0))
        elif name.startswith("retrieval:"):
            vector[name] = _bounded(retrieval_scores.get(name.removeprefix("retrieval:"), 0.0))
        elif name.startswith("source_family:"):
            vector[name] = families.get(name.removeprefix("source_family:"), 0.0)
        elif name == "meta:candidate_source_count":
            vector[name] = min(len(item.candidate_sources) / 8.0, 1.0)
        elif name == "meta:retrieval_score_max":
            vector[name] = max((_bounded(value) for value in retrieval_scores.values()), default=0.0)
        else:
            vector[name] = _bounded(score_components.get(name, retrieval_scores.get(name, 0.0)))
    return vector


def sigmoid(value: float) -> float:
    if value >= 0:
        z = math.exp(-value)
        return 1.0 / (1.0 + z)
    z = math.exp(value)
    return z / (1.0 + z)


def _bounded(value: object) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return 0.0
    if math.isnan(number) or math.isinf(number):
        return 0.0
    return min(max(number, -1.0), 1.0)
