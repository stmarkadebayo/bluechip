from __future__ import annotations

import json
import os
import time
from functools import lru_cache
from pathlib import Path

from app.models.schemas import (
    AgentTraceStep,
    CandidateDiagnostics,
    RecommendationRequest,
    RecommendationResponse,
)
from app.services.generation.generator import generate_recommendation_reason
from app.services.generation.providers import generation_provider_name
from app.services.profiling.user_profile import build_user_profile
from app.services.ranking.recommendation import load_recommendation_weights, rank_candidates
from app.services.retrieval.candidates import generate_candidate_pool
from app.stores.trace_store import trace_store


class RecommendationAgent:
    """Agentic workflow for Task B."""

    def run(self, request: RecommendationRequest) -> RecommendationResponse:
        started = time.perf_counter()
        trace: list[AgentTraceStep] = []
        user_profile = build_user_profile(
            persona=request.user_persona,
            history=request.user_history,
            locale=request.locale,
        )

        strategy = "history_aware"
        if not request.user_history:
            strategy = "cold_start"
        elif request.context:
            strategy = "contextual_history_aware"

        trace.append(
            AgentTraceStep(
                step="profile_user",
                status="ok",
                detail=f"{strategy}, confidence {user_profile.confidence:.2f}",
            )
        )

        collaborative_index = _load_collaborative_index()
        candidate_pool = generate_candidate_pool(
            user_profile=user_profile,
            history=request.user_history,
            items=request.candidate_items,
            context=request.context,
            collaborative_index=collaborative_index,
            limit=min(len(request.candidate_items), 100),
        )
        trace.append(
            AgentTraceStep(
                step="retrieve_candidates",
                status="ok",
                detail=(
                    f"{len(candidate_pool.items)} candidates from {len(request.candidate_items)} "
                    f"input items via {candidate_pool.source_counts()}"
                ),
            )
        )

        ranked = rank_candidates(
            user_profile=user_profile,
            context=request.context,
            candidate_items=candidate_pool.items,
            limit=request.limit,
            weights=_load_ranker_weights(),
            candidate_sources=candidate_pool.sources,
            candidate_source_scores=candidate_pool.source_scores,
        )
        trace.append(
            AgentTraceStep(
                step="rank_candidates",
                status="ok",
                detail=f"ranked top {len(ranked)}",
            )
        )

        recommendations = []
        for item in ranked:
            recommendations.append(
                item.model_copy(
                    update={
                        "reason": generate_recommendation_reason(
                            user_profile=user_profile,
                            recommendation=item,
                            context=request.context,
                        ),
                    }
                )
            )
        trace.append(
            AgentTraceStep(
                step="explain_recommendations",
                status="ok",
                detail="grounded explanations generated",
            )
        )
        generated_text = " ".join(item.reason for item in recommendations)
        trace_record = trace_store.append(
            endpoint="recommend",
            latency_ms=(time.perf_counter() - started) * 1000,
            steps=trace,
            generation_provider=generation_provider_name(),
            estimated_generation_tokens=max(len(generated_text) // 4, 1),
        )

        return RecommendationResponse(
            trace_id=trace_record.trace_id,
            recommendations=recommendations,
            agent_trace=trace,
            candidate_diagnostics=CandidateDiagnostics(
                strategy=strategy,
                input_count=len(request.candidate_items),
                candidate_count=len(candidate_pool.items),
                source_counts=candidate_pool.source_counts(),
                used_collaborative=bool(collaborative_index),
            ),
        )


@lru_cache(maxsize=1)
def _load_collaborative_index() -> dict | None:
    configured = os.getenv("TASK_B_RETRIEVAL_INDEX")
    candidates = [
        Path(configured) if configured else None,
        Path("data/processed/all_categories/collaborative_retrieval.json"),
        Path("data/processed/collaborative_retrieval.json"),
        Path("data/processed/all_categories/item_neighbors.json"),
        Path("data/processed/item_neighbors.json"),
    ]
    for path in candidates:
        if not path or not path.exists():
            continue
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        if payload.get("item_neighbors") and not payload.get("items"):
            return _attach_review_term_index(payload, path)
        if payload.get("items"):
            return _attach_review_term_index(
                {"type": "legacy_item_neighbors", "item_neighbors": payload["items"]},
                path,
            )
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


@lru_cache(maxsize=1)
def _load_ranker_weights() -> dict[str, float] | None:
    return load_recommendation_weights(os.getenv("TASK_B_RANKER_WEIGHTS"))
