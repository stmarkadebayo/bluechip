from __future__ import annotations

import json
import time
from functools import lru_cache

from app.models.schemas import (
    AgentTraceStep,
    CandidateDiagnostics,
    RecommendationRequest,
    RecommendationResponse,
)
from app.platform.feature_store import get_feature_store
from app.platform.model_registry import get_model_registry
from app.services.generation.generator import generate_recommendation_reason_result
from app.services.generation.providers import generation_provider_name
from app.services.profiling.user_profile import build_user_profile
from app.services.ranking.recommendation import rank_candidates
from app.services.retrieval.candidates import generate_candidate_pool
from app.services.validation.evidence_critic import recommendation_evidence_issues
from app.stores.trace_store import trace_store


class RecommendationAgent:
    """Serving orchestrator for Task B recommendations."""

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
        explanation_issues = []
        generation_fallbacks = []
        generation_provider = generation_provider_name()
        for item in ranked:
            generated_reason = generate_recommendation_reason_result(
                user_profile=user_profile,
                recommendation=item,
                context=request.context,
            )
            reason = generated_reason.text
            generation_provider = generated_reason.provider or generation_provider
            if generated_reason.used_fallback:
                generation_fallbacks.append(generated_reason.error or "generation fallback")
            explanation_issues.extend(
                recommendation_evidence_issues(
                    reason=reason,
                    user_profile=user_profile,
                    recommendation=item,
                    context=request.context,
                )
            )
            recommendations.append(
                item.model_copy(
                    update={"reason": reason}
                )
            )
        trace.append(
            AgentTraceStep(
                step="explain_recommendations",
                status="warning" if explanation_issues else "ok",
                detail=(
                    "; ".join(sorted(set(explanation_issues))[:3])
                    if explanation_issues
                    else "grounded explanations generated"
                ),
            )
        )
        generated_text = " ".join(item.reason for item in recommendations)
        trace_record = trace_store.append(
            endpoint="recommend",
            latency_ms=(time.perf_counter() - started) * 1000,
            steps=trace,
            generation_provider=generation_provider,
            estimated_generation_tokens=max(len(generated_text) // 4, 1),
            model_versions={"task_b_ranker": _task_b_ranker_version()},
            index_versions=_task_b_index_versions(),
            retrieval_source_counts=candidate_pool.source_counts(),
            validation_status="warning" if explanation_issues else "ok",
            fallback_reason="; ".join(sorted(set(generation_fallbacks))) or None,
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
    path = get_model_registry().resolve_path("task_b_retrieval_index")
    if not path:
        return None
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    if "item_neighbors" in payload and not payload.get("items"):
        return _attach_optional_indexes(payload)
    if "items" in payload:
        return _attach_optional_indexes(
            {"type": "legacy_item_neighbors", "item_neighbors": payload["items"]},
        )
    return None


def _attach_optional_indexes(payload: dict) -> dict:
    payload = _attach_review_term_index(payload)
    return _attach_evidence_graph_index(payload)


def _attach_review_term_index(payload: dict) -> dict:
    review_term_path = get_model_registry().resolve_path("task_b_review_term_index")
    if not review_term_path or not review_term_path.exists():
        return payload
    try:
        review_term_payload = json.loads(review_term_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return payload
    if review_term_payload.get("term_items"):
        payload = dict(payload)
        payload["review_term_retrieval"] = review_term_payload
    return payload


def _attach_evidence_graph_index(payload: dict) -> dict:
    evidence_graph_path = get_model_registry().resolve_path("task_b_evidence_graph_index")
    if not evidence_graph_path or not evidence_graph_path.exists():
        return payload
    try:
        evidence_graph_payload = json.loads(evidence_graph_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return payload
    if evidence_graph_payload.get("type") == "evidence_graph":
        payload = dict(payload)
        payload["evidence_graph_retrieval"] = evidence_graph_payload
    return payload


def _task_b_ranker_version() -> str:
    return "default_hybrid"


def _task_b_index_versions() -> dict[str, str]:
    versions = get_model_registry().versions(
        "task_b_retrieval_index",
        "task_b_review_term_index",
        "task_b_evidence_graph_index",
    )
    versions["feature_store"] = get_feature_store().version()
    return versions or {"candidate_items": "request_payload"}
