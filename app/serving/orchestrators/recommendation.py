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
from app.services.agentic.recommender_agent import RecommenderReasoner
from app.services.generation.generator import generate_recommendation_reason_result
from app.services.generation.providers import generation_provider_name
from app.services.nigerian.context import NigerianContextEngine
from app.services.profiling.user_profile import build_user_profile
from app.services.ranking.recommendation import rank_candidates
from app.services.retrieval.candidates import generate_candidate_pool
from app.services.retrieval.embeddings import neural_available
from app.services.retrieval.vector_store import FAISSVectorStore
from app.services.validation.evidence_critic import recommendation_evidence_issues
from app.stores.trace_store import trace_store


class RecommendationAgent:
    """Serving orchestrator for Task B recommendations with LLM agentic workflow.

    Integrates the LLM-driven RecommenderReasoner for preference analysis,
    candidate re-ranking, cold-start inference, cross-domain transfer, and
    authentic explanation generation. Nigerian context enriches all outputs.
    Falls back to deterministic pipeline when LLM is unavailable.
    """

    def __init__(self) -> None:
        self._reasoner = RecommenderReasoner()
        self._nigerian_engine = NigerianContextEngine()

    def run(self, request: RecommendationRequest) -> RecommendationResponse:
        started = time.perf_counter()
        trace: list[AgentTraceStep] = []

        user_profile = build_user_profile(
            persona=request.user_persona,
            history=request.user_history,
            locale=request.locale,
            enhance_with_llm=request.enhance_with_llm,
        )

        strategy = "history_aware"
        if not request.user_history:
            strategy = "cold_start"
            cold_start = self._reasoner.handle_cold_start(request.user_persona)
            trace.append(
                AgentTraceStep(
                    step="cold_start_inference",
                    status="ok",
                    detail=(
                        "LLM inferred preferences"
                        if cold_start.llm_augmented
                        else "deterministic cold-start fallback"
                    ),
                )
            )
        elif request.context:
            strategy = "contextual_history_aware"

        trace.append(
            AgentTraceStep(
                step="profile_user",
                status=(
                    "ok"
                    if user_profile.profile_enhancement is None
                    or user_profile.profile_enhancement.llm_augmented
                    else "fallback"
                ),
                detail=(
                    f"{strategy}, confidence {user_profile.confidence:.2f}"
                    + (
                        f", profile enhancer {user_profile.profile_enhancement.provider}"
                        if user_profile.profile_enhancement
                        else ""
                    )
                ),
            )
        )

        nigerian_result = NigerianContextEngine.score_nigerian_relevance(user_profile)
        trace.append(
            AgentTraceStep(
                step="nigerian_context",
                status="ok",
                detail=f"relevance score {nigerian_result:.2f}",
            )
        )

        if not request.user_history:
            preference_analysis = None
        else:
            try:
                preference_analysis = self._reasoner.reason_about_preferences(
                    user_profile, request.context
                )
                trace.append(
                    AgentTraceStep(
                        step="agentic_preference_reasoning",
                        status="ok" if preference_analysis.llm_augmented else "fallback",
                        detail=(
                            f"LLM analysed {len(preference_analysis.core_preferences)} core preferences"
                            if preference_analysis.llm_augmented
                            else "deterministic preference fallback"
                        ),
                    )
                )
            except Exception:
                preference_analysis = None

        collaborative_index = _load_collaborative_index()
        neural_index = _load_neural_index()
        if neural_index is not None:
            neural_index = neural_index.bind_items(request.candidate_items)
        candidate_pool = generate_candidate_pool(
            user_profile=user_profile,
            history=request.user_history,
            items=request.candidate_items,
            context=request.context,
            collaborative_index=collaborative_index,
            neural_retriever=neural_index,
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

        try:
            reranked = self._reasoner.rerank_candidates(
                user_profile=user_profile,
                context=request.context,
                candidates=ranked,
                limit=request.limit,
            )
            if reranked and any(r.llm_augmented for r in reranked):
                reranked_ids = {r.item.item_id for r in reranked}
                reranked_ranked = [
                    item for item in ranked
                    if item.item_id in reranked_ids
                ]
                plus_others = []
                for item in ranked:
                    if item.item_id not in reranked_ids:
                        plus_others.append(item)
                ranked = reranked_ranked + plus_others
                trace.append(
                    AgentTraceStep(
                        step="agentic_rerank",
                        status="ok",
                        detail=f"LLM re-ranked {len(reranked_ranked)} candidates",
                    )
                )
        except Exception:
            pass

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


@lru_cache(maxsize=1)
def _load_neural_index() -> FAISSVectorStore | None:
    path = get_model_registry().resolve_path("task_b_neural_index")
    if not path or not path.exists() or not neural_available():
        return None
    try:
        return FAISSVectorStore.deserialize(str(path), [])
    except (OSError, json.JSONDecodeError, RuntimeError):
        return None


def _task_b_index_versions() -> dict[str, str]:
    versions = get_model_registry().versions(
        "task_b_retrieval_index",
        "task_b_review_term_index",
        "task_b_evidence_graph_index",
    )
    versions["feature_store"] = get_feature_store().version()
    return versions or {"candidate_items": "request_payload"}
