from __future__ import annotations

import json
import os
import time
from functools import lru_cache

from app.models.schemas import (
    AgentTraceStep,
    CandidateDiagnostics,
    RecommendationRequest,
    RecommendationResponse,
    UserProfile,
)
from app.platform.feature_store import get_feature_store
from app.platform.model_registry import get_model_registry
from app.platform.artifacts import read_json_artifact
from app.services.agentic.recommender_agent import ColdStartInference, RecommenderReasoner
from app.services.generation.generator import generate_recommendation_reason_result
from app.services.generation.providers import generation_provider_name
from app.services.nigerian.context import NigerianContextEngine
from app.services.profiling.user_profile import build_user_profile
from app.services.ranking.learned_task_b import TaskBLinearRanker
from app.services.ranking.recommendation import adaptive_recommendation_policy, rank_candidates
from app.services.retrieval.candidates import generate_candidate_pool
from app.services.retrieval.embeddings import embedding_text, hashed_embedding
from app.services.retrieval.embeddings import neural_available
from app.services.retrieval.item_similarity import SQLiteItemNeighborIndex
from app.services.retrieval.source_registry import adaptive_disabled_retrieval_sources
from app.services.retrieval.vector_store import LocalVectorRetriever
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
        cold_start = None
        if not request.user_history:
            strategy = "cold_start"
            cold_start = self._reasoner.handle_cold_start(request.user_persona)
            user_profile = _apply_cold_start_to_profile(
                user_profile=user_profile,
                persona=request.user_persona,
                cold_start=cold_start,
            )
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

        ranking_policy = adaptive_recommendation_policy(
            user_profile=user_profile,
            context=request.context,
            strategy=strategy,
            preference_analysis=preference_analysis,
        )
        trace.append(
            AgentTraceStep(
                step="ranking_policy",
                status="ok",
                detail=ranking_policy.name,
            )
        )

        disabled_sources = _disabled_retrieval_sources(request.context, user_profile, strategy)
        collaborative_index = _load_collaborative_index()
        neural_index = _load_neural_index()
        if neural_index is not None:
            neural_index = neural_index.bind_items(request.candidate_items)
        vector_retriever = (
            LocalVectorRetriever(request.candidate_items)
            if "vector_profile" not in disabled_sources
            else None
        )
        candidate_pool = generate_candidate_pool(
            user_profile=user_profile,
            history=request.user_history,
            items=request.candidate_items,
            context=request.context,
            collaborative_index=collaborative_index,
            vector_retriever=vector_retriever,
            neural_retriever=neural_index,
            disabled_sources=disabled_sources,
            excluded_item_ids=set(request.rejected_item_ids),
            limit=min(len(request.candidate_items), 100),
        )
        trace.append(
            AgentTraceStep(
                step="retrieve_candidates",
                status="ok",
                detail=(
                    f"{len(candidate_pool.items)} candidates from {len(request.candidate_items)} "
                    f"input items via {candidate_pool.source_counts()}"
                    + (
                        f"; disabled {sorted(disabled_sources)}"
                        if disabled_sources
                        else ""
                    )
                ),
            )
        )

        learned_ranker = _load_task_b_ranker()
        deterministic_rank_limit = (
            len(candidate_pool.items) if learned_ranker is not None else request.limit
        )
        ranked = rank_candidates(
            user_profile=user_profile,
            context=request.context,
            candidate_items=candidate_pool.items,
            limit=deterministic_rank_limit,
            policy=ranking_policy,
            candidate_sources=candidate_pool.sources,
            candidate_source_scores=candidate_pool.source_scores,
            accepted_item_ids=request.accepted_item_ids,
            rejected_item_ids=request.rejected_item_ids,
        )
        trace.append(
            AgentTraceStep(
                step="rank_candidates",
                status="ok",
                detail=f"ranked top {len(ranked)}",
            )
        )

        if learned_ranker is not None:
            ranked = learned_ranker.rerank(ranked, limit=request.limit)
            trace.append(
                AgentTraceStep(
                    step="learned_ranker",
                    status="ok",
                    detail=f"linear artifact re-ranked top {len(ranked)}",
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
                ranked_by_id = {item.item_id: item for item in ranked}
                reranked_ids = []
                for item in sorted(reranked, key=lambda row: row.rank):
                    item_id = item.item.item_id
                    if item_id in ranked_by_id and item_id not in reranked_ids:
                        reranked_ids.append(item_id)
                reranked_ranked = [
                    ranked_by_id[item_id]
                    for item_id in reranked_ids
                ]
                plus_others = []
                for item in ranked:
                    if item.item_id not in reranked_ids:
                        plus_others.append(item)
                ranked = reranked_ranked + plus_others
                for index, item in enumerate(ranked, start=1):
                    item.rank = index
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
                history=request.user_history,
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
                disabled_sources=sorted(disabled_sources),
                used_collaborative=bool(collaborative_index),
                ranking_policy=ranking_policy.name,
                semantic_retrieval_enabled=(
                    "vector_profile" not in disabled_sources
                    or "neural_vector" not in disabled_sources
                ),
            ),
        )


def _apply_cold_start_to_profile(
    user_profile: UserProfile,
    persona: str,
    cold_start: ColdStartInference,
) -> UserProfile:
    preferred_terms = _merge_values(user_profile.preferred_terms, cold_start.key_terms, limit=12)
    preferred_categories = _merge_values(
        user_profile.preferred_categories,
        cold_start.preferred_categories,
        limit=8,
    )
    category_affinity = dict(user_profile.category_affinity)
    for category in preferred_categories:
        category_affinity[category] = max(category_affinity.get(category, 0.0), 0.35)
    positive_aspects = list(user_profile.positive_aspects)
    if cold_start.price_sensitivity.lower() == "high":
        positive_aspects = _merge_values(positive_aspects, ["affordable", "value"], limit=10)
    if cold_start.quality_expectation.lower() == "high":
        positive_aspects = _merge_values(positive_aspects, ["quality", "reliable"], limit=10)
    confidence = round(max(user_profile.confidence, min(cold_start.confidence, 0.72)), 2)
    voice_style = (
        cold_start.likely_voice_style
        if cold_start.likely_voice_style and user_profile.voice_style == "balanced"
        else user_profile.voice_style
    )
    signals = list(user_profile.signals)
    signals.append(
        "cold-start profile inference: "
        + ("LLM" if cold_start.llm_augmented else "deterministic")
    )
    if cold_start.key_terms:
        signals.append("cold-start terms: " + ", ".join(cold_start.key_terms[:5]))
    embedding = hashed_embedding(
        embedding_text(
            persona,
            preferred_terms,
            preferred_categories,
            positive_aspects,
            user_profile.recent_terms,
        )
    )
    return user_profile.model_copy(
        update={
            "preferred_terms": preferred_terms,
            "preferred_categories": preferred_categories,
            "category_affinity": category_affinity,
            "positive_aspects": positive_aspects,
            "confidence": confidence,
            "voice_style": voice_style,
            "signals": signals[: len(user_profile.signals) + 3],
            "embedding": embedding,
        }
    )


def _merge_values(existing: list[str], additions: list[str], limit: int) -> list[str]:
    output = list(existing)
    normalized = {value.lower() for value in output}
    for value in additions:
        cleaned = str(value or "").strip()
        if not cleaned or cleaned.lower() in normalized:
            continue
        output.append(cleaned)
        normalized.add(cleaned.lower())
        if len(output) >= limit:
            break
    return output


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
    payload = _attach_implicit_item_index(payload)
    return _attach_evidence_graph_index(payload)


def _attach_review_term_index(payload: dict) -> dict:
    review_term_path = get_model_registry().resolve_path("task_b_review_term_index")
    if not review_term_path or not review_term_path.exists():
        return payload
    try:
        review_term_payload = read_json_artifact(review_term_path)
    except (OSError, json.JSONDecodeError):
        return payload
    if review_term_payload.get("term_items"):
        payload = dict(payload)
        payload["review_term_retrieval"] = review_term_payload
    return payload


def _attach_implicit_item_index(payload: dict) -> dict:
    implicit_path = get_model_registry().resolve_path("task_b_implicit_item_index")
    if not implicit_path or not implicit_path.exists():
        return payload
    payload = dict(payload)
    if implicit_path.suffix == ".sqlite":
        payload["implicit_item_neighbors"] = SQLiteItemNeighborIndex(implicit_path)
        return payload
    try:
        implicit_payload = read_json_artifact(implicit_path)
    except (OSError, json.JSONDecodeError):
        return payload
    neighbors = implicit_payload.get("neighbors") if isinstance(implicit_payload, dict) else None
    if neighbors:
        payload["implicit_item_neighbors"] = neighbors
    return payload


def _attach_evidence_graph_index(payload: dict) -> dict:
    evidence_graph_path = get_model_registry().resolve_path("task_b_evidence_graph_index")
    if not evidence_graph_path or not evidence_graph_path.exists():
        return payload
    try:
        evidence_graph_payload = read_json_artifact(evidence_graph_path)
    except (OSError, json.JSONDecodeError):
        return payload
    if evidence_graph_payload.get("type") == "evidence_graph":
        payload = dict(payload)
        payload["evidence_graph_retrieval"] = evidence_graph_payload
    return payload


def _task_b_ranker_version() -> str:
    versions = get_model_registry().versions("task_b_ranker_artifact")
    artifact_version = versions.get("task_b_ranker_artifact")
    if artifact_version:
        return f"adaptive_hybrid_policy_v2+task_b_linear_ranker:{artifact_version}"
    return "adaptive_hybrid_policy_v2"


def _disabled_retrieval_sources(
    context: str,
    user_profile: UserProfile,
    strategy: str,
) -> set[str]:
    configured = os.getenv("BLUECHIP_DISABLED_RETRIEVAL_SOURCES")
    if configured is not None:
        return {
            source.strip()
            for source in configured.split(",")
            if source.strip()
        }
    return adaptive_disabled_retrieval_sources(
        context=context,
        user_profile=user_profile,
        strategy=strategy,
    )


@lru_cache(maxsize=1)
def _load_neural_index() -> FAISSVectorStore | None:
    path = get_model_registry().resolve_path("task_b_neural_index")
    if not path or not path.exists() or not neural_available():
        return None
    try:
        return FAISSVectorStore.deserialize(str(path), [])
    except (OSError, json.JSONDecodeError, RuntimeError):
        return None


@lru_cache(maxsize=1)
def _load_task_b_ranker() -> TaskBLinearRanker | None:
    path = get_model_registry().resolve_path("task_b_ranker_artifact")
    if not path or not path.exists():
        return None
    try:
        return TaskBLinearRanker.from_json(path)
    except (OSError, json.JSONDecodeError, RuntimeError, ValueError):
        return None


def _task_b_index_versions() -> dict[str, str]:
    versions = get_model_registry().versions(
        "task_b_retrieval_index",
        "task_b_implicit_item_index",
        "task_b_review_term_index",
        "task_b_evidence_graph_index",
    )
    versions["feature_store"] = get_feature_store().version()
    return versions or {"candidate_items": "request_payload"}
