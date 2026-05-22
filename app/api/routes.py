from __future__ import annotations

from fastapi import APIRouter
from starlette.concurrency import run_in_threadpool

from app.models.schemas import (
    ColdStartInferenceResponse,
    ConversationSummary,
    ConversationTurnRequest,
    ConversationTurnResponse,
    CrossDomainTransferRequest,
    CrossDomainTransferResponse,
    HealthResponse,
    NigerianContextResponse,
    ProfileUserRequest,
    RecommendationRequest,
    RecommendationResponse,
    RuntimeMetrics,
    SimulateReviewRequest,
    SimulateReviewResponse,
    TraceRecord,
    UserProfileResponse,
)
from app.platform.feature_store import get_feature_store
from app.platform.model_registry import get_model_registry
from app.serving.orchestrators.recommendation import RecommendationAgent
from app.serving.orchestrators.review_simulation import ReviewSimulationAgent
from app.services.agentic.recommender_agent import RecommenderReasoner
from app.services.conversation.state import get_conversation_manager
from app.services.nigerian.context import NigerianContextEngine
from app.services.profiling.user_profile import build_user_profile
from app.stores.trace_store import trace_store

router = APIRouter()


@router.get("/health", response_model=HealthResponse)
async def health() -> HealthResponse:
    return HealthResponse(status="ok")


@router.post("/profile-user", response_model=UserProfileResponse)
async def profile_user(request: ProfileUserRequest) -> UserProfileResponse:
    return await run_in_threadpool(
        lambda: UserProfileResponse(
            profile=build_user_profile(
                persona=request.user_persona,
                history=request.user_history,
                locale=request.locale,
                enhance_with_llm=request.enhance_with_llm,
            )
        )
    )


@router.post("/simulate-review", response_model=SimulateReviewResponse)
async def simulate_review(request: SimulateReviewRequest) -> SimulateReviewResponse:
    return await run_in_threadpool(lambda: ReviewSimulationAgent().run(request))


@router.post("/recommend", response_model=RecommendationResponse)
async def recommend(request: RecommendationRequest) -> RecommendationResponse:
    return await run_in_threadpool(lambda: RecommendationAgent().run(request))


@router.post("/conversation/turn", response_model=ConversationTurnResponse)
async def conversation_turn(request: ConversationTurnRequest) -> ConversationTurnResponse:
    return await run_in_threadpool(lambda: _conversation_turn_sync(request))


def _conversation_turn_sync(request: ConversationTurnRequest) -> ConversationTurnResponse:
    manager = get_conversation_manager()
    user_profile = build_user_profile(
        persona=request.user_persona,
        history=request.user_history,
        locale=request.locale,
        enhance_with_llm=request.enhance_with_llm,
    )

    conversation = manager.get_or_create(
        conversation_id=request.conversation_id or "",
        user_profile=user_profile,
        context=request.context,
    )
    conversation.user_profile = user_profile

    if request.context:
        conversation.refine_context(request.context)

    response = RecommendationAgent().run(
        RecommendationRequest(
            user_persona=request.user_persona,
            user_history=request.user_history,
            context=conversation.context,
            candidate_items=request.candidate_items,
            locale=request.locale,
            limit=request.limit,
            enhance_with_llm=request.enhance_with_llm,
            accepted_item_ids=conversation.accepted_recommendations,
            rejected_item_ids=conversation.rejected_recommendations,
        )
    )

    agent_response = (
        f"Here are {len(response.recommendations)} recommendations based on "
        f"your preferences"
        + (f" for: {conversation.context}" if conversation.context else "")
        + "."
    )

    conversation.add_turn(
        user_message=request.user_message or request.context,
        agent_response=agent_response,
        recommendations=response.recommendations,
    )

    return ConversationTurnResponse(
        conversation_id=conversation.conversation_id,
        turn_index=conversation.turn_count,
        recommendations=response.recommendations,
        agent_response=agent_response,
        agent_trace=response.agent_trace,
        candidate_diagnostics=response.candidate_diagnostics,
    )


@router.post("/conversation/{conversation_id}/feedback")
async def conversation_feedback(
    conversation_id: str,
    item_id: str = "",
    accepted: bool = True,
) -> dict:
    return await run_in_threadpool(
        lambda: _conversation_feedback_sync(conversation_id, item_id, accepted)
    )


def _conversation_feedback_sync(
    conversation_id: str,
    item_id: str = "",
    accepted: bool = True,
) -> dict:
    manager = get_conversation_manager()
    conversation = manager.get(conversation_id)
    if conversation is None:
        return {"status": "error", "detail": "conversation not found"}
    conversation.record_feedback(item_id=item_id, accepted=accepted)
    return {
        "status": "ok",
        "conversation_id": conversation_id,
        "accepted_count": len(conversation.accepted_recommendations),
        "rejected_count": len(conversation.rejected_recommendations),
    }


@router.get("/conversation/{conversation_id}", response_model=ConversationSummary)
async def get_conversation(conversation_id: str) -> ConversationSummary:
    return await run_in_threadpool(lambda: _get_conversation_sync(conversation_id))


def _get_conversation_sync(conversation_id: str) -> ConversationSummary:
    manager = get_conversation_manager()
    conversation = manager.get(conversation_id)
    if conversation is None:
        return ConversationSummary(
            conversation_id=conversation_id,
            turn_count=0,
            context="",
            accepted_count=0,
            rejected_count=0,
            last_turn="",
            last_updated=0.0,
        )
    return ConversationSummary(**conversation.summary())


@router.get("/conversations", response_model=list[ConversationSummary])
async def list_conversations(limit: int = 20) -> list[ConversationSummary]:
    return await run_in_threadpool(lambda: _list_conversations_sync(limit))


def _list_conversations_sync(limit: int = 20) -> list[ConversationSummary]:
    manager = get_conversation_manager()
    return [
        ConversationSummary(**summary)
        for summary in manager.list_conversations(limit=limit)
    ]


@router.post("/infer-cold-start", response_model=ColdStartInferenceResponse)
async def infer_cold_start(persona: str = "") -> ColdStartInferenceResponse:
    return await run_in_threadpool(lambda: _infer_cold_start_sync(persona))


def _infer_cold_start_sync(persona: str = "") -> ColdStartInferenceResponse:
    reasoner = RecommenderReasoner()
    result = reasoner.handle_cold_start(persona)
    return ColdStartInferenceResponse(
        preferred_categories=result.preferred_categories,
        price_sensitivity=result.price_sensitivity,
        quality_expectation=result.quality_expectation,
        likely_voice_style=result.likely_voice_style,
        key_terms=result.key_terms,
        confidence=result.confidence,
        provider=result.provider,
        llm_augmented=result.llm_augmented,
    )


@router.post("/transfer-cross-domain", response_model=CrossDomainTransferResponse)
async def transfer_cross_domain(
    request: CrossDomainTransferRequest,
) -> CrossDomainTransferResponse:
    return await run_in_threadpool(lambda: _transfer_cross_domain_sync(request))


def _transfer_cross_domain_sync(
    request: CrossDomainTransferRequest,
) -> CrossDomainTransferResponse:
    reasoner = RecommenderReasoner()
    result = reasoner.handle_cross_domain(
        source_domain=request.source_domain,
        target_domain=request.target_domain,
        preferences=request.preferences,
    )
    return CrossDomainTransferResponse(
        transferred_preferences=result.transferred_preferences,
        mapped_categories=result.mapped_categories,
        adjusted_terms=result.adjusted_terms,
        confidence=result.confidence,
        reasoning=result.reasoning,
        provider=result.provider,
        llm_augmented=result.llm_augmented,
    )


@router.post("/nigerian/context", response_model=NigerianContextResponse)
async def nigerian_context(
    persona: str = "",
    locale: str | None = None,
) -> NigerianContextResponse:
    return await run_in_threadpool(lambda: _nigerian_context_sync(persona, locale))


def _nigerian_context_sync(
    persona: str = "",
    locale: str | None = None,
) -> NigerianContextResponse:
    del locale
    engine = NigerianContextEngine()
    result = engine.inject_nigerian_context(persona=persona, history=[])
    return NigerianContextResponse(
        detected_markers=result.detected_markers,
        locale_signals=result.locale_signals,
        regional_context=result.regional_context,
        behavioral_indicators=result.behavioral_indicators,
        cultural_confidence=result.cultural_confidence,
        enriched_persona=result.enriched_persona,
    )


@router.get("/metrics", response_model=RuntimeMetrics)
async def metrics() -> RuntimeMetrics:
    return await run_in_threadpool(trace_store.metrics)


@router.get("/traces", response_model=list[TraceRecord])
async def traces(limit: int = 20) -> list[TraceRecord]:
    return await run_in_threadpool(lambda: trace_store.recent(limit=max(1, min(limit, 100))))


@router.get("/runtime/registry")
async def runtime_registry() -> dict:
    return await run_in_threadpool(lambda: get_model_registry().payload())


@router.get("/runtime/feature-store")
async def runtime_feature_store() -> dict:
    return await run_in_threadpool(_runtime_feature_store_sync)


def _runtime_feature_store_sync() -> dict:
    summary = get_feature_store().summary()
    return {
        "root": summary.root,
        "version": summary.version,
        "available": summary.available,
        "artifacts": summary.artifacts,
        "counts": summary.counts,
    }
