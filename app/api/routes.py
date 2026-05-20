from __future__ import annotations

from fastapi import APIRouter

from app.models.schemas import (
    HealthResponse,
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
from app.services.profiling.user_profile import build_user_profile
from app.stores.trace_store import trace_store

router = APIRouter()


@router.get("/health", response_model=HealthResponse)
def health() -> HealthResponse:
    return HealthResponse(status="ok")


@router.post("/profile-user", response_model=UserProfileResponse)
def profile_user(request: ProfileUserRequest) -> UserProfileResponse:
    profile = build_user_profile(
        persona=request.user_persona,
        history=request.user_history,
        locale=request.locale,
    )
    return UserProfileResponse(profile=profile)


@router.post("/simulate-review", response_model=SimulateReviewResponse)
def simulate_review(request: SimulateReviewRequest) -> SimulateReviewResponse:
    return ReviewSimulationAgent().run(request)


@router.post("/recommend", response_model=RecommendationResponse)
def recommend(request: RecommendationRequest) -> RecommendationResponse:
    return RecommendationAgent().run(request)


@router.get("/metrics", response_model=RuntimeMetrics)
def metrics() -> RuntimeMetrics:
    return trace_store.metrics()


@router.get("/traces", response_model=list[TraceRecord])
def traces(limit: int = 20) -> list[TraceRecord]:
    return trace_store.recent(limit=max(1, min(limit, 100)))


@router.get("/runtime/registry")
def runtime_registry() -> dict:
    return get_model_registry().payload()


@router.get("/runtime/feature-store")
def runtime_feature_store() -> dict:
    summary = get_feature_store().summary()
    return {
        "root": summary.root,
        "version": summary.version,
        "available": summary.available,
        "artifacts": summary.artifacts,
        "counts": summary.counts,
    }
