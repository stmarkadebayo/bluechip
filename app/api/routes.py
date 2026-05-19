from __future__ import annotations

from fastapi import APIRouter

from app.agents.recommendation_agent import RecommendationAgent
from app.agents.review_simulation_agent import ReviewSimulationAgent
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
