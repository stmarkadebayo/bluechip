from __future__ import annotations

from fastapi import APIRouter

from app.agents.recommendation_agent import RecommendationAgent
from app.agents.review_simulation_agent import ReviewSimulationAgent
from app.models.schemas import (
    HealthResponse,
    ProfileUserRequest,
    RecommendationRequest,
    RecommendationResponse,
    SimulateReviewRequest,
    SimulateReviewResponse,
    UserProfileResponse,
)
from app.services.profiling.user_profile import build_user_profile

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
