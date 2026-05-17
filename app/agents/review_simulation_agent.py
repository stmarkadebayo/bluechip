from __future__ import annotations

from app.models.schemas import AgentTraceStep, SimulateReviewRequest, SimulateReviewResponse
from app.services.generation.generator import generate_review
from app.services.profiling.item_profile import build_item_profile
from app.services.profiling.user_profile import build_user_profile
from app.services.ranking.rating import predict_rating
from app.services.validation.critic import validate_review_simulation


class ReviewSimulationAgent:
    """Agentic workflow for Task A.

    The agent owns orchestration and branching. The underlying services remain
    deterministic, testable tools.
    """

    def run(self, request: SimulateReviewRequest) -> SimulateReviewResponse:
        trace: list[AgentTraceStep] = []

        user_profile = build_user_profile(
            persona=request.user_persona,
            history=request.user_history,
            locale=request.locale,
        )
        trace.append(
            AgentTraceStep(
                step="profile_user",
                status="ok",
                detail=f"{user_profile.evidence_count} history items, confidence {user_profile.confidence:.2f}",
            )
        )

        item_profile = build_item_profile(request.target_item)
        trace.append(
            AgentTraceStep(
                step="profile_item",
                status="ok",
                detail=f"{item_profile.category} with quality {item_profile.quality_score:.2f}",
            )
        )

        rating_result = predict_rating(user_profile=user_profile, item_profile=item_profile)
        trace.append(
            AgentTraceStep(
                step="predict_rating",
                status="ok",
                detail=f"predicted {rating_result.predicted_rating}/5",
            )
        )

        review = generate_review(
            user_profile=user_profile,
            item_profile=item_profile,
            predicted_rating=rating_result.predicted_rating,
        )
        trace.append(AgentTraceStep(step="generate_review", status="ok", detail="rating-conditioned"))

        validation = validate_review_simulation(
            predicted_rating=rating_result.predicted_rating,
            review=review,
            user_profile=user_profile,
            item_profile=item_profile,
        )
        trace.append(
            AgentTraceStep(
                step="validate",
                status="ok" if validation.is_consistent else "warning",
                detail=", ".join(validation.issues) if validation.issues else "no issues",
            )
        )

        return SimulateReviewResponse(
            predicted_rating=rating_result.predicted_rating,
            review=review,
            confidence=rating_result.confidence,
            user_signals=rating_result.user_signals,
            item_signals=rating_result.item_signals,
            validation=validation,
            agent_trace=trace,
        )

