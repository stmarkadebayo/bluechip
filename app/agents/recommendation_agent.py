from __future__ import annotations

from app.models.schemas import AgentTraceStep, RecommendationRequest, RecommendationResponse
from app.services.generation.generator import generate_recommendation_reason
from app.services.profiling.user_profile import build_user_profile
from app.services.ranking.recommendation import rank_candidates
from app.services.retrieval.candidates import generate_candidates


class RecommendationAgent:
    """Agentic workflow for Task B."""

    def run(self, request: RecommendationRequest) -> RecommendationResponse:
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

        retrieved = generate_candidates(
            user_profile=user_profile,
            history=request.user_history,
            items=request.candidate_items,
            context=request.context,
            limit=min(len(request.candidate_items), 100),
        )
        trace.append(
            AgentTraceStep(
                step="retrieve_candidates",
                status="ok",
                detail=f"{len(retrieved)} candidates from {len(request.candidate_items)} input items",
            )
        )

        ranked = rank_candidates(
            user_profile=user_profile,
            context=request.context,
            candidate_items=retrieved,
            limit=request.limit,
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

        return RecommendationResponse(recommendations=recommendations, agent_trace=trace)
