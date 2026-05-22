from __future__ import annotations

import time

from app.models.schemas import AgentTraceStep, SimulateReviewRequest, SimulateReviewResponse
from app.platform.feature_store import get_feature_store
from app.platform.model_registry import get_model_registry
from app.services.agentic.user_simulator import UserSimulator
from app.services.generation.generator import generate_review_result
from app.services.generation.providers import generation_provider_name
from app.services.nigerian.context import NigerianContextEngine
from app.services.nigerian.pidgin import NigerianVoiceInjector
from app.services.profiling.item_profile import build_item_profile
from app.services.profiling.user_profile import build_user_profile
from app.services.ranking.rating import predict_rating
from app.services.validation.critic import validate_review_simulation
from app.stores.trace_store import trace_store


class ReviewSimulationAgent:
    """Serving orchestrator for Task A review simulation with LLM agentic workflow.

    Integrates the LLM-driven UserSimulator for authentic review generation,
    Nigerian context enrichment, and pidgin voice injection. Falls back to
    deterministic pipeline when LLM is unavailable.
    """

    def __init__(self) -> None:
        self._simulator = UserSimulator()
        self._nigerian_engine = NigerianContextEngine()
        self._voice_injector = NigerianVoiceInjector()

    def run(self, request: SimulateReviewRequest) -> SimulateReviewResponse:
        started = time.perf_counter()
        trace: list[AgentTraceStep] = []

        user_profile = build_user_profile(
            persona=request.user_persona,
            history=request.user_history,
            locale=request.locale,
            enhance_with_llm=request.enhance_with_llm,
        )
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
                    f"{user_profile.evidence_count} history items, confidence {user_profile.confidence:.2f}"
                    + (
                        f", profile enhancer {user_profile.profile_enhancement.provider}"
                        if user_profile.profile_enhancement
                        else ""
                    )
                ),
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

        nigerian_result = self._nigerian_engine.inject_nigerian_context(
            persona=request.user_persona,
            history=[
                {"item_name": h.item_name, "rating": h.rating, "review": h.review,
                 "category": h.category or ""}
                for h in request.user_history
            ],
        )
        nigerian_relevance = NigerianContextEngine.score_nigerian_relevance(user_profile)
        trace.append(
            AgentTraceStep(
                step="nigerian_context",
                status="ok",
                detail=(
                    f"confidence {nigerian_result.cultural_confidence:.2f}, "
                    f"relevance {nigerian_relevance:.2f}"
                ),
            )
        )

        rating_result = predict_rating(
            user_profile=user_profile,
            item_profile=item_profile,
            user_id=request.user_id,
        )

        decision = self._simulator.simulate_review_decision(user_profile, item_profile)
        agentic_rating = decision.predicted_rating if decision.llm_augmented else None

        # The promoted rating head owns the numeric metric; LLM reasoning may
        # shape wording, but should not override the RMSE-validated rating.
        final_rating = rating_result.predicted_rating
        trace.append(
            AgentTraceStep(
                step="predict_rating",
                status="ok",
                detail=(
                    f"predicted {final_rating}/5"
                    f" using {rating_result.model_name or 'default rating model'}"
                    f"{' with LLM advisory ignored for rating' if agentic_rating is not None else ''}"
                ),
            )
        )

        generated_review = generate_review_result(
            user_profile=user_profile,
            item_profile=item_profile,
            predicted_rating=final_rating,
        )

        if decision.llm_augmented and generated_review.used_fallback:
            agentic_review = self._simulator.generate_authentic_review(
                user_profile=user_profile,
                item_profile=item_profile,
                rating=final_rating,
                decision_context=decision,
            )
            if agentic_review:
                review = agentic_review
            else:
                review = generated_review.text
        else:
            review = generated_review.text

        if nigerian_relevance > 0.25:
            review = self._voice_injector.nigerianize_review(
                review, intensity=min(nigerian_relevance, 0.80)
            )
            trace.append(
                AgentTraceStep(
                    step="nigerianize_review",
                    status="ok",
                    detail=f"injected Nigerian voice at intensity {min(nigerian_relevance, 0.80):.2f}",
                )
            )

        trace.append(
            AgentTraceStep(
                step="generate_review",
                status="ok",
                detail=(
                    "LLM agentic authentic review"
                    if decision.llm_augmented
                    else "rating-conditioned generation"
                ),
            )
        )

        validation = validate_review_simulation(
            predicted_rating=final_rating,
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
        trace_record = trace_store.append(
            endpoint="simulate-review",
            latency_ms=(time.perf_counter() - started) * 1000,
            steps=trace,
            generation_provider=generated_review.provider or generation_provider_name(),
            estimated_generation_tokens=max(len(review) // 4, 1),
            model_versions=_task_a_model_versions(rating_result.model_name),
            index_versions={"feature_store": get_feature_store().version()},
            validation_status="ok" if validation.is_consistent else "warning",
            fallback_reason=generated_review.error if generated_review.used_fallback else None,
        )

        return SimulateReviewResponse(
            trace_id=trace_record.trace_id,
            predicted_rating=final_rating,
            predicted_score=rating_result.predicted_score,
            review=review,
            confidence=rating_result.confidence,
            model_name=rating_result.model_name,
            user_signals=rating_result.user_signals,
            item_signals=rating_result.item_signals,
            validation=validation,
            agent_trace=trace,
        )


def _task_a_model_versions(serving_head: str | None) -> dict[str, str]:
    versions = get_model_registry().versions(
        "task_a_model",
        "task_a_rating_stats",
        "task_a_serving_policy",
    )
    versions["task_a_serving_head"] = serving_head or "unknown"
    return versions
