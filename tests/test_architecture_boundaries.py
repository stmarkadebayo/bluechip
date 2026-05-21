from __future__ import annotations

from app.agents.recommendation_agent import RecommendationAgent as LegacyRecommendationAgent
from app.agents.review_simulation_agent import ReviewSimulationAgent as LegacyReviewSimulationAgent
from app.api import routes
from app.serving.orchestrators.recommendation import RecommendationAgent
from app.serving.orchestrators.review_simulation import ReviewSimulationAgent


def test_api_routes_use_serving_orchestrators() -> None:
    assert routes.RecommendationAgent is RecommendationAgent
    assert routes.ReviewSimulationAgent is ReviewSimulationAgent


def test_legacy_agent_imports_remain_compatibility_shims() -> None:
    assert LegacyRecommendationAgent is RecommendationAgent
    assert LegacyReviewSimulationAgent is ReviewSimulationAgent
