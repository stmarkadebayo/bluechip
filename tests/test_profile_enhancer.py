from __future__ import annotations

from app.models.schemas import UserHistoryItem
from app.services.generation.providers import TemplateGenerationProvider
from app.services.profiling import profile_enhancer
from app.services.profiling.user_profile import build_user_profile


class JsonProvider:
    def generate(self, instructions: str, prompt: str) -> str:
        del instructions, prompt
        return """
        {
          "preferred_terms": ["delivery", "budget", "authentic", "extra-term"],
          "positive_aspects": ["durable", "value"],
          "preferred_categories": ["skincare"],
          "voice_style": "price-aware and specific",
          "price_sensitivity": "high",
          "confidence": 0.9,
          "rationale": "Persona and reviews show value seeking."
        }
        """


def test_profile_enhancer_merges_bounded_llm_fields(monkeypatch) -> None:
    monkeypatch.setattr(profile_enhancer, "get_generation_provider", lambda: JsonProvider())
    monkeypatch.setattr(profile_enhancer, "generation_provider_name", lambda: "test-llm")

    history = [
        UserHistoryItem(
            item_id="cream-1",
            item_name="Gentle Cream",
            rating=5,
            review="Affordable and durable for daily use.",
            category="beauty",
        )
    ]

    profile = build_user_profile(
        "Lagos student who wants affordable original skincare.",
        history,
        locale="Nigeria",
        enhance_with_llm=True,
    )

    assert profile.profile_enhancement is not None
    assert profile.profile_enhancement.llm_augmented
    assert profile.profile_enhancement.provider == "test-llm"
    assert profile.profile_enhancement.added_terms["preferred_terms"] == [
        "delivery",
        "budget",
        "authentic",
    ]
    assert "skincare" in profile.preferred_categories
    assert profile.confidence <= 0.98
    assert any("LLM profile enhancement" in signal for signal in profile.signals)


def test_profile_enhancer_falls_back_without_provider(monkeypatch) -> None:
    monkeypatch.setattr(
        profile_enhancer,
        "get_generation_provider",
        lambda: TemplateGenerationProvider(),
    )
    monkeypatch.setattr(profile_enhancer, "generation_provider_name", lambda: "template")

    profile = build_user_profile(
        "Practical buyer.",
        [],
        enhance_with_llm=True,
    )

    assert profile.profile_enhancement is not None
    assert not profile.profile_enhancement.llm_augmented
    assert profile.profile_enhancement.fallback_reason == "template provider"
