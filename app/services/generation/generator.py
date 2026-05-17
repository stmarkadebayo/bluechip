from __future__ import annotations

from app.models.schemas import ItemProfile, RecommendationItem, UserProfile
from app.services.generation.providers import TemplateGenerationProvider, get_generation_provider


def generate_review(
    user_profile: UserProfile,
    item_profile: ItemProfile,
    predicted_rating: int,
) -> str:
    tone = "practical"
    if user_profile.locale and user_profile.locale.lower() == "nigeria":
        tone = "natural Nigerian English"

    positive_signal = (item_profile.signals[-1] if item_profile.signals else item_profile.name).rstrip(".")
    if predicted_rating >= 4:
        verdict = "it fits what I usually look for"
    elif predicted_rating == 3:
        verdict = "it has some useful strengths, but I would be selective about recommending it"
    else:
        verdict = "it misses too many of the things I care about"

    fallback = (
        f"I would rate {item_profile.name} {predicted_rating} out of 5. "
        f"Based on the available details, {positive_signal}. "
        f"For my preferences, {verdict}. "
        f"The tone should stay {tone}, {user_profile.voice_style}, and grounded in the facts provided."
    )
    provider = get_generation_provider()
    if isinstance(provider, TemplateGenerationProvider):
        return fallback

    instructions = (
        "You generate concise personalized reviews. Ground every claim in the provided "
        "user profile, item profile, and predicted rating. Do not invent item facts."
    )
    prompt = (
        f"Predicted rating: {predicted_rating}/5\n"
        f"User voice: {user_profile.voice_style}\n"
        f"User signals: {user_profile.signals}\n"
        f"Item: {item_profile.name}\n"
        f"Item signals: {item_profile.signals}\n"
        f"Locale: {user_profile.locale or 'unspecified'}\n"
        "Write one review paragraph only."
    )
    try:
        generated = provider.generate(instructions=instructions, prompt=prompt)
        return generated or fallback
    except RuntimeError:
        return fallback


def generate_recommendation_reason(
    user_profile: UserProfile,
    recommendation: RecommendationItem,
    context: str,
) -> str:
    signals = ", ".join(recommendation.matched_signals[:3]) or (
        ", ".join((user_profile.positive_aspects or user_profile.preferred_terms)[:3])
        or "the stated persona"
    )
    context_clause = f" The current context is: {context}" if context else ""
    fallback = (
        f"{recommendation.name} ranks well because it matches {signals} "
        f"and has supporting item signals in the candidate profile.{context_clause}"
    )
    provider = get_generation_provider()
    if isinstance(provider, TemplateGenerationProvider):
        return fallback

    instructions = (
        "You explain personalized recommendations in one concise sentence. "
        "Use only the provided user and item signals."
    )
    prompt = (
        f"User signals: {user_profile.signals}\n"
        f"Candidate: {recommendation.name}\n"
        f"Candidate signals: {recommendation.signals}\n"
        f"Score: {recommendation.score}\n"
        f"Context: {context}\n"
        "Explain why this item fits and mention any obvious tradeoff."
    )
    try:
        generated = provider.generate(instructions=instructions, prompt=prompt)
        return generated or fallback
    except RuntimeError:
        return fallback
