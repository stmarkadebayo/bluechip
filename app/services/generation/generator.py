from __future__ import annotations

from dataclasses import dataclass

from app.models.schemas import ItemProfile, RecommendationItem, UserProfile
from app.services.generation.providers import (
    TemplateGenerationProvider,
    generation_provider_name,
    get_generation_provider,
)


@dataclass(frozen=True)
class GeneratedText:
    text: str
    provider: str
    used_fallback: bool = False
    error: str = ""


def generate_review(
    user_profile: UserProfile,
    item_profile: ItemProfile,
    predicted_rating: int,
) -> str:
    return generate_review_result(
        user_profile=user_profile,
        item_profile=item_profile,
        predicted_rating=predicted_rating,
    ).text


def generate_review_result(
    user_profile: UserProfile,
    item_profile: ItemProfile,
    predicted_rating: int,
    strict_provider: bool = False,
) -> GeneratedText:
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
    provider_name = generation_provider_name()
    if isinstance(provider, TemplateGenerationProvider):
        return GeneratedText(text=fallback, provider=provider_name)

    instructions = (
        "You generate concise personalized reviews. Ground every claim in the provided "
        "user profile, item profile, and predicted rating. Do not invent item facts. "
        "The first sentence must explicitly state the target item and rating in the form "
        "'I would rate <item> <rating> out of 5.'"
    )
    prompt = (
        f"Predicted rating: {predicted_rating}/5\n"
        f"User voice: {user_profile.voice_style}\n"
        f"User signals: {user_profile.signals}\n"
        f"Item: {item_profile.name}\n"
        f"Item signals: {item_profile.signals}\n"
        f"Locale: {user_profile.locale or 'unspecified'}\n"
        f"Required first sentence: I would rate {item_profile.name} {predicted_rating} out of 5.\n"
        "Write one review paragraph only."
    )
    try:
        generated = provider.generate(instructions=instructions, prompt=prompt)
        if generated:
            repaired = _repair_review_contract(
                generated,
                item_profile=item_profile,
                predicted_rating=predicted_rating,
                user_profile=user_profile,
            )
            return GeneratedText(text=repaired, provider=provider_name)
        if strict_provider:
            raise RuntimeError(f"{provider_name} returned an empty generation")
        return GeneratedText(
            text=fallback,
            provider=provider_name,
            used_fallback=True,
            error="empty generation",
        )
    except RuntimeError as exc:
        if strict_provider:
            raise
        return GeneratedText(
            text=fallback,
            provider=provider_name,
            used_fallback=True,
            error=str(exc),
        )


def _repair_review_contract(
    text: str,
    item_profile: ItemProfile,
    predicted_rating: int,
    user_profile: UserProfile,
) -> str:
    review = " ".join(text.split())
    required = f"I would rate {item_profile.name} {predicted_rating} out of 5."
    mentions_item = item_profile.name.lower() in review.lower()
    mentions_rating = str(predicted_rating) in review
    if not mentions_item or not mentions_rating:
        review = f"{required} {review}"
    if user_profile.locale and user_profile.locale.lower() == "nigeria" and "Nigerian" not in review:
        review = f"{review} The tone is natural Nigerian English."
    return review


def generate_recommendation_reason(
    user_profile: UserProfile,
    recommendation: RecommendationItem,
    context: str,
) -> str:
    return generate_recommendation_reason_result(
        user_profile=user_profile,
        recommendation=recommendation,
        context=context,
    ).text


def generate_recommendation_reason_result(
    user_profile: UserProfile,
    recommendation: RecommendationItem,
    context: str,
    strict_provider: bool = False,
) -> GeneratedText:
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
    provider_name = generation_provider_name()
    if isinstance(provider, TemplateGenerationProvider):
        return GeneratedText(text=fallback, provider=provider_name)

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
        if generated:
            return GeneratedText(text=generated, provider=provider_name)
        if strict_provider:
            raise RuntimeError(f"{provider_name} returned an empty generation")
        return GeneratedText(
            text=fallback,
            provider=provider_name,
            used_fallback=True,
            error="empty generation",
        )
    except RuntimeError as exc:
        if strict_provider:
            raise
        return GeneratedText(
            text=fallback,
            provider=provider_name,
            used_fallback=True,
            error=str(exc),
        )
