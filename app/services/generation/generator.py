from __future__ import annotations

from dataclasses import dataclass

from app.models.schemas import ItemProfile, RecommendationItem, UserHistoryItem, UserProfile
from app.services.generation.providers import (
    TemplateGenerationProvider,
    generation_provider_name,
    get_generation_provider,
)
from app.services.generation.review_plan import build_review_plan, fallback_review_from_plan


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
    history: list[UserHistoryItem] | None = None,
) -> str:
    return generate_review_result(
        user_profile=user_profile,
        item_profile=item_profile,
        predicted_rating=predicted_rating,
        history=history,
    ).text


def generate_review_result(
    user_profile: UserProfile,
    item_profile: ItemProfile,
    predicted_rating: int,
    strict_provider: bool = False,
    history: list[UserHistoryItem] | None = None,
) -> GeneratedText:
    plan = build_review_plan(user_profile, item_profile, predicted_rating)
    fallback = fallback_review_from_plan(item_profile, plan)
    provider = get_generation_provider()
    provider_name = generation_provider_name()
    if isinstance(provider, TemplateGenerationProvider):
        return GeneratedText(text=fallback, provider=provider_name)

    instructions = (
        "You are simulating a real customer's product review, not completing a template. "
        "Match the user's prior review style, length, directness, and vocabulary when examples "
        "are provided. Ground every claim in the user profile, item profile, review plan, and "
        "predicted rating. Do not invent item facts. Mention the target item and the numeric "
        "rating naturally somewhere in the review, but do not force the first sentence to be "
        "the rating. Avoid boilerplate phrases like 'supports that rating' and "
        "'fits what I usually look for'."
    )
    prompt = (
        f"{plan.prompt_block()}\n"
        f"User voice: {user_profile.voice_style}\n"
        f"Archetype guidance: {_archetype_guidance(plan.archetype)}\n"
        f"User signals: {_format_list(user_profile.signals)}\n"
        f"Prior review examples:\n{_history_examples(history)}\n"
        f"Item: {item_profile.name}\n"
        f"Item signals: {_format_list(item_profile.signals)}\n"
        f"Locale: {user_profile.locale or 'unspecified'}\n"
        f"Predicted rating: {predicted_rating} out of 5. Express it naturally, not as a forced opener.\n"
        "Write one human review paragraph only. It may be short, blunt, uneven, or specific "
        "if that matches the user's examples. Do not use the same opening or closing every time."
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
    mentions_item = item_profile.name.lower() in review.lower()
    mentions_rating = str(predicted_rating) in review
    if not mentions_item and not mentions_rating:
        review = (
            f"{review} For {item_profile.name}, that lands at "
            f"{predicted_rating} out of 5 for me."
        )
    elif not mentions_rating:
        review = f"{review} That puts it at {predicted_rating} out of 5 for me."
    elif not mentions_item:
        review = f"{review} That is my take on {item_profile.name}."
    if (
        user_profile.locale
        and user_profile.locale.lower() == "nigeria"
        and not _has_nigerian_marker(review)
    ):
        review = f"{review} As a Nigerian shopper, value still matters here."
    return review


def _history_examples(history: list[UserHistoryItem] | None, limit: int = 5) -> str:
    if not history:
        return "- No prior examples available; infer a natural review style from the profile."
    examples = []
    ordered = sorted(history, key=lambda item: item.timestamp or 0)
    for item in ordered[-limit:]:
        review = " ".join(item.review.split())
        examples.append(
            f"- {item.item_name} ({item.rating:g}/5): {review[:500]}"
        )
    return "\n".join(examples)


def _format_list(values: list[str], limit: int = 12) -> str:
    clean = [str(value).strip() for value in values if str(value).strip()]
    return "; ".join(clean[:limit]) if clean else "none"


def _archetype_guidance(archetype: str) -> str:
    guidance = {
        "SHORT_DIRECT": "Keep it brief and plain, like a quick note after using the product.",
        "EMOTIONAL": "Let the reaction feel personal without becoming dramatic or salesy.",
        "DETAILED_ANALYTICAL": "Use specific tradeoffs and measured language, but keep it human.",
        "CASUAL": "Sound relaxed and natural, with ordinary customer wording.",
        "CRITICAL": "Be candid about reservations while staying fair to the evidence.",
        "HYPE_REVIEWER": "Let the enthusiasm show, but avoid promotional language.",
    }
    return guidance.get(archetype, guidance["CASUAL"])


def _has_nigerian_marker(text: str) -> bool:
    lower = text.lower()
    return any(
        marker in lower
        for marker in (
            "nigeria",
            "nigerian",
            "lagos",
            "naira",
            "delivery",
            "seller",
            "original",
            "value",
        )
    )


def generate_recommendation_reason(
    user_profile: UserProfile,
    recommendation: RecommendationItem,
    context: str,
    history: list[UserHistoryItem] | None = None,
) -> str:
    return generate_recommendation_reason_result(
        user_profile=user_profile,
        recommendation=recommendation,
        context=context,
        history=history,
    ).text


def generate_recommendation_reason_result(
    user_profile: UserProfile,
    recommendation: RecommendationItem,
    context: str,
    strict_provider: bool = False,
    history: list[UserHistoryItem] | None = None,
) -> GeneratedText:
    signals = ", ".join(recommendation.matched_signals[:3]) or (
        ", ".join((user_profile.positive_aspects or user_profile.preferred_terms)[:3])
        or "the stated persona"
    )
    fallback = _recommendation_fallback(recommendation, signals, context)
    provider = get_generation_provider()
    provider_name = generation_provider_name()
    if isinstance(provider, TemplateGenerationProvider):
        return GeneratedText(text=fallback, provider=provider_name)

    instructions = (
        "Write a short, natural recommendation blurb. Sound conversational and human, "
        "not like a ranking system. Use only the provided user, item, and context signals. "
        "Do not mention score components, candidate profiles, or retrieval logic."
    )
    prompt = (
        f"User voice: {user_profile.voice_style}\n"
        f"Prior review examples:\n{_history_examples(history, limit=3)}\n"
        f"User signals: {_format_list(user_profile.signals)}\n"
        f"Candidate: {recommendation.name}\n"
        f"Candidate signals: {_format_list(recommendation.signals)}\n"
        f"Matched signals: {_format_list(recommendation.matched_signals)}\n"
        f"Known tradeoff: {recommendation.tradeoffs or 'none'}\n"
        f"Context: {context}\n"
        "Write one sentence a user would actually accept in a product assistant."
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


def _recommendation_fallback(
    recommendation: RecommendationItem,
    signals: str,
    context: str,
) -> str:
    context_clause = _context_clause(context)
    seed = sum(ord(char) for char in recommendation.item_id or recommendation.name)
    templates = (
        f"{recommendation.name} looks like the practical pick{context_clause}, especially around {signals}.",
        f"I would keep {recommendation.name} near the top{context_clause} because the strongest match is {signals}.",
        f"{recommendation.name} feels like the safer first choice{context_clause}; it lines up with {signals}.",
    )
    return templates[seed % len(templates)]


def _context_clause(context: str) -> str:
    clean = " ".join(context.split()).strip(" .;:,")
    for delimiter in (".", ";"):
        if delimiter not in clean:
            continue
        first_sentence = clean.split(delimiter, 1)[0].strip(" .;:,")
        if first_sentence:
            clean = first_sentence
            break
    if not clean:
        return ""
    if len(clean) > 90:
        clean = clean[:87].rsplit(" ", 1)[0].strip(" .;:,")
    return f" for {clean}"
