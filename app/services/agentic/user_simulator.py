from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from typing import Any

from app.models.schemas import ItemProfile, UserHistoryItem, UserProfile
from app.services.generation.providers import (
    TemplateGenerationProvider,
    get_generation_provider,
    generation_provider_name,
)
from app.services.generation.review_plan import (
    ReviewPlan,
    build_review_plan,
    fallback_review_from_plan,
)
from app.services.agentic.reasoner import get_reasoner


@dataclass
class ReviewDecision:
    """LLM-reasoned review decision with intermediate traces."""

    predicted_rating: int
    rating_rationale: str
    thought_process: str
    key_concerns: list[str]
    first_notice: str
    emotional_reaction: str
    provider: str = ""
    llm_augmented: bool = False
    trace: list[dict[str, str]] = field(default_factory=list)


class UserSimulator:
    """LLM-driven user simulation for Task A: review generation.

    Uses the LLM to reason about how a specific user would evaluate an item,
    then generates an authentic review in the user's voice.  All methods
    fall back to deterministic behaviour when the LLM is unavailable.
    """

    def __init__(self) -> None:
        self._provider = get_generation_provider()
        self._provider_name = generation_provider_name()
        self._reasoner = get_reasoner()

    # ------------------------------------------------------------------
    # Review decision simulation
    # ------------------------------------------------------------------

    def simulate_review_decision(
        self,
        user_profile: UserProfile,
        item_profile: ItemProfile,
    ) -> ReviewDecision:
        """LLM reasons about how this specific user would evaluate this item.

        Returns the thought process, predicted rating rationale, key concerns,
        what the user would notice first, and their emotional reaction.
        """
        plan = build_review_plan(
            user_profile=user_profile,
            item_profile=item_profile,
            predicted_rating=_baseline_rating(user_profile, item_profile),
        )
        baseline = plan.predicted_rating

        if isinstance(self._provider, TemplateGenerationProvider):
            return self._deterministic_decision(user_profile, item_profile, plan)

        prompt = (
            f"You are simulating a real user's decision process for a product review.\n\n"
            f"USER PROFILE:\n"
            f"  Voice style: {user_profile.voice_style}\n"
            f"  Rating strictness: {user_profile.rating_strictness}\n"
            f"  Average rating tendency: {user_profile.average_rating:.2f}\n"
            f"  Preferred terms: {', '.join(user_profile.preferred_terms[:5]) or 'none'}\n"
            f"  Disliked terms: {', '.join(user_profile.disliked_terms[:5]) or 'none'}\n"
            f"  Preferred categories: {', '.join(user_profile.preferred_categories[:3]) or 'none'}\n"
            f"  Positive aspects: {', '.join(user_profile.positive_aspects[:5]) or 'none'}\n"
            f"  Negative sensitivities: {', '.join(user_profile.negative_aspects[:5]) or 'none'}\n"
            f"  Confidence: {user_profile.confidence:.2f}\n"
            f"  Locale: {user_profile.locale or 'unspecified'}\n\n"
            f"ITEM PROFILE:\n"
            f"  Name: {item_profile.name}\n"
            f"  Category: {item_profile.category}\n"
            f"  Quality: {item_profile.quality_score:.2f}\n"
            f"  Positive aspects: {', '.join(item_profile.positive_aspects[:5]) or 'none'}\n"
            f"  Negative aspects: {', '.join(item_profile.negative_aspects[:5]) or 'none'}\n"
            f"  Signals: {', '.join(item_profile.signals[:3])}\n"
            f"  Baseline rating prediction: {baseline}/5\n\n"
            "Reason step by step about how this specific user would approach this item.\n"
            "Return a JSON object with these keys:\n"
            '  "predicted_rating": integer 1-5\n'
            '  "rating_rationale": one sentence explaining the rating\n'
            '  "thought_process": the user\u2019s internal monologue (2-3 sentences)\n'
            '  "key_concerns": list of 2-4 specific concerns this user would have\n'
            '  "first_notice": what this user would notice first about this item\n'
            '  "emotional_reaction": how this user feels (specific emotion phrases)\n'
            "Respond ONLY with valid JSON."
        )
        system = (
            "You are simulating a real consumer's internal decision process. "
            "Ground every judgement in the provided user profile and item data. "
            "Do NOT invent item facts. Think like this specific person would think. "
            "Return only valid JSON."
        )
        try:
            raw = self._provider.generate(instructions=system, prompt=prompt)
            parsed = _extract_json(raw) if raw else {}
        except Exception:
            parsed = {}

        if not parsed:
            return self._deterministic_decision(user_profile, item_profile, plan)

        rating = max(1, min(5, int(parsed.get("predicted_rating", baseline))))

        decision = ReviewDecision(
            predicted_rating=rating,
            rating_rationale=str(parsed.get("rating_rationale", "")),
            thought_process=str(parsed.get("thought_process", "")),
            key_concerns=_as_str_list(parsed.get("key_concerns", [])),
            first_notice=str(parsed.get("first_notice", "")),
            emotional_reaction=str(parsed.get("emotional_reaction", "")),
            provider=self._provider_name,
            llm_augmented=True,
            trace=[
                {"step": "reason", "status": "ok", "detail": "LLM reasoned review decision"},
            ],
        )
        return decision

    # ------------------------------------------------------------------
    # Authentic review generation
    # ------------------------------------------------------------------

    def generate_authentic_review(
        self,
        user_profile: UserProfile,
        item_profile: ItemProfile,
        rating: int,
        decision_context: ReviewDecision | None = None,
        history: list[UserHistoryItem] | None = None,
    ) -> str:
        """Generate a review in the user's authentic voice.

        Uses the behavioural model from the reasoner and the review plan
        system to produce a linguistically faithful review.
        """
        plan = build_review_plan(user_profile, item_profile, rating)
        fallback = fallback_review_from_plan(item_profile, plan)

        if isinstance(self._provider, TemplateGenerationProvider):
            return fallback

        mental_model = self._reasoner.build_user_mental_model(
            persona=_user_persona_text(user_profile),
            history=[],  # profile already encodes history
        )

        decision_text = ""
        if decision_context:
            decision_text = (
                f"User's thought process: {decision_context.thought_process}\n"
                f"First notice: {decision_context.first_notice}\n"
                f"Emotional reaction: {decision_context.emotional_reaction}\n"
                f"Key concerns: {', '.join(decision_context.key_concerns)}\n"
            )

        instructions = (
            "You generate a review in the exact voice of a specific user. This is a "
            "simulation of a human review, not a template fill. Ground every claim in "
            "the provided profile, item, and review plan. Do not invent item facts. "
            "Mention the item and numeric rating naturally somewhere, but do not force "
            "the first sentence to be the rating. Mirror the user's prior review examples: "
            "short if they write short, detailed if they write detailed, blunt if they "
            "write blunt. Avoid boilerplate phrases like 'supports that rating' and "
            "'fits what I usually look for'."
        )
        prompt = (
            f"{plan.prompt_block()}\n"
            f"User voice style: {user_profile.voice_style}\n"
            f"User signals: {_format_list(user_profile.signals)}\n"
            f"Prior review examples:\n{_history_examples(history)}\n"
            f"User mental model:\n{json.dumps(mental_model, indent=2)}\n"
            f"{decision_text}\n"
            f"Item: {item_profile.name}\n"
            f"Item signals: {', '.join(item_profile.signals)}\n"
            f"Locale: {user_profile.locale or 'unspecified'}\n"
            f"Predicted rating to express naturally: {rating} out of 5.\n"
            f"IMPORTANT: Write in the user's authentic voice. Use phrasing this user "
            f"would naturally use. Write ONE review paragraph only, with varied opening "
            f"and ending."
        )
        try:
            generated = self._provider.generate(instructions=instructions, prompt=prompt)
            if generated:
                repaired = _repair_review(generated, item_profile, rating, user_profile)
                return repaired
            return fallback
        except Exception:
            return fallback

    # ------------------------------------------------------------------
    # Rating simulation
    # ------------------------------------------------------------------

    def simulate_rating(
        self,
        user_profile: UserProfile,
        item_profile: ItemProfile,
    ) -> int:
        """Predict what rating this user would give, with a reasoning chain.

        Falls back to the baseline heuristic when the LLM is unavailable.
        """
        plan = build_review_plan(
            user_profile, item_profile,
            predicted_rating=_baseline_rating(user_profile, item_profile),
        )
        baseline = plan.predicted_rating

        if isinstance(self._provider, TemplateGenerationProvider):
            return baseline

        reasoner_prompt = (
            "You are predicting what star rating (1-5) a specific user would give "
            "to an item. Think step by step, then output ONLY the integer rating.\n\n"
            f"User: {_user_persona_text(user_profile)}\n"
            f"Item: {item_profile.name} ({item_profile.category})\n"
            f"Quality: {item_profile.quality_score:.2f}\n"
            f"Positive: {', '.join(item_profile.positive_aspects[:3]) or 'none'}\n"
            f"Negative: {', '.join(item_profile.negative_aspects[:3]) or 'none'}\n"
            f"User avg rating: {user_profile.average_rating:.2f}\n"
            f"User strictness: {user_profile.rating_strictness}\n"
            f"Baseline prediction: {baseline}/5\n\n"
            f"Respond with ONLY the integer rating number (1 through 5)."
        )
        system = "You predict user ratings. Output only the integer rating."

        try:
            raw = self._provider.generate(instructions=system, prompt=reasoner_prompt)
            if raw:
                match = re.search(r"\b([1-5])\b", raw.strip())
                if match:
                    return int(match.group(1))
            return baseline
        except Exception:
            return baseline

    # ------------------------------------------------------------------
    # Deterministic fallbacks
    # ------------------------------------------------------------------

    def _deterministic_decision(
        self,
        user_profile: UserProfile,
        item_profile: ItemProfile,
        plan: ReviewPlan,
    ) -> ReviewDecision:
        rating = plan.predicted_rating
        positives = plan.positive_evidence[:3] or ["general quality"]
        negatives = plan.negative_evidence[:2] or ["nothing specific"]
        first = (
            f"the {item_profile.category} category match"
            if item_profile.category in user_profile.preferred_categories
            else f"the name and description of {item_profile.name}"
        )
        return ReviewDecision(
            predicted_rating=rating,
            rating_rationale=(
                f"Rated {rating}/5 because it aligns with the user's "
                f"{user_profile.rating_strictness} rating style and matches "
                f"preferences for {', '.join(positives[:2])}."
            ),
            thought_process=(
                f"This {item_profile.category} item looks "
                + ("promising" if rating >= 4 else "decent" if rating >= 3 else "concerning")
                + f" for someone who values {', '.join(positives[:2])}. "
                + (f"However, {', '.join(negatives[:1])} is a concern." if rating <= 3 else "The positives outweigh any negatives.")
            ),
            key_concerns=negatives[:3],
            first_notice=first,
            emotional_reaction=(
                "excited and curious" if rating >= 4
                else "cautiously optimistic" if rating >= 3
                else "sceptical and guarded"
            ),
            provider=self._provider_name,
            llm_augmented=False,
            trace=[
                {"step": "reason", "status": "fallback", "detail": "Deterministic review decision"},
            ],
        )


# ------------------------------------------------------------------
# Helpers
# ------------------------------------------------------------------

def _baseline_rating(user_profile: UserProfile, item_profile: ItemProfile) -> int:
    avg = user_profile.average_rating
    quality = item_profile.quality_score
    raw = (avg * 0.55 + quality * 5 * 0.30 + 3.0 * 0.15)
    return max(1, min(5, int(round(raw))))


def _user_persona_text(user_profile: UserProfile) -> str:
    return (
        f"Voice: {user_profile.voice_style}. "
        f"Strictness: {user_profile.rating_strictness}. "
        f"Average rating: {user_profile.average_rating:.2f}. "
        f"Category affinities: {json.dumps(user_profile.category_affinity)}. "
        f"Preferences: {', '.join(user_profile.preferred_terms[:5] or ['none'])}. "
        f"Aversions: {', '.join(user_profile.disliked_terms[:5] or ['none'])}."
    )


def _extract_json(raw: str | None) -> dict[str, Any]:
    if not raw:
        return {}
    text = raw.strip()
    match = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, re.DOTALL)
    if match:
        text = match.group(1)
    for start in re.finditer(r"\{", text):
        depth = 0
        for i in range(start.start(), len(text)):
            if text[i] == "{":
                depth += 1
            elif text[i] == "}":
                depth -= 1
                if depth == 0:
                    try:
                        return json.loads(text[start.start() : i + 1])
                    except json.JSONDecodeError:
                        break
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        return {}


def _as_str_list(value: Any) -> list[str]:
    if isinstance(value, list):
        return [str(v) for v in value]
    return [str(value)]


def _repair_review(
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
        return "- No prior examples available; infer style from the profile."
    examples = []
    ordered = sorted(history, key=lambda item: item.timestamp or 0)
    for item in ordered[-limit:]:
        review = " ".join(item.review.split())
        examples.append(f"- {item.item_name} ({item.rating:g}/5): {review[:500]}")
    return "\n".join(examples)


def _format_list(values: list[str], limit: int = 12) -> str:
    clean = [str(value).strip() for value in values if str(value).strip()]
    return "; ".join(clean[:limit]) if clean else "none"


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
