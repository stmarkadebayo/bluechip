from __future__ import annotations

from dataclasses import dataclass

from app.models.schemas import ItemProfile, UserProfile


@dataclass(frozen=True)
class ReviewPlan:
    predicted_rating: int
    verdict: str
    voice: str
    archetype: str
    locale_tone: str
    positive_evidence: list[str]
    negative_evidence: list[str]
    aspect_scores: dict[str, float]
    must_mention: list[str]

    def prompt_block(self) -> str:
        return (
            f"Predicted rating: {self.predicted_rating}/5\n"
            f"Verdict: {self.verdict}\n"
            f"Voice: {self.voice}\n"
            f"Review archetype: {self.archetype}\n"
            f"Locale tone: {self.locale_tone}\n"
            f"Positive evidence: {self.positive_evidence}\n"
            f"Negative evidence: {self.negative_evidence}\n"
            f"Aspect scores: {self.aspect_scores}\n"
            f"Must mention: {self.must_mention}"
        )


def build_review_plan(
    user_profile: UserProfile,
    item_profile: ItemProfile,
    predicted_rating: int,
) -> ReviewPlan:
    aspect_scores = _aspect_scores(user_profile, item_profile)
    positive_evidence = _ordered_unique(
        item_profile.positive_aspects
        + item_profile.nigerian_context
        + _human_evidence_from_signals(item_profile.signals)
        + user_profile.positive_aspects[:3],
        limit=6,
    )
    negative_evidence = _ordered_unique(
        [
            term
            for term in item_profile.negative_aspects + user_profile.negative_aspects
            if term not in positive_evidence
        ],
        limit=4,
    )
    locale_tone = "natural Nigerian English" if (user_profile.locale or "").lower() == "nigeria" else "plain English"
    return ReviewPlan(
        predicted_rating=predicted_rating,
        verdict=_verdict(predicted_rating),
        voice=user_profile.voice_style,
        archetype=_review_archetype(user_profile, item_profile, predicted_rating),
        locale_tone=locale_tone,
        positive_evidence=positive_evidence or _human_evidence_from_signals(item_profile.signals),
        negative_evidence=negative_evidence,
        aspect_scores=aspect_scores,
        must_mention=[item_profile.name, f"{predicted_rating} out of 5"],
    )


def fallback_review_from_plan(item_profile: ItemProfile, plan: ReviewPlan) -> str:
    positive = _sentence_fragment(plan.positive_evidence) or "the available product details"
    if plan.negative_evidence and plan.predicted_rating <= 3:
        tradeoff = " I would still watch out for " + _sentence_fragment(plan.negative_evidence) + "."
    elif plan.negative_evidence:
        tradeoff = " The only thing I would keep in mind is " + _sentence_fragment(plan.negative_evidence) + "."
    else:
        tradeoff = ""
    local = _locale_sentence(plan.locale_tone, plan.predicted_rating)
    variant = _fallback_variant(item_profile.item_id or item_profile.name, plan.predicted_rating)
    if variant == 0:
        return (
            f"{item_profile.name} feels like a {plan.predicted_rating} out of 5 for me. "
            f"{positive.capitalize()} stood out first, and {plan.verdict}.{tradeoff}{local}"
        )
    if variant == 1:
        return (
            f"I am at {plan.predicted_rating} out of 5 on {item_profile.name}. "
            f"{positive.capitalize()} is what carries it for me, though the overall call is that "
            f"{plan.verdict}.{tradeoff}{local}"
        )
    return (
        f"{positive.capitalize()} is the main reason {item_profile.name} lands at "
        f"{plan.predicted_rating} out of 5. For my taste, {plan.verdict}.{tradeoff}{local}"
    )


def _aspect_scores(user_profile: UserProfile, item_profile: ItemProfile) -> dict[str, float]:
    scores = {}
    for aspect, user_score in user_profile.aspect_scores.items():
        item_score = item_profile.aspect_scores.get(aspect, 0.0)
        if user_score or item_score:
            scores[aspect] = round((user_score + item_score) / 2, 4)
    for aspect, item_score in item_profile.aspect_scores.items():
        scores.setdefault(aspect, round(item_score / 2, 4))
    return dict(sorted(scores.items(), key=lambda row: abs(row[1]), reverse=True)[:8])


def _verdict(predicted_rating: int) -> str:
    if predicted_rating >= 4:
        return "this would probably work for me"
    if predicted_rating == 3:
        return "I see the useful parts, but I am not fully sold"
    return "I would be cautious about it"


def _review_archetype(
    user_profile: UserProfile,
    item_profile: ItemProfile,
    predicted_rating: int,
) -> str:
    del item_profile
    voice = user_profile.voice_style.lower()
    strict = user_profile.rating_strictness.lower()
    positive_terms = {
        term.lower()
        for term in user_profile.positive_aspects
        + user_profile.preferred_terms
        + user_profile.recent_terms
    }
    if predicted_rating <= 2 or user_profile.negative_rating_share >= 0.35:
        return "CRITICAL"
    if user_profile.review_length_mean and user_profile.review_length_mean <= 14:
        return "SHORT_DIRECT"
    if "analytical" in voice or "strict" in strict or user_profile.rating_std >= 1.15:
        return "DETAILED_ANALYTICAL"
    if predicted_rating >= 5 and positive_terms.intersection({"love", "amazing", "excellent", "perfect"}):
        return "HYPE_REVIEWER"
    if predicted_rating >= 4 and positive_terms.intersection({"beautiful", "comfortable", "friendly", "fresh"}):
        return "EMOTIONAL"
    return "CASUAL"


def _ordered_unique(values: list[str], limit: int) -> list[str]:
    output = []
    for value in values:
        text = str(value).strip()
        if text and text not in output:
            output.append(text)
        if len(output) >= limit:
            break
    return output


def _human_evidence_from_signals(signals: list[str]) -> list[str]:
    evidence = []
    for signal in signals:
        text = str(signal).strip()
        lower = text.lower()
        if not text:
            continue
        if lower.startswith(("category:", "quality score:", "average rating:", "review count:")):
            continue
        evidence.append(text)
    return evidence[:3]


def _sentence_fragment(values: list[str]) -> str:
    clean = [str(value).strip().rstrip(".") for value in values if str(value).strip()]
    if not clean:
        return ""
    if len(clean) == 1:
        return clean[0]
    return ", ".join(clean[:-1]) + ", and " + clean[-1]


def _fallback_variant(seed: str, rating: int) -> int:
    return (sum(ord(char) for char in seed) + rating) % 3


def _locale_sentence(locale_tone: str, rating: int) -> str:
    if locale_tone != "natural Nigerian English":
        return ""
    if rating >= 4:
        return " As a Nigerian shopper, I would still call it practical if the price stays fair."
    if rating == 3:
        return " As a Nigerian shopper, I would compare price and delivery before buying again."
    return " As a Nigerian shopper, I would not ignore those value concerns."
