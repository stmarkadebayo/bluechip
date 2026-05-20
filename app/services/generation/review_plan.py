from __future__ import annotations

from dataclasses import dataclass

from app.models.schemas import ItemProfile, UserProfile


@dataclass(frozen=True)
class ReviewPlan:
    predicted_rating: int
    verdict: str
    voice: str
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
        + item_profile.signals[:2]
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
        locale_tone=locale_tone,
        positive_evidence=positive_evidence or item_profile.signals[:2],
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
    local = ""
    if plan.locale_tone == "natural Nigerian English":
        local = " It feels practical for a Nigerian shopper who cares about real value."
    return (
        f"I would rate {item_profile.name} {plan.predicted_rating} out of 5. "
        f"{positive.capitalize()} supports that rating, and {plan.verdict}."
        f"{tradeoff}{local}"
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
        return "it fits what I usually look for"
    if predicted_rating == 3:
        return "it has useful strengths but also a few tradeoffs"
    return "it misses too many things I care about"


def _ordered_unique(values: list[str], limit: int) -> list[str]:
    output = []
    for value in values:
        text = str(value).strip()
        if text and text not in output:
            output.append(text)
        if len(output) >= limit:
            break
    return output


def _sentence_fragment(values: list[str]) -> str:
    clean = [str(value).strip().rstrip(".") for value in values if str(value).strip()]
    if not clean:
        return ""
    if len(clean) == 1:
        return clean[0]
    return ", ".join(clean[:-1]) + ", and " + clean[-1]
