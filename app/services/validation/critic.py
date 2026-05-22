from __future__ import annotations

from app.models.schemas import ItemProfile, UserProfile, ValidationResult
from app.services.validation.evidence_critic import review_evidence_issues


def validate_review_simulation(
    predicted_rating: int,
    review: str,
    user_profile: UserProfile,
    item_profile: ItemProfile,
) -> ValidationResult:
    issues = []

    if str(predicted_rating) not in review:
        issues.append("review does not naturally include the predicted rating")

    if item_profile.name.lower() not in review.lower():
        issues.append("review does not mention the target item")

    if predicted_rating <= 2 and any(word in review.lower() for word in ["excellent", "perfect"]):
        issues.append("positive review language conflicts with low predicted rating")

    if predicted_rating >= 4 and any(word in review.lower() for word in ["terrible", "awful"]):
        issues.append("negative review language conflicts with high predicted rating")

    if (
        user_profile.locale
        and user_profile.locale.lower() == "nigeria"
        and not _has_nigerian_marker(review)
    ):
        issues.append("locale was requested but not reflected in the generated review")

    issues.extend(review_evidence_issues(review, user_profile, item_profile))

    return ValidationResult(is_consistent=not issues, issues=issues)


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
