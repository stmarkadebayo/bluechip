from __future__ import annotations


def context_for_task_b_row(row: dict, history: list) -> str:
    category = row.get("category") or "products"
    positive_text = " ".join(
        f"{item.item_name} {item.review}" for item in history if item.rating >= 4
    ).lower()
    if category == "All_Beauty":
        if any(term in positive_text for term in ("skin", "serum", "face", "cream")):
            return (
                "Needs a beauty product for a gentle skincare routine; "
                "avoid harsh-feeling items."
            )
        if any(term in positive_text for term in ("hair", "wig", "shampoo", "brush")):
            return "Needs a practical hair or styling product for regular use."
        if any(term in positive_text for term in ("nail", "manicure", "polish")):
            return "Needs a nail-care or manicure product that feels useful, not gimmicky."
        return "Needs a practical beauty item aligned with recent positive purchases."
    if category == "Digital_Music":
        return "Wants music that fits the user's recent taste and is easy to replay."
    if "Card" in category or category in {"Restaurants", "For Him", "For Her", "Specialty Cards"}:
        return "Needs a low-risk gift option that matches recent gifting behavior."
    return f"Needs a recommendation in or near {category} that fits prior positive reviews."
