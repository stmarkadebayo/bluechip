from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path

from app.models.schemas import Item


GIFT_CATEGORIES = {"For Him", "For Her", "Gift Cards", "Restaurants", "Specialty Cards"}


@dataclass(frozen=True)
class ContextIntentRule:
    name: str
    priority: int
    trigger_terms: frozenset[str]
    item_terms: frozenset[str]
    category_hint: str
    boost: float = 0.32
    penalty: float = 0.55
    penalty_category: str | None = None


@dataclass(frozen=True)
class CategoryHintRule:
    category: str
    priority: int
    terms: frozenset[str]


FALLBACK_INTENT_RULES: tuple[ContextIntentRule, ...] = (
    ContextIntentRule(
        name="hair",
        priority=100,
        trigger_terms=frozenset({"hair", "hairspray", "shampoo", "styling", "style", "wig"}),
        item_terms=frozenset(
            {
                "body_wave",
                "brush",
                "comb",
                "conditioner",
                "curl",
                "extension",
                "extensions",
                "hair",
                "hairbrush",
                "hairspray",
                "scalp",
                "shampoo",
                "spray",
                "styling",
                "texturizing",
                "weave",
                "wig",
            }
        ),
        category_hint="All_Beauty",
        penalty_category="All_Beauty",
    ),
    ContextIntentRule(
        name="skin",
        priority=100,
        trigger_terms=frozenset({"face", "facial", "gentle", "serum", "skin", "skincare"}),
        item_terms=frozenset(
            {
                "cleanser",
                "cream",
                "face",
                "facial",
                "gentle",
                "lotion",
                "moisturizer",
                "serum",
                "skin",
                "skincare",
                "sunscreen",
            }
        ),
        category_hint="All_Beauty",
        penalty_category="All_Beauty",
    ),
    ContextIntentRule(
        name="nail",
        priority=100,
        trigger_terms=frozenset({"manicure", "nail", "nails", "polish"}),
        item_terms=frozenset({"acrylic", "gel", "manicure", "nail", "nails", "polish"}),
        category_hint="All_Beauty",
        penalty_category="All_Beauty",
    ),
    ContextIntentRule(
        name="gift",
        priority=70,
        trigger_terms=frozenset({"gift", "gifting", "low-risk"}),
        item_terms=frozenset(),
        category_hint="gift",
    ),
)


FALLBACK_CATEGORY_HINT_RULES: tuple[CategoryHintRule, ...] = (
    CategoryHintRule("gift", 100, frozenset({"gift", "gifting", "low-risk"})),
    CategoryHintRule(
        "All_Beauty",
        80,
        frozenset(
            {
                "beauty",
                "hair",
                "makeup",
                "manicure",
                "nail",
                "skincare",
                "skin",
                "styling",
            }
        ),
    ),
    CategoryHintRule(
        "Digital_Music",
        80,
        frozenset({"music", "playlist", "replay", "song", "songs"}),
    ),
)

def context_category_hint(context_terms: list[str]) -> str | None:
    terms = set(context_terms)
    matching_rules = [
        rule
        for rule in CATEGORY_HINT_RULES
        if terms & rule.terms
    ]
    if matching_rules:
        return max(matching_rules, key=lambda rule: rule.priority).category
    return None


def context_intent_rule(context_terms: list[str]) -> ContextIntentRule | None:
    terms = set(context_terms)
    matching_rules = [
        rule
        for rule in INTENT_RULES
        if terms & rule.trigger_terms
    ]
    return max(matching_rules, key=lambda rule: rule.priority) if matching_rules else None


def context_intent_boost(context_terms: list[str], item: Item) -> float:
    rule = context_intent_rule(context_terms)
    if rule is None:
        return 0.0
    return rule.boost if item_matches_intent(item, rule) else 0.0


def context_intent_penalty(context_terms: list[str], item: Item) -> float:
    rule = context_intent_rule(context_terms)
    if rule is None:
        return 0.0
    if rule.name == "gift":
        return 0.0 if is_gift_category(item.category) else rule.penalty
    if rule.penalty_category and item.category != rule.penalty_category:
        return 0.0
    return 0.0 if item_matches_intent(item, rule) else rule.penalty


def item_matches_intent(item: Item, rule: ContextIntentRule) -> bool:
    if rule.name == "gift":
        return is_gift_category(item.category)
    return bool(rule.item_terms & item_text_terms(item))


def item_text_terms(item: Item) -> set[str]:
    text = " ".join(
        [
            item.name,
            item.category,
            item.summary,
            " ".join(str(value) for value in item.metadata.values()),
        ]
    ).lower()
    return set(re.findall(r"[a-zA-Z][a-zA-Z0-9_-]{2,}", text))


def is_gift_category(category: str) -> bool:
    return category in GIFT_CATEGORIES


def _load_context_intent_rules() -> tuple[
    tuple[ContextIntentRule, ...],
    tuple[CategoryHintRule, ...],
]:
    path = Path(__file__).with_suffix(".json")
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return FALLBACK_INTENT_RULES, FALLBACK_CATEGORY_HINT_RULES
    try:
        intent_rules = tuple(
            sorted(
                (
                    ContextIntentRule(
                        name=str(rule["name"]),
                        priority=int(rule.get("priority", 0)),
                        trigger_terms=frozenset(str(term) for term in rule["trigger_terms"]),
                        item_terms=frozenset(str(term) for term in rule.get("item_terms", [])),
                        category_hint=str(rule["category_hint"]),
                        boost=float(rule.get("boost", 0.32)),
                        penalty=float(rule.get("penalty", 0.55)),
                        penalty_category=(
                            str(rule["penalty_category"])
                            if rule.get("penalty_category") is not None
                            else None
                        ),
                    )
                    for rule in payload["intents"]
                ),
                key=lambda rule: rule.priority,
                reverse=True,
            )
        )
        category_rules = tuple(
            sorted(
                (
                    CategoryHintRule(
                        category=str(rule["category"]),
                        priority=int(rule.get("priority", 0)),
                        terms=frozenset(str(term) for term in rule["terms"]),
                    )
                    for rule in payload["category_hints"]
                ),
                key=lambda rule: rule.priority,
                reverse=True,
            )
        )
    except (KeyError, TypeError, ValueError):
        return FALLBACK_INTENT_RULES, FALLBACK_CATEGORY_HINT_RULES
    return intent_rules, category_rules


INTENT_RULES, CATEGORY_HINT_RULES = _load_context_intent_rules()
