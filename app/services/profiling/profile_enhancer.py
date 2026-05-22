from __future__ import annotations

import json
import re
from typing import Any

from app.models.schemas import ProfileEnhancement, UserHistoryItem, UserProfile
from app.services.generation.providers import (
    TemplateGenerationProvider,
    generation_provider_name,
    get_generation_provider,
)


class ProfileEnhancer:
    """Bounded LLM profile enrichment.

    The deterministic profile remains the source of truth. This enhancer may
    only add a small number of inferred terms/aspects/categories and a bounded
    confidence boost. It never removes deterministic evidence.
    """

    MAX_NEW_VALUES = 3
    MAX_LIST_SIZE = 10
    MAX_CONFIDENCE_BOOST = 0.08

    def enhance(
        self,
        profile: UserProfile,
        persona: str,
        history: list[UserHistoryItem],
        locale: str | None = None,
    ) -> UserProfile:
        provider = get_generation_provider()
        provider_name = generation_provider_name()
        if isinstance(provider, TemplateGenerationProvider):
            return _with_fallback(profile, provider_name, "template provider")

        prompt = _build_prompt(profile=profile, persona=persona, history=history, locale=locale)
        try:
            raw = provider.generate(
                instructions=(
                    "You enrich consumer profiles for a recommendation and review simulation "
                    "system. Return only valid JSON. Do not invent item facts."
                ),
                prompt=prompt,
            )
            parsed = _extract_json(raw)
        except Exception as exc:
            return _with_fallback(profile, provider_name, str(exc)[:160])

        if not parsed:
            return _with_fallback(profile, provider_name, "empty or invalid JSON")

        return self._merge(profile, parsed, provider_name)

    def _merge(self, profile: UserProfile, payload: dict[str, Any], provider_name: str) -> UserProfile:
        updates: dict[str, Any] = {}
        added_terms: dict[str, list[str]] = {}
        adjusted_fields: list[str] = []

        for field in (
            "preferred_terms",
            "disliked_terms",
            "preferred_categories",
            "positive_aspects",
            "negative_aspects",
            "nigerian_context",
        ):
            existing = list(getattr(profile, field))
            additions = _bounded_additions(existing, payload.get(field))
            if additions:
                updates[field] = _merge_values(existing, additions, self.MAX_LIST_SIZE)
                added_terms[field] = additions
                adjusted_fields.append(field)

        voice_style = _clean_phrase(payload.get("voice_style"))
        if voice_style and voice_style.lower() != profile.voice_style.lower():
            updates["voice_style"] = _merge_voice_style(profile.voice_style, voice_style)
            adjusted_fields.append("voice_style")

        category_affinity = dict(profile.category_affinity)
        for category in added_terms.get("preferred_categories", []):
            category_affinity.setdefault(category, 0.15)
        if category_affinity != profile.category_affinity:
            updates["category_affinity"] = category_affinity
            adjusted_fields.append("category_affinity")

        llm_confidence = _bounded_float(payload.get("confidence"), default=0.5)
        if adjusted_fields:
            updates["confidence"] = round(
                min(profile.confidence + self.MAX_CONFIDENCE_BOOST * llm_confidence, 0.98),
                2,
            )

        rationale = _clean_phrase(payload.get("rationale")) or "LLM inferred bounded profile enrichments."
        signals = list(profile.signals)
        if adjusted_fields:
            signals.append(
                "LLM profile enhancement added: " + ", ".join(adjusted_fields[:5])
            )
        price_sensitivity = _clean_phrase(payload.get("price_sensitivity"))
        if price_sensitivity:
            signals.append(f"LLM inferred price sensitivity: {price_sensitivity}")
        updates["signals"] = signals[: len(profile.signals) + 3]
        updates["profile_enhancement"] = ProfileEnhancement(
            provider=provider_name,
            llm_augmented=bool(adjusted_fields),
            confidence=llm_confidence if adjusted_fields else 0.0,
            added_terms=added_terms,
            adjusted_fields=adjusted_fields,
            rationale=rationale,
            fallback_reason=None if adjusted_fields else "no bounded additions accepted",
        )
        return profile.model_copy(update=updates)


def _build_prompt(
    profile: UserProfile,
    persona: str,
    history: list[UserHistoryItem],
    locale: str | None,
) -> str:
    history_rows = [
        {
            "item": item.item_name,
            "category": item.category,
            "rating": item.rating,
            "review": item.review[:240],
        }
        for item in history[-8:]
    ]
    return (
        "Enrich this deterministic user profile with only high-confidence inferred fields.\n"
        "Do not repeat existing values unless needed. Do not remove anything.\n\n"
        f"Persona: {persona}\n"
        f"Locale: {locale or profile.locale or 'unspecified'}\n"
        f"Deterministic profile: {profile.model_dump(exclude={'embedding', 'profile_enhancement'})}\n"
        f"Recent history: {json.dumps(history_rows, ensure_ascii=True)}\n\n"
        "Return JSON with optional keys: preferred_terms, disliked_terms, preferred_categories, "
        "positive_aspects, negative_aspects, nigerian_context, voice_style, price_sensitivity, "
        "confidence, rationale. Keep every list to at most 5 short lowercase strings."
    )


def _with_fallback(profile: UserProfile, provider_name: str, reason: str) -> UserProfile:
    return profile.model_copy(
        update={
            "profile_enhancement": ProfileEnhancement(
                provider=provider_name,
                llm_augmented=False,
                confidence=0.0,
                added_terms={},
                adjusted_fields=[],
                rationale="Deterministic profile retained.",
                fallback_reason=reason,
            )
        }
    )


def _bounded_additions(existing: list[str], raw_values: Any) -> list[str]:
    if not isinstance(raw_values, list):
        return []
    existing_normalized = {_normalize(value) for value in existing}
    additions = []
    for value in raw_values:
        cleaned = _clean_token(value)
        if not cleaned or cleaned in existing_normalized or cleaned in additions:
            continue
        additions.append(cleaned)
        if len(additions) >= ProfileEnhancer.MAX_NEW_VALUES:
            break
    return additions


def _merge_values(existing: list[str], additions: list[str], limit: int) -> list[str]:
    merged = list(existing)
    for value in additions:
        if value not in merged:
            merged.append(value)
    return merged[:limit]


def _merge_voice_style(existing: str, addition: str) -> str:
    if addition.lower() in existing.lower():
        return existing
    merged = f"{existing}; LLM notes {addition}"
    return merged[:160]


def _clean_token(value: Any) -> str:
    text = str(value or "").strip().lower()
    text = re.sub(r"[^a-z0-9_ -]", "", text)
    text = re.sub(r"\s+", "_", text)
    if len(text) < 3 or len(text) > 32:
        return ""
    return text


def _clean_phrase(value: Any) -> str:
    text = str(value or "").strip()
    text = re.sub(r"\s+", " ", text)
    return text[:160]


def _normalize(value: str) -> str:
    return _clean_token(value)


def _bounded_float(value: Any, default: float) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return default
    return round(min(max(number, 0.0), 1.0), 3)


def _extract_json(raw: str | None) -> dict[str, Any]:
    if not raw:
        return {}
    text = raw.strip()
    match = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, re.DOTALL)
    if match:
        text = match.group(1)
    for start in re.finditer(r"\{", text):
        depth = 0
        for index in range(start.start(), len(text)):
            if text[index] == "{":
                depth += 1
            elif text[index] == "}":
                depth -= 1
                if depth == 0:
                    try:
                        return json.loads(text[start.start() : index + 1])
                    except json.JSONDecodeError:
                        break
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        return {}
