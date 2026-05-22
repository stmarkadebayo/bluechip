from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from typing import Any

from app.models.schemas import Item, RecommendationItem, UserProfile
from app.services.generation.providers import (
    TemplateGenerationProvider,
    get_generation_provider,
    generation_provider_name,
)
from app.services.agentic.reasoner import get_reasoner


@dataclass
class PreferenceAnalysis:
    """LLM-analysed preference decomposition."""

    core_preferences: list[str]
    contextual_needs: list[str]
    price_tolerance: str
    quality_priority: str
    category_affinities: dict[str, float]
    exploration_openness: str
    provider: str = ""
    llm_augmented: bool = False
    trace: list[dict[str, str]] = field(default_factory=list)


@dataclass
class RerankedItem:
    """LLM re-ranked item with reasoning."""

    item: Item
    rank: int
    score: float
    reasoning: str
    tradeoffs: str
    explanation: str
    provider: str = ""
    llm_augmented: bool = False


@dataclass
class ColdStartInference:
    """LLM-inferred preferences from a cold-start persona."""

    preferred_categories: list[str]
    price_sensitivity: str
    quality_expectation: str
    likely_voice_style: str
    key_terms: list[str]
    confidence: float
    provider: str = ""
    llm_augmented: bool = False


@dataclass
class CrossDomainTransfer:
    """LLM-mediated cross-domain preference transfer."""

    transferred_preferences: list[str]
    mapped_categories: list[str]
    adjusted_terms: list[str]
    confidence: float
    reasoning: str
    provider: str = ""
    llm_augmented: bool = False


class RecommenderReasoner:
    """LLM-driven recommendation reasoning for Task B.

    Analyses user preferences, re-ranks candidates with per-item reasoning,
    generates personalised explanations, handles cold-start inference, and
    transfers preferences across domains.  All methods fall back to
    deterministic behaviour.
    """

    def __init__(self) -> None:
        self._provider = get_generation_provider()
        self._provider_name = generation_provider_name()
        self._reasoner = get_reasoner()

    # ------------------------------------------------------------------
    # Preference reasoning
    # ------------------------------------------------------------------

    def reason_about_preferences(
        self,
        user_profile: UserProfile,
        context: str,
    ) -> PreferenceAnalysis:
        """LLM analyses what this user truly wants given the context."""
        prompt = (
            "Analyse this user in context to determine what they truly want.\n\n"
            f"User:\n"
            f"  Voice style: {user_profile.voice_style}\n"
            f"  Rating strictness: {user_profile.rating_strictness}\n"
            f"  Average rating: {user_profile.average_rating:.2f}\n"
            f"  Preferred terms: {', '.join(user_profile.preferred_terms[:8] or ['none'])}\n"
            f"  Disliked terms: {', '.join(user_profile.disliked_terms[:8] or ['none'])}\n"
            f"  Preferred categories: {', '.join(user_profile.preferred_categories[:5] or ['none'])}\n"
            f"  Category affinity: {json.dumps(user_profile.category_affinity)}\n"
            f"  Positive aspects: {', '.join(user_profile.positive_aspects[:6] or ['none'])}\n"
            f"  Negative sensitivities: {', '.join(user_profile.negative_aspects[:6] or ['none'])}\n"
            f"  Confidence: {user_profile.confidence:.2f}\n"
            f"  Locale: {user_profile.locale or 'unspecified'}\n\n"
            f"Context: {context or 'no specific context'}\n\n"
            f"Return a JSON object:\n"
            f'  "core_preferences": list of 3-5 things this user consistently values\n'
            f'  "contextual_needs": what the context implies they need now\n'
            f'  "price_tolerance": how price-sensitive they are (low/medium/high)\n'
            f'  "quality_priority": whether quality matters more than price (yes/no/balanced)\n'
            f'  "exploration_openness": willingness to try new things (low/medium/high)\n'
            f"Respond ONLY with valid JSON."
        )
        system = "You analyse consumer preferences contextually. Return only JSON."

        if isinstance(self._provider, TemplateGenerationProvider):
            return self._deterministic_preferences(user_profile, context)

        try:
            raw = self._provider.generate(instructions=system, prompt=prompt)
            parsed = _extract_json(raw) if raw else {}
        except Exception:
            parsed = {}

        if not parsed:
            return self._deterministic_preferences(user_profile, context)

        return PreferenceAnalysis(
            core_preferences=_as_str_list(parsed.get("core_preferences", [])),
            contextual_needs=_as_str_list(parsed.get("contextual_needs", [])),
            price_tolerance=str(parsed.get("price_tolerance", "medium")),
            quality_priority=str(parsed.get("quality_priority", "balanced")),
            category_affinities=user_profile.category_affinity,
            exploration_openness=str(parsed.get("exploration_openness", "medium")),
            provider=self._provider_name,
            llm_augmented=True,
            trace=[
                {"step": "preference_reasoning", "status": "ok", "detail": "LLM analysed preferences"},
            ],
        )

    # ------------------------------------------------------------------
    # Candidate re-ranking
    # ------------------------------------------------------------------

    def rerank_candidates(
        self,
        user_profile: UserProfile,
        context: str,
        candidates: list[RecommendationItem],
        limit: int = 5,
    ) -> list[RerankedItem]:
        """LLM re-ranks candidates with per-item reasoning.

        Falls back to score-based sorting when the LLM is unavailable.
        """
        if not candidates:
            return []

        if isinstance(self._provider, TemplateGenerationProvider):
            return self._deterministic_rerank(candidates, limit)

        if len(candidates) <= 3:
            return self._rerank_small_batch(user_profile, context, candidates, limit)

        # Use LLM for deeper re-ranking on the top subset
        top_n = min(len(candidates), max(limit * 3, 10))
        top_candidates = sorted(candidates, key=lambda c: c.score, reverse=True)[:top_n]

        items_text = "\n".join(
            f"  [{idx}] {c.name} (category: {c.name.split()[-1] if ' ' in c.name else 'various'})"
            f" | score: {c.score:.2f}"
            f" | signals: {', '.join(c.matched_signals[:3])}"
            for idx, c in enumerate(top_candidates)
        )

        prompt = (
            "Re-rank these recommendation candidates for this specific user.\n\n"
            f"User signals: {', '.join(user_profile.signals[:5])}\n"
            f"Context: {context or 'none'}\n"
            f"Candidates:\n{items_text}\n\n"
            f"Return a JSON object with key 'rankings': an array of objects, each with:\n"
            f'  "index": the original [idx] number\n'
            f'  "new_rank": new rank position (1-based)\n'
            f'  "reasoning": one sentence why this position\n'
            f"Only include up to {limit} items. Respond ONLY with valid JSON."
        )
        system = "You re-rank product recommendations. Return only valid JSON."

        try:
            raw = self._provider.generate(instructions=system, prompt=prompt)
            parsed = _extract_json(raw) if raw else {}
        except Exception:
            parsed = {}

        rankings = parsed.get("rankings", []) if parsed else []
        if not rankings:
            return self._deterministic_rerank(candidates, limit)

        index_map = {idx: c for idx, c in enumerate(top_candidates)}
        reranked = []
        for entry in rankings:
            idx = entry.get("index", 0)
            candidate = index_map.get(idx)
            if candidate and len(reranked) < limit:
                # Build a lightweight Item from the RecommendationItem
                item = Item(
                    item_id=candidate.item_id,
                    name=candidate.name,
                    category=candidate.name.split()[-1] if " " in candidate.name else "unknown",
                )
                reranked.append(RerankedItem(
                    item=item,
                    rank=entry.get("new_rank", len(reranked) + 1),
                    score=candidate.score,
                    reasoning=str(entry.get("reasoning", "")),
                    tradeoffs=candidate.tradeoffs,
                    explanation="",
                    provider=self._provider_name,
                    llm_augmented=True,
                ))

        if len(reranked) < limit:
            for candidate in candidates:
                if not any(r.item.item_id == candidate.item_id for r in reranked):
                    item = Item(
                        item_id=candidate.item_id,
                        name=candidate.name,
                        category=candidate.name.split()[-1] if " " in candidate.name else "unknown",
                    )
                    reranked.append(RerankedItem(
                        item=item,
                        rank=len(reranked) + 1,
                        score=candidate.score,
                        reasoning="Score-based fallback position.",
                        tradeoffs=candidate.tradeoffs,
                        explanation="",
                        provider=self._provider_name,
                        llm_augmented=False,
                    ))
                    if len(reranked) >= limit:
                        break

        reranked.sort(key=lambda r: r.rank)
        return reranked[:limit]

    def _rerank_small_batch(
        self,
        user_profile: UserProfile,
        context: str,
        candidates: list[RecommendationItem],
        limit: int,
    ) -> list[RerankedItem]:
        reranked = []
        for idx, candidate in enumerate(candidates[:limit]):
            reasoning = self._reason_single_item(user_profile, context, candidate)
            item = Item(
                item_id=candidate.item_id,
                name=candidate.name,
                category=candidate.name.split()[-1] if " " in candidate.name else "unknown",
            )
            reranked.append(RerankedItem(
                item=item,
                rank=idx + 1,
                score=candidate.score,
                reasoning=reasoning,
                tradeoffs=candidate.tradeoffs,
                explanation="",
                provider=self._provider_name,
                llm_augmented=True,
            ))
        return reranked

    def _reason_single_item(
        self,
        user_profile: UserProfile,
        context: str,
        candidate: RecommendationItem,
    ) -> str:
        try:
            prompt = (
                f"Explain in one sentence why {candidate.name} fits user with "
                f"preferences: {', '.join(user_profile.preferred_terms[:3] or ['general quality'])}. "
                f"Context: {context or 'none'}. "
                f"Item signals: {', '.join(candidate.matched_signals[:3] or ['various'])}."
            )
            raw = self._provider.generate(
                instructions="One-sentence explanation.", prompt=prompt,
            )
            return raw.strip() if raw else f"Matches user preferences with score {candidate.score:.2f}."
        except Exception:
            return f"Matches user preferences with score {candidate.score:.2f}."

    # ------------------------------------------------------------------
    # Explanation generation
    # ------------------------------------------------------------------

    def generate_recommendation_explanation(
        self,
        user_profile: UserProfile,
        item: Item,
        context: str,
        reasoning: str = "",
    ) -> str:
        """Generate a personal, convincing explanation for a recommendation."""
        fallback = (
            f"{item.name} ranks well because it matches your preferences "
            f"for {', '.join(user_profile.preferred_terms[:3] or ['quality and value'])}. "
            f"{'Considered in context: ' + context + '.' if context else ''}"
        )

        if isinstance(self._provider, TemplateGenerationProvider):
            return fallback

        instructions = (
            "You explain recommendations personally and persuasively. "
            "Use the user's stated preferences. Be concise (1-2 sentences)."
        )
        prompt = (
            f"User preferences: {', '.join(user_profile.preferred_terms[:5] or ['quality'])}\n"
            f"User style: {user_profile.voice_style}\n"
            f"Item: {item.name} ({item.category})\n"
            f"Context: {context or 'none'}\n"
            f"Reasoning: {reasoning or 'strong preference match'}\n"
            f"Generate a personal explanation for why this item is recommended."
        )
        try:
            generated = self._provider.generate(instructions=instructions, prompt=prompt)
            if generated:
                return generated.strip()
            return fallback
        except Exception:
            return fallback

    # ------------------------------------------------------------------
    # Cold start
    # ------------------------------------------------------------------

    def handle_cold_start(self, persona: str) -> ColdStartInference:
        """LLM infers preferences from a persona description alone."""
        if isinstance(self._provider, TemplateGenerationProvider):
            return self._deterministic_cold_start(persona)

        prompt = (
            "You are inferring a user's likely preferences from their persona description "
            "for a recommendation system.\n\n"
            f"Persona: {persona}\n\n"
            "Return a JSON object:\n"
            '  "preferred_categories": list of 3-5 likely category interests\n'
            '  "price_sensitivity": low, medium, or high\n'
            '  "quality_expectation": low, medium, or high\n'
            '  "likely_voice_style": how they would write reviews\n'
            '  "key_terms": list of 5-8 terms they would care about\n'
            '  "confidence": 0-1 float for how confident this inference is\n'
            "Respond ONLY with valid JSON."
        )
        system = "You infer user preferences from descriptions. Return only JSON."

        try:
            raw = self._provider.generate(instructions=system, prompt=prompt)
            parsed = _extract_json(raw) if raw else {}
        except Exception:
            parsed = {}

        if not parsed:
            return self._deterministic_cold_start(persona)

        return ColdStartInference(
            preferred_categories=_as_str_list(parsed.get("preferred_categories", [])),
            price_sensitivity=str(parsed.get("price_sensitivity", "medium")),
            quality_expectation=str(parsed.get("quality_expectation", "medium")),
            likely_voice_style=str(parsed.get("likely_voice_style", "concise")),
            key_terms=_as_str_list(parsed.get("key_terms", [])),
            confidence=float(parsed.get("confidence", 0.5)),
            provider=self._provider_name,
            llm_augmented=True,
        )

    # ------------------------------------------------------------------
    # Cross-domain transfer
    # ------------------------------------------------------------------

    def handle_cross_domain(
        self,
        source_domain: str,
        target_domain: str,
        preferences: list[str],
    ) -> CrossDomainTransfer:
        """LLM transfers preferences across domains (e.g. food -> beauty)."""
        if isinstance(self._provider, TemplateGenerationProvider):
            return self._deterministic_cross_domain(source_domain, target_domain, preferences)

        prompt = (
            f"Map preferences from {source_domain} to {target_domain}.\n\n"
            f"Source preferences: {', '.join(preferences[:8])}\n\n"
            f"Return a JSON object:\n"
            f'  "transferred_preferences": list of mapped preference terms for {target_domain}\n'
            f'  "mapped_categories": list of {target_domain} subcategories that match\n'
            f'  "adjusted_terms": list of domain-appropriate search terms\n'
            f'  "confidence": 0-1 float for transfer confidence\n'
            f'  "reasoning": one sentence explaining the mapping logic\n'
            f"Respond ONLY with valid JSON."
        )
        system = "You map preferences across product domains. Return only JSON."

        try:
            raw = self._provider.generate(instructions=system, prompt=prompt)
            parsed = _extract_json(raw) if raw else {}
        except Exception:
            parsed = {}

        if not parsed:
            return self._deterministic_cross_domain(source_domain, target_domain, preferences)

        return CrossDomainTransfer(
            transferred_preferences=_as_str_list(parsed.get("transferred_preferences", [])),
            mapped_categories=_as_str_list(parsed.get("mapped_categories", [])),
            adjusted_terms=_as_str_list(parsed.get("adjusted_terms", [])),
            confidence=float(parsed.get("confidence", 0.5)),
            reasoning=str(parsed.get("reasoning", "")),
            provider=self._provider_name,
            llm_augmented=True,
        )

    # ------------------------------------------------------------------
    # Deterministic fallbacks
    # ------------------------------------------------------------------

    def _deterministic_preferences(
        self,
        user_profile: UserProfile,
        context: str,
    ) -> PreferenceAnalysis:
        context_needs = _extract_context_terms(context)
        return PreferenceAnalysis(
            core_preferences=user_profile.preferred_terms[:5] or ["quality", "value"],
            contextual_needs=context_needs or ["general-purpose recommendation"],
            price_tolerance=(
                "high" if user_profile.average_rating > 4.0
                else "low" if user_profile.average_rating < 3.2
                else "medium"
            ),
            quality_priority="yes" if user_profile.average_rating <= 3.5 else "balanced",
            category_affinities=user_profile.category_affinity,
            exploration_openness=(
                "high" if user_profile.rating_std > 1.0
                else "low" if len(user_profile.preferred_categories) <= 1
                else "medium"
            ),
            provider=self._provider_name,
            llm_augmented=False,
            trace=[
                {"step": "preference_reasoning", "status": "fallback",
                 "detail": "Deterministic preference analysis"},
            ],
        )

    def _deterministic_rerank(
        self,
        candidates: list[RecommendationItem],
        limit: int,
    ) -> list[RerankedItem]:
        sorted_candidates = sorted(candidates, key=lambda c: c.score, reverse=True)
        return [
            RerankedItem(
                item=Item(
                    item_id=c.item_id,
                    name=c.name,
                    category=c.name.split()[-1] if " " in c.name else "unknown",
                ),
                rank=idx + 1,
                score=c.score,
                reasoning=f"Ranked by score ({c.score:.2f}) with deterministic ordering.",
                tradeoffs=c.tradeoffs,
                explanation="",
                provider=self._provider_name,
                llm_augmented=False,
            )
            for idx, c in enumerate(sorted_candidates[:limit])
        ]

    def _deterministic_cold_start(self, persona: str) -> ColdStartInference:
        words = re.findall(r"[a-zA-Z]{3,}", persona.lower())
        common_categories = {
            "beauty": ["All_Beauty"],
            "hair": ["All_Beauty"],
            "music": ["Digital_Music"],
            "food": ["Grocery_Gourmet_Food"],
            "book": ["Books"],
            "restaurant": ["Restaurants"],
            "grocery": ["Grocery_Gourmet_Food"],
            "student": ["Books", "All_Beauty", "Grocery_Gourmet_Food"],
            "skin": ["All_Beauty"],
            "makeup": ["All_Beauty"],
        }
        categories = []
        for word in words:
            if word in common_categories:
                categories.extend(common_categories[word])
        categories = list(dict.fromkeys(categories))[:5] or ["All_Beauty"]

        return ColdStartInference(
            preferred_categories=categories,
            price_sensitivity="medium",
            quality_expectation="medium",
            likely_voice_style="concise and preference-focused",
            key_terms=[w for w in words if len(w) > 3][:8],
            confidence=min(0.15 + len(words) * 0.02, 0.60),
            provider=self._provider_name,
            llm_augmented=False,
        )

    def _deterministic_cross_domain(
        self,
        source_domain: str,
        target_domain: str,
        preferences: list[str],
    ) -> CrossDomainTransfer:
        return CrossDomainTransfer(
            transferred_preferences=preferences[:5],
            mapped_categories=[target_domain],
            adjusted_terms=preferences[:5],
            confidence=0.30,
            reasoning=f"Direct term transfer from {source_domain} to {target_domain} via deterministic mapping.",
            provider=self._provider_name,
            llm_augmented=False,
        )


# ------------------------------------------------------------------
# Helpers
# ------------------------------------------------------------------

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
        return [str(v) for v in value if str(v).strip()]
    if isinstance(value, str):
        return [value] if value.strip() else []
    return []


def _extract_context_terms(context: str) -> list[str]:
    if not context:
        return []
    words = re.findall(r"[a-zA-Z]{3,}", context.lower())
    stop = {"the", "and", "for", "with", "that", "this", "was", "were", "from",
            "they", "their", "would", "could", "about", "based", "item", "items",
            "product", "products", "review", "reviews", "very", "really"}
    return list(dict.fromkeys(w for w in words if w not in stop))[:6]
