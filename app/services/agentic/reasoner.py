from __future__ import annotations

import json
import re
from collections import Counter
from typing import Any

from app.services.generation.providers import (
    GenerationProvider,
    TemplateGenerationProvider,
    generation_provider_name,
    get_generation_provider,
)

_STOPWORDS = {
    "about", "after", "also", "and", "are", "based", "because", "been", "but",
    "can", "could", "does", "doesn", "for", "from", "had", "has", "have",
    "her", "him", "his", "how", "into", "its", "item", "items", "just",
    "like", "much", "not", "only", "out", "product", "products", "really",
    "review", "reviews", "she", "that", "the", "their", "them", "then",
    "these", "they", "this", "very", "was", "were", "what", "when",
    "will", "with", "would", "you", "your",
}

_RATING_MARKERS = {
    "always", "usually", "typically", "tend", "often", "rarely", "never",
    "generally", "normally", "frequently", "consistently",
}

_EMOTION_MARKERS = {
    "disappointed", "excited", "frustrated", "happy", "love", "loved",
    "pleased", "regret", "satisfied", "surprised", "upset", "wow",
}

_WRITING_MARKERS = {
    "brief", "detailed", "direct", "elaborate", "short", "long",
    "concrete", "abstract", "analytical", "emotional", "practical",
}


class LLMReasoner:
    """Singleton LLM reasoning core for agentic workflows.

    Wraps the generation provider to enable structured reasoning, user mental
    model construction, and behavioural dimension extraction.  All methods
    fall back to deterministic behaviour when the LLM is unavailable or
    returns unusable output.
    """

    _instance: LLMReasoner | None = None

    def __new__(cls) -> "LLMReasoner":
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._provider = get_generation_provider()
            cls._instance._provider_name = generation_provider_name()
        return cls._instance

    # ------------------------------------------------------------------
    # Core reasoning
    # ------------------------------------------------------------------

    def reason(
        self,
        prompt: str,
        system: str = "You are a precise reasoning assistant.",
        output_format: str = "text",
    ) -> str:
        """Call the LLM with a system prompt and return structured output.

        Returns deterministic fallback text on failure.
        """
        provider = _fresh_provider()
        if isinstance(provider, TemplateGenerationProvider):
            return _deterministic_reason(prompt)
        try:
            generated = provider.generate(instructions=system, prompt=prompt)
            if generated:
                return generated.strip()
            return _deterministic_reason(prompt)
        except Exception:
            return _deterministic_reason(prompt)

    def reason_structured(
        self,
        prompt: str,
        system: str = "You are a precise reasoning assistant that always outputs valid JSON.",
        schema: dict | None = None,
    ) -> dict[str, Any]:
        """Call the LLM, parse the JSON response, and return a dict.

        Falls back to a deterministic dict on failure.
        """
        raw = self.reason(prompt, system, output_format="json")
        parsed = _extract_json(raw)
        if parsed:
            return parsed
        return _deterministic_structured(prompt)

    # ------------------------------------------------------------------
    # Behavioural modelling
    # ------------------------------------------------------------------

    def build_user_mental_model(
        self,
        persona: str,
        history: list[dict[str, Any]],
    ) -> dict[str, Any]:
        """Analyse review history to extract a deep behavioural model.

        Returns rating philosophy, writing style traits, emotional patterns,
        vocabulary preferences, and domain knowledge indicators.
        """
        history_text = _summarise_history(history)
        prompt = (
            "You are building a psychological profile from review data.\n\n"
            f"Persona: {persona}\n\n"
            f"Review history:\n{history_text}\n\n"
            "Return a JSON object with these keys:\n"
            '  "rating_philosophy": how this person decides ratings (1-2 sentences)\n'
            '  "writing_style_traits": list of 3-5 dominant writing traits\n'
            '  "emotional_patterns": recurring emotional responses in reviews\n'
            '  "vocabulary_preferences": words/phrases this user favours\n'
            '  "domain_knowledge": areas where this user shows expertise\n'
            '  "review_length_profile": typical review length pattern\n'
            '  "price_sensitivity": how price-conscious this user is\n'
            '  "quality_expectation": quality-for-money expectation level (low/medium/high)\n'
            "Respond ONLY with valid JSON."
        )
        system = (
            "You are an expert behavioural psychologist who analyses consumer "
            "review patterns to build accurate user mental models. Return only JSON."
        )
        result = self.reason_structured(prompt, system)
        result["_provider"] = self._provider_name
        result["_llm_augmented"] = not isinstance(
            _fresh_provider(), TemplateGenerationProvider
        )
        return result

    def extract_behavioral_dimensions(
        self,
        history: list[dict[str, Any]],
    ) -> dict[str, Any]:
        """Extract an N-dimensional behavioural fingerprint from review history.

        Dimensions include: rating consistency, polarity ratio, review length
        variance, aspect focus breadth, temporal stability, emotional range,
        and vocabulary diversity.
        """
        if not history:
            return _empty_behavioral_dimensions()

        ratings = [h.get("rating", 3) for h in history if h.get("rating")]

        prompt = (
            "Analyse this user's review history to extract behavioural dimensions.\n\n"
            f"Review count: {len(history)}\n"
            f"Rating range: {min(ratings, default=0)}-{max(ratings, default=0)}\n"
            f"Average rating: {round(sum(ratings) / len(ratings), 2) if ratings else 'N/A'}\n"
            f"Recent reviews:\n{_summarise_history(history[-5:])}\n\n"
            "Return a JSON object with these keys:\n"
            '  "rating_consistency": how consistent or varied their ratings are (low/medium/high)\n'
            '  "polarity_ratio": ratio of positive to negative reviews\n'
            '  "review_length_pattern": typical review length description\n'
            '  "aspect_focus_breadth": how many different aspects they notice (narrow/broad)\n'
            '  "temporal_stability": how stable their preferences are over time\n'
            '  "emotional_range": range of emotions expressed (narrow/moderate/wide)\n'
            '  "vocabulary_diversity": lexical variety in reviews (low/medium/high)\n'
            '  "critical_thinking": how analytical vs impulse-driven their reviews are\n'
            "Respond ONLY with valid JSON."
        )
        system = (
            "You are a computational behavioural analyst. Analyse review patterns "
            "and return a structured behavioural fingerprint as JSON."
        )
        result = self.reason_structured(prompt, system)
        result["_provider"] = self._provider_name
        result["_llm_augmented"] = not isinstance(
            _fresh_provider(), TemplateGenerationProvider
        )
        return result


# ------------------------------------------------------------------
# Singleton access
# ------------------------------------------------------------------

def get_reasoner() -> LLMReasoner:
    return LLMReasoner()


# ------------------------------------------------------------------
# Deterministic fallbacks
# ------------------------------------------------------------------

def _fresh_provider() -> GenerationProvider:
    """Return a fresh provider reference so singleton doesn't cache."""
    return LLMReasoner()._provider


def _deterministic_reason(prompt: str) -> str:
    """Return a simple keyword-based reasoning fallback."""
    words = re.findall(r"[a-zA-Z]{3,}", prompt.lower())
    counts = Counter(words)
    top = [w for w, _ in counts.most_common(5) if w not in _STOPWORDS]
    if not top:
        return "Cannot determine reasoning from the provided prompt."
    return (
        "Based on the provided information, the key factors to consider are: "
        + ", ".join(top)
        + ". A reasonable inference requires weighing these elements against "
        "the stated constraints."
    )


def _deterministic_structured(prompt: str) -> dict[str, Any]:
    words = re.findall(r"[a-zA-Z]{3,}", prompt.lower())
    counts = Counter(words)
    top = [w for w, _ in counts.most_common(8) if w not in _STOPWORDS]
    ratings = [int(n) for n in re.findall(r"\b([1-5])\b", prompt)]
    return {
        "predicted_rating": int(round(sum(ratings) / len(ratings))) if ratings else 3,
        "key_factors": top[:5] if top else ["unknown"],
        "confidence": round(min(0.3 + len(ratings) * 0.05, 0.85), 2),
        "reasoning": f"Deterministic analysis based on {len(words)} input tokens.",
        "_fallback": True,
    }


def _extract_json(raw: str) -> dict[str, Any] | None:
    """Extract a JSON object from text, tolerating markdown fences."""
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
        return None


def _summarise_history(history: list[dict[str, Any]], max_items: int = 20) -> str:
    lines = []
    for item in history[-max_items:]:
        name = item.get("item_name", item.get("name", "unknown"))
        rating = item.get("rating", "N/A")
        review = str(item.get("review", "") or "")
        category = item.get("category", "")
        lines.append(
            f"  - {name} ({category}): rated {rating}/5. "
            f'Review: "{review[:200]}"'
        )
    return "\n".join(lines) if lines else "(no history)"


def _empty_behavioral_dimensions() -> dict[str, Any]:
    return {
        "rating_consistency": "unknown",
        "polarity_ratio": 0.0,
        "review_length_pattern": "unknown",
        "aspect_focus_breadth": "unknown",
        "temporal_stability": "unknown",
        "emotional_range": "unknown",
        "vocabulary_diversity": "unknown",
        "critical_thinking": "unknown",
        "_provider": "",
        "_llm_augmented": False,
        "_fallback": True,
    }
