from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any

from app.models.schemas import UserProfile
from app.services.intelligence.aspects import NIGERIAN_CONTEXT_TERMS

NIGERIAN_LOCALES = [
    "lagos",
    "abuja",
    "port_harcourt",
    "ibadan",
    "kano",
    "benin_city",
    "enugu",
    "aba",
    "owerri",
    "jos",
    "calabar",
    "uyo",
    "warri",
    "ilorin",
    "maiduguri",
    "kaduna",
    "abeokuta",
    "onitsha",
    "asaba",
    "akure",
]

NIGERIAN_CITIES = {
    "lagos": {"region": "southwest", "vibe": "fast-paced, commercial hub, status-conscious"},
    "abuja": {"region": "northcentral", "vibe": "cosmopolitan, government, aspirational"},
    "port_harcourt": {"region": "southsouth", "vibe": "oil-rich, pragmatic, brand-aware"},
    "ibadan": {"region": "southwest", "vibe": "traditional, value-conscious, practical"},
    "kano": {"region": "northwest", "vibe": "commercial northern hub, conservative, quality-focused"},
    "benin_city": {"region": "southsouth", "vibe": "cultural, student-oriented, price-sensitive"},
    "enugu": {"region": "southeast", "vibe": "coal city, calm, quality-over-quantity"},
    "aba": {"region": "southeast", "vibe": "entrepreneurial, industrious, value-seeking"},
    "owerri": {"region": "southeast", "vibe": "laid-back, entertainment-focused, trendy"},
    "jos": {"region": "northcentral", "vibe": "cool climate, relaxed, local-quality preference"},
    "calabar": {"region": "southsouth", "vibe": "tourism, calm, experience-driven"},
    "uyo": {"region": "southsouth", "vibe": "emerging, ambitious, quality-conscious"},
    "warri": {"region": "southsouth", "vibe": "oil city, bold, status-driven"},
    "ilorin": {"region": "northcentral", "vibe": "balanced, practical, education-focused"},
    "maiduguri": {"region": "northeast", "vibe": "resilient, essential-focused, community-driven"},
    "kaduna": {"region": "northwest", "vibe": "industrial, pragmatic, value-for-money"},
    "abeokuta": {"region": "southwest", "vibe": "traditional-meets-modern, craft-aware"},
    "onitsha": {"region": "southeast", "vibe": "market hub, entrepreneurial, bargain-conscious"},
    "asaba": {"region": "southsouth", "vibe": "growing, aspirational, lifestyle-aware"},
    "akure": {"region": "southwest", "vibe": "academic, modest, practical-spending"},
}

NIGERIAN_SHOPPING_CONTEXTS = [
    "open_market",
    "supermarket",
    "online_delivery",
    "okrika_thrift",
    "boutique",
    "mall",
    "roadside_vendor",
    "specialty_store",
]


@dataclass
class NigerianContextResult:
    """Enriched Nigerian cultural context signals."""

    detected_markers: list[str]
    locale_signals: list[str]
    regional_context: dict[str, Any]
    behavioral_indicators: dict[str, Any]
    cultural_confidence: float
    enriched_persona: str


class NigerianContextEngine:
    """Engine for detecting and enriching Nigerian cultural signals.

    Analyses user data for Nigerian cultural markers, injects context into
    profiles, and scores relevance of Nigerian-localised behaviour.
    """

    # ------------------------------------------------------------------
    # Nigerian behavioural patterns
    # ------------------------------------------------------------------

    @staticmethod
    def nigerian_behavioral_patterns() -> dict[str, Any]:
        """Common Nigerian consumer behavioural patterns.

        These are authentic patterns observed in Nigerian consumer behaviour,
        not stereotypes.  They reflect real market dynamics.
        """
        return {
            "price_sensitivity": {
                "description": (
                    "Nigerian consumers typically exhibit high price sensitivity due to "
                    "economic realities, but this co-exists with brand consciousness when "
                    "the brand signifies durable quality."
                ),
                "indicators": [
                    "explicit mention of price or cost",
                    "value-for-money language",
                    "comparison shopping references",
                    "durability and longevity concerns",
                ],
            },
            "social_proof_importance": {
                "description": (
                    "Recommendations from trusted sources (friends, family, community) "
                    "carry significant weight.  Online reviews are increasingly influential "
                    "especially among urban and younger demographics."
                ),
                "indicators": [
                    "references to recommendations from others",
                    "community-approved or widely-used language",
                    "mention of popular or trending items",
                    "brand reputation references",
                ],
            },
            "brand_consciousness": {
                "description": (
                    "Nigerian consumers are brand-aware: they associate established brands "
                    "with reliability and status.  However, willingness to pay premium "
                    "depends on perceived longevity of the product."
                ),
                "indicators": [
                    "brand-name mentions in reviews",
                    "original vs fake authenticity language",
                    "premium or luxury terminology",
                    "status-signalling language",
                ],
            },
            "quality_for_money_mindset": {
                "description": (
                    "The dominant consumer philosophy is 'quality for money'.  Nigerian "
                    "shoppers want items that last, perform well, and justify their cost.  "
                    "This is different from simple price sensitivity: it's a long-term "
                    "value calculation."
                ),
                "indicators": [
                    "durability and lasting quality mentions",
                    "worth-it or not-worth-it language",
                    "multi-purpose or versatile usage notes",
                    "replacement frequency concerns",
                ],
            },
            "haggling_culture": {
                "description": (
                    "In physical markets, negotiation is expected.  This translates online "
                    "into deal-seeking behaviour: waiting for discounts, comparing prices "
                    "across platforms, and valuing perceived fairness in pricing."
                ),
                "indicators": [
                    "bargain or deal-seeking language",
                    "price comparison with alternatives",
                    "mention of discounts or promotions",
                    "fair-price or overpriced language",
                ],
            },
            "delivery_awareness": {
                "description": (
                    "Logistics and delivery reliability are major concerns for Nigerian "
                    "online shoppers.  Reviews frequently mention delivery experience "
                    "alongside product quality."
                ),
                "indicators": [
                    "delivery time mentions",
                    "packaging condition notes",
                    "arrived-as-described language",
                    "logistics or shipping references",
                ],
            },
        }

    # ------------------------------------------------------------------
    # Nigerian context injection
    # ------------------------------------------------------------------

    def inject_nigerian_context(
        self,
        persona: str,
        history: list[dict[str, Any]],
    ) -> NigerianContextResult:
        """Detect and enrich Nigerian cultural signals from persona and history.

        Analyses the persona text and review history for Nigerian markers,
        maps them to regional contexts, and returns enriched signals.
        """
        combined = persona + " " + " ".join(
            str(h.get("review", "") or "") + " " + str(h.get("item_name", "") or "")
            for h in history
        )
        markers = self.detect_nigerian_markers(combined)

        locale = self._infer_locale(markers, combined)
        region_context = NIGERIAN_CITIES.get(locale, {"region": "unknown", "vibe": "general"})

        behavioural = self.nigerian_behavioral_patterns()
        matched_behaviours = {}
        for name, pattern in behavioural.items():
            indicator_hits = [
                ind for ind in pattern["indicators"]
                if any(word in combined.lower() for word in _indicator_words(ind))
            ]
            if indicator_hits:
                matched_behaviours[name] = {
                    "description": pattern["description"],
                    "matched_indicators": indicator_hits,
                    "strength": round(min(len(indicator_hits) / len(pattern["indicators"]), 1.0), 2),
                }

        confidence = round(
            min(0.30 + len(markers.get("lexical", [])) * 0.08
                + len(markers.get("contextual", [])) * 0.12
                + len(matched_behaviours) * 0.10,
                0.95),
            2,
        )

        enriched = persona
        if locale and locale != "unspecified":
            enriched += (
                f" This consumer shops in {locale.title()}, a "
                f"{region_context.get('vibe', 'typical Nigerian')} city."
            )
        if markers.get("contextual"):
            enriched += (
                f" Their shopping context reflects: "
                f"{', '.join(markers['contextual'][:3])}."
            )

        return NigerianContextResult(
            detected_markers=markers.get("lexical", []),
            locale_signals=markers.get("contextual", []),
            regional_context={"city": locale, **region_context},
            behavioral_indicators=matched_behaviours,
            cultural_confidence=confidence,
            enriched_persona=enriched,
        )

    # ------------------------------------------------------------------
    # Nigerian marker detection
    # ------------------------------------------------------------------

    @staticmethod
    def detect_nigerian_markers(text: str) -> dict[str, list[str]]:
        """Detect Nigerian English patterns, locale references, and cultural signals.

        Returns a dict with 'lexical' (words/phrases), 'contextual' (cultural
        context signals), and 'locale' (city/region references).
        """
        lowered = text.lower()
        words = set(re.findall(r"[a-zA-Z_]{3,}", lowered))

        lexical = [
            w for w in words
            if w in NIGERIAN_CONTEXT_TERMS
            or w in _NIGERIAN_EXPRESSION_WORDS
        ][:12]

        contextual = []
        for context_signal, phrases in _CONTEXTUAL_PATTERNS.items():
            if any(phrase in lowered for phrase in phrases):
                contextual.append(context_signal)
        contextual = contextual[:8]

        locale_hits = []
        for city in NIGERIAN_LOCALES:
            if city.replace("_", " ") in lowered or city in lowered:
                locale_hits.append(city)
        for state in _NIGERIAN_STATES:
            if state in lowered:
                locale_hits.append(state)

        return {
            "lexical": lexical,
            "contextual": contextual,
            "locale": locale_hits,
        }

    # ------------------------------------------------------------------
    # Nigerian relevance scoring
    # ------------------------------------------------------------------

    @staticmethod
    def score_nigerian_relevance(user_profile: UserProfile) -> float:
        """Score how Nigerian-relevant a user profile is (0-1).

        Considers locale, nigerian_context terms, and behavioural indicators.
        """
        score = 0.0

        if user_profile.locale and user_profile.locale.lower() in ("nigeria", "ng"):
            score += 0.30
        elif user_profile.locale and user_profile.locale.lower() in NIGERIAN_LOCALES:
            score += 0.15

        nigerian_terms = set(user_profile.nigerian_context)
        if nigerian_terms:
            score += min(len(nigerian_terms) * 0.05, 0.25)

        general_terms = set(user_profile.preferred_terms + user_profile.positive_aspects)
        nigerian_signal_terms = general_terms & NIGERIAN_CONTEXT_TERMS
        if nigerian_signal_terms:
            score += min(len(nigerian_signal_terms) * 0.04, 0.15)

        signals_text = " ".join(user_profile.signals).lower()
        locale_hits = [
            city for city in NIGERIAN_LOCALES
            if city.replace("_", " ") in signals_text
        ]
        if locale_hits:
            score += min(len(locale_hits) * 0.06, 0.15)

        price_terms = {"affordable", "budget", "naira", "price", "value", "worth", "cheap"}
        if general_terms & price_terms:
            score += 0.08

        return round(min(score, 0.95), 2)

    # ------------------------------------------------------------------
    # Cultural enrichment
    # ------------------------------------------------------------------

    @staticmethod
    def enrich_with_cultural_context(
        item: Any,
        user_profile: UserProfile,
    ) -> dict[str, Any]:
        """Add Nigerian cultural relevance signals to an item context.

        Analyses how an item resonates with Nigerian consumer values and
        returns enriched cultural signals.
        """
        relevance_score = NigerianContextEngine.score_nigerian_relevance(user_profile)

        item_signals = (
            item.signals if hasattr(item, "signals")
            else getattr(item, "terms", [])
        )
        item_text = " ".join(str(s) for s in item_signals).lower()

        cultural_signals = {
            "nigerian_relevance_score": relevance_score,
            "value_for_money_signal": bool(
                set(re.findall(r"[a-zA-Z_]{3,}", item_text))
                & {"affordable", "budget", "value", "durable", "quality", "worth"}
            ),
            "social_proof_signal": bool(
                set(re.findall(r"[a-zA-Z_]{3,}", item_text))
                & {"popular", "trending", "recommended", "trusted", "reviewed"}
            ),
            "brand_consciousness_signal": bool(
                set(re.findall(r"[a-zA-Z_]{3,}", item_text))
                & {"premium", "original", "authentic", "brand", "quality"}
            ),
            "delivery_relevant": bool(
                set(re.findall(r"[a-zA-Z_]{3,}", item_text))
                & {"delivery", "shipping", "arrived", "packaging", "logistics"}
            ),
            "nigerian_market_fit": "high" if relevance_score > 0.60 else "medium" if relevance_score > 0.30 else "low",
        }

        item_name = getattr(item, "name", "unknown")
        item_category = getattr(item, "category", "unknown")
        context_note = _item_category_nigerian_context(item_category)

        return {
            "item": item_name,
            "category": item_category,
            "cultural_signals": cultural_signals,
            "context_note": context_note,
        }

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _infer_locale(markers: dict[str, list[str]], text: str) -> str:
        locale_hits = markers.get("locale", [])
        if locale_hits:
            return locale_hits[0]

        lowered = text.lower()
        for city in NIGERIAN_LOCALES:
            if city.replace("_", " ") in lowered:
                return city

        return "lagos"


_NIGERIAN_STATES = [
    "abia", "adamawa", "akwa_ibom", "anambra", "bauchi", "bayelsa", "benue",
    "borno", "cross_river", "delta", "ebonyi", "edo", "ekiti", "enugu",
    "gombe", "imo", "jigawa", "kaduna", "kano", "katsina", "kebbi",
    "kogi", "kwara", "lagos", "nasarawa", "niger", "ogun", "ondo",
    "osun", "oyo", "plateau", "rivers", "sokoto", "taraba", "yobe", "zamfara",
    "fct", "abuja",
]

_NIGERIAN_EXPRESSION_WORDS = {
    "abeg", "na", "o", "sha", "sef", "nawa", "wahala", "chop", "oga",
    "okada", "danfo", "bole", "suya", "jollof", "amala", "eba", "fufu",
    "akara", "moi", "zobo", "kunu", "agbada", "ankara", "asoebi",
    "gele", "iro", "buba", "kente", "adire", "ase",
}

_CONTEXTUAL_PATTERNS: dict[str, list[str]] = {
    "market_shopping": [
        "market", "balogun", "computer village", "wuse", "tejuosho",
        "trade fair", "open market", "roadside", "hawker",
    ],
    "online_shopping": [
        "jumia", "konga", "online", "delivery to", "ordered from",
        "shipping to lagos", "waybill", "pickup",
    ],
    "okrika_thrift": [
        "okrika", "secondhand", "thrift", "bend down select",
        "fairly used", "tokunbo", "belgium", "first grade",
    ],
    "price_negotiation": [
        "haggled", "bargain", "last price", "first price",
        "cost how much", "too expensive", "fine price",
    ],
    "naira_economy": [
        "naira", "exchange rate", "dollar rate", "budget",
        "cost of living", "fuel price", "inflation",
    ],
    "social_recommendation": [
        "my friend said", "someone recommended", "everyone is using",
        "trending now", "viral", "people are talking",
    ],
    "quality_scepticism": [
        "fake", "original", "genuine", "authentic", "china",
        "abroad quality", "imported", "locally made",
    ],
    "delivery_experience": [
        "delivery was fast", "delivery was slow", "arrived on time",
        "packaging was good", "delivery man", "dispatch rider",
    ],
}


def _indicator_words(indicator: str) -> list[str]:
    return re.findall(r"[a-zA-Z]{3,}", indicator.lower())


def _item_category_nigerian_context(category: str) -> str:
    contexts = {
        "All_Beauty": (
            "Beauty products in Nigeria are often evaluated for suitability "
            "with melanin-rich skin, weather resilience, and authenticity "
            "(original vs fake).  Popular categories in Nigerian markets."
        ),
        "Grocery_Gourmet_Food": (
            "Food and grocery items in Nigeria are evaluated against local "
            "taste preferences, with strong demand for ethnic ingredients "
            "and Nigerian food staples."
        ),
        "Digital_Music": (
            "Music is central to Nigerian social life.  Afrobeats and local "
            "artists dominate preferences, alongside gospel and highlife."
        ),
        "Books": (
            "Nigerian readers value educational and professional development "
            "books highly, alongside African literature and motivational works."
        ),
        "Restaurants": (
            "Dining preferences in Nigeria span local bukkas to international "
            "chains.  Service quality and ambience matter for social occasions."
        ),
    }
    return contexts.get(category, (
        f"Nigerian consumers in the {category} space typically balance "
        f"price sensitivity with quality expectations, valuing durability "
        f"and practical utility."
    ))
