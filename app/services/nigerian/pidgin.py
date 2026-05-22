from __future__ import annotations

import random
import re
from dataclasses import dataclass

NIGERIAN_PIDGIN_PHRASES: dict[str, list[str]] = {
    "positive": [
        "e too correct!",
        "this one na banger!",
        "the thing good well well",
        "I no fit complain",
        "e sweet me die!",
        "no be small enjoyment",
        "God bless the person wey make this one",
        "this one pass my expectation",
        "na solid product be this",
        "e worth every kobo",
        "no dulling at all",
        "the thing choke!",
        "no regrets at all",
        "I go buy again",
        "na correct package",
        "the quality dey mad",
        "I dey enjoy am",
        "nothing wey person fit talk",
        "e dey deliver",
        "no lie, this one good",
    ],
    "negative": [
        "e no just make sense",
        "this one na scam",
        "the thing fall my hand",
        "I no fit shout",
        "na wahala be this",
        "waste of money",
        "e no worth am at all",
        "the thing weak",
        "I regret this purchase",
        "na fake dem sell me",
        "no be wetin I order",
        "the quality poor well well",
        "dem cheat me",
        "na so so packaging",
        "e disappoint me",
        "the thing no last",
        "na rubbish",
        "I want my money back",
        "e no correspond",
        "the headache too much",
    ],
    "neutral": [
        "e dey okay",
        "no be perfect but e manage",
        "na so so average",
        "e do the work sha",
        "not too bad",
        "e dey try small",
        "manageable",
        "I fit manage am",
        "no too sweet, no too sour",
        "middle of the road",
        "okay for the price",
        "e get as e be",
        "no pass, no fail",
        "e just dey there",
        "we dey manage",
    ],
}

NIGERIAN_EXPRESSIONS: dict[str, str] = {
    "abeg": "please",
    "na": "it is / that is",
    "o": "emphasis marker",
    "sha": "anyway / in any case",
    "sef": "even / also",
    "nawa": "expression of surprise or exasperation",
    "wahala": "trouble / problem",
    "chop": "eat / enjoy",
    "oga": "boss / leader",
    "kobo": "smallest currency unit (like cent)",
    "naira": "Nigerian currency",
    "no be so": "that is not how it is",
    "how far": "how are you / what's up",
    "how body": "how are you",
    "I dey": "I am (doing fine)",
    "make I": "let me",
    "we go": "we will",
    "no wahala": "no problem",
    "oya": "let's go / come on",
    "chop life": "enjoy life",
    "e choke": "it's impressive / overwhelming",
    "gobe": "problem / serious issue",
    "para": "angry / upset",
    "ginger": "motivate / energise",
    "yawa": "trouble / disgrace",
    "gbedu": "good music / party",
    "japa": "leave / escape",
    "comot": "leave / go away",
    "gist me": "tell me the news",
    "no fall my hand": "don't disappoint me",
    "god when": "prayer for divine intervention",
    "who dey breathe": "who cares / I don't care",
    "e don cast": "it's ruined / it's gone bad",
    "k-leg": "problem / complication",
    "belle": "stomach",
    "waka": "walk / journey",
    "troway": "throw away / waste",
    "pepper rest": "strong spicy food hit (positive or negative)",
    "packaging na die": "the packaging is extremely impressive",
    "na cruise": "it's just for fun / not serious",
}

NIGERIAN_CONTEXTUAL_MARKERS: dict[str, list[str]] = {
    "urban_lagos": [
        "island", "mainland", "lekki", "vi", "ikoyi", "surulere",
        "yaba", "ikeja", "festac", "ajah", "sangotedo",
    ],
    "naira_economy": [
        "naira", "kobo", "exchange rate", "black market rate",
        "official rate", "naira value", "naira equivalent",
    ],
    "nigerian_brands": [
        "jumia", "konga", "paystack", "flutterwave", "piggyvest",
        "opay", "moniepoint", "palmall", "slot", "spar",
        "shoprite", "game", "justrite", "market square",
    ],
    "logistics": [
        "waybill", "terminal", "park", "dispatch", "keke",
        "okada", "danfo", "bike man", "delivery to lagos",
        "pick up", "drop off", "last mile",
    ],
    "social_life": [
        "owambe", "party", "asoebi", "wedding", "burial",
        "naming ceremony", "house warming", "church", "mosque",
        "vigil", "crusade", "fellowship",
    ],
    "food_culture": [
        "jollof", "suya", "amala", "eba", "pounded yam",
        "egusi", "ogbono", "afang", "edikaikong", "nkwobi",
        "isiewu", "banga", "starch", "banga soup",
        "pepper soup", "moi moi", "akara", "puff puff",
        "chin chin", "zobo", "kunu", "chapman",
    ],
    "entertainment": [
        "afrobeats", "naija", "nollywood", "bbnaija", "big brother",
        "gulder ultimate search", "nigerian idol", "the voice nigeria",
        "mtn project fame", "african magic",
    ],
}


@dataclass
class VoiceInjectionResult:
    original: str
    nigerianized: str
    intensity: float
    modifications: list[str]


class NigerianVoiceInjector:
    """Injects authentic Nigerian English and pidgin patterns into text.

    Provides culturally-appropriate voice injection at configurable intensity
    levels, along with Nigerian-specific phrases and greetings.
    """

    # ------------------------------------------------------------------
    # Nigerianize review
    # ------------------------------------------------------------------

    @staticmethod
    def nigerianize_review(review: str, intensity: float = 0.35) -> str:
        """Inject Nigerian English patterns into a review at the given intensity.

        Intensity 0.0 = no change, 1.0 = maximum nigerianisation.
        The injection preserves the original meaning while adding authentic
        Nigerian voice patterns.
        """
        if intensity <= 0.0 or not review.strip():
            return review

        intensity = min(max(intensity, 0.0), 1.0)
        text = review.strip()

        modifications = []

        # 1. Add a Nigerian greeting opener (if no greeting present)
        if intensity >= 0.4 and not _has_greeting(text):
            greeting = NigerianVoiceInjector.get_nigerian_greeting()
            if random.random() < intensity * 0.6:
                text = f"{greeting}! {text}"
                modifications.append("added_greeting")

        # 2. Inject pidgin emphasis at the end
        if intensity >= 0.3 and random.random() < intensity * 0.5:
            positive_phrases = NIGERIAN_PIDGIN_PHRASES["positive"]
            neutral_phrases = NIGERIAN_PIDGIN_PHRASES["neutral"]
            ending = random.choice(positive_phrases + neutral_phrases)
            if ending not in text:
                text = f"{text.rstrip('.!')}, {ending}."
                modifications.append("added_pidgin_emphasis")

        # 3. Replace common words with Nigerian equivalents
        if intensity >= 0.5:
            replacements = _nigerian_word_map(intensity)
            for standard, nigerian in replacements.items():
                if random.random() < intensity * 0.4:
                    pattern = re.compile(rf"\b{re.escape(standard)}\b", re.IGNORECASE)

                    def _replace(match: re.Match) -> str:
                        original = match.group(0)
                        if original[0].isupper():
                            return nigerian.capitalize()
                        return nigerian

                    new_text = pattern.sub(_replace, text)
                    if new_text != text:
                        text = new_text
                        modifications.append(f"replaced:{standard}->{nigerian}")

        # 4. Add 'o' or 'sha' emphasis markers
        if intensity >= 0.6 and random.random() < intensity * 0.3:
            sentences = re.split(r"(?<=[.!?])\s+", text)
            if sentences:
                last = sentences[-1].rstrip(".!?")
                marker = random.choice(["o", "sha", "sef"])
                if marker not in last:
                    sentences[-1] = f"{last} {marker}."
                text = " ".join(sentences)
                modifications.append("added_emphasis_marker")

        # 5. Add Naira reference for price/value mentions
        if intensity >= 0.5:
            price_words = {"price", "cost", "expensive", "cheap", "affordable", "worth", "value"}
            words_in_text = set(re.findall(r"[a-zA-Z]{3,}", text.lower()))
            if words_in_text & price_words and random.random() < intensity * 0.35:
                naira_phrases = NigerianVoiceInjector.nigerian_price_sensitivity_phrases()
                phrase = random.choice(naira_phrases)
                if phrase not in text.lower():
                    text = f"{text.rstrip('.!')}. {phrase}."
                    modifications.append("added_naira_context")

        return text

    # ------------------------------------------------------------------
    # Nigerian greeting
    # ------------------------------------------------------------------

    @staticmethod
    def get_nigerian_greeting() -> str:
        """Return a culture-appropriate Nigerian greeting."""
        greetings = [
            "How far",
            "How body",
            "Oya",
            "Ah ah",
            "See ehn",
            "Abeg",
            "Na so",
            "My people",
            "My brother",
            "My sister",
            "Chairman",
            "Madam",
            "Oga",
            "Boss",
        ]
        return random.choice(greetings)

    # ------------------------------------------------------------------
    # Price sensitivity phrases
    # ------------------------------------------------------------------

    @staticmethod
    def nigerian_price_sensitivity_phrases() -> list[str]:
        """Common Nigerian price-related expressions."""
        return [
            "For this price, e suppose better pass this one",
            "The price dey okay but quality matter pass money",
            "If the price small, I for collect two",
            "Value for money na the real deal",
            "No be the price matter, na wetin you get for your money",
            "With how naira dey now, every kobo must count",
            "The thing worth the money sha",
            "Not bad for the money wey I pay",
            "I fit pay more if the quality sure pass",
            "Na the kind thing wey you go use, no be to just buy",
            "Good product but the price fit make person think twice",
            "For this economy, you need value for every naira",
            "Cheap no mean fake o, and expensive no mean original",
            "Wetin you pay na wetin you go get",
            "If you want quality, you must open wallet well well",
        ]

    # ------------------------------------------------------------------
    # Voice injection result
    # ------------------------------------------------------------------

    @staticmethod
    def inject_with_result(
        review: str,
        intensity: float = 0.35,
    ) -> VoiceInjectionResult:
        """Nigerianize a review and return the result with metadata."""
        nigerianized = NigerianVoiceInjector.nigerianize_review(review, intensity)

        modifications = []
        if nigerianized != review:
            orig_words = set(re.findall(r"\w+", review.lower()))
            new_words = set(re.findall(r"\w+", nigerianized.lower()))
            added = new_words - orig_words
            if added:
                modifications.append(f"added_terms: {', '.join(sorted(added)[:5])}")

        return VoiceInjectionResult(
            original=review,
            nigerianized=nigerianized,
            intensity=round(intensity, 2),
            modifications=modifications,
        )


# ------------------------------------------------------------------
# Helpers
# ------------------------------------------------------------------

def _nigerian_word_map(intensity: float) -> dict[str, str]:
    """Map standard English words to Nigerian equivalents based on intensity."""
    full_map = {
        "excellent": "correct",
        "terrible": "rubbish",
        "wonderful": "sweet",
        "very": "well well",
        "really": "well well",
        "disappointed": "vexed",
        "happy": "gingered",
        "expensive": "costly well well",
        "cheap": "affordable sha",
        "quality": "correct quality",
        "bad": "fall my hand",
    }
    if intensity < 0.6:
        return dict(list(full_map.items())[:5])
    return full_map


def _has_greeting(text: str) -> bool:
    """Check if text already has a Nigerian-style greeting."""
    greetings = {"how far", "how body", "oya", "ah ah", "see ehn", "abeg", "na so",
                 "my people", "my brother", "my sister", "chairman", "madam", "oga"}
    first_words = " ".join(text.lower().split()[:3])
    return any(g in first_words for g in greetings)
