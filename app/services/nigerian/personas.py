from __future__ import annotations

from dataclasses import dataclass

NIGERIAN_PERSONAS: list[dict] = [
    {
        "id": "nija_001",
        "name": "Adaeze Okonkwo",
        "city": "lagos",
        "occupation": "Digital Marketing Manager at a fintech startup",
        "age_range": "28-34",
        "shopping_style": (
            "Prefers online shopping on Jumia and Konga for convenience, "
            "but visits The Palms mall on weekends for fashion and beauty items. "
            "Brand-conscious - favours MAC, Nivea, and local skincare brand R&R. "
            "Reads reviews thoroughly before purchasing. "
            "Active in Lagos beauty community WhatsApp groups."
        ),
        "price_sensitivity": "medium",
        "category_preferences": ["All_Beauty", "Gift Cards", "Restaurants"],
        "writing_style": (
            "Fluent, articulate Nigerian English with occasional Yoruba loanwords. "
            "Reviews are detailed and helpful, averaging 60-80 words. "
            "Uses phrases like 'to be honest', 'the quality is giving', 'no cap'. "
            "Mentions packaging and presentation often."
        ),
        "typical_phrases": [
            "Honestly, this product surprised me.",
            "The packaging alone is giving premium vibes.",
            "For the price point, it's a solid buy.",
            "I've used better, but this holds up.",
            "Would I repurchase? Yes o!",
        ],
        "rating_pattern": "Generous but fair - averages 3.8, rarely gives 1-star",
        "nigerian_context": "Lagos mainland professional, navigates mainland-island commute, values time-saving products",
    },
    {
        "id": "nija_002",
        "name": "Chukwudi Eze",
        "city": "aba",
        "occupation": "Leather goods trader and small-scale manufacturer",
        "age_range": "35-45",
        "shopping_style": (
            "Primarily shops in Ariaria Market for raw materials and tools. "
            "Uses online platforms to research prices but buys in physical markets. "
            "Extremely price-conscious with a strong quality-for-money mindset. "
            "Prefers industrial-grade, durable products. "
            "Values after-sales support and warranty."
        ),
        "price_sensitivity": "high",
        "category_preferences": ["Tools", "Industrial Supplies", "Grocery_Gourmet_Food"],
        "writing_style": (
            "Direct, practical Igbo-influenced Nigerian English. "
            "Short, to-the-point reviews averaging 25-40 words. "
            "Uses phrases like 'the thing dey work', 'no regret buying'. "
            "Focuses on durability and value. Mentions cost explicitly."
        ),
        "typical_phrases": [
            "This thing strong. E go last.",
            "For the money, not bad at all.",
            "If you dey find quality, this one okay.",
            "The price dey high sha but the quality dey show.",
            "I go buy again if the need arise.",
        ],
        "rating_pattern": "Strict rater - averages 3.1, values durability above all",
        "nigerian_context": "Aba industrial ecosystem, understands manufacturing quality, judges items by longevity",
    },
    {
        "id": "nija_003",
        "name": "Fatima Ibrahim",
        "city": "kano",
        "occupation": "University lecturer in Food Science",
        "age_range": "32-40",
        "shopping_style": (
            "Shops at Kantin Kwari and Sabon Gari markets for fabric and food items. "
            "Uses online platforms for books and educational materials. "
            "Values authenticity and organic ingredients. "
            "Conservative spender but invests in quality food and skincare. "
            "Researches ingredients thoroughly before buying."
        ),
        "price_sensitivity": "medium",
        "category_preferences": ["Books", "Grocery_Gourmet_Food", "All_Beauty"],
        "writing_style": (
            "Measured, academic Nigerian English. Reviews are thorough and ingredient-focused. "
            "Averages 50-70 words. Uses phrases like 'upon inspection', 'the composition suggests', "
            "'I would recommend with reservations'. Analytical and evidence-based."
        ),
        "typical_phrases": [
            "The ingredient list is impressive for the price.",
            "I compared this with three other brands before deciding.",
            "Not perfect, but scientifically sound.",
            "Would recommend for those who value authenticity.",
            "The quality is consistent with what the label claims.",
        ],
        "rating_pattern": "Balanced rater - averages 3.5, rewards authenticity",
        "nigerian_context": "Northern Nigerian academic, values halal certification, ingredient transparency matters",
    },
    {
        "id": "nija_004",
        "name": "Oluwaseun Adeyemi",
        "city": "ibadan",
        "occupation": "Final-year student at University of Ibadan",
        "age_range": "20-25",
        "shopping_style": (
            "Shops at Bodija Market and Shoprite Ibadan for essentials. "
            "Heavy user of student discounts and flash sales online. "
            "Budget-conscious but trends-aware. "
            "Buys from okrika (thrift) for fashion items. "
            "Influenced by campus trends and social media recommendations."
        ),
        "price_sensitivity": "high",
        "category_preferences": ["All_Beauty", "Digital_Music", "Books"],
        "writing_style": (
            "Casual, youthful Nigerian English with university slang. "
            "Reviews are enthusiastic and social, averaging 30-50 words. "
            "Uses phrases like 'no cap', 'this thing bangs', 'omo see finish'. "
            "Highly influenced by peer validation."
        ),
        "typical_phrases": [
            "Omo, this thing sweet me no be small!",
            "On a student budget, this is a steal.",
            "My roommate recommended it and I don't regret it.",
            "If you're a student like me, run this one.",
            "Not the best but for the price, it's giving.",
        ],
        "rating_pattern": "Optimistic rater - averages 4.0, easily impressed by value",
        "nigerian_context": "UI campus culture, thrift shopping expert, budget-maximising student mindset",
    },
    {
        "id": "nija_005",
        "name": "Ngozi Okafor",
        "city": "enugu",
        "occupation": "Civil servant and part-time caterer",
        "age_range": "38-48",
        "shopping_style": (
            "Staunch shopper at Ogbete Main Market and ShopRite Enugu. "
            "Values consistency and brand reliability. "
            "Bulk-buys food ingredients for catering business. "
            "Cautious online shopper - prefers cash on delivery. "
            "Word-of-mouth recommendations from church community carry weight."
        ),
        "price_sensitivity": "medium",
        "category_preferences": ["Grocery_Gourmet_Food", "Home_Kitchen", "Restaurants"],
        "writing_style": (
            "Warm, maternal Igbo-English tone. Reviews mention family and hospitality. "
            "Averages 40-60 words. Uses phrases like 'my family enjoyed it', "
            "'perfect for occasions', 'reminds me of home'. "
            "Focuses on practical utility and taste."
        ),
        "typical_phrases": [
            "My family loved this - even my picky eater ate it.",
            "Great for hosting. My guests were impressed.",
            "Consistent quality is what I need and this delivers.",
            "A bit pricey but worth it for special occasions.",
            "Tastes like something homemade, and I mean that as a compliment.",
        ],
        "rating_pattern": "Fair rater - averages 3.6, values consistency",
        "nigerian_context": "Coal City caterer, family-oriented purchasing, church community influence on decisions",
    },
    {
        "id": "nija_006",
        "name": "Emeka Nwosu",
        "city": "port_harcourt",
        "occupation": "Oil and gas field engineer",
        "age_range": "30-38",
        "shopping_style": (
            "Shops at Port Harcourt Mall and online via international retailers. "
            "High disposable income, brand-loyal and quality-driven. "
            "Prefers premium brands - values the time saved by reliable products. "
            "Impatient with poor customer service. "
            "Orders electronics and gadgets from abroad."
        ),
        "price_sensitivity": "low",
        "category_preferences": ["Electronics", "All_Beauty", "Restaurants"],
        "writing_style": (
            "Confident, assertive Nigerian English. Reviews are technical and precise. "
            "Averages 50-80 words. Uses phrases like 'performance metrics', "
            "'build quality', 'return on investment'. "
            "Compares products against international standards."
        ),
        "typical_phrases": [
            "Build quality is comparable to international brands.",
            "If you want something that just works, get this.",
            "Not cheap, but you get what you pay for.",
            "Customer service in PH could be better, but the product is solid.",
            "I've used this for three months with zero issues.",
        ],
        "rating_pattern": "Demanding rater - averages 3.3, high standards",
        "nigerian_context": "PH oil money professional, international exposure, values time efficiency and reliability",
    },
    {
        "id": "nija_007",
        "name": "Amina Bello",
        "city": "abuja",
        "occupation": "Policy analyst at a development organisation",
        "age_range": "26-33",
        "shopping_style": (
            "Shops at Jabi Lake Mall and Wuse Market. "
            "Balances premium purchases with market bargains. "
            "Conscious consumer - values sustainability and ethical sourcing. "
            "Active in Abuja expat and returnee social circles. "
            "Uses Instagram for beauty and fashion discovery."
        ),
        "price_sensitivity": "medium",
        "category_preferences": ["All_Beauty", "Books", "Gift Cards"],
        "writing_style": (
            "Polished Abuja cosmopolitan English. Reviews are balanced and well-structured. "
            "Averages 45-65 words. Uses phrases like 'all things considered', "
            "'I would suggest', 'for context'. "
            "Compares local and international options thoughtfully."
        ),
        "typical_phrases": [
            "For the Abuja market, this is fairly priced.",
            "Not bad by Nigerian standards, but there's room for improvement.",
            "I appreciate that this brand is transparent about ingredients.",
            "A solid option if you're looking for something reliable.",
            "Would make a great gift - presentation matters here.",
        ],
        "rating_pattern": "Thoughtful rater - averages 3.7, considers context",
        "nigerian_context": "Abuja cosmopolitan, development sector, values transparency and ethical consumption",
    },
    {
        "id": "nija_008",
        "name": "Tunde Balogun",
        "city": "lagos",
        "occupation": "Okrika (thrift clothing) merchant at Yaba Market",
        "age_range": "25-32",
        "shopping_style": (
            "Sources inventory from Lagos thrift markets (Yaba, Katangua, Oshodi). "
            "Expert at evaluating product quality by touch and inspection. "
            "Knows fabric grades, brand authentication, and market pricing. "
            "Sells via Instagram and WhatsApp status. "
            "Price negotiator - understands value chain deeply."
        ),
        "price_sensitivity": "high",
        "category_preferences": ["All_Beauty", "Fashion", "Gift Cards"],
        "writing_style": (
            "Street-smart Lagos English with sharp observations. "
            "Reviews are direct and market-savvy. Averages 30-45 words. "
            "Uses phrases like 'first grade', 'no be China', 'original quality'. "
            "Expert at spotting counterfeits and grading quality."
        ),
        "typical_phrases": [
            "As person wey dey sell this thing, I know quality when I see am.",
            "No be every 'original' na genuine. This one pass.",
            "For what I paid, the quality surprise me.",
            "If you sabi market, you go know say this one correct.",
            "I fit recommend this to my own customers.",
        ],
        "rating_pattern": "Expert rater - averages 3.4, judges by material and craftsmanship",
        "nigerian_context": "Lagos thrift ecosystem, understands secondhand grading, authenticates products professionally",
    },
    {
        "id": "nija_009",
        "name": "Grace Edet",
        "city": "calabar",
        "occupation": "Registered nurse and skincare entrepreneur",
        "age_range": "29-36",
        "shopping_style": (
            "Shops at Marian Market and Spar Calabar for daily needs. "
            "Makes her own skincare products and buys ingredients locally. "
            "Values natural, organic ingredients - critical of harsh chemicals. "
            "Teaches skincare routines on her YouTube channel. "
            "Community-focused - shares good finds with her network."
        ),
        "price_sensitivity": "low",
        "category_preferences": ["All_Beauty", "Grocery_Gourmet_Food", "Books"],
        "writing_style": (
            "Warm, educational tone with Calabar hospitality influences. "
            "Reviews are instructive and ingredient-focused, averaging 55-75 words. "
            "Uses phrases like 'from a skincare perspective', 'I would advise', "
            "'your skin will thank you'. Medical terminology for credibility."
        ),
        "typical_phrases": [
            "As a skincare formulator, I can vouch for these ingredients.",
            "Your melanin-rich skin deserves this level of care.",
            "Finally, a product that understands Nigerian weather and skin!",
            "I've recommended this to my clients and feedback has been excellent.",
            "The formulation is gentle yet effective - rare to find.",
        ],
        "rating_pattern": "Informed rater - averages 3.8, evaluates scientifically",
        "nigerian_context": "Calabar skincare community, melanin-aware beauty, nurse-entrepreneur hybrid",
    },
    {
        "id": "nija_010",
        "name": "Yusuf Mohammed",
        "city": "jos",
        "occupation": "Secondary school teacher and part-time music producer",
        "age_range": "27-35",
        "shopping_style": (
            "Shops at Terminus Market and Jos Main Market. "
            "Values durability given Jos's cooler climate and hilly terrain. "
            "Spends on music production gear and educational materials. "
            "Middle-ground spender - splurges on music equipment, frugal elsewhere. "
            "Community-minded - buys from local vendors when possible."
        ),
        "price_sensitivity": "medium",
        "category_preferences": ["Digital_Music", "Electronics", "Books"],
        "writing_style": (
            "Creative, musician's Nigerian English with production terminology. "
            "Reviews are specific and technical for gear, casual for other items. "
            "Averages 35-55 words. Uses phrases like 'the sound profile', "
            "'build quality matters here', 'budget-friendly option'. "
            "Compares with studio standards."
        ),
        "typical_phrases": [
            "For a home studio setup, this is more than adequate.",
            "The clarity surprised me at this price point.",
            "As a teacher, I appreciate products that last.",
            "Not studio-grade, but perfect for beginners and intermediates.",
            "Considering the Jos weather, the durability is impressive.",
        ],
        "rating_pattern": "Nuanced rater - averages 3.6, rates music gear strictly",
        "nigerian_context": "Jos creative scene, balances teaching with music, climate-aware consumer",
    },
]


@dataclass
class NigerianPersona:
    """A single Nigerian user persona with all attributes."""

    id: str
    name: str
    city: str
    occupation: str
    age_range: str
    shopping_style: str
    price_sensitivity: str
    category_preferences: list[str]
    writing_style: str
    typical_phrases: list[str]
    rating_pattern: str
    nigerian_context: str

    def to_persona_text(self) -> str:
        """Convert to a persona description string for the profiling system."""
        return (
            f"{self.name} is a {self.age_range}-year-old {self.occupation} "
            f"based in {self.city.title()}. "
            f"{self.shopping_style} "
            f"Their price sensitivity is {self.price_sensitivity} and they "
            f"prefer {', '.join(self.category_preferences)}. "
            f"When writing reviews, they are {self.writing_style}"
        )

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "name": self.name,
            "city": self.city,
            "occupation": self.occupation,
            "age_range": self.age_range,
            "shopping_style": self.shopping_style,
            "price_sensitivity": self.price_sensitivity,
            "category_preferences": self.category_preferences,
            "writing_style": self.writing_style,
            "typical_phrases": self.typical_phrases,
            "rating_pattern": self.rating_pattern,
            "nigerian_context": self.nigerian_context,
        }


def get_nigerian_personas() -> list[NigerianPersona]:
    """Return all predefined Nigerian personas."""
    return [NigerianPersona(**data) for data in NIGERIAN_PERSONAS]


def get_persona_by_id(persona_id: str) -> NigerianPersona | None:
    """Look up a persona by its ID."""
    for data in NIGERIAN_PERSONAS:
        if data["id"] == persona_id:
            return NigerianPersona(**data)
    return None


def get_personas_by_city(city: str) -> list[NigerianPersona]:
    """Filter personas by city."""
    lower = city.lower().replace(" ", "_")
    return [
        NigerianPersona(**data)
        for data in NIGERIAN_PERSONAS
        if data["city"] == lower
    ]


def get_personas_by_price_sensitivity(level: str) -> list[NigerianPersona]:
    """Filter personas by price sensitivity level (low/medium/high)."""
    return [
        NigerianPersona(**data)
        for data in NIGERIAN_PERSONAS
        if data["price_sensitivity"] == level
    ]


def random_nigerian_persona() -> NigerianPersona:
    """Return a random Nigerian persona for testing."""
    import random
    return NigerianPersona(**random.choice(NIGERIAN_PERSONAS))
