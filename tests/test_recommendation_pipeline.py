from __future__ import annotations

from app.agents.recommendation_agent import RecommendationAgent
from app.models.schemas import Item, RecommendationRequest, UserHistoryItem
from app.serving.orchestrators import recommendation as recommendation_orchestrator
from app.services.profiling.user_profile import build_user_profile
from app.services.ranking.context_intents import context_category_hint, context_intent_expansion
from app.services.ranking.learned_task_b import TaskBLinearRanker, TaskBRankerArtifact
from app.services.ranking.recommendation import adaptive_recommendation_policy, rank_candidates
from app.services.retrieval.candidates import generate_candidate_pool
from app.services.retrieval.item_similarity import (
    build_collaborative_retrieval_index,
    build_review_term_retrieval_index,
)
from scripts.build_splits import build_temporal_splits


def test_recommendation_agent_filters_seen_items() -> None:
    request = RecommendationRequest(
        user_persona="A practical listener who likes calm acoustic music.",
        user_history=[
            UserHistoryItem(
                item_id="song_seen",
                item_name="Quiet Guitar",
                rating=5,
                review="Calm acoustic guitar and a very relaxing listen.",
                category="music",
            )
        ],
        candidate_items=[
            Item(
                item_id="song_seen",
                name="Quiet Guitar",
                category="music",
                summary="Calm acoustic guitar and relaxing production.",
                average_rating=4.8,
            ),
            Item(
                item_id="song_new",
                name="Soft Piano",
                category="music",
                summary="Calm instrumental piano for focused listening.",
                average_rating=4.7,
            ),
        ],
        limit=2,
    )

    response = RecommendationAgent().run(request)

    assert [item.item_id for item in response.recommendations] == ["song_new"]


def test_cold_start_policy_enriches_active_ranking_path(monkeypatch) -> None:
    monkeypatch.setattr(recommendation_orchestrator, "_load_neural_index", lambda: None)
    request = RecommendationRequest(
        user_persona=(
            "A Lagos-based student on a tight budget who wants calm places, "
            "fast service, spicy food, and no surprise fees."
        ),
        user_history=[],
        context="Dinner with two friends after class. Keep it affordable and conversation-friendly.",
        candidate_items=[
            Item(
                item_id="popular_loud",
                name="Party Yard",
                category="restaurant",
                metadata={"price": "premium", "ambience": "loud", "review_count": 900},
                summary="Loud premium outdoor seating with a longer wait.",
                average_rating=4.8,
            ),
            Item(
                item_id="quiet_value",
                name="Campus Pepper Bowl",
                category="restaurant",
                metadata={"price": "low", "ambience": "quiet", "review_count": 120},
                summary="Affordable spicy bowls, quiet seating, and fast counter service.",
                average_rating=4.2,
            ),
        ],
        limit=2,
    )

    response = RecommendationAgent().run(request)

    assert response.candidate_diagnostics is not None
    assert "cold_start" in response.candidate_diagnostics.ranking_policy
    assert response.candidate_diagnostics.semantic_retrieval_enabled
    assert response.recommendations[0].score_components["personalization_weight"] >= 0.22


def test_rejected_item_ids_are_removed_before_ranking(monkeypatch) -> None:
    monkeypatch.setattr(recommendation_orchestrator, "_load_neural_index", lambda: None)
    request = RecommendationRequest(
        user_persona="A practical diner who wants quiet affordable restaurants.",
        user_history=[],
        context="Dinner where conversation is easy.",
        candidate_items=[
            Item(
                item_id="rejected_loud",
                name="Loud Premium Yard",
                category="restaurant",
                metadata={"review_count": 5000},
                summary="Very popular but loud premium seating.",
                average_rating=4.9,
            ),
            Item(
                item_id="quiet_value",
                name="Quiet Value Grill",
                category="restaurant",
                metadata={"review_count": 50},
                summary="Quiet affordable grill for conversation.",
                average_rating=4.2,
            ),
        ],
        rejected_item_ids=["rejected_loud"],
        limit=2,
    )

    response = RecommendationAgent().run(request)

    assert [item.item_id for item in response.recommendations] == ["quiet_value"]


def test_serving_applies_optional_learned_task_b_ranker(monkeypatch) -> None:
    monkeypatch.setattr(recommendation_orchestrator, "_load_neural_index", lambda: None)
    monkeypatch.setattr(
        recommendation_orchestrator.RecommenderReasoner,
        "rerank_candidates",
        lambda self, **kwargs: [],
    )
    ranker = TaskBLinearRanker(
        TaskBRankerArtifact(
            feature_names=("component:raw_score", "retrieval:context_intent_profile"),
            weights={
                "component:raw_score": -0.5,
                "retrieval:context_intent_profile": 4.0,
            },
        )
    )
    monkeypatch.setattr(recommendation_orchestrator, "_load_task_b_ranker", lambda: ranker)
    request = RecommendationRequest(
        user_persona="A budget shopper looking for practical beauty products.",
        user_history=[],
        context="Need a hair styling product for curls.",
        candidate_items=[
            Item(
                item_id="generic_popular",
                name="Popular Body Mist",
                category="All_Beauty",
                metadata={"review_count": 5000},
                summary="Popular fragrance body mist.",
                average_rating=4.9,
            ),
            Item(
                item_id="hair_target",
                name="Curl Styling Cream",
                category="All_Beauty",
                metadata={"review_count": 35},
                summary="Hair styling cream for curls and natural hair.",
                average_rating=4.1,
            ),
        ],
        limit=2,
    )

    response = RecommendationAgent().run(request)

    assert response.recommendations[0].item_id == "hair_target"
    assert "learned_ranker_score" in response.recommendations[0].score_components
    assert any(step.step == "learned_ranker" for step in response.agent_trace)


def test_profile_keeps_cold_start_identity_terms() -> None:
    profile = build_user_profile(
        "A Lagos-based student who needs quiet affordable places near campus.",
        [],
    )

    assert "lagos-based" in profile.preferred_terms
    assert "student" in profile.preferred_terms


def test_adaptive_policy_changes_for_sparse_contextual_profiles() -> None:
    profile = build_user_profile(
        "A budget-conscious student who likes affordable quiet places.",
        [],
    )

    policy = adaptive_recommendation_policy(
        user_profile=profile,
        context="Affordable dinner where conversation is easy.",
        strategy="cold_start",
    )

    assert "cold_start" in policy.name
    assert "budget_sensitive" in policy.name
    assert policy.base_popularity < 0.85
    assert policy.weights.context > 0.16


def test_candidate_pool_uses_collaborative_sources() -> None:
    train = [
        {
            "user_id": "u1",
            "item_id": "seed",
            "item_name": "Quiet Guitar",
            "rating": 5,
            "review": "Calm acoustic guitar.",
            "category": "music",
            "timestamp": 1,
        },
        {
            "user_id": "u1",
            "item_id": "neighbor",
            "item_name": "Soft Piano",
            "rating": 5,
            "review": "Calm piano.",
            "category": "music",
            "timestamp": 2,
        },
        {
            "user_id": "u2",
            "item_id": "seed",
            "item_name": "Quiet Guitar",
            "rating": 5,
            "review": "Relaxing.",
            "category": "music",
            "timestamp": 3,
        },
        {
            "user_id": "u2",
            "item_id": "neighbor",
            "item_name": "Soft Piano",
            "rating": 5,
            "review": "Focused listening.",
            "category": "music",
            "timestamp": 4,
        },
    ]
    collaborative_index = build_collaborative_retrieval_index(train, top_k=5)
    history = [
        UserHistoryItem(
            item_id="seed",
            item_name="Quiet Guitar",
            rating=5,
            review="Calm acoustic guitar and a very relaxing listen.",
            category="music",
        )
    ]
    user_profile = build_user_profile("A listener who likes calm acoustic music.", history)
    items = [
        Item(item_id="seed", name="Quiet Guitar", category="music"),
        Item(item_id="neighbor", name="Soft Piano", category="music"),
    ]

    pool = generate_candidate_pool(
        user_profile=user_profile,
        history=history,
        items=items,
        context="",
        collaborative_index=collaborative_index,
        limit=5,
    )

    assert [item.item_id for item in pool.items] == ["neighbor"]
    assert set(pool.sources["neighbor"]) & {"co_visitation", "user_neighbor"}


def test_candidate_pool_uses_implicit_item_item_source() -> None:
    history = [
        UserHistoryItem(
            item_id="seed",
            item_name="Quiet Guitar",
            rating=5,
            review="Calm acoustic guitar and a very relaxing listen.",
            category="music",
        )
    ]
    user_profile = build_user_profile("A listener who likes calm acoustic music.", history)
    items = [
        Item(item_id="seed", name="Quiet Guitar", category="music"),
        Item(item_id="implicit_neighbor", name="Soft Piano", category="music"),
        Item(item_id="unrelated", name="Fast Drums", category="music"),
    ]

    pool = generate_candidate_pool(
        user_profile=user_profile,
        history=history,
        items=items,
        context="",
        collaborative_index={
            "implicit_item_neighbors": {
                "seed": [["implicit_neighbor", 0.91]],
            }
        },
        limit=3,
    )

    assert "implicit_neighbor" in {item.item_id for item in pool.items}
    assert "implicit_item_item" in pool.sources["implicit_neighbor"]
    assert pool.source_scores["implicit_neighbor"]["implicit_item_item"] == 1.0


def test_candidate_pool_uses_review_term_sources_for_lexical_neighbors() -> None:
    train = [
        {
            "user_id": "u1",
            "item_id": "seed",
            "item_name": "Gentle Face Cream",
            "rating": 5,
            "review": "Gentle fragrance free cream for sensitive skin.",
            "category": "All_Beauty",
            "timestamp": 1,
        },
        {
            "user_id": "u2",
            "item_id": "target",
            "item_name": "Sensitive Skin Moisturizer",
            "rating": 5,
            "review": "Fragrance free moisturizer calms sensitive skin.",
            "category": "All_Beauty",
            "timestamp": 2,
        },
    ]
    items = [
        Item(item_id="seed", name="Gentle Face Cream", category="All_Beauty"),
        Item(
            item_id="target",
            name="Sensitive Skin Moisturizer",
            category="All_Beauty",
            summary="Fragrance free moisturizer for sensitive skin.",
            average_rating=4.8,
        ),
        Item(item_id="other", name="Party Glitter Spray", category="All_Beauty"),
    ]
    review_term_index = build_review_term_retrieval_index(
        train,
        items=[item.model_dump() for item in items],
    )
    history = [
        UserHistoryItem(
            item_id="seed",
            item_name="Gentle Face Cream",
            rating=5,
            review="Gentle fragrance free cream for sensitive skin.",
            category="All_Beauty",
        )
    ]
    user_profile = build_user_profile("Needs fragrance free products for sensitive skin.", history)

    pool = generate_candidate_pool(
        user_profile=user_profile,
        history=history,
        items=items,
        context="",
        collaborative_index={"review_term_retrieval": review_term_index},
        limit=3,
    )

    assert "target" in {item.item_id for item in pool.items}
    assert set(pool.sources["target"]) & {
        "beauty_review_term_profile",
        "beauty_lexical_item_neighbor",
    }


def test_context_intent_expansion_reuses_rule_terms() -> None:
    expansion = context_intent_expansion(["need", "hair", "styling", "under", "budget"])

    assert expansion is not None
    assert expansion.intent_name == "hair"
    assert expansion.category_hint == "All_Beauty"
    assert "conditioner" in expansion.retrieval_terms
    assert "styling" in expansion.required_terms


def test_candidate_pool_retrieves_context_intent_profile_candidates() -> None:
    history: list[UserHistoryItem] = []
    user_profile = build_user_profile(
        "A budget shopper looking for practical beauty products.",
        history,
    )
    items = [
        Item(
            item_id="generic_popular",
            name="Popular Body Mist",
            category="All_Beauty",
            metadata={"review_count": 5000},
            summary="Popular fragrance body mist.",
            average_rating=4.8,
        ),
        Item(
            item_id="hair_target",
            name="Curl Styling Cream",
            category="All_Beauty",
            metadata={"review_count": 35},
            summary="Hair styling cream for curls and natural hair.",
            average_rating=4.1,
        ),
    ]

    pool = generate_candidate_pool(
        user_profile=user_profile,
        history=history,
        items=items,
        context="Need a hair styling product for curls.",
        disabled_sources={"global_popular", "category_affinity_popular", "category_popular"},
        limit=5,
    )

    assert "hair_target" in {item.item_id for item in pool.items}
    assert "context_intent_profile" in pool.sources["hair_target"]


def test_candidate_pool_retrieves_context_intent_category_candidates() -> None:
    history: list[UserHistoryItem] = []
    user_profile = build_user_profile("A careful buyer looking for low-risk gifts.", history)
    items = [
        Item(
            item_id="gift_target",
            name="Elegant Choice",
            category="For Her",
            metadata={"review_count": 40},
            summary="A safe pick with broad appeal.",
            average_rating=4.2,
        ),
        Item(
            item_id="unrelated",
            name="Everyday Shampoo",
            category="All_Beauty",
            metadata={"review_count": 4000},
            summary="Shampoo for daily washing.",
            average_rating=4.7,
        ),
    ]

    pool = generate_candidate_pool(
        user_profile=user_profile,
        history=history,
        items=items,
        context="Need a low-risk gift.",
        disabled_sources={"global_popular", "category_affinity_popular", "category_popular"},
        limit=5,
    )

    assert "gift_target" in {item.item_id for item in pool.items}
    assert "context_intent_category" in pool.sources["gift_target"]


def test_candidate_pool_includes_beauty_taxonomy_source() -> None:
    history = [
        UserHistoryItem(
            item_id="seen_skin_cream",
            item_name="Gentle Skin Cream",
            rating=5,
            review="Gentle cream calmed sensitive skin without fragrance.",
            category="All_Beauty",
        )
    ]
    user_profile = build_user_profile("Needs gentle beauty products for skin care.", history)
    items = [
        Item(
            item_id="seen_skin_cream",
            name="Gentle Skin Cream",
            category="All_Beauty",
            summary="Fragrance free cream for sensitive skin.",
            average_rating=4.8,
            metadata={"review_count": 120},
        ),
        Item(
            item_id="taxonomy_skin_serum",
            name="Hyaluronic Face Serum",
            category="All_Beauty",
            summary="Light hyaluronic serum for face care.",
            average_rating=4.7,
            metadata={"review_count": 40},
        ),
        Item(
            item_id="nail_art_kit",
            name="Nail Rhinestone Kit",
            category="All_Beauty",
            summary="Acrylic nail art rhinestones and polish tools.",
            average_rating=4.6,
            metadata={"review_count": 80},
        ),
    ]

    pool = generate_candidate_pool(
        user_profile=user_profile,
        history=history,
        items=items,
        context="",
        limit=3,
    )

    assert "taxonomy_skin_serum" in {item.item_id for item in pool.items}
    assert "beauty_taxonomy_aspect" in pool.sources["taxonomy_skin_serum"]
    assert pool.source_scores["taxonomy_skin_serum"]["beauty_taxonomy_aspect"] > 0


def test_candidate_pool_uses_category_affinity_source() -> None:
    history = [
        UserHistoryItem(
            item_id="book_seen",
            item_name="Quiet Study Guide",
            rating=5,
            review="Clear practical book with strong examples.",
            category="books",
        )
    ]
    user_profile = build_user_profile("A practical reader who likes useful books.", history)
    items = [
        Item(
            item_id="book_seen",
            name="Quiet Study Guide",
            category="books",
            metadata={"review_count": 100},
            average_rating=4.8,
        ),
        Item(
            item_id="book_new",
            name="Exam Prep Manual",
            category="books",
            metadata={"review_count": 80},
            summary="Structured revision chapters.",
            average_rating=4.6,
        ),
        Item(
            item_id="music_popular",
            name="Chart Playlist",
            category="music",
            metadata={"review_count": 5000},
            average_rating=4.7,
        ),
    ]

    pool = generate_candidate_pool(
        user_profile=user_profile,
        history=history,
        items=items,
        context="",
        limit=2,
    )

    assert "book_new" in {item.item_id for item in pool.items}
    assert "category_affinity_popular" in pool.sources["book_new"]


def test_candidate_pool_preserves_global_popularity_floor() -> None:
    history = [
        UserHistoryItem(
            item_id="seen_beauty",
            item_name="Hair Styling Cream",
            rating=5,
            review="Useful hair cream for daily styling.",
            category="All_Beauty",
        )
    ]
    user_profile = build_user_profile("Needs practical beauty products.", history)
    items = [
        Item(
            item_id="seen_beauty",
            name="Hair Styling Cream",
            category="All_Beauty",
            summary="Useful hair cream for daily styling.",
            metadata={"review_count": 2000},
            average_rating=4.8,
        ),
        *[
            Item(
                item_id=f"higher_popular_{index}",
                name=f"Very Popular Gift {index}",
                category="Specialty Cards",
                metadata={"review_count": 2000 - index},
                average_rating=3.2,
            )
            for index in range(14)
        ],
        Item(
            item_id="popular_cross_domain",
            name="Popular Cross Domain Gift",
            category="Specialty Cards",
            metadata={"review_count": 1500},
            average_rating=3.2,
        ),
        *[
            Item(
                item_id=f"high_quality_tail_{index}",
                name=f"High Quality Tail Gift {index}",
                category="Specialty Cards",
                metadata={"review_count": 100 - index},
                average_rating=5.0,
            )
            for index in range(30)
        ],
        *[
            Item(
                item_id=f"beauty_candidate_{index}",
                name=f"Hair Brush Styling Tool {index}",
                category="All_Beauty",
                summary="Hair brush styling tool for daily hair care.",
                metadata={"review_count": 120 + index},
                average_rating=4.8,
            )
            for index in range(120)
        ],
    ]

    pool = generate_candidate_pool(
        user_profile=user_profile,
        history=history,
        items=items,
        context="",
        limit=100,
    )

    assert "popular_cross_domain" in {item.item_id for item in pool.items}
    assert pool.sources["popular_cross_domain"] == ["global_popular"]


def test_context_category_guard_keeps_contextual_recommendations_on_topic() -> None:
    history = [
        UserHistoryItem(
            item_id="seen_beauty",
            item_name="Gentle Hair Cream",
            rating=5,
            review="Useful hair cream for regular styling.",
            category="All_Beauty",
        )
    ]
    user_profile = build_user_profile("Needs practical beauty products.", history)
    ranked = rank_candidates(
        user_profile=user_profile,
        context="Needs a practical beauty product for hair styling.",
        candidate_items=[
            Item(
                item_id="gift_card",
                name="Popular Gift Card",
                category="Specialty Cards",
                metadata={"review_count": 10000},
                average_rating=4.9,
            ),
            Item(
                item_id="hair_brush",
                name="Soft Hair Brush",
                category="All_Beauty",
                metadata={"review_count": 100},
                average_rating=4.5,
            ),
        ],
        limit=2,
    )

    assert ranked[0].item_id == "hair_brush"
    assert ranked[0].score_components["context_category_boost"] > 0
    assert ranked[1].score_components["context_category_penalty"] > 0


def test_context_subintent_guard_prefers_hair_item_over_generic_beauty() -> None:
    history = [
        UserHistoryItem(
            item_id="seen_beauty",
            item_name="Gentle Hair Cream",
            rating=5,
            review="Useful hair cream for regular styling.",
            category="All_Beauty",
        )
    ]
    user_profile = build_user_profile("Needs practical beauty products.", history)
    ranked = rank_candidates(
        user_profile=user_profile,
        context="Needs a practical hair or styling product for regular use.",
        candidate_items=[
            Item(
                item_id="generic_beauty",
                name="Japanese Skin Bath Cloth",
                category="All_Beauty",
                metadata={"review_count": 9000},
                average_rating=4.9,
            ),
            Item(
                item_id="hair_spray",
                name="Texturizing Hair Spray",
                category="All_Beauty",
                metadata={"review_count": 90},
                average_rating=4.4,
            ),
        ],
        limit=2,
    )

    assert ranked[0].item_id == "hair_spray"
    assert ranked[0].score_components["context_intent_boost"] > 0
    assert ranked[1].score_components["context_intent_penalty"] > 0


def test_context_hint_prioritizes_gift_when_context_mentions_beauty_gift() -> None:
    assert context_category_hint(["beauty", "gift"]) == "gift"


def test_ranker_sorts_by_raw_score_not_rounded_display_score() -> None:
    history = [
        UserHistoryItem(
            item_id=f"seen_{index}",
            item_name="Calm Study Manual",
            rating=5 if index % 2 else 4,
            review="calm useful practical reliable quiet",
            category="books",
            timestamp=index,
        )
        for index in range(8)
    ]
    user_profile = build_user_profile(
        "A user who likes calm useful practical reliable books for study.",
        history,
    )

    ranked = rank_candidates(
        user_profile=user_profile,
        context="",
        candidate_items=[
            Item(
                item_id="lower_raw",
                name="Useful Calm Guide",
                category="books",
                summary="calm useful practical reliable study",
                metadata={"review_count": 100},
                average_rating=4.5,
            ),
            Item(
                item_id="higher_raw",
                name="Useful Calm Guide",
                category="books",
                summary="calm useful practical reliable study",
                metadata={"review_count": 100},
                average_rating=4.5,
            ),
        ],
        limit=2,
        candidate_source_scores={
            "lower_raw": {"review_term_profile": 0.90},
            "higher_raw": {"review_term_profile": 0.94},
        },
    )

    assert ranked[0].item_id == "higher_raw"
    assert ranked[0].score == ranked[1].score
    assert ranked[0].score_components["raw_score"] > ranked[1].score_components["raw_score"]


def test_temporal_split_holds_out_latest_review() -> None:
    reviews = [
        {
            "review_id": "r1",
            "user_id": "u1",
            "item_id": "a",
            "item_name": "A",
            "rating": 4,
            "review": "first",
            "timestamp": 10,
        },
        {
            "review_id": "r2",
            "user_id": "u1",
            "item_id": "b",
            "item_name": "B",
            "rating": 5,
            "review": "second",
            "timestamp": 20,
        },
    ]

    train, test_a, test_b = build_temporal_splits(reviews, min_history=1)

    assert [row["review_id"] for row in train] == ["r1"]
    assert [row["review_id"] for row in test_a] == ["r2"]
    assert test_b == test_a
