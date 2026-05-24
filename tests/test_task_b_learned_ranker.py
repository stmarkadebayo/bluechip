from __future__ import annotations

from app.models.schemas import RecommendationItem
from app.services.ranking.learned_task_b import (
    DEFAULT_FEATURE_NAMES,
    TaskBLinearRanker,
    TaskBRankerArtifact,
    feature_vector,
)
from eval.train_task_b_ranker import TrainingExample, _train


def _recommendation(
    item_id: str,
    raw_score: float,
    retrieval_score: float = 0.0,
) -> RecommendationItem:
    return RecommendationItem(
        rank=1,
        item_id=item_id,
        name=item_id,
        score=max(min(raw_score, 1.0), 0.0),
        reason="",
        tradeoffs="",
        signals=[],
        candidate_sources=["review_term_profile"] if retrieval_score else [],
        retrieval_scores={"review_term_profile": retrieval_score} if retrieval_score else {},
        score_components={
            "raw_score": raw_score,
            "preference_match": raw_score,
            "retrieval_match": retrieval_score,
        },
    )


def test_task_b_feature_vector_reads_components_and_retrieval_scores() -> None:
    item = _recommendation("target", raw_score=0.7, retrieval_score=0.9)

    vector = feature_vector(
        item,
        (
            "component:raw_score",
            "retrieval:review_term_profile",
            "source_family:lexical_review_term",
            "meta:candidate_source_count",
            "meta:retrieval_score_max",
        ),
    )

    assert vector["component:raw_score"] == 0.7
    assert vector["retrieval:review_term_profile"] == 0.9
    assert vector["source_family:lexical_review_term"] == 0.9
    assert vector["meta:candidate_source_count"] > 0
    assert vector["meta:retrieval_score_max"] == 0.9


def test_task_b_linear_ranker_reranks_without_mutating_inputs() -> None:
    artifact = TaskBRankerArtifact(
        feature_names=("component:raw_score", "retrieval:review_term_profile"),
        weights={"component:raw_score": -1.0, "retrieval:review_term_profile": 3.0},
    )
    ranker = TaskBLinearRanker(artifact)
    original = [
        _recommendation("baseline_top", raw_score=0.9, retrieval_score=0.0),
        _recommendation("learned_top", raw_score=0.4, retrieval_score=0.8),
    ]

    reranked = ranker.rerank(original)

    assert [item.item_id for item in reranked] == ["learned_top", "baseline_top"]
    assert [item.rank for item in reranked] == [1, 2]
    assert "learned_ranker_score" in reranked[0].score_components
    assert "learned_ranker_score" not in original[0].score_components


def test_task_b_ranker_artifact_round_trips(tmp_path) -> None:
    path = tmp_path / "ranker.json"
    artifact = TaskBRankerArtifact(
        feature_names=("component:raw_score",),
        weights={"component:raw_score": 1.25},
        intercept=-0.5,
        training={"examples": 2},
    )

    artifact.write_json(path)
    loaded = TaskBRankerArtifact.from_json(path)

    assert loaded.feature_names == ("component:raw_score",)
    assert loaded.weights == {"component:raw_score": 1.25}
    assert loaded.intercept == -0.5
    assert loaded.training["examples"] == 2


def test_task_b_train_learns_positive_feature_direction() -> None:
    examples = [
        TrainingExample(
            features=feature_vector(_recommendation("positive", 0.9), DEFAULT_FEATURE_NAMES),
            label=1,
        ),
        TrainingExample(
            features=feature_vector(_recommendation("negative", 0.1), DEFAULT_FEATURE_NAMES),
            label=0,
        ),
    ]

    artifact = _train(
        examples=examples,
        feature_names=DEFAULT_FEATURE_NAMES,
        epochs=25,
        learning_rate=0.1,
        l2=0.0,
        training={},
    )
    ranker = TaskBLinearRanker(artifact)

    assert ranker.score(_recommendation("positive", 0.9)) > ranker.score(
        _recommendation("negative", 0.1)
    )
