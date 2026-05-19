from __future__ import annotations

from eval.eval_task_a_generation import rouge_l_f1, sentiment_aligned, tokenize, unigram_f1


def test_generation_eval_text_metrics_reward_overlap() -> None:
    reference = "quiet affordable grill with fast service"
    candidate = "quiet grill with reliable fast service"

    assert tokenize(reference) == ["quiet", "affordable", "grill", "with", "fast", "service"]
    assert rouge_l_f1(reference, candidate) > 0.6
    assert unigram_f1(reference, candidate) > 0.6


def test_generation_eval_sentiment_alignment_uses_rating_direction() -> None:
    assert sentiment_aligned(5, "This is great and useful.")
    assert not sentiment_aligned(5, "This is terrible and awful.")
    assert sentiment_aligned(1, "I would be cautious because it misses my needs.")
