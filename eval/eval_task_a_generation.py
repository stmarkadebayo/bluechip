from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.models.schemas import Item  # noqa: E402
from app.models.schemas import UserHistoryItem  # noqa: E402
from app.services.generation.generator import generate_review_result  # noqa: E402
from app.services.generation.providers import generation_provider_name  # noqa: E402
from app.services.profiling.item_profile import build_item_profile  # noqa: E402
from app.services.profiling.user_profile import build_user_profile  # noqa: E402
from app.services.ranking.rating import predict_rating  # noqa: E402
from app.services.ranking.rating_features import build_rating_stats, clamp_rating  # noqa: E402
from app.services.ranking.task_a_model import load_task_a_model  # noqa: E402
from app.services.validation.critic import validate_review_simulation  # noqa: E402
from eval.common import histories_by_user, load_eval_data, persona_from_history, print_report, write_report  # noqa: E402
from eval.metrics import rounded  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate Task A generated review quality.")
    parser.add_argument("--reviews", default="data/sample/reviews.jsonl")
    parser.add_argument("--items", default="data/sample/items.jsonl")
    parser.add_argument("--processed-dir", default="data/processed")
    parser.add_argument("--output", default="runs/eval/task_a_generation_report.json")
    parser.add_argument("--model-path", default="", help="Optional trained Task A rating model.")
    parser.add_argument("--max-examples", type=int, default=25)
    parser.add_argument(
        "--strict-provider",
        action="store_true",
        help="Fail each generation on provider errors instead of using deterministic fallback.",
    )
    parser.add_argument(
        "--external-data-policy",
        choices=["deny", "redact", "allow"],
        default="deny",
        help=(
            "Data handling for external providers. 'deny' blocks non-sample rows, "
            "'redact' sends generic synthetic prompts, and 'allow' sends eval rows."
        ),
    )
    args = parser.parse_args()
    provider_name = generation_provider_name()
    if args.strict_provider and provider_name == "template":
        raise SystemExit(
            "Strict generation eval requires a configured provider. "
            "Set OPENROUTER_API_KEY or use LLM_PROVIDER=mock for local smoke tests."
        )
    external_provider = provider_name in {"deepseek", "openrouter", "openai"}
    if external_provider and args.external_data_policy == "deny" and _is_non_sample_dataset(args):
        raise SystemExit(
            "External generation eval would send local eval rows to an LLM provider. "
            "Use --external-data-policy redact for a safe provider smoke, or --external-data-policy allow "
            "only after explicit approval to export eval row content."
        )

    train, test_a, _, items = load_eval_data(
        reviews_path=Path(args.reviews),
        items_path=Path(args.items),
        processed_dir=Path(args.processed_dir),
    )
    if args.max_examples:
        test_a = test_a[: args.max_examples]
    rating_stats = build_rating_stats(train)
    histories = histories_by_user(train)
    model = load_task_a_model(Path(args.model_path)) if args.model_path else None

    scored = []
    failures = []
    for row in test_a:
        history = histories.get(row["user_id"], [])
        user_profile = build_user_profile(persona_from_history(history), history, locale=None)
        item_profile = build_item_profile(_target_item(row, items, rating_stats.global_mean))
        predicted_rating = _predicted_rating(
            row=row,
            user_profile=user_profile,
            item_profile=item_profile,
            rating_stats=rating_stats,
            model=model,
        )
        try:
            generation_user_profile = user_profile
            generation_item_profile = item_profile
            reference_review = row.get("review") or ""
            if external_provider and args.external_data_policy == "redact":
                generation_user_profile, generation_item_profile, reference_review = _redacted_generation_inputs(
                    category=row.get("category") or item_profile.category,
                    predicted_rating=predicted_rating,
                )
            generated = generate_review_result(
                user_profile=generation_user_profile,
                item_profile=generation_item_profile,
                predicted_rating=predicted_rating,
                strict_provider=args.strict_provider,
            )
        except RuntimeError as exc:
            failures.append(
                {
                    "user_id": row["user_id"],
                    "item_id": row["item_id"],
                    "error": str(exc),
                }
            )
            continue

        validation = validate_review_simulation(
            predicted_rating=predicted_rating,
            review=generated.text,
            user_profile=generation_user_profile,
            item_profile=generation_item_profile,
        )
        scored.append(
            {
                "user_id": row["user_id"],
                "item_id": row["item_id"],
                "item_name": item_profile.name,
                "actual_rating": row["rating"],
                "predicted_rating": predicted_rating,
                "reference_review": reference_review,
                "generated_review": generated.text,
                "provider": generated.provider,
                "external_data_policy": args.external_data_policy,
                "used_fallback": generated.used_fallback,
                "validation_consistent": validation.is_consistent,
                "validation_issues": validation.issues,
                "rating_mentioned": str(predicted_rating) in generated.text,
                "item_mentioned": generation_item_profile.name.lower() in generated.text.lower(),
                "rouge_l_f1": rouge_l_f1(reference_review, generated.text),
                "unigram_f1": unigram_f1(reference_review, generated.text),
                "sentiment_aligned": sentiment_aligned(predicted_rating, generated.text),
            }
        )

    metrics = _metrics(scored=scored, failures=failures, total=len(test_a))
    payload = {
        "task": "Task A Generation",
        "dataset": str(Path(args.processed_dir)),
        "examples": len(test_a),
        "provider": provider_name,
        "strict_provider": args.strict_provider,
        "external_data_policy": args.external_data_policy,
        "metrics": metrics,
        "samples": scored[:5],
        "failures": failures[:5],
        "notes": [
            "ROUGE-L and unigram F1 are dependency-free lexical proxies for review text quality.",
            "Consistency checks verify rating mention, target item grounding, and basic rating-sentiment alignment.",
            "Use --strict-provider for DeepSeek/OpenRouter/OpenAI runs so provider failures are visible instead of hidden by fallback text.",
            "External providers default to deny for non-sample datasets; redact mode sends synthetic prompts only.",
        ],
    }
    write_report(Path(args.output), payload)
    print_report(payload)


def _predicted_rating(row, user_profile, item_profile, rating_stats, model) -> int:
    if model is not None:
        score = model.predict_raw(
            user_profile=user_profile,
            item_profile=item_profile,
            stats=rating_stats,
            user_id=row["user_id"],
        )
        return int(round(clamp_rating(score)))
    return predict_rating(user_profile, item_profile, user_id=row["user_id"]).predicted_rating


def _target_item(row: dict, items: dict[str, Item], fallback_rating: float) -> Item:
    target = items.get(row["item_id"])
    if target is not None:
        return target
    return Item(
        item_id=row["item_id"],
        name=row["item_name"],
        category=row.get("category") or "unknown",
        metadata={},
        summary=row.get("review") or "",
        average_rating=fallback_rating,
    )


def _is_non_sample_dataset(args: argparse.Namespace) -> bool:
    processed_dir = Path(args.processed_dir)
    reviews_path = Path(args.reviews)
    return "sample" not in str(processed_dir) and "sample" not in str(reviews_path)


def _redacted_generation_inputs(
    category: str,
    predicted_rating: int,
) -> tuple[object, object, str]:
    sentiment = "liked" if predicted_rating >= 4 else "mixed" if predicted_rating == 3 else "disliked"
    history = [
        UserHistoryItem(
            item_id="synthetic-history",
            item_name="Prior Item",
            rating=max(min(predicted_rating, 5), 1),
            review=f"The user had a {sentiment} prior experience in this category.",
            category=category,
        )
    ]
    user_profile = build_user_profile(
        persona=f"A synthetic evaluator profile for {category} with no private review text.",
        history=history,
    )
    item_profile = build_item_profile(
        Item(
            item_id="synthetic-target",
            name="Target Item",
            category=category,
            metadata={},
            summary=f"A redacted {category} item used for provider smoke testing.",
            average_rating=float(predicted_rating),
        )
    )
    reference = f"Synthetic reference: {sentiment} {category} item."
    return user_profile, item_profile, reference


def _metrics(scored: list[dict], failures: list[dict], total: int) -> dict[str, float]:
    success_count = len(scored)
    if not scored:
        return {
            "generation_success_rate": 0.0,
            "provider_failure_rate": rounded(len(failures) / total) if total else 0.0,
            "fallback_rate": 0.0,
            "validation_consistency_rate": 0.0,
            "rating_mention_rate": 0.0,
            "item_mention_rate": 0.0,
            "sentiment_alignment_rate": 0.0,
            "rouge_l_f1": 0.0,
            "unigram_f1": 0.0,
        }
    return {
        "generation_success_rate": rounded(success_count / total) if total else 0.0,
        "provider_failure_rate": rounded(len(failures) / total) if total else 0.0,
        "fallback_rate": rounded(_mean([row["used_fallback"] for row in scored])),
        "validation_consistency_rate": rounded(_mean([row["validation_consistent"] for row in scored])),
        "rating_mention_rate": rounded(_mean([row["rating_mentioned"] for row in scored])),
        "item_mention_rate": rounded(_mean([row["item_mentioned"] for row in scored])),
        "sentiment_alignment_rate": rounded(_mean([row["sentiment_aligned"] for row in scored])),
        "rouge_l_f1": rounded(_mean([row["rouge_l_f1"] for row in scored])),
        "unigram_f1": rounded(_mean([row["unigram_f1"] for row in scored])),
    }


def rouge_l_f1(reference: str, candidate: str) -> float:
    reference_tokens = tokenize(reference)
    candidate_tokens = tokenize(candidate)
    if not reference_tokens or not candidate_tokens:
        return 0.0
    lcs = _lcs_length(reference_tokens, candidate_tokens)
    precision = lcs / len(candidate_tokens)
    recall = lcs / len(reference_tokens)
    return _f1(precision, recall)


def unigram_f1(reference: str, candidate: str) -> float:
    reference_tokens = tokenize(reference)
    candidate_tokens = tokenize(candidate)
    if not reference_tokens or not candidate_tokens:
        return 0.0
    reference_counts = {}
    for token in reference_tokens:
        reference_counts[token] = reference_counts.get(token, 0) + 1
    overlap = 0
    for token in candidate_tokens:
        if reference_counts.get(token, 0) > 0:
            overlap += 1
            reference_counts[token] -= 1
    precision = overlap / len(candidate_tokens)
    recall = overlap / len(reference_tokens)
    return _f1(precision, recall)


def sentiment_aligned(predicted_rating: int, text: str) -> bool:
    lowered = text.lower()
    positive_terms = {"excellent", "perfect", "great", "good", "useful", "helpful", "love", "fits"}
    negative_terms = {"terrible", "awful", "bad", "poor", "hate", "misses", "cautious", "disappointing"}
    has_positive = any(term in lowered for term in positive_terms)
    has_negative = any(term in lowered for term in negative_terms)
    if predicted_rating >= 4:
        return has_positive or not has_negative
    if predicted_rating <= 2:
        return has_negative or not has_positive
    return True


def tokenize(text: str) -> list[str]:
    return re.findall(r"[a-zA-Z0-9]+", text.lower())


def _lcs_length(left: list[str], right: list[str]) -> int:
    previous = [0] * (len(right) + 1)
    for left_token in left:
        current = [0]
        for index, right_token in enumerate(right, start=1):
            if left_token == right_token:
                current.append(previous[index - 1] + 1)
            else:
                current.append(max(previous[index], current[-1]))
        previous = current
    return previous[-1]


def _f1(precision: float, recall: float) -> float:
    if precision + recall == 0:
        return 0.0
    return 2 * precision * recall / (precision + recall)


def _mean(values: list[float | bool]) -> float:
    return sum(float(value) for value in values) / len(values) if values else 0.0


if __name__ == "__main__":
    main()
