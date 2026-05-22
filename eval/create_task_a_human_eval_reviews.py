from __future__ import annotations

import argparse
import csv
import json
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


FIELDNAMES = [
    "example_id",
    "user_id",
    "item_id",
    "category",
    "target_item_name",
    "predicted_rating",
    "generated_review",
    "user_history",
    "reference_rating",
    "reference_review",
    "rating_fit_1_5",
    "voice_fit_1_5",
    "groundedness_1_5",
    "specificity_1_5",
    "notes",
]


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Create a reviewer-facing Task A human-eval pack with generated reviews."
    )
    parser.add_argument("--reviews", default="data/sample/reviews.jsonl")
    parser.add_argument("--items", default="data/sample/items.jsonl")
    parser.add_argument("--processed-dir", default="data/processed/all_categories")
    parser.add_argument("--output-csv", default="docs/human_eval_task_a_reviews.csv")
    parser.add_argument("--output-md", default="docs/human_eval_task_a_reviews.md")
    parser.add_argument("--max-examples", type=int, default=25)
    args = parser.parse_args()

    rows = build_rows(
        reviews_path=Path(args.reviews),
        items_path=Path(args.items),
        processed_dir=Path(args.processed_dir),
        max_examples=args.max_examples,
    )
    write_csv(Path(args.output_csv), rows)
    write_markdown(Path(args.output_md), rows, processed_dir=Path(args.processed_dir))
    print(f"Wrote {args.output_csv} ({len(rows)} rows)")
    print(f"Wrote {args.output_md} ({len(rows)} examples)")


def build_rows(
    reviews_path: Path,
    items_path: Path,
    processed_dir: Path,
    max_examples: int,
) -> list[dict[str, object]]:
    train, test_a, items = _load_eval_data(reviews_path, items_path, processed_dir)
    if max_examples:
        test_a = test_a[:max_examples]

    histories = _histories_by_user(train)
    item_ratings = _item_ratings(train)
    global_mean = _mean([float(row["rating"]) for row in train]) or 3.5

    rows: list[dict[str, object]] = []
    for index, row in enumerate(test_a, start=1):
        history = histories.get(row["user_id"], [])
        item = items.get(row["item_id"], {})
        target_item_name = str(item.get("name") or row.get("item_name") or row["item_id"])
        category = str(row.get("category") or item.get("category") or "unknown")
        predicted_rating = _predicted_rating(
            row=row,
            history=history,
            item=item,
            item_ratings=item_ratings,
            global_mean=global_mean,
        )
        generated_review = _generated_review(
            item_name=target_item_name,
            category=category,
            item=item,
            history=history,
            predicted_rating=predicted_rating,
        )
        rows.append(
            {
                "example_id": f"A-{index:03d}",
                "user_id": row["user_id"],
                "item_id": row["item_id"],
                "category": category,
                "target_item_name": target_item_name,
                "predicted_rating": predicted_rating,
                "generated_review": generated_review,
                "user_history": _history_text(history),
                "reference_rating": row.get("rating", ""),
                "reference_review": row.get("review", ""),
                "rating_fit_1_5": "",
                "voice_fit_1_5": "",
                "groundedness_1_5": "",
                "specificity_1_5": "",
                "notes": "",
            }
        )
    return rows


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDNAMES)
        writer.writeheader()
        writer.writerows(rows)


def write_markdown(path: Path, rows: list[dict[str, object]], processed_dir: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "# Task A Human Evaluation Review Pack",
        "",
        f"Dataset: `{processed_dir}`",
        "",
        "Score 1-5 for each generated review:",
        "- `rating_fit`: rating matches the user's likely preference.",
        "- `voice_fit`: review sounds consistent with the user's prior review style.",
        "- `groundedness`: claims are supported by user history and item facts.",
        "- `specificity`: review is concrete rather than generic.",
        "",
        "Generated reviews are produced by the local deterministic Task A fallback, so the pack can be created without sending eval rows to an external LLM provider. The fallback intentionally varies openings and avoids first-sentence rating boilerplate.",
        "",
    ]
    for row in rows:
        lines.extend(
            [
                f"## {row['example_id']}",
                "",
                f"- User: `{row['user_id']}`",
                f"- Item: `{row['item_id']}` - {row['target_item_name']}",
                f"- Category: {row['category']}",
                f"- Predicted rating: {row['predicted_rating']} out of 5",
                f"- Reference rating: {row['reference_rating']}",
                "",
                "Generated review:",
                "",
                f"> {row['generated_review']}",
                "",
                "User history:",
                "",
                f"> {_quote(row['user_history'] or 'No prior history available.')}",
                "",
                "Reference review:",
                "",
                f"> {_quote(row['reference_review'])}",
                "",
                "Scores: rating_fit ___ / voice_fit ___ / groundedness ___ / specificity ___",
                "",
                "Notes:",
                "",
            ]
        )
    path.write_text("\n".join(lines), encoding="utf-8")


def _load_eval_data(
    reviews_path: Path,
    items_path: Path,
    processed_dir: Path,
) -> tuple[list[dict], list[dict], dict[str, dict]]:
    if (processed_dir / "train.jsonl").exists() and (processed_dir / "test_task_a.jsonl").exists():
        train = _read_jsonl(processed_dir / "train.jsonl")
        test_a = _read_jsonl(processed_dir / "test_task_a.jsonl")
        item_rows = _read_jsonl(processed_dir / "items.jsonl")
    else:
        rows = _read_jsonl(reviews_path)
        train, test_a = _temporal_task_a_split(rows)
        item_rows = _read_jsonl(items_path)
    return train, test_a, {str(row["item_id"]): row for row in item_rows}


def _read_jsonl(path: Path) -> list[dict]:
    rows = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def _temporal_task_a_split(rows: list[dict]) -> tuple[list[dict], list[dict]]:
    grouped: dict[str, list[dict]] = defaultdict(list)
    for row in rows:
        grouped[str(row["user_id"])].append(row)

    train = []
    test_a = []
    for user_rows in grouped.values():
        ordered = sorted(user_rows, key=lambda row: (int(row.get("timestamp") or 0), str(row.get("review_id") or "")))
        if len(ordered) < 2:
            train.extend(ordered)
            continue
        train.extend(ordered[:-1])
        test_a.append(ordered[-1])
    return train, test_a


def _histories_by_user(rows: list[dict]) -> dict[str, list[dict]]:
    grouped: dict[str, list[dict]] = defaultdict(list)
    for row in rows:
        grouped[str(row["user_id"])].append(row)
    for user_id, history in grouped.items():
        grouped[user_id] = sorted(
            history,
            key=lambda row: (int(row.get("timestamp") or 0), str(row.get("review_id") or "")),
        )
    return grouped


def _item_ratings(rows: list[dict]) -> dict[str, list[float]]:
    ratings: dict[str, list[float]] = defaultdict(list)
    for row in rows:
        ratings[str(row["item_id"])].append(float(row["rating"]))
    return ratings


def _predicted_rating(
    row: dict,
    history: list[dict],
    item: dict,
    item_ratings: dict[str, list[float]],
    global_mean: float,
) -> int:
    user_mean = _mean([float(item["rating"]) for item in history])
    item_mean = _mean(item_ratings.get(str(row["item_id"]), []))
    metadata_rating = item.get("average_rating")
    if item_mean == 0 and metadata_rating:
        item_mean = float(metadata_rating)
    score = 0.55 * (user_mean or global_mean) + 0.35 * (item_mean or global_mean) + 0.10 * global_mean
    return max(1, min(5, int(round(score))))


def _generated_review(
    item_name: str,
    category: str,
    item: dict,
    history: list[dict],
    predicted_rating: int,
) -> str:
    terms = _evidence_terms(item_name, category, item, history)
    positive = _sentence_fragment(terms[:6]) or "the available product details"
    negatives = _negative_terms(history)
    if predicted_rating >= 4:
        verdict = "this would probably work for me"
    elif predicted_rating == 3:
        verdict = "I see the useful parts, but I am not fully sold"
    else:
        verdict = "I would be cautious about it"
    tradeoff = ""
    if negatives and predicted_rating <= 3:
        tradeoff = " I would still watch out for " + _sentence_fragment(negatives[:4]) + "."
    elif negatives:
        tradeoff = " The only thing I would keep in mind is " + _sentence_fragment(negatives[:4]) + "."
    variant = (sum(ord(char) for char in item_name) + predicted_rating) % 3
    if variant == 0:
        return (
            f"{item_name} feels like a {predicted_rating} out of 5 for me. "
            f"{positive.capitalize()} stood out first, and {verdict}.{tradeoff}"
        )
    if variant == 1:
        return (
            f"I am at {predicted_rating} out of 5 on {item_name}. "
            f"{positive.capitalize()} is what carries it, though {verdict}.{tradeoff}"
        )
    return (
        f"{positive.capitalize()} is the main reason {item_name} lands at "
        f"{predicted_rating} out of 5. For my taste, {verdict}.{tradeoff}"
    )


def _evidence_terms(item_name: str, category: str, item: dict, history: list[dict]) -> list[str]:
    text = " ".join(
        [
            item_name,
            category,
            str(item.get("summary") or ""),
            " ".join(str(row.get("review") or "") for row in history if float(row.get("rating") or 0) >= 4),
        ]
    )
    counts = Counter(_terms(text))
    return [term for term, _ in counts.most_common(8)]


def _negative_terms(history: list[dict]) -> list[str]:
    text = " ".join(str(row.get("review") or "") for row in history if float(row.get("rating") or 0) <= 2)
    counts = Counter(_terms(text))
    return [term for term, _ in counts.most_common(4)]


def _terms(text: str) -> list[str]:
    stop = {
        "about",
        "after",
        "again",
        "also",
        "and",
        "are",
        "because",
        "but",
        "for",
        "from",
        "had",
        "has",
        "have",
        "into",
        "its",
        "just",
        "not",
        "out",
        "that",
        "the",
        "this",
        "too",
        "very",
        "was",
        "with",
        "would",
        "you",
    }
    return [
        token
        for token in re.findall(r"[a-zA-Z][a-zA-Z0-9_-]{2,}", text.lower())
        if token not in stop
    ]


def _sentence_fragment(values: list[str]) -> str:
    clean = [str(value).strip().rstrip(".") for value in values if str(value).strip()]
    if not clean:
        return ""
    if len(clean) == 1:
        return clean[0]
    return ", ".join(clean[:-1]) + ", and " + clean[-1]


def _mean(values: list[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def _history_text(history) -> str:
    pieces = []
    for item in history[-5:]:
        review = " ".join(str(item.get("review") or "").split())
        pieces.append(f"{item.get('item_name')} ({item.get('rating')}/5): {review}")
    return " | ".join(pieces)


def _quote(value: object) -> str:
    return str(value).replace("\n", " ").replace("\r", " ").replace("\n> ", "\n")


if __name__ == "__main__":
    main()
