from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from eval.common import load_eval_data  # noqa: E402


TASK_A_COLUMNS = [
    "example_id",
    "user_id",
    "item_id",
    "category",
    "rating_fit_1_5",
    "voice_fit_1_5",
    "groundedness_1_5",
    "specificity_1_5",
    "notes",
]

TASK_B_COLUMNS = [
    "example_id",
    "user_id",
    "target_item_id",
    "target_category",
    "top10_relevance_1_5",
    "context_fit_1_5",
    "diversity_1_5",
    "explanation_quality_1_5",
    "notes",
]


def main() -> None:
    parser = argparse.ArgumentParser(description="Create human-eval scoring tables for Tasks A and B.")
    parser.add_argument("--reviews", default="data/sample/reviews.jsonl")
    parser.add_argument("--items", default="data/sample/items.jsonl")
    parser.add_argument("--processed-dir", default="data/processed")
    parser.add_argument("--output-dir", default="docs")
    parser.add_argument("--max-examples", type=int, default=25)
    args = parser.parse_args()

    _, test_a, test_b, _ = load_eval_data(
        reviews_path=Path(args.reviews),
        items_path=Path(args.items),
        processed_dir=Path(args.processed_dir),
    )
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    _write_markdown(
        output_dir / "human_eval_task_a.md",
        title="Task A Human Evaluation Table",
        rubric=_task_a_rubric(),
        columns=TASK_A_COLUMNS,
        rows=[
            [
                f"A-{index + 1:03d}",
                row["user_id"],
                row["item_id"],
                row.get("category") or "unknown",
                "",
                "",
                "",
                "",
                "",
            ]
            for index, row in enumerate(test_a[: args.max_examples])
        ],
    )
    _write_markdown(
        output_dir / "human_eval_task_b.md",
        title="Task B Human Evaluation Table",
        rubric=_task_b_rubric(),
        columns=TASK_B_COLUMNS,
        rows=[
            [
                f"B-{index + 1:03d}",
                row["user_id"],
                row["item_id"],
                row.get("category") or "unknown",
                "",
                "",
                "",
                "",
                "",
            ]
            for index, row in enumerate(test_b[: args.max_examples])
        ],
    )


def _write_markdown(
    path: Path,
    title: str,
    rubric: list[str],
    columns: list[str],
    rows: list[list[str]],
) -> None:
    lines = [f"# {title}", ""]
    lines.extend(rubric)
    lines.extend(["", "| " + " | ".join(columns) + " |", "| " + " | ".join("---" for _ in columns) + " |"])
    for row in rows:
        lines.append("| " + " | ".join(_cell(value) for value in row) + " |")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _task_a_rubric() -> list[str]:
    return [
        "Score 1-5 for each generated review:",
        "- `rating_fit`: rating matches the user's likely preference.",
        "- `voice_fit`: review sounds consistent with the user's prior review style.",
        "- `groundedness`: claims are supported by user history and item facts.",
        "- `specificity`: review is concrete rather than generic.",
    ]


def _task_b_rubric() -> list[str]:
    return [
        "Score 1-5 for each recommendation list:",
        "- `top10_relevance`: top results contain items the user would plausibly choose.",
        "- `context_fit`: list respects occasion, price, avoid, locale, and mood constraints.",
        "- `diversity`: results are not redundant unless the context requires narrow focus.",
        "- `explanation_quality`: explanations cite real candidate sources and tradeoffs.",
    ]


def _cell(value: object) -> str:
    return str(value).replace("|", "\\|")


if __name__ == "__main__":
    main()
