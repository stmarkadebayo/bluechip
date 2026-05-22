from __future__ import annotations

import argparse
import csv
import json
import statistics
from collections import Counter
from pathlib import Path


DEFAULT_RUBRIC_COLUMNS = (
    "top10_relevance_1_5",
    "context_fit_1_5",
    "diversity_1_5",
    "explanation_quality_1_5",
)


def main() -> None:
    parser = argparse.ArgumentParser(description="Summarize human-eval CSV rubric scores.")
    parser.add_argument("csv_path")
    parser.add_argument("--output-json", default="")
    parser.add_argument("--output-md", default="")
    parser.add_argument("--rubric-columns", default=",".join(DEFAULT_RUBRIC_COLUMNS))
    args = parser.parse_args()

    rubric_columns = tuple(
        column.strip()
        for column in args.rubric_columns.split(",")
        if column.strip()
    )
    rows = _read_csv(Path(args.csv_path))
    summary = summarize(rows, rubric_columns)
    print(json.dumps(summary, ensure_ascii=True, indent=2))

    if args.output_json:
        output_json = Path(args.output_json)
        output_json.parent.mkdir(parents=True, exist_ok=True)
        output_json.write_text(
            json.dumps(summary, ensure_ascii=True, indent=2) + "\n",
            encoding="utf-8",
        )
    if args.output_md:
        output_md = Path(args.output_md)
        output_md.parent.mkdir(parents=True, exist_ok=True)
        output_md.write_text(_markdown(summary), encoding="utf-8")


def summarize(rows: list[dict[str, str]], rubric_columns: tuple[str, ...]) -> dict:
    rubric = {}
    for column in rubric_columns:
        values = _numeric_values(rows, column)
        rubric[column] = _metric_summary(values)

    complete_rows = []
    for row in rows:
        values = []
        for column in rubric_columns:
            value = _float_or_none(row.get(column))
            if value is not None:
                values.append(value)
        if len(values) == len(rubric_columns):
            complete_rows.append(sum(values) / len(values))

    target_values = [
        str(row.get("target_in_top10") or "").strip().lower()
        for row in rows
        if str(row.get("target_in_top10") or "").strip()
    ]
    target_counts = Counter(target_values)
    notes = [
        str(row.get("human_notes") or "").strip()
        for row in rows
        if str(row.get("human_notes") or "").strip()
    ]

    return {
        "task": "Task B contextual human evaluation",
        "examples": len(rows),
        "rubric_columns": list(rubric_columns),
        "rubric": rubric,
        "composite_1_5": _metric_summary(complete_rows),
        "estimated_contextual_relevance_points_20": round(
            (_mean(complete_rows) / 5.0) * 20.0,
            2,
        ) if complete_rows else 0.0,
        "target_in_top10": {
            "counts": dict(sorted(target_counts.items())),
            "yes_rate": round(target_counts.get("yes", 0) / max(len(target_values), 1), 4),
        },
        "notes": {
            "count": len(notes),
            "top": [
                {"note": note, "count": count}
                for note, count in Counter(notes).most_common(20)
            ],
        },
    }


def _read_csv(path: Path) -> list[dict[str, str]]:
    last_error: UnicodeDecodeError | None = None
    for encoding in ("utf-8-sig", "cp1252", "latin-1"):
        try:
            with path.open(newline="", encoding=encoding) as handle:
                return list(csv.DictReader(handle))
        except UnicodeDecodeError as exc:
            last_error = exc
    if last_error:
        raise last_error
    return []


def _numeric_values(rows: list[dict[str, str]], column: str) -> list[float]:
    values = []
    for row in rows:
        value = _float_or_none(row.get(column))
        if value is not None:
            values.append(value)
    return values


def _float_or_none(value: object) -> float | None:
    text = str(value or "").strip()
    if not text:
        return None
    try:
        return float(text)
    except ValueError:
        return None


def _metric_summary(values: list[float]) -> dict:
    if not values:
        return {"n": 0}
    return {
        "n": len(values),
        "mean": round(_mean(values), 4),
        "median": round(statistics.median(values), 4),
        "min": round(min(values), 4),
        "max": round(max(values), 4),
        "distribution": {
            str(key): count
            for key, count in sorted(Counter(values).items())
        },
    }


def _mean(values: list[float]) -> float:
    return statistics.mean(values) if values else 0.0


def _markdown(summary: dict) -> str:
    lines = [
        "# Task B Contextual Human Eval Results",
        "",
        f"Examples: `{summary['examples']}`",
        "",
        "## Rubric Scores",
        "",
        "| Metric | N | Mean | Median | Min | Max |",
        "| --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    for metric, values in summary["rubric"].items():
        lines.append(
            f"| `{metric}` | {values.get('n', 0)} | {values.get('mean', '')} | "
            f"{values.get('median', '')} | {values.get('min', '')} | {values.get('max', '')} |"
        )
    composite = summary["composite_1_5"]
    lines.extend(
        [
            f"| `composite_1_5` | {composite.get('n', 0)} | {composite.get('mean', '')} | "
            f"{composite.get('median', '')} | {composite.get('min', '')} | {composite.get('max', '')} |",
            "",
            "## Competition Read",
            "",
            (
                "Estimated contextual relevance score: "
                f"`{summary['estimated_contextual_relevance_points_20']}/20` "
                "if linearly mapped from the 1-5 rubric average."
            ),
            (
                "Target in top-10 yes-rate: "
                f"`{summary['target_in_top10']['yes_rate']}` "
                f"from counts `{summary['target_in_top10']['counts']}`."
            ),
            "",
            "## Notes",
            "",
        ]
    )
    for item in summary["notes"]["top"]:
        lines.append(f"- `{item['count']}`: {item['note']}")
    return "\n".join(lines) + "\n"


if __name__ == "__main__":
    main()
