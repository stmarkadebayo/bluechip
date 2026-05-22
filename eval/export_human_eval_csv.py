from __future__ import annotations

import argparse
import csv
from pathlib import Path


DEFAULT_INPUTS = [
    "docs/human_eval_task_a.md",
    "docs/human_eval_task_b.md",
    "docs/human_eval_task_b_contextual.md",
]


def main() -> None:
    parser = argparse.ArgumentParser(description="Export human-eval markdown tables to CSV.")
    parser.add_argument(
        "inputs",
        nargs="*",
        default=DEFAULT_INPUTS,
        help="Markdown files containing a single human-eval table.",
    )
    parser.add_argument(
        "--output-dir",
        default="docs",
        help="Directory for generated CSV files. Defaults to docs/.",
    )
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    for input_name in args.inputs:
        input_path = Path(input_name)
        rows = _extract_table(input_path)
        output_path = output_dir / f"{input_path.stem}.csv"
        with output_path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.writer(handle)
            writer.writerows(rows)
        print(f"Wrote {output_path} ({max(len(rows) - 1, 0)} rows)")


def _extract_table(path: Path) -> list[list[str]]:
    lines = path.read_text(encoding="utf-8").splitlines()
    table_lines = [line for line in lines if line.startswith("|")]
    if len(table_lines) < 2:
        raise ValueError(f"No markdown table found in {path}")

    rows: list[list[str]] = []
    for index, line in enumerate(table_lines):
        cells = _split_markdown_row(line)
        if index == 1 and all(_is_separator_cell(cell) for cell in cells):
            continue
        rows.append([_clean_cell(cell) for cell in cells])
    return rows


def _split_markdown_row(line: str) -> list[str]:
    if line.startswith("|"):
        line = line[1:]
    if line.endswith("|"):
        line = line[:-1]

    cells: list[str] = []
    current: list[str] = []
    index = 0
    while index < len(line):
        char = line[index]
        if char == "\\" and index + 1 < len(line) and line[index + 1] == "|":
            current.append("|")
            index += 2
            continue
        if char == "|":
            cells.append("".join(current).strip())
            current = []
            index += 1
            continue
        current.append(char)
        index += 1
    cells.append("".join(current).strip())
    return cells


def _is_separator_cell(cell: str) -> bool:
    stripped = cell.strip()
    return bool(stripped) and set(stripped) <= {"-", ":"}


def _clean_cell(cell: str) -> str:
    return cell.replace("\\|", "|").replace("<br>", "\n")


if __name__ == "__main__":
    main()
