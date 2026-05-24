from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.services.profiling.user_profile import build_user_profile  # noqa: E402
from app.services.ranking.recommendation import rank_candidates  # noqa: E402
from app.services.retrieval.candidates import (  # noqa: E402
    CandidateCatalog,
    generate_candidate_pool,
)
from app.services.retrieval.source_registry import default_disabled_retrieval_sources  # noqa: E402
from app.services.retrieval.text import BM25Retriever  # noqa: E402
from app.services.retrieval.vector_store import LocalVectorRetriever  # noqa: E402
from eval.common import histories_by_user, load_eval_data, persona_from_history  # noqa: E402
from eval.eval_task_b import _items_with_train_popularity, _load_collaborative_index  # noqa: E402
from eval.task_b_context import context_for_task_b_row  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description="Create a Task B contextual human-eval pack.")
    parser.add_argument("--reviews", default="data/sample/reviews.jsonl")
    parser.add_argument("--items", default="data/sample/items.jsonl")
    parser.add_argument("--processed-dir", default="data/processed")
    parser.add_argument("--collaborative-index", default="")
    parser.add_argument("--output", default="docs/human_eval_task_b_contextual.md")
    parser.add_argument("--max-examples", type=int, default=20)
    parser.add_argument("--candidate-limit", type=int, default=1000)
    parser.add_argument("--build-collaborative", action="store_true")
    parser.add_argument(
        "--disabled-sources",
        default=None,
        help=(
            "Comma-separated retrieval sources to disable. "
            "Defaults to the shared serving/eval lean policy."
        ),
    )
    args = parser.parse_args()

    train, _, test_b, items = load_eval_data(
        reviews_path=Path(args.reviews),
        items_path=Path(args.items),
        processed_dir=Path(args.processed_dir),
    )
    item_list = _items_with_train_popularity(items, train)
    history_map = histories_by_user(train)
    bm25 = BM25Retriever.from_items(item_list)
    configured_disabled = _configured_disabled_sources(args.disabled_sources)
    initial_disabled = (
        configured_disabled
        if configured_disabled is not None
        else default_disabled_retrieval_sources("contextual request")
    )
    vectors = None
    if "vector_profile" not in initial_disabled:
        vectors = LocalVectorRetriever(item_list)
    catalog = CandidateCatalog.from_items(item_list)
    collaborative_index = _load_collaborative_index(args, train)

    rows = []
    for index, row in enumerate(test_b[: args.max_examples], start=1):
        history = history_map.get(row["user_id"], [])
        context = _context_for(row, history)
        disabled_sources = (
            _disabled_sources_for_context(configured_disabled, context)
        )
        user_profile = build_user_profile(persona_from_history(history), history)
        pool = generate_candidate_pool(
            user_profile=user_profile,
            history=history,
            items=item_list,
            context=context,
            collaborative_index=collaborative_index,
            bm25_retriever=bm25,
            vector_retriever=vectors,
            catalog=catalog,
            disabled_sources=disabled_sources,
            limit=min(args.candidate_limit, len(item_list)),
        )
        ranked = rank_candidates(
            user_profile=user_profile,
            context=context,
            candidate_items=pool.items,
            limit=10,
            candidate_sources=pool.sources,
            candidate_source_scores=pool.source_scores,
        )
        target = items.get(row["item_id"])
        ranked_ids = [item.item_id for item in ranked]
        rows.append(
            {
                "example_id": f"CTX-B-{index:03d}",
                "user_id": row["user_id"],
                "context": context,
                "target": _item_label(
                    row["item_id"],
                    target.name if target else row.get("item_name"),
                ),
                "target_category": (
                    target.category if target else row.get("category")
                ) or "unknown",
                "history": _history_summary(history),
                "top10": _top10_summary(ranked),
                "target_in_top10": "yes" if row["item_id"] in ranked_ids else "no",
                "candidate_sources": _source_summary(ranked),
                "disabled_sources": ", ".join(sorted(disabled_sources)) or "none",
            }
        )

    _write_markdown(Path(args.output), rows)


def _context_for(row: dict, history: list) -> str:
    return context_for_task_b_row(row, history)


def _configured_disabled_sources(value: str | None) -> set[str] | None:
    if value is None:
        return None
    return {source.strip() for source in value.split(",") if source.strip()}


def _disabled_sources_for_context(configured: set[str] | None, context: str) -> set[str]:
    if configured is not None:
        return configured
    return default_disabled_retrieval_sources(context)


def _history_summary(history: list, limit: int = 3) -> str:
    positives = [item for item in history if item.rating >= 4]
    if not positives:
        positives = history
    snippets = [
        f"{item.item_name} ({item.category or 'unknown'}, {item.rating} stars)"
        for item in positives[:limit]
    ]
    return "; ".join(snippets) if snippets else "cold-start/no prior positive reviews"


def _top10_summary(ranked: list) -> str:
    return "<br>".join(
        f"{item.rank}. {_item_label(item.item_id, item.name)} "
        f"[{', '.join(item.candidate_sources[:3])}]"
        for item in ranked
    )


def _source_summary(ranked: list) -> str:
    sources = {}
    for item in ranked:
        for source in item.candidate_sources:
            sources[source] = sources.get(source, 0) + 1
    return ", ".join(f"{name}: {count}" for name, count in sorted(sources.items())) or "none"


def _item_label(item_id: str, name: str | None) -> str:
    label = (name or item_id).replace("|", "\\|")
    if len(label) > 90:
        label = label[:87] + "..."
    return f"{label} (`{item_id}`)"


def _write_markdown(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    columns = [
        "example_id",
        "user_id",
        "context",
        "target",
        "target_category",
        "history",
        "top10",
        "target_in_top10",
        "candidate_sources",
        "disabled_sources",
        "top10_relevance_1_5",
        "context_fit_1_5",
        "diversity_1_5",
        "explanation_quality_1_5",
        "human_notes",
    ]
    lines = [
        "# Task B Contextual Human Evaluation Pack",
        "",
        "Score each recommendation list 1-5 against the rubric columns.",
        "Leave scores blank until a human reviewer fills them in.",
        "",
        "| " + " | ".join(columns) + " |",
        "| " + " | ".join("---" for _ in columns) + " |",
    ]
    for row in rows:
        values = [row.get(column, "") for column in columns]
        lines.append("| " + " | ".join(_cell(value) for value in values) + " |")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _cell(value: object) -> str:
    return str(value).replace("\n", " ").replace("|", "\\|")


if __name__ == "__main__":
    main()
