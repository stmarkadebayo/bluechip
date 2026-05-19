from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.services.profiling.user_profile import build_user_profile  # noqa: E402
from app.services.ranking.recommendation import rank_candidates  # noqa: E402
from app.services.retrieval.candidates import CandidateCatalog, generate_candidate_pool  # noqa: E402
from app.services.retrieval.text import BM25Retriever  # noqa: E402
from app.services.retrieval.vector_store import LocalVectorRetriever  # noqa: E402
from eval.common import histories_by_user, load_eval_data, persona_from_history  # noqa: E402
from eval.eval_task_b import _items_with_train_popularity, _load_collaborative_index  # noqa: E402


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
    args = parser.parse_args()

    train, _, test_b, items = load_eval_data(
        reviews_path=Path(args.reviews),
        items_path=Path(args.items),
        processed_dir=Path(args.processed_dir),
    )
    item_list = _items_with_train_popularity(items, train)
    history_map = histories_by_user(train)
    bm25 = BM25Retriever.from_items(item_list)
    vectors = LocalVectorRetriever(item_list)
    catalog = CandidateCatalog.from_items(item_list)
    collaborative_index = _load_collaborative_index(args, train)

    rows = []
    for index, row in enumerate(test_b[: args.max_examples], start=1):
        history = history_map.get(row["user_id"], [])
        context = _context_for(row, history)
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
                "target": _item_label(row["item_id"], target.name if target else row.get("item_name")),
                "target_category": (target.category if target else row.get("category")) or "unknown",
                "history": _history_summary(history),
                "top10": _top10_summary(ranked),
                "target_in_top10": "yes" if row["item_id"] in ranked_ids else "no",
                "candidate_sources": _source_summary(ranked),
            }
        )

    _write_markdown(Path(args.output), rows)


def _context_for(row: dict, history: list) -> str:
    category = row.get("category") or "products"
    positive_text = " ".join(
        f"{item.item_name} {item.review}" for item in history if item.rating >= 4
    ).lower()
    if category == "All_Beauty":
        if any(term in positive_text for term in ("skin", "serum", "face", "cream")):
            return "Needs a beauty product for a gentle skincare routine; avoid harsh-feeling items."
        if any(term in positive_text for term in ("hair", "wig", "shampoo", "brush")):
            return "Needs a practical hair or styling product for regular use."
        if any(term in positive_text for term in ("nail", "manicure", "polish")):
            return "Needs a nail-care or manicure product that feels useful, not gimmicky."
        return "Needs a practical beauty item aligned with recent positive purchases."
    if category == "Digital_Music":
        return "Wants music that fits the user's recent taste and is easy to replay."
    if "Card" in category or category in {"Restaurants", "For Him"}:
        return "Needs a low-risk gift option that matches recent gifting behavior."
    return f"Needs a recommendation in or near {category} that fits prior positive reviews."


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
        f"{item.rank}. {_item_label(item.item_id, item.name)} [{', '.join(item.candidate_sources[:3])}]"
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
