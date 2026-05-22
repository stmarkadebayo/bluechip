from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from eval.common import (  # noqa: E402
    histories_by_user,
    load_eval_data,
    persona_from_history,
    popularity_ranking,
    print_report,
    write_report,
)
from eval.metrics import hit_rate_at_k, ndcg_at_k, recall_at_k, rounded  # noqa: E402
from app.models.schemas import Item  # noqa: E402
from app.platform.artifacts import read_json_artifact  # noqa: E402
from app.services.profiling.user_profile import build_user_profile  # noqa: E402
from app.services.ranking.recommendation import rank_candidates  # noqa: E402
from app.services.retrieval.candidates import (  # noqa: E402
    CandidateCatalog,
    CandidatePool,
    generate_candidate_pool,
)
from app.services.retrieval.item_similarity import (  # noqa: E402
    SQLiteItemNeighborIndex,
    build_collaborative_retrieval_index,
)
from app.services.retrieval.source_registry import (  # noqa: E402
    SOURCE_FAMILY_LABELS,
    SOURCE_FAMILY_ORDER,
    retrieval_source_family,
)
from app.services.retrieval.text import BM25Retriever  # noqa: E402
from app.services.retrieval.vector_store import (  # noqa: E402
    FAISSVectorStore,
    LocalVectorRetriever,
    create_retriever,
)


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate Task B recommendation baselines.")
    parser.add_argument("--reviews", default="data/sample/reviews.jsonl")
    parser.add_argument("--items", default="data/sample/items.jsonl")
    parser.add_argument("--processed-dir", default="data/processed")
    parser.add_argument("--collaborative-index", default="")
    parser.add_argument("--output", default="runs/eval/task_b_report.json")
    parser.add_argument("--k", type=int, default=10)
    parser.add_argument("--candidate-limit", type=int, default=200)
    parser.add_argument("--max-examples", type=int, default=0)
    parser.add_argument(
        "--sample-strategy",
        choices=["first", "stride"],
        default="first",
        help="How to select --max-examples rows; stride reduces first-N ordering bias.",
    )
    parser.add_argument("--build-collaborative", action="store_true")
    parser.add_argument(
        "--hybrid-only",
        action="store_true",
        help="Skip unchanged baseline and cold-start rankings for larger hybrid-only sweeps.",
    )
    parser.add_argument(
        "--candidate-recall-only",
        action="store_true",
        help="Skip ranking and measure candidate recall/source diagnostics only.",
    )
    parser.add_argument(
        "--disabled-sources",
        default="",
        help="Comma-separated retrieval sources to disable for ablation/pruning runs.",
    )
    parser.add_argument("--miss-output", default="")
    parser.add_argument("--max-misses", type=int, default=200)
    parser.add_argument(
        "--retriever",
        choices=["legacy", "neural"],
        default="legacy",
        help=(
            "Vector retriever backend: legacy (LocalVectorRetriever) "
            "or neural (FAISSVectorStore)."
        ),
    )
    args = parser.parse_args()

    EvalRunner(args).run()


@dataclass
class EvalDataset:
    train: list[dict]
    test_b: list[dict]
    items: dict[str, Item]
    item_list: list[Item]
    positives: list[str]
    history_map: dict
    retriever: BM25Retriever
    vector_retriever: LocalVectorRetriever | None
    neural_retriever: FAISSVectorStore | None
    catalog: CandidateCatalog
    popularity: list[str]
    train_item_counts: Counter
    collaborative_index: dict | None
    item_neighbor_ids: set[str]
    review_term_item_ids: set[str]
    disabled_sources: set[str]
    item_profile_cache: dict


@dataclass
class EvalResult:
    rankings: dict[str, list[list[str]]]
    metrics: dict[str, float]
    slices: dict
    source_counts: dict[str, int]
    source_diagnostics: dict[str, dict]
    source_family_diagnostics: dict[str, dict]
    miss_report: dict
    hybrid_pools: list[CandidatePool]
    hybrid_ranked_ids: list[list[str]]


class EvalDatasetBuilder:
    def __init__(self, args: argparse.Namespace) -> None:
        self.args = args

    def build(self) -> EvalDataset:
        train, _, test_b, items = load_eval_data(
            reviews_path=Path(self.args.reviews),
            items_path=Path(self.args.items),
            processed_dir=Path(self.args.processed_dir),
        )
        item_list = _items_with_train_popularity(items, train)
        if self.args.max_examples:
            test_b = _sample_eval_rows(
                test_b,
                self.args.max_examples,
                self.args.sample_strategy,
            )
        disabled_sources = {
            source.strip()
            for source in self.args.disabled_sources.split(",")
            if source.strip()
        }
        positives = [row["item_id"] for row in test_b]
        history_map = histories_by_user(train)
        retriever = BM25Retriever.from_items(item_list)
        vector_retriever = (
            None if "vector_profile" in disabled_sources else LocalVectorRetriever(item_list)
        )
        neural_retriever = self._build_neural_retriever(item_list)
        catalog = CandidateCatalog.from_items(item_list)
        popularity = _popularity_rank(train, item_list)
        train_item_counts = Counter(row["item_id"] for row in train if row["rating"] >= 4)
        collaborative_index = _load_collaborative_index(self.args, train)
        return EvalDataset(
            train=train,
            test_b=test_b,
            items=items,
            item_list=item_list,
            positives=positives,
            history_map=history_map,
            retriever=retriever,
            vector_retriever=vector_retriever,
            neural_retriever=neural_retriever,
            catalog=catalog,
            popularity=popularity,
            train_item_counts=train_item_counts,
            collaborative_index=collaborative_index,
            item_neighbor_ids=_item_neighbor_ids(collaborative_index),
            review_term_item_ids=_review_term_item_ids(collaborative_index),
            disabled_sources=disabled_sources,
            item_profile_cache={},
        )

    def _build_neural_retriever(self, item_list: list[Item]) -> FAISSVectorStore | None:
        if self.args.retriever != "neural":
            return None
        index_path = Path(self.args.processed_dir) / "neural_index.faiss"
        if index_path.exists():
            neural_retriever = FAISSVectorStore.deserialize(str(index_path), item_list)
        else:
            neural_retriever = create_retriever(item_list, method="neural")
        if isinstance(neural_retriever, FAISSVectorStore) and neural_retriever._built:
            print(
                "Neural FAISS retriever initialised: "
                f"{neural_retriever.index.ntotal} items indexed."
            )
            return neural_retriever
        print("Neural FAISS retriever unavailable; falling back to LocalVectorRetriever only.")
        return None


class RecommendationEvaluator:
    def __init__(self, args: argparse.Namespace, dataset: EvalDataset) -> None:
        self.args = args
        self.dataset = dataset

    def evaluate(self) -> EvalResult:
        base_pools, hybrid_pools, hybrid_ranked_ids, cold_start_ranked_ids = self._run_rows()
        rankings = self._build_rankings(
            base_pools,
            hybrid_pools,
            hybrid_ranked_ids,
            cold_start_ranked_ids,
        )
        metrics = self._metrics(rankings)
        slices = self._slices(rankings, hybrid_ranked_ids)
        source_counts = _aggregate_source_counts(hybrid_pools)
        source_diagnostics = _source_diagnostics(
            pools=hybrid_pools,
            positives=self.dataset.positives,
            k=self.args.candidate_limit,
        )
        source_family_diagnostics = _source_family_diagnostics(
            pools=hybrid_pools,
            positives=self.dataset.positives,
            k=self.args.candidate_limit,
        )
        miss_report = self._miss_report(
            base_pools=base_pools,
            hybrid_pools=hybrid_pools,
            hybrid_ranked_ids=hybrid_ranked_ids,
        )
        return EvalResult(
            rankings=rankings,
            metrics=metrics,
            slices=slices,
            source_counts=source_counts,
            source_diagnostics=source_diagnostics,
            source_family_diagnostics=source_family_diagnostics,
            miss_report=miss_report,
            hybrid_pools=hybrid_pools,
            hybrid_ranked_ids=hybrid_ranked_ids,
        )

    def _run_rows(
        self,
    ) -> tuple[list[CandidatePool], list[CandidatePool], list[list[str]], list[list[str]]]:
        base_pools: list[CandidatePool] = []
        hybrid_pools: list[CandidatePool] = []
        hybrid_ranked_ids: list[list[str]] = []
        cold_start_ranked_ids: list[list[str]] = []

        for row in self.dataset.test_b:
            history = self.dataset.history_map.get(row["user_id"], [])
            base_pool = CandidatePool(items=[])
            if not self.args.hybrid_only:
                base_pool = self._candidate_pool(row=row, history=history, collaborative_index=None)
            hybrid_pool = self._candidate_pool(
                row=row,
                history=history,
                collaborative_index=self.dataset.collaborative_index,
            )
            base_pools.append(base_pool)
            hybrid_pools.append(hybrid_pool)
            if not self.args.candidate_recall_only:
                hybrid_ranked_ids.append(
                    _rank_pool(
                        row=row,
                        history=history,
                        pool=hybrid_pool,
                        limit=max(self.args.k, len(hybrid_pool.items)),
                        item_profile_cache=self.dataset.item_profile_cache,
                    )
                )
            if not self.args.hybrid_only and not self.args.candidate_recall_only:
                cold_start_ranked_ids.append(self._cold_start_rank(row))
        return base_pools, hybrid_pools, hybrid_ranked_ids, cold_start_ranked_ids

    def _candidate_pool(
        self,
        row: dict,
        history: list,
        collaborative_index: dict | None,
    ) -> CandidatePool:
        return _candidate_pool(
            row=row,
            history=history,
            retriever=self.dataset.retriever,
            vector_retriever=self.dataset.vector_retriever,
            catalog=self.dataset.catalog,
            item_list=self.dataset.item_list,
            candidate_limit=self.args.candidate_limit,
            collaborative_index=collaborative_index,
            disabled_sources=self.dataset.disabled_sources,
            neural_retriever=self.dataset.neural_retriever,
        )

    def _cold_start_rank(self, row: dict) -> list[str]:
        return _cold_start_rank(
            row=row,
            retriever=self.dataset.retriever,
            vector_retriever=self.dataset.vector_retriever,
            catalog=self.dataset.catalog,
            item_list=self.dataset.item_list,
            candidate_limit=self.args.candidate_limit,
            limit=max(self.args.k, min(self.args.candidate_limit, len(self.dataset.item_list))),
            disabled_sources=self.dataset.disabled_sources,
            item_profile_cache=self.dataset.item_profile_cache,
            neural_retriever=self.dataset.neural_retriever,
        )

    def _build_rankings(
        self,
        base_pools: list[CandidatePool],
        hybrid_pools: list[CandidatePool],
        hybrid_ranked_ids: list[list[str]],
        cold_start_ranked_ids: list[list[str]],
    ) -> dict[str, list[list[str]]]:
        rankings = {
            "hybrid_candidate_recall": [
                [item.item_id for item in pool.items] for pool in hybrid_pools
            ],
        }
        if not self.args.candidate_recall_only:
            rankings["hybrid_ranker"] = hybrid_ranked_ids
        if not self.args.hybrid_only:
            rankings.update(
                {
                    "popularity": [self.dataset.popularity for _ in self.dataset.test_b],
                    "filtered_popularity": [
                        _filtered_popularity_rank(
                            row,
                            self.dataset.history_map,
                            self.dataset.popularity,
                        )
                        for row in self.dataset.test_b
                    ],
                    "base_candidate_recall": [
                        [item.item_id for item in pool.items] for pool in base_pools
                    ],
                    "cold_start_persona_only": cold_start_ranked_ids,
                }
            )
            if "bm25_profile" not in self.dataset.disabled_sources:
                rankings["bm25_profile"] = [
                    _bm25_rank(
                        row,
                        self.dataset.history_map,
                        self.dataset.retriever,
                        self.args.candidate_limit,
                    )
                    for row in self.dataset.test_b
                ]
            if self.dataset.vector_retriever is not None:
                rankings["vector_profile"] = [
                    _vector_rank(
                        row,
                        self.dataset.history_map,
                        self.dataset.vector_retriever,
                        self.args.candidate_limit,
                    )
                    for row in self.dataset.test_b
                ]
        return rankings

    def _metrics(self, rankings: dict[str, list[list[str]]]) -> dict[str, float]:
        metrics = {}
        for name, ranked_ids in rankings.items():
            metrics[f"{name}_hit_rate@{self.args.k}"] = rounded(
                hit_rate_at_k(ranked_ids, self.dataset.positives, self.args.k)
            )
            metrics[f"{name}_recall@{self.args.k}"] = rounded(
                recall_at_k(ranked_ids, self.dataset.positives, self.args.k)
            )
            metrics[f"{name}_ndcg@{self.args.k}"] = rounded(
                ndcg_at_k(ranked_ids, self.dataset.positives, self.args.k)
            )
        for recall_k in (50, 100, self.args.candidate_limit):
            if recall_k <= 0:
                continue
            if "base_candidate_recall" in rankings:
                metrics[f"base_candidate_recall@{recall_k}"] = rounded(
                    recall_at_k(
                        rankings["base_candidate_recall"],
                        self.dataset.positives,
                        recall_k,
                    )
                )
            metrics[f"hybrid_candidate_recall@{recall_k}"] = rounded(
                recall_at_k(
                    rankings["hybrid_candidate_recall"],
                    self.dataset.positives,
                    recall_k,
                )
            )
        return metrics

    def _slices(
        self,
        rankings: dict[str, list[list[str]]],
        hybrid_ranked_ids: list[list[str]],
    ) -> dict:
        if self.args.candidate_recall_only:
            return _slice_candidate_metrics(
                test_b=self.dataset.test_b,
                history_map=self.dataset.history_map,
                positives=self.dataset.positives,
                candidate_ids=rankings["hybrid_candidate_recall"],
                candidate_k=self.args.candidate_limit,
            )
        return _slice_metrics(
            test_b=self.dataset.test_b,
            history_map=self.dataset.history_map,
            positives=self.dataset.positives,
            ranked_ids=hybrid_ranked_ids,
            candidate_ids=rankings["hybrid_candidate_recall"],
            k=self.args.k,
            candidate_k=self.args.candidate_limit,
        )

    def _miss_report(
        self,
        base_pools: list[CandidatePool],
        hybrid_pools: list[CandidatePool],
        hybrid_ranked_ids: list[list[str]],
    ) -> dict:
        if self.args.candidate_recall_only:
            return _build_candidate_only_miss_report(
                test_b=self.dataset.test_b,
                history_map=self.dataset.history_map,
                items=self.dataset.items,
                hybrid_pools=hybrid_pools,
                candidate_limit=self.args.candidate_limit,
                max_misses=self.args.max_misses,
            )
        if self.args.hybrid_only:
            return _build_hybrid_only_miss_report(
                test_b=self.dataset.test_b,
                history_map=self.dataset.history_map,
                items=self.dataset.items,
                hybrid_pools=hybrid_pools,
                hybrid_ranked_ids=hybrid_ranked_ids,
                candidate_limit=self.args.candidate_limit,
                max_misses=self.args.max_misses,
            )
        return _build_miss_report(
            test_b=self.dataset.test_b,
            history_map=self.dataset.history_map,
            items=self.dataset.items,
            train_item_counts=self.dataset.train_item_counts,
            item_neighbor_ids=self.dataset.item_neighbor_ids,
            review_term_item_ids=self.dataset.review_term_item_ids,
            base_pools=base_pools,
            hybrid_pools=hybrid_pools,
            hybrid_ranked_ids=hybrid_ranked_ids,
            candidate_limit=self.args.candidate_limit,
            max_misses=self.args.max_misses,
        )


class EvalRunner:
    def __init__(self, args: argparse.Namespace) -> None:
        self.args = args

    def run(self) -> None:
        dataset = EvalDatasetBuilder(self.args).build()
        result = RecommendationEvaluator(self.args, dataset).evaluate()
        if self.args.miss_output:
            Path(self.args.miss_output).parent.mkdir(parents=True, exist_ok=True)
            Path(self.args.miss_output).write_text(
                json.dumps(result.miss_report, ensure_ascii=True, indent=2) + "\n",
                encoding="utf-8",
            )
        payload = self._payload(dataset, result)
        write_report(Path(self.args.output), payload)
        print_report(payload)

    def _payload(self, dataset: EvalDataset, result: EvalResult) -> dict:
        return {
            "task": "Task B",
            "dataset": str(
                Path(self.args.processed_dir)
                if Path(self.args.processed_dir).exists()
                else Path(self.args.reviews)
            ),
            "examples": len(dataset.test_b),
            "retriever": self.args.retriever,
            "neural_retriever_active": (
                dataset.neural_retriever is not None and dataset.neural_retriever._built
                if dataset.neural_retriever
                else False
            ),
            "disabled_sources": sorted(dataset.disabled_sources),
            "metrics": result.metrics,
            "slices": result.slices,
            "retrieval_sources": result.source_counts,
            "retrieval_source_diagnostics": result.source_diagnostics,
            "retrieval_source_families": result.source_family_diagnostics,
            "miss_analysis": result.miss_report["summary"],
            "notes": self._notes(dataset),
        }

    def _notes(self, dataset: EvalDataset) -> list[str]:
        if dataset.neural_retriever is not None and dataset.neural_retriever._built:
            retriever_note = (
                "Neural FAISS vector retriever active; neural_vector source contributes "
                "additional candidates and is tracked separately in source diagnostics."
            )
        elif dataset.vector_retriever is None:
            retriever_note = "Semantic vector retrieval disabled for this pruning run."
        else:
            retriever_note = "Legacy LocalVectorRetriever used for semantic vector retrieval."
        return [
            note
            for note in [
                "Positive item is the held-out next review for each eligible user.",
                (
                    "Filtered popularity removes items already seen in the user's training history."
                    if not self.args.hybrid_only
                    else ""
                ),
                (
                    "Candidate recall measures whether retrieval surfaced the held-out item "
                    "before ranking."
                ),
                (
                    "Hybrid candidates blend co-visitation, implicit item-item, user-neighbor CF, "
                    "review-term, evidence graph, BM25, vector, category-affinity, and popularity "
                    "sources when artifacts are available."
                ),
                (
                    "Cold-start persona-only uses the derived persona without history items."
                    if not self.args.hybrid_only
                    else ""
                ),
                (
                    "Hybrid ranker uses preference, context, category, aspect, sequential, "
                    "evidence graph, vector, collaborative, quality, novelty, "
                    "and confidence signals."
                ),
                "Hybrid-only eval mode skipped unchanged baseline and cold-start rankings."
                if self.args.hybrid_only
                else "",
                "Candidate-recall-only mode skipped ranking."
                if self.args.candidate_recall_only
                else "",
                (
                    f"Disabled retrieval sources: {', '.join(sorted(dataset.disabled_sources))}."
                    if dataset.disabled_sources
                    else ""
                ),
                retriever_note,
            ]
            if note
        ]


def _popularity_rank(train: list[dict], items: list[Item]) -> list[str]:
    return popularity_ranking(train, [item.item_id for item in items])


def _sample_eval_rows(rows: list[dict], max_examples: int, strategy: str) -> list[dict]:
    if max_examples <= 0 or len(rows) <= max_examples:
        return rows
    if strategy == "stride":
        step = len(rows) / max_examples
        return [rows[min(int(index * step), len(rows) - 1)] for index in range(max_examples)]
    return rows[:max_examples]


def _filtered_popularity_rank(
    row: dict,
    history_map: dict[str, list],
    popularity: list[str],
) -> list[str]:
    seen = {item.item_id for item in history_map.get(row["user_id"], [])}
    return [item_id for item_id in popularity if item_id not in seen]


def _items_with_train_popularity(items: dict[str, Item], train: list[dict]) -> list[Item]:
    counts = Counter(row["item_id"] for row in train if row["rating"] >= 4)
    enriched = []
    for item in items.values():
        metadata = dict(item.metadata)
        metadata["review_count"] = counts[item.item_id]
        metadata["rating_number"] = counts[item.item_id]
        enriched.append(item.model_copy(update={"metadata": metadata}))
    return enriched


def _bm25_rank(
    row: dict,
    history_map: dict[str, list],
    retriever: BM25Retriever,
    candidate_limit: int,
) -> list[str]:
    history = history_map.get(row["user_id"], [])
    query = persona_from_history(history)
    return [item.item_id for item in retriever.search(query, limit=candidate_limit)]


def _vector_rank(
    row: dict,
    history_map: dict[str, list],
    retriever: LocalVectorRetriever,
    candidate_limit: int,
) -> list[str]:
    history = history_map.get(row["user_id"], [])
    query = persona_from_history(history)
    return [item.item_id for item in retriever.search(query, limit=candidate_limit)]


def _candidate_pool(
    row: dict,
    history: list,
    retriever: BM25Retriever,
    vector_retriever: LocalVectorRetriever | None,
    catalog: CandidateCatalog,
    item_list: list[Item],
    candidate_limit: int,
    collaborative_index: dict | None,
    disabled_sources: set[str] | None = None,
    neural_retriever: FAISSVectorStore | None = None,
) -> CandidatePool:
    persona = persona_from_history(history)
    user_profile = build_user_profile(persona=persona, history=history, locale=None)
    return generate_candidate_pool(
        user_profile=user_profile,
        history=history,
        items=item_list,
        context="",
        collaborative_index=collaborative_index,
        bm25_retriever=retriever,
        vector_retriever=vector_retriever,
        catalog=catalog,
        disabled_sources=disabled_sources,
        limit=min(candidate_limit, len(item_list)),
        neural_retriever=neural_retriever,
    )


def _rank_pool(
    row: dict,
    history: list,
    pool: CandidatePool,
    limit: int,
    item_profile_cache: dict | None = None,
) -> list[str]:
    persona = persona_from_history(history)
    user_profile = build_user_profile(persona=persona, history=history, locale=None)
    ranked = rank_candidates(
        user_profile=user_profile,
        context="",
        candidate_items=pool.items,
        limit=limit,
        candidate_sources=pool.sources,
        candidate_source_scores=pool.source_scores,
        item_profile_cache=item_profile_cache,
    )
    return [item.item_id for item in ranked]


def _cold_start_rank(
    row: dict,
    retriever: BM25Retriever,
    vector_retriever: LocalVectorRetriever | None,
    catalog: CandidateCatalog,
    item_list: list[Item],
    candidate_limit: int,
    limit: int,
    disabled_sources: set[str] | None = None,
    item_profile_cache: dict | None = None,
    neural_retriever: FAISSVectorStore | None = None,
) -> list[str]:
    persona = _cold_start_persona(row)
    user_profile = build_user_profile(persona=persona, history=[], locale=None)
    pool = generate_candidate_pool(
        user_profile=user_profile,
        history=[],
        items=item_list,
        context="",
        bm25_retriever=retriever,
        vector_retriever=vector_retriever,
        catalog=catalog,
        disabled_sources=disabled_sources,
        limit=min(candidate_limit, len(item_list)),
        neural_retriever=neural_retriever,
    )
    ranked = rank_candidates(
        user_profile=user_profile,
        context="",
        candidate_items=pool.items,
        limit=limit,
        candidate_sources=pool.sources,
        candidate_source_scores=pool.source_scores,
        item_profile_cache=item_profile_cache,
    )
    return [item.item_id for item in ranked]


def _cold_start_persona(row: dict) -> str:
    category = row.get("category") or "products"
    review = row.get("review") or ""
    return f"A user interested in {category}. Preference hints from signup text: {review[:240]}"


def _load_collaborative_index(args: argparse.Namespace, train: list[dict]) -> dict | None:
    configured = Path(args.collaborative_index) if args.collaborative_index else None
    candidates = [
        configured,
        Path(args.processed_dir) / "collaborative_retrieval.json",
        Path(args.processed_dir) / "item_neighbors.json",
    ]
    for path in candidates:
        if not path or not path.exists():
            continue
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        if "item_neighbors" in payload:
            return _attach_review_term_index(payload, path)
        if "items" in payload:
            return _attach_review_term_index(
                {"type": "legacy_item_neighbors", "item_neighbors": payload["items"]},
                path,
            )
    for path in (
        Path(args.processed_dir) / "implicit_item_neighbors.sqlite",
        Path(args.processed_dir) / "implicit_item_neighbors.json.gz",
        Path(args.processed_dir) / "implicit_item_neighbors.json",
    ):
        if path.exists():
            return _attach_implicit_item_index({"type": "implicit_item_item"}, path)
    if args.build_collaborative or len(train) <= 100_000:
        return build_collaborative_retrieval_index(train, top_k=50)
    return None


def _attach_review_term_index(payload: dict, source_path: Path) -> dict:
    review_term_path = source_path.parent / "review_term_retrieval.json"
    if not review_term_path.exists():
        return _attach_implicit_item_index(payload, source_path)
    try:
        review_term_payload = read_json_artifact(review_term_path)
    except (OSError, json.JSONDecodeError):
        return _attach_implicit_item_index(payload, source_path)
    if review_term_payload.get("term_items"):
        payload = dict(payload)
        payload["review_term_retrieval"] = review_term_payload
    return _attach_implicit_item_index(payload, source_path)


def _attach_implicit_item_index(payload: dict, source_path: Path) -> dict:
    for path in (
        source_path.parent / "implicit_item_neighbors.sqlite",
        source_path.parent / "implicit_item_neighbors.json.gz",
        source_path.parent / "implicit_item_neighbors.json",
    ):
        if not path.exists():
            continue
        payload = dict(payload)
        if path.suffix == ".sqlite":
            payload["implicit_item_neighbors"] = SQLiteItemNeighborIndex(path)
            return _attach_evidence_graph_index(payload, source_path)
        try:
            implicit_payload = read_json_artifact(path)
        except (OSError, json.JSONDecodeError):
            continue
        neighbors = (
            implicit_payload.get("neighbors")
            if isinstance(implicit_payload, dict)
            else None
        )
        if neighbors:
            payload["implicit_item_neighbors"] = neighbors
            return _attach_evidence_graph_index(payload, source_path)
    return _attach_evidence_graph_index(payload, source_path)


def _attach_evidence_graph_index(payload: dict, source_path: Path) -> dict:
    evidence_graph_path = source_path.parent / "evidence_graph_retrieval.json"
    if not evidence_graph_path.exists():
        return payload
    try:
        evidence_graph_payload = read_json_artifact(evidence_graph_path)
    except (OSError, json.JSONDecodeError):
        return payload
    if evidence_graph_payload.get("type") == "evidence_graph":
        payload = dict(payload)
        payload["evidence_graph_retrieval"] = evidence_graph_payload
    return payload


def _item_neighbor_ids(collaborative_index: dict | None) -> set[str]:
    if not collaborative_index:
        return set()
    if collaborative_index.get("item_neighbors"):
        return set(collaborative_index["item_neighbors"])
    if collaborative_index.get("items"):
        return set(collaborative_index["items"])
    return set()


def _review_term_item_ids(collaborative_index: dict | None) -> set[str]:
    if not collaborative_index:
        return set()
    review_term_index = collaborative_index.get("review_term_retrieval") or {}
    return set((review_term_index.get("item_terms") or {}).keys())


def _build_miss_report(
    test_b: list[dict],
    history_map: dict[str, list],
    items: dict[str, Item],
    train_item_counts: Counter,
    item_neighbor_ids: set[str],
    review_term_item_ids: set[str],
    base_pools: list[CandidatePool],
    hybrid_pools: list[CandidatePool],
    hybrid_ranked_ids: list[list[str]],
    candidate_limit: int,
    max_misses: int,
) -> dict:
    misses = []
    summary_counts = Counter()
    category_counts = Counter()
    history_buckets = Counter()
    for index, row in enumerate(test_b):
        positive = row["item_id"]
        history = history_map.get(row["user_id"], [])
        base_ids = [item.item_id for item in base_pools[index].items]
        hybrid_ids = [item.item_id for item in hybrid_pools[index].items]
        base_rank = _rank_position(base_ids, positive)
        hybrid_rank = _rank_position(hybrid_ids, positive)
        final_rank = _rank_position(hybrid_ranked_ids[index], positive)
        if hybrid_rank is not None:
            continue

        target = items.get(positive)
        target_category = (target.category if target else row.get("category")) or "unknown"
        target_popularity = int(train_item_counts[positive])
        history_len = len(history)
        bucket = _history_bucket(history_len)
        positive_categories = {
            item.category
            for item in history
            if item.category and item.rating >= 4
        }
        likely_causes = _miss_causes(
            target_popularity=target_popularity,
            target_category=target_category,
            positive_categories=positive_categories,
            target_has_neighbors=positive in item_neighbor_ids,
            target_has_review_terms=positive in review_term_item_ids,
            base_rank=base_rank,
        )

        summary_counts.update(likely_causes)
        category_counts[target_category] += 1
        history_buckets[bucket] += 1
        if len(misses) < max_misses:
            misses.append(
                {
                    "user_id": row["user_id"],
                    "target_item_id": positive,
                    "target_name": target.name if target else row.get("item_name"),
                    "target_category": target_category,
                    "target_train_positive_count": target_popularity,
                    "history_length": history_len,
                    "history_bucket": bucket,
                    "positive_history_categories": sorted(positive_categories),
                    "target_has_item_neighbors": positive in item_neighbor_ids,
                    "target_has_review_terms": positive in review_term_item_ids,
                    "base_candidate_rank": base_rank,
                    "hybrid_candidate_rank": hybrid_rank,
                    "hybrid_final_rank": final_rank,
                    "likely_causes": likely_causes,
                }
            )

    return {
        "summary": {
            "candidate_limit": candidate_limit,
            "candidate_misses": sum(category_counts.values()),
            "miss_cause_counts": dict(sorted(summary_counts.items())),
            "miss_category_counts": dict(category_counts.most_common(10)),
            "miss_history_buckets": dict(sorted(history_buckets.items())),
        },
        "misses": misses,
    }


def _build_hybrid_only_miss_report(
    test_b: list[dict],
    history_map: dict[str, list],
    items: dict[str, Item],
    hybrid_pools: list[CandidatePool],
    hybrid_ranked_ids: list[list[str]],
    candidate_limit: int,
    max_misses: int,
) -> dict:
    misses = []
    category_counts = Counter()
    history_buckets = Counter()
    ranker_misses = 0
    for index, row in enumerate(test_b):
        positive = row["item_id"]
        history = history_map.get(row["user_id"], [])
        hybrid_ids = [item.item_id for item in hybrid_pools[index].items]
        hybrid_rank = _rank_position(hybrid_ids, positive)
        final_rank = _rank_position(hybrid_ranked_ids[index], positive)
        if hybrid_rank is not None:
            if final_rank is None or final_rank > 10:
                ranker_misses += 1
            continue

        target = items.get(positive)
        target_category = (target.category if target else row.get("category")) or "unknown"
        bucket = _history_bucket(len(history))
        category_counts[target_category] += 1
        history_buckets[bucket] += 1
        if len(misses) < max_misses:
            misses.append(
                {
                    "user_id": row["user_id"],
                    "target_item_id": positive,
                    "target_name": target.name if target else row.get("item_name"),
                    "target_category": target_category,
                    "history_length": len(history),
                    "history_bucket": bucket,
                    "hybrid_candidate_rank": hybrid_rank,
                    "hybrid_final_rank": final_rank,
                    "likely_causes": ["hybrid_retrieval_miss"],
                }
            )

    return {
        "summary": {
            "candidate_limit": candidate_limit,
            "candidate_misses": sum(category_counts.values()),
            "ranker_misses_after_retrieval_hit": ranker_misses,
            "miss_cause_counts": {"hybrid_retrieval_miss": sum(category_counts.values())},
            "miss_category_counts": dict(category_counts.most_common(10)),
            "miss_history_buckets": dict(sorted(history_buckets.items())),
        },
        "misses": misses,
    }


def _build_candidate_only_miss_report(
    test_b: list[dict],
    history_map: dict[str, list],
    items: dict[str, Item],
    hybrid_pools: list[CandidatePool],
    candidate_limit: int,
    max_misses: int,
) -> dict:
    misses = []
    category_counts = Counter()
    history_buckets = Counter()
    for index, row in enumerate(test_b):
        positive = row["item_id"]
        hybrid_ids = [item.item_id for item in hybrid_pools[index].items]
        hybrid_rank = _rank_position(hybrid_ids, positive)
        if hybrid_rank is not None:
            continue
        history = history_map.get(row["user_id"], [])
        target = items.get(positive)
        target_category = (target.category if target else row.get("category")) or "unknown"
        bucket = _history_bucket(len(history))
        category_counts[target_category] += 1
        history_buckets[bucket] += 1
        if len(misses) < max_misses:
            misses.append(
                {
                    "user_id": row["user_id"],
                    "target_item_id": positive,
                    "target_name": target.name if target else row.get("item_name"),
                    "target_category": target_category,
                    "history_length": len(history),
                    "history_bucket": bucket,
                    "hybrid_candidate_rank": hybrid_rank,
                    "likely_causes": ["hybrid_retrieval_miss"],
                }
            )
    return {
        "summary": {
            "candidate_limit": candidate_limit,
            "candidate_misses": sum(category_counts.values()),
            "miss_cause_counts": {"hybrid_retrieval_miss": sum(category_counts.values())},
            "miss_category_counts": dict(category_counts.most_common(10)),
            "miss_history_buckets": dict(sorted(history_buckets.items())),
        },
        "misses": misses,
    }


def _rank_position(ranking: list[str], item_id: str) -> int | None:
    try:
        return ranking.index(item_id) + 1
    except ValueError:
        return None


def _history_bucket(history_len: int) -> str:
    if history_len <= 2:
        return "sparse_1_2"
    if history_len <= 7:
        return "medium_3_7"
    return "warm_8_plus"


def _miss_causes(
    target_popularity: int,
    target_category: str,
    positive_categories: set[str],
    target_has_neighbors: bool,
    target_has_review_terms: bool,
    base_rank: int | None,
) -> list[str]:
    causes = []
    if target_popularity == 0:
        causes.append("target_absent_from_positive_train")
    elif target_popularity <= 2:
        causes.append("target_sparse_in_train")
    if target_category not in positive_categories:
        causes.append("cross_domain_or_no_category_affinity")
    if not target_has_neighbors:
        causes.append("no_item_neighbor_path")
    if not target_has_review_terms:
        causes.append("no_review_term_path")
    if base_rank is None:
        causes.append("base_retrieval_miss")
    return causes or ["ranker_miss"]


def _slice_metrics(
    test_b: list[dict],
    history_map: dict[str, list],
    positives: list[str],
    ranked_ids: list[list[str]],
    k: int,
    candidate_ids: list[list[str]] | None = None,
    candidate_k: int = 0,
) -> dict[str, dict[str, float]]:
    slice_indices = {
        "sparse_history_1_2": [],
        "medium_history_3_7": [],
        "warm_history_8_plus": [],
        "cross_domain": [],
    }
    for index, row in enumerate(test_b):
        history = history_map.get(row["user_id"], [])
        history_len = len(history)
        if history_len <= 2:
            slice_indices["sparse_history_1_2"].append(index)
        elif history_len <= 7:
            slice_indices["medium_history_3_7"].append(index)
        else:
            slice_indices["warm_history_8_plus"].append(index)
        if _is_cross_domain(row, history):
            slice_indices["cross_domain"].append(index)

    return {
        name: _metrics_for_indices(
            indices=indices,
            positives=positives,
            ranked_ids=ranked_ids,
            k=k,
            candidate_ids=candidate_ids,
            candidate_k=candidate_k,
        )
        for name, indices in slice_indices.items()
        if indices
    }


def _slice_candidate_metrics(
    test_b: list[dict],
    history_map: dict[str, list],
    positives: list[str],
    candidate_ids: list[list[str]],
    candidate_k: int,
) -> dict[str, dict[str, float]]:
    slice_indices = {
        "sparse_history_1_2": [],
        "medium_history_3_7": [],
        "warm_history_8_plus": [],
        "cross_domain": [],
    }
    for index, row in enumerate(test_b):
        history = history_map.get(row["user_id"], [])
        history_len = len(history)
        if history_len <= 2:
            slice_indices["sparse_history_1_2"].append(index)
        elif history_len <= 7:
            slice_indices["medium_history_3_7"].append(index)
        else:
            slice_indices["warm_history_8_plus"].append(index)
        if _is_cross_domain(row, history):
            slice_indices["cross_domain"].append(index)
    return {
        name: {
            "examples": len(indices),
            f"hybrid_candidate_recall@{candidate_k}": rounded(
                recall_at_k(
                    [candidate_ids[index] for index in indices],
                    [positives[index] for index in indices],
                    candidate_k,
                )
            ),
        }
        for name, indices in slice_indices.items()
        if indices
    }


def _metrics_for_indices(
    indices: list[int],
    positives: list[str],
    ranked_ids: list[list[str]],
    k: int,
    candidate_ids: list[list[str]] | None = None,
    candidate_k: int = 0,
) -> dict[str, float]:
    sliced_rankings = [ranked_ids[index] for index in indices]
    sliced_positives = [positives[index] for index in indices]
    metrics = {
        "examples": len(indices),
        f"hybrid_ranker_hit_rate@{k}": rounded(hit_rate_at_k(sliced_rankings, sliced_positives, k)),
        f"hybrid_ranker_ndcg@{k}": rounded(ndcg_at_k(sliced_rankings, sliced_positives, k)),
    }
    if candidate_ids is not None and candidate_k > 0:
        sliced_candidates = [candidate_ids[index] for index in indices]
        metrics[f"hybrid_candidate_recall@{candidate_k}"] = rounded(
            recall_at_k(sliced_candidates, sliced_positives, candidate_k)
        )
    return metrics


def _is_cross_domain(row: dict, history: list) -> bool:
    target_category = row.get("category")
    if not target_category:
        return False
    positive_categories = {
        item.category
        for item in history
        if item.category and item.rating >= 4
    }
    return bool(positive_categories) and target_category not in positive_categories


def _aggregate_source_counts(pools: list[CandidatePool]) -> dict[str, int]:
    counts = Counter()
    for pool in pools:
        counts.update(pool.source_counts())
    return dict(sorted(counts.items()))


def _source_diagnostics(
    pools: list[CandidatePool],
    positives: list[str],
    k: int,
) -> dict[str, dict]:
    sources = sorted(
        {
            source
            for pool in pools
            for source_names in pool.sources.values()
            for source in source_names
        }
    )
    source_counts = Counter()
    source_rankings = {source: [] for source in sources}

    for pool in pools:
        pool_source_ids = {source: [] for source in sources}
        for item in pool.items[:k]:
            for source in sorted(pool.sources.get(item.item_id, [])):
                if source in pool_source_ids:
                    pool_source_ids[source].append(item.item_id)
        for source in sources:
            source_rankings[source].append(pool_source_ids[source])
        for source_names in pool.sources.values():
            source_counts.update(source_names)

    return {
        source: _diagnostic_payload(
            count=source_counts[source],
            rankings=source_rankings[source],
            positives=positives,
            k=k,
            extra={"family": _source_family(source)},
        )
        for source in sources
    }


def _source_family_diagnostics(
    pools: list[CandidatePool],
    positives: list[str],
    k: int,
) -> dict[str, dict]:
    family_counts = Counter()
    family_source_counts = {family: Counter() for family in SOURCE_FAMILY_ORDER}
    family_rankings = {family: [] for family in SOURCE_FAMILY_ORDER}

    for pool in pools:
        pool_family_ids = {family: [] for family in SOURCE_FAMILY_ORDER}
        for item in pool.items[:k]:
            item_families = {
                _source_family(source)
                for source in pool.sources.get(item.item_id, [])
            }
            for family in sorted(item_families):
                pool_family_ids[family].append(item.item_id)

        for family in SOURCE_FAMILY_ORDER:
            family_rankings[family].append(pool_family_ids[family])

        for source_names in pool.sources.values():
            for source in source_names:
                family = _source_family(source)
                family_counts[family] += 1
                family_source_counts[family][source] += 1

    return {
        family: {
            "label": SOURCE_FAMILY_LABELS[family],
            **_diagnostic_payload(
                count=family_counts[family],
                rankings=family_rankings[family],
                positives=positives,
                k=k,
            ),
            "sources": dict(sorted(family_source_counts[family].items())),
        }
        for family in SOURCE_FAMILY_ORDER
    }


def _diagnostic_payload(
    count: int,
    rankings: list[list[str]],
    positives: list[str],
    k: int,
    extra: dict | None = None,
) -> dict:
    hits = _hit_count_at_k(rankings, positives, k)
    payload = {
        "count": count,
        f"hits@{k}": hits,
        f"misses@{k}": max(len(positives) - hits, 0),
        f"candidate_recall@{k}": rounded(recall_at_k(rankings, positives, k)),
    }
    if extra:
        payload.update(extra)
    return payload


def _hit_count_at_k(ranked_ids: list[list[str]], positives: list[str], k: int) -> int:
    return sum(
        1
        for ranking, positive in zip(ranked_ids, positives)
        if positive in ranking[:k]
    )


def _source_family(source: str) -> str:
    return retrieval_source_family(source)


if __name__ == "__main__":
    main()
