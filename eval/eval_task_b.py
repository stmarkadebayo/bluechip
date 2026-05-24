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
from app.models.schemas import Item, RecommendationItem  # noqa: E402
from app.platform.artifacts import read_json_artifact  # noqa: E402
from app.services.profiling.user_profile import build_user_profile  # noqa: E402
from app.services.ranking.learned_task_b import TaskBLinearRanker  # noqa: E402
from app.services.ranking.context_intents import context_intent_rule  # noqa: E402
from app.services.ranking.recommendation import rank_candidates  # noqa: E402
from app.services.retrieval.embeddings import terms  # noqa: E402
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
from eval.task_b_context import context_for_task_b_row  # noqa: E402

TASK_B_REQUIRED_GATE_SLICES = ("all", "sparse_history_1_2", "cross_domain")
TASK_B_OPTIONAL_GATE_SLICES = ("cold_start", "context_heavy", "intent_heavy")
TASK_B_CANDIDATE_RECALL_GATES = {
    "all": 0.34,
    "sparse_history_1_2": 0.3611,
    "cross_domain": 0.5484,
    "cold_start": 0.34,
    "context_heavy": 0.34,
    "intent_heavy": 0.34,
}
TASK_B_RANKING_GATES = {
    "hybrid_candidate_recall@50": 0.13,
    "hybrid_candidate_recall@100": 0.18,
    "hybrid_ranker_hit_rate@10": 0.10,
    "hybrid_ranker_ndcg@10": 0.0766,
}
EVAL_ROW_CACHE_SCHEMA_VERSION = 1


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate Task B recommendation baselines.")
    parser.add_argument("--reviews", default="data/sample/reviews.jsonl")
    parser.add_argument("--items", default="data/sample/items.jsonl")
    parser.add_argument("--processed-dir", default="data/processed")
    parser.add_argument("--collaborative-index", default="")
    parser.add_argument("--output", default="runs/eval/task_b_report.json")
    parser.add_argument(
        "--learned-ranker-artifact",
        default="",
        help="Optional Task B learned linear ranker artifact to add as a separate metric path.",
    )
    parser.add_argument("--k", type=int, default=10)
    parser.add_argument("--candidate-limit", type=int, default=200)
    parser.add_argument(
        "--rank-depth",
        type=int,
        default=0,
        help=(
            "Number of retrieved candidates to score with the ranker. "
            "0 preserves full-pool ranking; use a bounded value for full-split @k gates."
        ),
    )
    parser.add_argument("--max-examples", type=int, default=0)
    parser.add_argument(
        "--shard-count",
        type=int,
        default=1,
        help="Split eval rows into N deterministic shards for parallel full-split runs.",
    )
    parser.add_argument(
        "--shard-index",
        type=int,
        default=0,
        help="Zero-based shard index to evaluate when --shard-count is greater than 1.",
    )
    parser.add_argument(
        "--progress-every",
        type=int,
        default=1000,
        help="Print progress to stderr every N Task B rows; set 0 to disable.",
    )
    parser.add_argument(
        "--context-mode",
        choices=["none", "synthetic"],
        default="none",
        help=(
            "Context to pass through retrieval/ranking. 'none' preserves the standard "
            "next-item offline gate; 'synthetic' reuses the contextual human-eval intent builder."
        ),
    )
    parser.add_argument(
        "--sample-strategy",
        choices=["first", "stride"],
        default="first",
        help="How to select --max-examples rows; stride reduces first-N ordering bias.",
    )
    parser.add_argument(
        "--target-mode",
        choices=["all_interactions", "positive_recommendation"],
        default="all_interactions",
        help=(
            "Which held-out Task B rows to score: all next interactions, or only "
            "rating >= 4 positive recommendation targets."
        ),
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
        "--row-cache",
        default="",
        help="Optional JSONL cache for full Task B eval candidate pools and rankings.",
    )
    parser.add_argument(
        "--rebuild-row-cache",
        action="store_true",
        help="Ignore an existing --row-cache and rebuild it.",
    )
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
    contexts: list[str]
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
    learned_ranker: TaskBLinearRanker | None


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
    learned_ranked_ids: list[list[str]]


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
        test_b = _filter_task_b_targets(test_b, self.args.target_mode)
        if self.args.max_examples:
            test_b = _sample_eval_rows(
                test_b,
                self.args.max_examples,
                self.args.sample_strategy,
            )
        test_b = _shard_eval_rows(test_b, self.args.shard_count, self.args.shard_index)
        disabled_sources = {
            source.strip()
            for source in self.args.disabled_sources.split(",")
            if source.strip()
        }
        positives = [row["item_id"] for row in test_b]
        history_map = histories_by_user(train)
        contexts = [
            _context_for_eval(row, history_map.get(row["user_id"], []), self.args.context_mode)
            for row in test_b
        ]
        retriever = BM25Retriever.from_items(item_list)
        vector_retriever = (
            None if "vector_profile" in disabled_sources else LocalVectorRetriever(item_list)
        )
        neural_retriever = self._build_neural_retriever(item_list)
        catalog = CandidateCatalog.from_items(item_list)
        popularity = _popularity_rank(train, item_list)
        train_item_counts = Counter(row["item_id"] for row in train if row["rating"] >= 4)
        collaborative_index = _load_collaborative_index(self.args, train)
        learned_ranker = _load_learned_ranker(getattr(self.args, "learned_ranker_artifact", ""))
        return EvalDataset(
            train=train,
            test_b=test_b,
            items=items,
            item_list=item_list,
            positives=positives,
            contexts=contexts,
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
            learned_ranker=learned_ranker,
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
        cached_rows = _load_eval_row_cache(self.args, self.dataset)
        if cached_rows is None:
            (
                base_pools,
                hybrid_pools,
                hybrid_ranked_ids,
                learned_ranked_ids,
                cold_start_ranked_ids,
            ) = self._run_rows()
            _write_eval_row_cache(
                self.args,
                self.dataset,
                base_pools,
                hybrid_pools,
                hybrid_ranked_ids,
                learned_ranked_ids,
                cold_start_ranked_ids,
            )
        else:
            (
                base_pools,
                hybrid_pools,
                hybrid_ranked_ids,
                learned_ranked_ids,
                cold_start_ranked_ids,
            ) = cached_rows
        rankings = self._build_rankings(
            base_pools,
            hybrid_pools,
            hybrid_ranked_ids,
            learned_ranked_ids,
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
            learned_ranked_ids=learned_ranked_ids,
        )

    def _run_rows(
        self,
    ) -> tuple[
        list[CandidatePool],
        list[CandidatePool],
        list[list[str]],
        list[list[str]],
        list[list[str]],
    ]:
        base_pools: list[CandidatePool] = []
        hybrid_pools: list[CandidatePool] = []
        hybrid_ranked_ids: list[list[str]] = []
        learned_ranked_ids: list[list[str]] = []
        cold_start_ranked_ids: list[list[str]] = []

        for index, row in enumerate(self.dataset.test_b):
            if self.args.progress_every and (index + 1) % self.args.progress_every == 0:
                print(
                    f"Task B eval progress: {index + 1}/{len(self.dataset.test_b)} rows",
                    file=sys.stderr,
                    flush=True,
                )
            history = self.dataset.history_map.get(row["user_id"], [])
            context = self.dataset.contexts[index]
            base_pool = CandidatePool(items=[])
            if not self.args.hybrid_only:
                base_pool = self._candidate_pool(
                    row=row,
                    history=history,
                    context=context,
                    collaborative_index=None,
                )
            hybrid_pool = self._candidate_pool(
                row=row,
                history=history,
                context=context,
                collaborative_index=self.dataset.collaborative_index,
            )
            base_pools.append(base_pool)
            hybrid_pools.append(hybrid_pool)
            if not self.args.candidate_recall_only:
                ranked_items = _rank_pool(
                    row=row,
                    history=history,
                    context=context,
                    pool=hybrid_pool,
                    limit=max(self.args.k, len(hybrid_pool.items)),
                    rank_depth=self.args.rank_depth,
                    item_profile_cache=self.dataset.item_profile_cache,
                )
                hybrid_ranked_ids.append([item.item_id for item in ranked_items])
                if self.dataset.learned_ranker is not None:
                    learned_ranked_ids.append(
                        [
                            item.item_id
                            for item in self.dataset.learned_ranker.rerank(
                                ranked_items,
                                limit=len(ranked_items),
                            )
                        ]
                    )
            if not self.args.hybrid_only and not self.args.candidate_recall_only:
                cold_start_ranked_ids.append(self._cold_start_rank(row, context))
        return base_pools, hybrid_pools, hybrid_ranked_ids, learned_ranked_ids, cold_start_ranked_ids

    def _candidate_pool(
        self,
        row: dict,
        history: list,
        context: str,
        collaborative_index: dict | None,
    ) -> CandidatePool:
        return _candidate_pool(
            row=row,
            history=history,
            context=context,
            retriever=self.dataset.retriever,
            vector_retriever=self.dataset.vector_retriever,
            catalog=self.dataset.catalog,
            item_list=self.dataset.item_list,
            candidate_limit=self.args.candidate_limit,
            collaborative_index=collaborative_index,
            disabled_sources=self.dataset.disabled_sources,
            neural_retriever=self.dataset.neural_retriever,
        )

    def _cold_start_rank(self, row: dict, context: str) -> list[str]:
        return _cold_start_rank(
            row=row,
            context=context,
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
        learned_ranked_ids: list[list[str]],
        cold_start_ranked_ids: list[list[str]],
    ) -> dict[str, list[list[str]]]:
        rankings = {
            "hybrid_candidate_recall": [
                [item.item_id for item in pool.items] for pool in hybrid_pools
            ],
        }
        if not self.args.candidate_recall_only:
            rankings["hybrid_ranker"] = hybrid_ranked_ids
            if learned_ranked_ids:
                rankings["hybrid_learned_ranker"] = learned_ranked_ids
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
                contexts=self.dataset.contexts,
                positives=self.dataset.positives,
                candidate_ids=rankings["hybrid_candidate_recall"],
                candidate_k=self.args.candidate_limit,
            )
        return _slice_metrics(
            test_b=self.dataset.test_b,
            history_map=self.dataset.history_map,
            contexts=self.dataset.contexts,
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
            "context_mode": self.args.context_mode,
            "target_mode": self.args.target_mode,
            "shard": {
                "index": self.args.shard_index,
                "count": self.args.shard_count,
            },
            "target_rating_distribution": _target_rating_distribution(dataset.test_b),
            "rank_depth": self.args.rank_depth,
            "learned_ranker_active": dataset.learned_ranker is not None,
            "neural_retriever_active": (
                dataset.neural_retriever is not None and dataset.neural_retriever._built
                if dataset.neural_retriever
                else False
            ),
            "disabled_sources": sorted(dataset.disabled_sources),
            "metrics": result.metrics,
            "slices": result.slices,
            "promotion_gate": _task_b_promotion_gate(
                metrics=result.metrics,
                slices=result.slices,
                examples=len(dataset.test_b),
                k=self.args.k,
                candidate_k=self.args.candidate_limit,
                ranker_metric_prefix=(
                    "hybrid_learned_ranker"
                    if dataset.learned_ranker is not None
                    else "hybrid_ranker"
                ),
            ),
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
                    "Target mode is positive_recommendation, so Task B rows with rating < 4 "
                    "were excluded from this eval."
                    if self.args.target_mode == "positive_recommendation"
                    else "Target mode is all_interactions, so every held-out next review is a target."
                ),
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
                    "Synthetic context mode reuses the Task B contextual human-eval intent builder."
                    if self.args.context_mode == "synthetic"
                    else ""
                ),
                (
                    "Hybrid candidates blend co-visitation, implicit item-item, user-neighbor CF, "
                    "review-term, context-intent, evidence graph, BM25, vector, category-affinity, "
                    "and popularity sources when artifacts are available."
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
                    f"Shard {self.args.shard_index + 1}/{self.args.shard_count} "
                    "of the filtered Task B rows."
                    if self.args.shard_count > 1
                    else ""
                ),
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


def _shard_eval_rows(rows: list[dict], shard_count: int, shard_index: int) -> list[dict]:
    if shard_count < 1:
        raise ValueError("--shard-count must be >= 1")
    if shard_index < 0 or shard_index >= shard_count:
        raise ValueError("--shard-index must be in [0, shard_count)")
    if shard_count == 1:
        return rows
    return [row for index, row in enumerate(rows) if index % shard_count == shard_index]


def _filter_task_b_targets(rows: list[dict], target_mode: str) -> list[dict]:
    if target_mode == "positive_recommendation":
        return [row for row in rows if float(row.get("rating") or 0) >= 4.0]
    return rows


def _target_rating_distribution(rows: list[dict]) -> dict:
    counts = Counter()
    for row in rows:
        rating = float(row.get("rating") or 0)
        bucket = "rating_4_5" if rating >= 4 else "rating_1_3"
        counts[bucket] += 1
    total = max(len(rows), 1)
    return {
        "total": len(rows),
        **{
            bucket: {
                "count": count,
                "share": rounded(count / total),
            }
            for bucket, count in sorted(counts.items())
        },
    }


def _context_for_eval(row: dict, history: list, context_mode: str) -> str:
    if context_mode == "synthetic":
        return context_for_task_b_row(row, history)
    return ""


def _load_learned_ranker(path: str) -> TaskBLinearRanker | None:
    if not path:
        return None
    artifact_path = Path(path)
    if not artifact_path.exists():
        raise FileNotFoundError(f"Task B learned ranker artifact not found: {artifact_path}")
    return TaskBLinearRanker.from_json(artifact_path)


def _load_eval_row_cache(
    args: argparse.Namespace,
    dataset: EvalDataset,
) -> tuple[
    list[CandidatePool],
    list[CandidatePool],
    list[list[str]],
    list[list[str]],
    list[list[str]],
] | None:
    path = Path(args.row_cache) if args.row_cache else None
    if path is None or args.rebuild_row_cache or not path.exists():
        return None
    signature = _eval_row_cache_signature(args, dataset)
    with path.open("r", encoding="utf-8") as handle:
        header_line = handle.readline()
        if not header_line:
            return None
        header = json.loads(header_line)
        if (
            header.get("type") != "task_b_eval_row_cache"
            or header.get("schema_version") != EVAL_ROW_CACHE_SCHEMA_VERSION
            or header.get("signature") != signature
        ):
            raise ValueError(f"Task B eval row cache config mismatch: {path}")
        base_pools = []
        hybrid_pools = []
        hybrid_ranked_ids = []
        learned_ranked_ids = []
        cold_start_ranked_ids = []
        for line in handle:
            if not line.strip():
                continue
            row = json.loads(line)
            base_pools.append(_pool_from_cache(row.get("base_pool") or {}, dataset.catalog))
            hybrid_pools.append(_pool_from_cache(row.get("hybrid_pool") or {}, dataset.catalog))
            hybrid_ranked_ids.append([str(item_id) for item_id in row.get("hybrid_ranked_ids", [])])
            if "learned_ranked_ids" in row:
                learned_ranked_ids.append(
                    [str(item_id) for item_id in row.get("learned_ranked_ids", [])]
                )
            if "cold_start_ranked_ids" in row:
                cold_start_ranked_ids.append(
                    [str(item_id) for item_id in row.get("cold_start_ranked_ids", [])]
                )
    print(f"Loaded {len(hybrid_pools)} Task B eval rows from cache {path}", file=sys.stderr)
    return base_pools, hybrid_pools, hybrid_ranked_ids, learned_ranked_ids, cold_start_ranked_ids


def _write_eval_row_cache(
    args: argparse.Namespace,
    dataset: EvalDataset,
    base_pools: list[CandidatePool],
    hybrid_pools: list[CandidatePool],
    hybrid_ranked_ids: list[list[str]],
    learned_ranked_ids: list[list[str]],
    cold_start_ranked_ids: list[list[str]],
) -> None:
    if not args.row_cache:
        return
    path = Path(args.row_cache)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp_path = path.with_name(path.name + ".tmp")
    header = {
        "type": "task_b_eval_row_cache",
        "schema_version": EVAL_ROW_CACHE_SCHEMA_VERSION,
        "signature": _eval_row_cache_signature(args, dataset),
        "metadata": {
            "examples": len(dataset.test_b),
            "items_count": len(dataset.item_list),
        },
    }
    with temp_path.open("w", encoding="utf-8") as handle:
        handle.write(json.dumps(header, ensure_ascii=True, separators=(",", ":")) + "\n")
        for index, hybrid_pool in enumerate(hybrid_pools):
            row = {
                "base_pool": _pool_to_cache(base_pools[index]) if index < len(base_pools) else {},
                "hybrid_pool": _pool_to_cache(hybrid_pool),
                "hybrid_ranked_ids": hybrid_ranked_ids[index]
                if index < len(hybrid_ranked_ids)
                else [],
            }
            if index < len(learned_ranked_ids):
                row["learned_ranked_ids"] = learned_ranked_ids[index]
            if index < len(cold_start_ranked_ids):
                row["cold_start_ranked_ids"] = cold_start_ranked_ids[index]
            handle.write(json.dumps(row, ensure_ascii=True, separators=(",", ":")) + "\n")
    temp_path.replace(path)
    print(f"Wrote {len(hybrid_pools)} Task B eval rows to cache {path}", file=sys.stderr)


def _eval_row_cache_signature(args: argparse.Namespace, dataset: EvalDataset) -> dict:
    return {
        "processed_dir": str(Path(args.processed_dir)),
        "reviews": str(Path(args.reviews)),
        "items": str(Path(args.items)),
        "collaborative_index": str(args.collaborative_index or ""),
        "candidate_limit": int(args.candidate_limit),
        "rank_depth": int(args.rank_depth),
        "max_examples": int(args.max_examples),
        "sample_strategy": str(args.sample_strategy),
        "shard_count": int(args.shard_count),
        "shard_index": int(args.shard_index),
        "target_mode": str(args.target_mode),
        "context_mode": str(args.context_mode),
        "retriever": str(args.retriever),
        "hybrid_only": bool(args.hybrid_only),
        "candidate_recall_only": bool(args.candidate_recall_only),
        "learned_ranker_artifact": str(args.learned_ranker_artifact or ""),
        "disabled_sources": sorted(dataset.disabled_sources),
    }


def _pool_to_cache(pool: CandidatePool) -> dict:
    item_ids = [item.item_id for item in pool.items]
    item_id_set = set(item_ids)
    return {
        "item_ids": item_ids,
        "sources": {
            item_id: values
            for item_id, values in pool.sources.items()
            if item_id in item_id_set
        },
        "source_scores": {
            item_id: values
            for item_id, values in pool.source_scores.items()
            if item_id in item_id_set
        },
    }


def _pool_from_cache(payload: dict, catalog: CandidateCatalog) -> CandidatePool:
    item_ids = [str(item_id) for item_id in payload.get("item_ids", [])]
    items = [catalog.by_id[item_id] for item_id in item_ids if item_id in catalog.by_id]
    item_id_set = {item.item_id for item in items}
    sources = {
        str(item_id): [str(source) for source in values]
        for item_id, values in (payload.get("sources") or {}).items()
        if str(item_id) in item_id_set
    }
    source_scores = {
        str(item_id): {
            str(source): float(score)
            for source, score in (values or {}).items()
        }
        for item_id, values in (payload.get("source_scores") or {}).items()
        if str(item_id) in item_id_set
    }
    return CandidatePool(items=items, sources=sources, source_scores=source_scores)


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
        metadata.setdefault(
            "catalog_review_count",
            item.metadata.get("review_count") or item.metadata.get("rating_number") or 0,
        )
        metadata["review_count"] = counts[item.item_id]
        metadata["rating_number"] = counts[item.item_id]
        metadata["train_positive_count"] = counts[item.item_id]
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
    context: str,
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
        context=context,
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
    context: str,
    pool: CandidatePool,
    limit: int,
    rank_depth: int = 0,
    item_profile_cache: dict | None = None,
) -> list[RecommendationItem]:
    persona = persona_from_history(history)
    user_profile = build_user_profile(persona=persona, history=history, locale=None)
    candidate_items = pool.items
    if rank_depth > 0:
        candidate_items = candidate_items[: max(rank_depth, 1)]
        limit = min(limit, len(candidate_items))
    ranked = rank_candidates(
        user_profile=user_profile,
        context=context,
        candidate_items=candidate_items,
        limit=limit,
        candidate_sources=pool.sources,
        candidate_source_scores=pool.source_scores,
        item_profile_cache=item_profile_cache,
    )
    return ranked


def _cold_start_rank(
    row: dict,
    context: str,
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
        context=context,
        bm25_retriever=retriever,
        vector_retriever=vector_retriever,
        catalog=catalog,
        disabled_sources=disabled_sources,
        limit=min(candidate_limit, len(item_list)),
        neural_retriever=neural_retriever,
    )
    ranked = rank_candidates(
        user_profile=user_profile,
        context=context,
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
    contexts: list[str],
    positives: list[str],
    ranked_ids: list[list[str]],
    k: int,
    candidate_ids: list[list[str]] | None = None,
    candidate_k: int = 0,
) -> dict[str, dict[str, float]]:
    slice_indices = {
        "all": list(range(len(test_b))),
        "sparse_history_1_2": [],
        "medium_history_3_7": [],
        "warm_history_8_plus": [],
        "cross_domain": [],
        "cold_start": [],
        "context_heavy": [],
        "intent_heavy": [],
    }
    for index, row in enumerate(test_b):
        history = history_map.get(row["user_id"], [])
        context = contexts[index] if index < len(contexts) else ""
        history_len = len(history)
        if history_len <= 2:
            slice_indices["sparse_history_1_2"].append(index)
        elif history_len <= 7:
            slice_indices["medium_history_3_7"].append(index)
        else:
            slice_indices["warm_history_8_plus"].append(index)
        if history_len == 0:
            slice_indices["cold_start"].append(index)
        if _is_cross_domain(row, history):
            slice_indices["cross_domain"].append(index)
        if context:
            slice_indices["context_heavy"].append(index)
            intent_name = _context_intent_name(context)
            if intent_name:
                slice_indices["intent_heavy"].append(index)
                slice_indices.setdefault(f"context_intent_{intent_name}", []).append(index)

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
    contexts: list[str],
    positives: list[str],
    candidate_ids: list[list[str]],
    candidate_k: int,
) -> dict[str, dict[str, float]]:
    slice_indices = {
        "all": list(range(len(test_b))),
        "sparse_history_1_2": [],
        "medium_history_3_7": [],
        "warm_history_8_plus": [],
        "cross_domain": [],
        "cold_start": [],
        "context_heavy": [],
        "intent_heavy": [],
    }
    for index, row in enumerate(test_b):
        history = history_map.get(row["user_id"], [])
        context = contexts[index] if index < len(contexts) else ""
        history_len = len(history)
        if history_len <= 2:
            slice_indices["sparse_history_1_2"].append(index)
        elif history_len <= 7:
            slice_indices["medium_history_3_7"].append(index)
        else:
            slice_indices["warm_history_8_plus"].append(index)
        if history_len == 0:
            slice_indices["cold_start"].append(index)
        if _is_cross_domain(row, history):
            slice_indices["cross_domain"].append(index)
        if context:
            slice_indices["context_heavy"].append(index)
            intent_name = _context_intent_name(context)
            if intent_name:
                slice_indices["intent_heavy"].append(index)
                slice_indices.setdefault(f"context_intent_{intent_name}", []).append(index)
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


def _context_intent_name(context: str) -> str:
    rule = context_intent_rule(terms(context))
    return rule.name if rule is not None else ""


def _task_b_promotion_gate(
    metrics: dict[str, float],
    slices: dict,
    examples: int,
    k: int,
    candidate_k: int,
    ranker_metric_prefix: str = "hybrid_ranker",
) -> dict:
    candidate_key = f"hybrid_candidate_recall@{candidate_k}"
    checks = _all_scope_gate_checks(metrics, k, candidate_key, ranker_metric_prefix)
    slice_availability = {}

    for name in (*TASK_B_REQUIRED_GATE_SLICES, *TASK_B_OPTIONAL_GATE_SLICES):
        values = metrics if name == "all" else slices.get(name, {})
        slice_examples = examples if name == "all" else int(values.get("examples", 0) or 0)
        required = name in TASK_B_REQUIRED_GATE_SLICES
        available = slice_examples > 0
        slice_availability[name] = {
            "required": required,
            "available": available,
            "examples": slice_examples,
        }
        if not available:
            if required:
                checks.append(
                    _gate_check(
                        scope=name,
                        metric=candidate_key,
                        actual=None,
                        threshold=TASK_B_CANDIDATE_RECALL_GATES[name],
                        required=True,
                        reason="required slice unavailable",
                    )
                )
            continue
        checks.append(
            _gate_check(
                scope=name,
                metric=candidate_key,
                actual=values.get(candidate_key),
                threshold=TASK_B_CANDIDATE_RECALL_GATES[name],
                required=True,
            )
        )

    failed_checks = [check for check in checks if check["status"] != "pass"]
    promotion_ready = not failed_checks
    return {
        "decision": "pass" if promotion_ready else "reject",
        "promotion_ready": promotion_ready,
        "candidate_recall_gate_k": candidate_k,
        "ranker_gate_k": k,
        "ranker_metric_prefix": ranker_metric_prefix,
        "required_slices": list(TASK_B_REQUIRED_GATE_SLICES),
        "optional_slices_if_available": list(TASK_B_OPTIONAL_GATE_SLICES),
        "slice_availability": slice_availability,
        "thresholds": {
            "ranking": _ranker_gate_thresholds(ranker_metric_prefix),
            f"candidate_recall@{candidate_k}": TASK_B_CANDIDATE_RECALL_GATES,
            "hybrid_must_beat_filtered_popularity": [f"hit_rate@{k}", f"ndcg@{k}"],
            "hybrid_candidate_recall_must_not_regress_vs_base": True,
        },
        "checks": checks,
        "failed_checks": failed_checks,
        "notes": [
            "Required Task B gates are non-negotiable for all, sparse, and cross-domain slices.",
            "Cold-start, context-heavy, and intent-heavy gates are enforced when those rows exist.",
            "Candidate-recall-only runs cannot pass ranking gates because ranker metrics are absent.",
        ],
    }


def _all_scope_gate_checks(
    metrics: dict[str, float],
    k: int,
    candidate_key: str,
    ranker_metric_prefix: str = "hybrid_ranker",
) -> list[dict]:
    checks = []
    for metric, threshold in _ranker_gate_thresholds(ranker_metric_prefix).items():
        checks.append(
            _gate_check(
                scope="all",
                metric=metric,
                actual=metrics.get(metric),
                threshold=threshold,
                required=True,
            )
        )

    base_key = candidate_key.replace("hybrid_", "base_", 1)
    if base_key in metrics:
        checks.append(
            _gate_check(
                scope="all",
                metric=candidate_key,
                actual=metrics.get(candidate_key),
                threshold=metrics.get(base_key),
                required=True,
                reason=f"must be >= {base_key}",
            )
        )

    for metric_suffix in (f"hit_rate@{k}", f"ndcg@{k}"):
        hybrid_key = f"{ranker_metric_prefix}_{metric_suffix}"
        filtered_key = f"filtered_popularity_{metric_suffix}"
        if filtered_key in metrics:
            checks.append(
                _gate_check(
                    scope="all",
                    metric=hybrid_key,
                    actual=metrics.get(hybrid_key),
                    threshold=metrics.get(filtered_key),
                    required=True,
                    reason=f"must beat {filtered_key}",
                    strict=True,
                )
            )
    return checks


def _ranker_gate_thresholds(ranker_metric_prefix: str) -> dict[str, float]:
    thresholds = {}
    for metric, threshold in TASK_B_RANKING_GATES.items():
        if metric.startswith("hybrid_ranker_"):
            metric = metric.replace("hybrid_ranker_", f"{ranker_metric_prefix}_", 1)
        thresholds[metric] = threshold
    return thresholds


def _gate_check(
    scope: str,
    metric: str,
    actual: float | int | None,
    threshold: float | int | None,
    required: bool,
    reason: str = "",
    strict: bool = False,
) -> dict:
    if actual is None or threshold is None:
        status = "fail"
        reason = reason or "metric unavailable"
    elif strict:
        status = "pass" if actual > threshold else "fail"
    else:
        status = "pass" if actual >= threshold else "fail"
    return {
        "scope": scope,
        "metric": metric,
        "actual": actual,
        "operator": ">" if strict else ">=",
        "threshold": threshold,
        "required": required,
        "status": status,
        "reason": reason,
    }


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
