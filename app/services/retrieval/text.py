from __future__ import annotations

import math
import re
from collections import Counter

from app.models.schemas import Item


class BM25Retriever:
    """Dependency-free BM25 retriever for local candidate generation."""

    def __init__(self, documents: list[tuple[str, str, Item]], k1: float = 1.5, b: float = 0.75):
        self.documents = documents
        self.k1 = k1
        self.b = b
        self.term_counts = [Counter(_terms(text)) for _, text, _ in documents]
        self.postings: dict[str, set[int]] = {}
        self.lengths = [sum(counts.values()) for counts in self.term_counts]
        self.avgdl = sum(self.lengths) / len(self.lengths) if self.lengths else 1.0
        doc_freq = Counter()
        for index, counts in enumerate(self.term_counts):
            doc_freq.update(counts.keys())
            for term in counts:
                self.postings.setdefault(term, set()).add(index)
        n_docs = max(len(documents), 1)
        self.idf = {
            term: math.log(1 + (n_docs - freq + 0.5) / (freq + 0.5))
            for term, freq in doc_freq.items()
        }

    @classmethod
    def from_items(cls, items: list[Item]) -> "BM25Retriever":
        documents = []
        for item in items:
            metadata_text = " ".join(str(value) for value in item.metadata.values())
            text = " ".join([item.name, item.category, item.summary, metadata_text])
            documents.append((item.item_id, text, item))
        return cls(documents)

    def search(self, query: str, limit: int = 20) -> list[Item]:
        return [item for item, _ in self.search_with_scores(query, limit=limit)]

    def search_with_scores(self, query: str, limit: int = 20) -> list[tuple[Item, float]]:
        query_terms = _terms(query)
        if not query_terms:
            return [(item, 0.0) for _, _, item in self.documents[:limit]]

        candidate_indices = self._candidate_indices(query_terms, limit)
        scored = []
        for index in candidate_indices:
            _, _, item = self.documents[index]
            score = self._score(query_terms, index)
            scored.append((score, item))
        scored.sort(key=lambda row: row[0], reverse=True)

        max_score = max((score for score, _ in scored), default=0.0)
        positive = [
            (item, _normalize_score(score, max_score)) for score, item in scored if score > 0
        ]
        if positive:
            return positive[:limit]
        return [(item, 0.0) for _, item in scored[:limit]]

    def _candidate_indices(self, query_terms: list[str], limit: int) -> list[int]:
        candidates: set[int] = set()
        max_candidates = min(max(limit * 4, 2_000), 3_000)
        terms_by_rarity = sorted(
            set(query_terms),
            key=lambda term: len(self.postings.get(term, set())),
        )
        for term in terms_by_rarity:
            postings = self.postings.get(term, set())
            if not postings:
                continue
            candidates.update(postings)
            if len(candidates) >= max_candidates:
                break
        if not candidates:
            return list(range(len(self.documents)))
        return sorted(candidates)[:max_candidates]

    def _score(self, query_terms: list[str], doc_index: int) -> float:
        counts = self.term_counts[doc_index]
        length = self.lengths[doc_index] or 1
        score = 0.0
        for term in query_terms:
            freq = counts.get(term, 0)
            if not freq:
                continue
            idf = self.idf.get(term, 0.0)
            numerator = freq * (self.k1 + 1)
            denominator = freq + self.k1 * (1 - self.b + self.b * length / self.avgdl)
            score += idf * numerator / denominator
        return score


def _terms(text: str) -> list[str]:
    return re.findall(r"[a-zA-Z][a-zA-Z0-9_-]{2,}", text.lower())


def _normalize_score(score: float, max_score: float) -> float:
    if score <= 0 or max_score <= 0:
        return 0.0
    return round(min(score / max_score, 1.0), 4)
