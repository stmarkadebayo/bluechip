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
        self.lengths = [sum(counts.values()) for counts in self.term_counts]
        self.avgdl = sum(self.lengths) / len(self.lengths) if self.lengths else 1.0
        doc_freq = Counter()
        for counts in self.term_counts:
            doc_freq.update(counts.keys())
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
        query_terms = _terms(query)
        if not query_terms:
            return [item for _, _, item in self.documents[:limit]]

        scored = []
        for index, (_, _, item) in enumerate(self.documents):
            score = self._score(query_terms, index)
            scored.append((score, item))
        scored.sort(key=lambda row: row[0], reverse=True)

        positive = [item for score, item in scored if score > 0]
        if positive:
            return positive[:limit]
        return [item for _, item in scored[:limit]]

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
