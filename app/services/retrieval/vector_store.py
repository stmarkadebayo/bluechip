from __future__ import annotations

from collections import defaultdict

from app.models.schemas import Item
from app.services.retrieval.embeddings import cosine_similarity, embedding_text, hashed_embedding, terms


class LocalKeywordRetriever:
    """Small local retriever used until embeddings/vector storage are wired in."""

    def __init__(self, items: list[Item]) -> None:
        self.items = items

    def search(self, query_terms: list[str], limit: int = 20) -> list[Item]:
        scored = []
        query = set(query_terms)
        for item in self.items:
            text = " ".join(
                [item.name, item.category, item.summary, " ".join(map(str, item.metadata.values()))]
            ).lower()
            score = sum(1 for term in query if term.lower() in text)
            scored.append((score, item))
        scored.sort(key=lambda row: row[0], reverse=True)
        return [item for score, item in scored[:limit] if score > 0] or self.items[:limit]


class LocalVectorRetriever:
    """Deterministic local vector retriever backed by hashing embeddings."""

    def __init__(self, items: list[Item]) -> None:
        self.items = items
        self.embeddings = {}
        self.token_index: dict[str, set[int]] = defaultdict(set)
        for index, item in enumerate(items):
            text = embedding_text(item.name, item.category, item.summary, item.metadata)
            self.embeddings[item.item_id] = hashed_embedding(text)
            for token in set(terms(text)):
                self.token_index[token].add(index)

    def search(self, query: str, limit: int = 20) -> list[Item]:
        return [item for item, _ in self.search_with_scores(query, limit=limit)]

    def search_with_scores(self, query: str, limit: int = 20) -> list[tuple[Item, float]]:
        query_embedding = hashed_embedding(query)
        candidate_indices = self._candidate_indices(query, limit)
        scored = [
            (cosine_similarity(query_embedding, self.embeddings.get(item.item_id, [])), item)
            for item in (self.items[index] for index in candidate_indices)
        ]
        scored.sort(key=lambda row: row[0], reverse=True)
        positive = [(item, round(max(score, 0.0), 4)) for score, item in scored if score > 0]
        return positive[:limit] or [(item, 0.0) for _, item in scored[:limit]]

    def _candidate_indices(self, query: str, limit: int) -> list[int]:
        query_terms = terms(query)
        candidates: set[int] = set()
        max_candidates = min(max(limit * 20, 5_000), 10_000)
        terms_by_rarity = sorted(
            set(query_terms),
            key=lambda token: len(self.token_index.get(token, set())),
        )
        for token in terms_by_rarity:
            postings = self.token_index.get(token, set())
            if not postings:
                continue
            candidates.update(postings)
            if len(candidates) >= max_candidates:
                break
        if not candidates:
            return list(range(len(self.items)))
        if len(candidates) < limit:
            candidates.update(range(min(len(self.items), limit * 4)))
        return sorted(candidates)[:max_candidates]
