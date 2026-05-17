from __future__ import annotations

from app.models.schemas import Item


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
