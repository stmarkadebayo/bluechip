from __future__ import annotations

import os
import json
from collections import defaultdict
from pathlib import Path
from typing import ClassVar

from app.models.schemas import Item
from app.services.retrieval.embeddings import (
    cosine_similarity,
    embedding_text,
    hashed_embedding,
    neural_available,
    terms,
)


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


class FAISSVectorStore:
    """Vector store backed by FAISS with neural embeddings."""

    _FAISS_AVAILABLE: ClassVar[bool | None] = None

    def __init__(self, items: list[Item]) -> None:
        self.items = items
        self._item_map: dict[int, Item] = {}
        self._item_ids: list[str] = []
        self.index = None
        self._dim: int = 0
        self._built = False
        if FAISSVectorStore._FAISS_AVAILABLE is None:
            FAISSVectorStore._FAISS_AVAILABLE = _check_faiss()
        if not FAISSVectorStore._FAISS_AVAILABLE or not neural_available():
            return
        self._build(items)

    def _build(self, items: list[Item]) -> None:
        if not items:
            return
        try:
            import numpy as np
        except ImportError:
            return

        from app.services.retrieval.neural_embeddings import encode_batch

        texts = [
            embedding_text(item.name, item.category, item.summary, item.metadata)
            for item in items
        ]
        embeddings = encode_batch(texts)
        if not embeddings:
            return

        self._dim = len(embeddings[0])
        vectors = np.array(embeddings, dtype=np.float32)

        try:
            import faiss

            self.index = faiss.IndexFlatIP(self._dim)
            self.index.add(vectors)
        except ImportError:
            return

        self._item_ids = [item.item_id for item in items]
        for idx, item in enumerate(items):
            self._item_map[idx] = item
        self._built = True

    def search(self, query: str, limit: int = 20) -> list[Item]:
        return [item for item, _ in self.search_with_scores(query, limit=limit)]

    def search_with_scores(self, query: str, limit: int = 20) -> list[tuple[Item, float]]:
        if not self._built or self.index is None:
            retriever = LocalVectorRetriever(self.items)
            return retriever.search_with_scores(query, limit=limit)

        try:
            import numpy as np
        except ImportError:
            retriever = LocalVectorRetriever(self.items)
            return retriever.search_with_scores(query, limit=limit)

        from app.services.retrieval.neural_embeddings import encode_text

        query_text = embedding_text(query)
        query_embedding = encode_text(query_text)
        if len(query_embedding) != self._dim:
            retriever = LocalVectorRetriever(self.items)
            return retriever.search_with_scores(query, limit=limit)
        query_vec = np.array([query_embedding], dtype=np.float32)

        k = min(self._search_k(limit), self.index.ntotal)
        if k == 0:
            return [(item, 0.0) for item in self.items[:limit]]

        scores, indices = self.index.search(query_vec, k)
        results: list[tuple[Item, float]] = []

        for score, idx in zip(scores[0], indices[0]):
            if idx == -1:
                continue
            item = self._item_map.get(int(idx))
            if item is not None:
                results.append((item, round(float(max(score, 0.0)), 4)))

        seen: set[str] = set()
        deduplicated: list[tuple[Item, float]] = []
        for item, score in results:
            if item.item_id not in seen:
                deduplicated.append((item, score))
                seen.add(item.item_id)
        return deduplicated[:limit]

    def serialize(self, path: str, ids_path: str | None = None) -> None:
        if not self._built or self.index is None:
            return
        try:
            import faiss

            faiss.write_index(self.index, path)
            _write_item_ids(ids_path or companion_ids_path(path), self._item_ids)
        except Exception:
            pass

    @classmethod
    def deserialize(
        cls,
        path: str,
        items: list[Item],
        ids_path: str | None = None,
    ) -> "FAISSVectorStore":
        return cls.deserialize_with_ids(
            path=path,
            items_by_id={item.item_id: item for item in items},
            ids_path=ids_path,
            fallback_items=items,
        )

    @classmethod
    def deserialize_with_ids(
        cls,
        path: str,
        items_by_id: dict[str, Item],
        ids_path: str | None = None,
        fallback_items: list[Item] | None = None,
    ) -> "FAISSVectorStore":
        store = cls.__new__(cls)
        store.items = list(items_by_id.values())
        store._item_map = {}
        store._item_ids = []
        store.index = None
        store._dim = 0
        store._built = False

        if not os.path.exists(path):
            return store

        try:
            import faiss

            store.index = faiss.read_index(path)
            store._dim = store.index.d
            store._built = True
            store._item_ids = _read_item_ids(ids_path or companion_ids_path(path))
            if store._item_ids:
                for idx, item_id in enumerate(store._item_ids):
                    item = items_by_id.get(item_id)
                    if item is not None:
                        store._item_map[idx] = item
            else:
                for idx, item in enumerate(fallback_items or []):
                    store._item_map[idx] = item
                store._item_ids = [item.item_id for item in fallback_items or []]
        except Exception:
            pass

        return store

    def bind_items(self, items: list[Item]) -> "FAISSVectorStore":
        store = self.__class__.__new__(self.__class__)
        store.items = items
        store._item_map = {}
        store._item_ids = list(self._item_ids)
        store.index = self.index
        store._dim = self._dim
        store._built = self._built

        items_by_id = {item.item_id: item for item in items}
        if store._item_ids:
            for idx, item_id in enumerate(store._item_ids):
                item = items_by_id.get(item_id)
                if item is not None:
                    store._item_map[idx] = item
        else:
            for idx, item in enumerate(items):
                store._item_map[idx] = item
        return store

    def _search_k(self, limit: int) -> int:
        if not self.index:
            return limit
        if len(self._item_map) >= self.index.ntotal:
            return limit
        mapped = max(len(self._item_map), 1)
        coverage = mapped / max(self.index.ntotal, 1)
        sparse_multiplier = min(max(int(1 / coverage), 20), 500)
        return max(limit, limit * sparse_multiplier)


def _check_faiss() -> bool:
    try:
        import faiss  # noqa: F401

        return True
    except ImportError:
        return False


def companion_ids_path(index_path: str | os.PathLike[str]) -> str:
    path = Path(index_path)
    return str(path.with_name(f"{path.stem}_ids.json"))


def _read_item_ids(path: str) -> list[str]:
    try:
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return []
    if isinstance(payload, list):
        return [str(item_id) for item_id in payload]
    if isinstance(payload, dict) and isinstance(payload.get("item_ids"), list):
        return [str(item_id) for item_id in payload["item_ids"]]
    return []


def _write_item_ids(path: str, item_ids: list[str]) -> None:
    Path(path).write_text(json.dumps(item_ids), encoding="utf-8")


def create_retriever(
    items: list[Item],
    method: str = "neural",
) -> LocalVectorRetriever | FAISSVectorStore:
    if method == "neural" and neural_available() and _check_faiss():
        return FAISSVectorStore(items)
    return LocalVectorRetriever(items)
