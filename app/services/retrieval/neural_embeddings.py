from __future__ import annotations

import functools
import math
import os
import threading
from typing import Any

from app.services.retrieval.embeddings import embedding_text, hashed_embedding

_MODEL_NAME = "all-MiniLM-L6-v2"
_FALLBACK_MODEL_NAME = "all-mpnet-base-v2"
_MODEL = None
_MODEL_LOCK = threading.Lock()
_CACHE: dict[str, list[float]] = {}
_CACHE_LOCK = threading.Lock()
_CACHE_MAX_SIZE = 50_000


def _get_model() -> Any | None:
    global _MODEL
    if _MODEL is not None:
        return _MODEL
    with _MODEL_LOCK:
        if _MODEL is not None:
            return _MODEL
        try:
            from sentence_transformers import SentenceTransformer

            allow_download = os.getenv("BLUECHIP_ALLOW_MODEL_DOWNLOAD", "").lower() in {
                "1",
                "true",
                "yes",
            }
            for name in (_MODEL_NAME, _FALLBACK_MODEL_NAME):
                try:
                    _MODEL = SentenceTransformer(name, local_files_only=not allow_download)
                    return _MODEL
                except Exception:
                    continue
        except ImportError:
            pass
        return None


def _model_dimensions() -> int:
    model = _get_model()
    if model is None:
        return 64
    try:
        return model.get_sentence_embedding_dimension() or 384
    except Exception:
        return 384


def _normalize(vector: list[float]) -> list[float]:
    norm = math.sqrt(sum(value * value for value in vector))
    if not norm:
        return vector
    return [round(value / norm, 6) for value in vector]


def neural_available() -> bool:
    return _get_model() is not None


def encode_text(text: str) -> list[float]:
    if not text.strip():
        return [0.0] * _model_dimensions()

    model = _get_model()
    if model is None:
        return hashed_embedding(text)

    cache_key = text
    with _CACHE_LOCK:
        if cache_key in _CACHE:
            return _CACHE[cache_key]

    try:
        embedding = model.encode(text, normalize_embeddings=True).tolist()
    except Exception:
        return hashed_embedding(text)

    result = [round(float(value), 6) for value in embedding]
    with _CACHE_LOCK:
        if len(_CACHE) < _CACHE_MAX_SIZE:
            _CACHE[cache_key] = result
    return result


def encode_batch(texts: list[str]) -> list[list[float]]:
    model = _get_model()
    if model is None:
        return [hashed_embedding(text) for text in texts]

    if not texts:
        return []

    try:
        embeddings = model.encode(
            texts,
            normalize_embeddings=True,
            show_progress_bar=False,
            batch_size=min(64, len(texts)),
        ).tolist()
    except Exception:
        return [hashed_embedding(text) for text in texts]

    return [[round(float(value), 6) for value in embedding] for embedding in embeddings]


@functools.lru_cache(maxsize=4096)
def neural_embedding_text(*parts: object) -> str:
    return embedding_text(*parts)
