from __future__ import annotations

import hashlib
import math
import re
from collections import Counter


TOKEN_RE = re.compile(r"[a-zA-Z][a-zA-Z0-9_-]{2,}")


def terms(text: str) -> list[str]:
    return TOKEN_RE.findall(text.lower())


def hashed_embedding(text: str, dimensions: int = 64) -> list[float]:
    """Return a deterministic normalized hashing-vector embedding.

    This is intentionally dependency-free for the hackathon submission. It gives
    us an embedding-backed retrieval and ranking signal that can be swapped for
    a model embedding provider without changing downstream contracts.
    """

    vector = [0.0] * dimensions
    counts = Counter(terms(text))
    for token, count in counts.items():
        digest = hashlib.sha256(token.encode("utf-8")).digest()
        index = int.from_bytes(digest[:4], "big") % dimensions
        sign = 1.0 if digest[4] % 2 == 0 else -1.0
        vector[index] += sign * (1.0 + math.log(count))
    return normalize(vector)


def cosine_similarity(left: list[float], right: list[float]) -> float:
    if not left or not right or len(left) != len(right):
        return 0.0
    return sum(a * b for a, b in zip(left, right))


def normalize(vector: list[float]) -> list[float]:
    norm = math.sqrt(sum(value * value for value in vector))
    if not norm:
        return vector
    return [round(value / norm, 6) for value in vector]


def embedding_text(*parts: object) -> str:
    flattened = []
    for part in parts:
        if part is None:
            continue
        if isinstance(part, dict):
            flattened.extend(str(value) for value in part.values())
        elif isinstance(part, (list, tuple, set)):
            flattened.extend(str(value) for value in part)
        else:
            flattened.append(str(part))
    return " ".join(flattened)
