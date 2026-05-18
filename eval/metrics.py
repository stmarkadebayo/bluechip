from __future__ import annotations

import math


def mae(actual: list[float], predicted: list[float]) -> float:
    if not actual:
        return 0.0
    return sum(abs(a - p) for a, p in zip(actual, predicted)) / len(actual)


def rmse(actual: list[float], predicted: list[float]) -> float:
    if not actual:
        return 0.0
    return math.sqrt(sum((a - p) ** 2 for a, p in zip(actual, predicted)) / len(actual))


def hit_rate_at_k(ranked_ids: list[list[str]], positives: list[str], k: int) -> float:
    if not positives:
        return 0.0
    hits = 0
    for ranking, positive in zip(ranked_ids, positives):
        if positive in ranking[:k]:
            hits += 1
    return hits / len(positives)


def recall_at_k(ranked_ids: list[list[str]], positives: list[str], k: int) -> float:
    return hit_rate_at_k(ranked_ids, positives, k)


def ndcg_at_k(ranked_ids: list[list[str]], positives: list[str], k: int) -> float:
    if not positives:
        return 0.0
    total = 0.0
    for ranking, positive in zip(ranked_ids, positives):
        try:
            rank = ranking[:k].index(positive)
        except ValueError:
            continue
        total += 1 / math.log2(rank + 2)
    return total / len(positives)


def rounded(value: float) -> float:
    return round(value, 4)

