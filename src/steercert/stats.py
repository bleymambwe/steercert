"""Small statistics helpers with no SciPy dependency."""

from __future__ import annotations

from collections.abc import Callable

import numpy as np

__all__ = ["bootstrap_interval", "rankdata", "spearman"]


def rankdata(values: np.ndarray) -> np.ndarray:
    """1-based ranks with ties assigned their average rank."""
    values = np.asarray(values, dtype=np.float64).ravel()
    order = np.argsort(values, kind="mergesort")
    ordered = values[order]
    ranks = np.empty(len(values), dtype=np.float64)
    start = 0
    while start < len(ordered):
        stop = start
        while stop + 1 < len(ordered) and ordered[stop + 1] == ordered[start]:
            stop += 1
        ranks[order[start : stop + 1]] = 0.5 * (start + stop) + 1.0
        start = stop + 1
    return ranks


def spearman(left: np.ndarray, right: np.ndarray) -> float:
    """Tie-corrected Spearman rank correlation; ``nan`` for fewer than 3 pairs."""
    left = np.asarray(left, dtype=np.float64).ravel()
    right = np.asarray(right, dtype=np.float64).ravel()
    if left.shape != right.shape:
        raise ValueError("inputs must have equal length")
    finite = np.isfinite(left) & np.isfinite(right)
    left, right = left[finite], right[finite]
    if len(left) < 3:
        return float("nan")
    a = rankdata(left) - (len(left) + 1) / 2.0
    b = rankdata(right) - (len(right) + 1) / 2.0
    denom = float(np.sqrt((a * a).sum() * (b * b).sum()))
    return float((a * b).sum() / denom) if denom > 0 else float("nan")


def bootstrap_interval(
    statistic: Callable[[np.ndarray], float],
    n: int,
    repeats: int = 1000,
    level: float = 0.95,
    seed: int = 0,
) -> tuple[float, float]:
    """Percentile bootstrap interval for ``statistic(indices)`` over ``n`` examples."""
    rng = np.random.default_rng(seed)
    draws = []
    for _ in range(repeats):
        value = statistic(rng.integers(0, n, size=n))
        if np.isfinite(value):
            draws.append(value)
    if not draws:
        return float("nan"), float("nan")
    tail = (1.0 - level) / 2.0
    return float(np.quantile(draws, tail)), float(np.quantile(draws, 1.0 - tail))
