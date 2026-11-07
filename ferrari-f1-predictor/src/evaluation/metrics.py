"""
Evaluation metrics for continuous, probabilistic, and ranking targets (FF27-10).
Implements metrics specified in Section 7 of docs/2027-architecture.md.
"""

from __future__ import annotations

import math
from typing import Dict, List, Sequence


def mae(y_true: Sequence[float], y_pred: Sequence[float]) -> float:
    """Mean Absolute Error."""
    if not y_true or len(y_true) != len(y_pred):
        raise ValueError("y_true and y_pred must be non-empty and of equal length")
    return sum(abs(float(t) - float(p)) for t, p in zip(y_true, y_pred)) / len(y_true)


def rmse(y_true: Sequence[float], y_pred: Sequence[float]) -> float:
    """Root Mean Squared Error."""
    if not y_true or len(y_true) != len(y_pred):
        raise ValueError("y_true and y_pred must be non-empty and of equal length")
    mse = sum((float(t) - float(p)) ** 2 for t, p in zip(y_true, y_pred)) / len(y_true)
    return math.sqrt(mse)


def brier_score(y_true: Sequence[int], y_prob: Sequence[float]) -> float:
    """Brier score for binary probabilistic predictions (lower is better)."""
    if not y_true or len(y_true) != len(y_prob):
        raise ValueError("y_true and y_prob must be non-empty and of equal length")
    return sum((float(t) - min(1.0, max(0.0, float(p)))) ** 2 for t, p in zip(y_true, y_prob)) / len(
        y_true
    )


def log_loss(y_true: Sequence[int], y_prob: Sequence[float], eps: float = 1e-6) -> float:
    """Binary cross-entropy / log loss with probability clipping."""
    if not y_true or len(y_true) != len(y_prob):
        raise ValueError("y_true and y_prob must be non-empty and of equal length")
    total = 0.0
    for t, p in zip(y_true, y_prob):
        p_clip = min(1.0 - eps, max(eps, float(p)))
        if int(t) == 1:
            total -= math.log(p_clip)
        else:
            total -= math.log(1.0 - p_clip)
    return total / len(y_true)


def _rank_vector(values: Sequence[float]) -> List[float]:
    """Assign average ranks (1-based, ascending) to a sequence of values."""
    n = len(values)
    indexed = sorted(enumerate(values), key=lambda x: float(x[1]))
    ranks = [0.0] * n
    i = 0
    while i < n:
        j = i
        while j + 1 < n and float(indexed[j + 1][1]) == float(indexed[i][1]):
            j += 1
        avg_rank = (i + j + 2) / 2.0
        for k in range(i, j + 1):
            ranks[indexed[k][0]] = avg_rank
        i = j + 1
    return ranks


def spearman_rank_correlation(
    true_positions: Sequence[float], predicted_positions: Sequence[float]
) -> float:
    """Compute Spearman rank correlation between true finishing order and predicted order."""
    if len(true_positions) < 2 or len(true_positions) != len(predicted_positions):
        return 0.0
    r_true = _rank_vector(true_positions)
    r_pred = _rank_vector(predicted_positions)
    n = len(r_true)
    mean_t = sum(r_true) / n
    mean_p = sum(r_pred) / n
    cov = sum((rt - mean_t) * (rp - mean_p) for rt, rp in zip(r_true, r_pred))
    var_t = sum((rt - mean_t) ** 2 for rt in r_true)
    var_p = sum((rp - mean_p) ** 2 for rp in r_pred)
    denom = math.sqrt(var_t * var_p)
    return cov / denom if denom > 1e-12 else 0.0


def top_k_overlap_accuracy(
    true_positions: Sequence[float], predicted_positions: Sequence[float], k: int = 3
) -> float:
    """Fraction of actual top-k finishers that are also in the predicted top-k."""
    if not true_positions or k <= 0:
        return 0.0
    n = len(true_positions)
    k_eff = min(k, n)
    true_top_k = {
        idx for idx, _ in sorted(enumerate(true_positions), key=lambda x: float(x[1]))[:k_eff]
    }
    pred_top_k = {
        idx for idx, _ in sorted(enumerate(predicted_positions), key=lambda x: float(x[1]))[:k_eff]
    }
    return len(true_top_k.intersection(pred_top_k)) / float(k_eff)

