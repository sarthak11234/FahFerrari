"""
Probability calibration diagnostics and reliability binning (FF27-10).
"""

from __future__ import annotations

import math
from typing import Any, Dict, List, Sequence


def reliability_diagram_bins(
    y_true: Sequence[int], y_prob: Sequence[float], n_bins: int = 5
) -> List[Dict[str, Any]]:
    """
    Group predicted probabilities into `n_bins` equal-width bins on [0, 1]
    and compute empirical observed frequency vs mean predicted probability per bin.
    """
    if not y_true or len(y_true) != len(y_prob):
        raise ValueError("y_true and y_prob must be non-empty and of equal length")

    bins: List[List[int]] = [[] for _ in range(n_bins)]
    prob_bins: List[List[float]] = [[] for _ in range(n_bins)]

    for t, p in zip(y_true, y_prob):
        p_clamped = min(1.0, max(0.0, float(p)))
        idx = min(n_bins - 1, int(p_clamped * n_bins))
        bins[idx].append(int(t))
        prob_bins[idx].append(p_clamped)

    summary: List[Dict[str, Any]] = []
    for i in range(n_bins):
        low = round(i / n_bins, 2)
        high = round((i + 1) / n_bins, 2)
        count = len(bins[i])
        if count > 0:
            mean_pred = sum(prob_bins[i]) / count
            obs_freq = sum(bins[i]) / count
            gap = abs(mean_pred - obs_freq)
        else:
            mean_pred = 0.0
            obs_freq = 0.0
            gap = 0.0
        summary.append(
            {
                "bin_index": i,
                "bin_range": [low, high],
                "count": count,
                "mean_predicted_prob": round(mean_pred, 4),
                "observed_frequency": round(obs_freq, 4),
                "calibration_gap": round(gap, 4),
            }
        )
    return summary


def expected_calibration_error(
    y_true: Sequence[int], y_prob: Sequence[float], n_bins: int = 5
) -> float:
    """Compute sample-weighted Expected Calibration Error (ECE) across `n_bins`."""
    n = len(y_true)
    if n == 0:
        return 0.0
    bins = reliability_diagram_bins(y_true, y_prob, n_bins=n_bins)
    ece = sum((b["count"] / float(n)) * b["calibration_gap"] for b in bins)
    return round(ece, 4)


def gaussian_cdf(x: float, mean: float, std: float) -> float:
    """Evaluate normal CDF P(X <= x) for X ~ N(mean, std^2)."""
    if std <= 1e-6:
        return 1.0 if x >= mean else 0.0
    z = (x - mean) / (std * math.sqrt(2.0))
    return 0.5 * (1.0 + math.erf(z))

