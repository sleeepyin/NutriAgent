"""Finite-sample split-conformal quantile: k-th order statistic, no interpolation."""
from __future__ import annotations

import math

import numpy as np


def finite_sample_quantile(scores, alpha: float = 0.10) -> float:
    """
    q = R_{(k)} with k = ceil((n+1)(1-alpha)).
    If k > n, return +inf (conservative convention).
    Uses np.partition; never np.quantile linear interpolation.
    """
    s = np.asarray(scores, dtype=float).reshape(-1)
    s = s[np.isfinite(s)]
    n = int(s.size)
    if n == 0:
        return float("inf")
    k = int(math.ceil((n + 1) * (1.0 - alpha)))
    if k > n:
        return float("inf")
    if k <= 0:
        return float(np.min(s))
    return float(np.partition(s, k - 1)[k - 1])


def conformal_interval(center, scale, q: float, lower_clip: float = 0.0):
    """[max(lower_clip, m - q h), m + q h]."""
    m = np.asarray(center, dtype=float)
    h = np.asarray(scale, dtype=float)
    h = np.maximum(h, 1e-8)
    if not np.isfinite(q):
        lo = np.full_like(m, lower_clip, dtype=float)
        hi = np.full_like(m, np.inf, dtype=float)
        return lo, hi
    lo = np.maximum(lower_clip, m - q * h)
    hi = m + q * h
    return lo, hi


def residual_scores(y, center, scale) -> np.ndarray:
    y = np.asarray(y, dtype=float)
    m = np.asarray(center, dtype=float)
    h = np.maximum(np.asarray(scale, dtype=float), 1e-8)
    return np.abs(y - m) / h
