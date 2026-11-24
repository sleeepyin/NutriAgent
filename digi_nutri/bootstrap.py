"""Plate-grouped percentile bootstrap of a per-dish difference."""

from __future__ import annotations

import numpy as np


def plate_grouped_ci(
    delta: np.ndarray,
    groups: np.ndarray,
    n_boot: int = 2000,
    seed: int = 0,
    alpha: float = 0.05,
) -> dict:
    """95% CI of the mean of `delta`, resampling physical plates.

    A plate sampled k times contributes its dishes k times. The point estimate
    is the full-sample mean. `stable_below_zero` is true only when the upper
    endpoint is strictly below 0.
    """

    values = np.asarray(delta, dtype=np.float64)
    labels = np.asarray(groups)
    if values.ndim != 1:
        raise ValueError("delta must be a 1-D per-dish vector")
    if labels.shape[0] != values.shape[0]:
        raise ValueError("groups and delta length differ")
    if values.shape[0] == 0:
        raise ValueError("empty delta")
    uniq = np.unique(labels)
    buckets = [np.flatnonzero(labels == group) for group in uniq]
    sums = np.array([values[idx].sum() for idx in buckets], dtype=np.float64)
    counts = np.array([idx.size for idx in buckets], dtype=np.float64)
    rng = np.random.default_rng(seed)
    draws = rng.integers(0, uniq.shape[0], size=(n_boot, uniq.shape[0]))
    stats = sums[draws].sum(axis=1) / counts[draws].sum(axis=1)
    low, high = np.quantile(stats, [alpha / 2.0, 1.0 - alpha / 2.0], method="linear")
    return {
        "mean": float(values.mean()),
        "ci_low": float(low),
        "ci_high": float(high),
        "stable_below_zero": bool(high < 0.0),
        "n_groups": int(uniq.shape[0]),
        "n_dishes": int(values.shape[0]),
        "n_boot": int(n_boot),
    }
