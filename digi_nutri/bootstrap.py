"""Plate-grouped percentile bootstrap."""

from __future__ import annotations

from collections import defaultdict

import numpy as np

from digi_nutri.metrics import METRIC_FNS
from digi_nutri.protocol import BOOTSTRAP_CI, BOOTSTRAP_SAMPLES, BOOTSTRAP_SEED


def _group_index(groups: list[str]) -> list[list[int]]:
    ordered: list[str] = []
    members: dict[str, list[int]] = defaultdict(list)
    for index, group in enumerate(groups):
        if group not in members:
            ordered.append(group)
        members[group].append(index)
    return [members[group] for group in ordered]


def contrast_ci(
    pred_method: np.ndarray,
    pred_base: np.ndarray,
    target: np.ndarray,
    groups: list[str],
    metric: str,
    n_samples: int = BOOTSTRAP_SAMPLES,
    seed: int = BOOTSTRAP_SEED,
) -> dict:
    """95% CI of (method - baseline) on one nutrient column.

    Groups are resampled with replacement. Plates inside a sampled group stay
    together, and a group drawn twice is included twice. The interval is the
    2.5 and 97.5 percentiles. A tie or a bound of 0 is not a pass.
    """
    fn = METRIC_FNS[metric]
    pred_method = np.asarray(pred_method, dtype=np.float64).reshape(-1)
    pred_base = np.asarray(pred_base, dtype=np.float64).reshape(-1)
    target = np.asarray(target, dtype=np.float64).reshape(-1)
    blocks = _group_index(list(groups))
    if len(blocks) == 0:
        raise ValueError("bootstrap received no plate groups")
    rng = np.random.default_rng(seed)
    diffs = np.empty(n_samples, dtype=np.float64)
    filled = 0
    attempts = 0
    max_attempts = n_samples * 5
    while filled < n_samples:
        attempts += 1
        if attempts > max_attempts:
            raise RuntimeError(f"bootstrap for {metric} produced too many undefined resamples")
        chosen = rng.integers(0, len(blocks), size=len(blocks))
        rows = np.concatenate([np.asarray(blocks[index], dtype=np.int64) for index in chosen])
        method_value = fn(pred_method[rows], target[rows])
        base_value = fn(pred_base[rows], target[rows])
        if not np.isfinite(method_value) or not np.isfinite(base_value):
            continue
        diffs[filled] = method_value - base_value
        filled += 1
    lo, hi = np.percentile(diffs, list(BOOTSTRAP_CI))
    point_method = fn(pred_method, target)
    point_base = fn(pred_base, target)
    return {
        "lo": float(lo),
        "hi": float(hi),
        "mean_diff": float(np.mean(diffs)),
        "point_diff": float(point_method - point_base),
        "method": float(point_method),
        "baseline": float(point_base),
        "n_groups": len(blocks),
        "n_samples": n_samples,
        "seed": seed,
    }
