"""Depth maps: invalid holes filled, then scaled to [0, 1]."""

from __future__ import annotations

import numpy as np


def fill_invalid_and_normalize(depth: np.ndarray, fill: str = "median") -> np.ndarray:
    """Normalize one depth map to [0, 1].

    Non-finite values and values <= 0 are holes. Holes are filled with the
    mean or median of the remaining pixels, then the filled map is min-max
    scaled per image. A map with no usable pixels, or with no contrast, is
    returned as zeros.
    """

    if fill not in {"median", "mean"}:
        raise ValueError("depth fill must be 'median' or 'mean'")
    values = np.array(depth, dtype=np.float32, copy=True)
    valid = np.isfinite(values) & (values > 0)
    if not np.any(valid):
        return np.zeros_like(values, dtype=np.float32)
    statistic = np.median(values[valid]) if fill == "median" else np.mean(values[valid])
    values[~valid] = np.float32(statistic)
    low = float(values.min())
    high = float(values.max())
    if not np.isfinite(low) or not np.isfinite(high) or high <= low:
        return np.zeros_like(values, dtype=np.float32)
    scaled = (values - np.float32(low)) / np.float32(high - low)
    return np.clip(scaled, 0.0, 1.0).astype(np.float32)
