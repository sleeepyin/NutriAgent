"""Five-output joint max-score conformalization."""
from __future__ import annotations

from typing import Any

import numpy as np

from .finite_sample import conformal_interval, finite_sample_quantile


def joint_max_score(y: np.ndarray, center: np.ndarray, scale: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    y = np.asarray(y, dtype=float)
    m = np.asarray(center, dtype=float)
    h = np.asarray(scale, dtype=float)
    if y.ndim == 1:
        y = y[:, None]
        m = m[:, None]
        h = h[:, None]
    r = np.abs(y - m) / (np.maximum(h, 0) + eps)
    return r.max(axis=1)


def apply_joint(center, scale, q: float, lower_clip: float = 0.0):
    return conformal_interval(center, scale, q, lower_clip=lower_clip)


def joint_calibrate(y, center, scale, alpha: float = 0.10) -> dict[str, Any]:
    scores = joint_max_score(y, center, scale)
    q = finite_sample_quantile(scores, alpha)
    return {"mode": "joint_max_score", "alpha": alpha, "n_cal": int(len(scores)), "q": q}
