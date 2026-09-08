"""All-tool-state x multi-output joint conformal envelope."""
from __future__ import annotations

from typing import Any

import numpy as np

from .finite_sample import conformal_interval, finite_sample_quantile


def all_state_scores(
    y_list: list[np.ndarray],
    centers: list[np.ndarray],
    scales: list[np.ndarray],
    eps: float = 1e-8,
) -> np.ndarray:
    """
    y_list[j] is output j (n,), centers[m][i, j] / scales[m][i, j]
    or simpler: each state provides (n, J) center/scale, y is (n, J).
    Here we accept:
      y: (n, J)
      state_centers: list of (n, J)
      state_scales: list of (n, J)
    R_i = max_m max_j |y_ij - m_mj| / (h_mj + eps)
    """
    raise NotImplementedError


def all_state_calibrate(
    y: np.ndarray,
    state_centers: list[np.ndarray],
    state_scales: list[np.ndarray],
    alpha: float = 0.10,
    eps: float = 1e-8,
) -> dict[str, Any]:
    y = np.asarray(y, dtype=float)
    if y.ndim == 1:
        y = y[:, None]
    n, j = y.shape
    scores = np.full(n, -np.inf, dtype=float)
    for c, s in zip(state_centers, state_scales):
        c = np.asarray(c, dtype=float)
        s = np.asarray(s, dtype=float)
        if c.ndim == 1:
            c = c[:, None]
            s = s[:, None]
        r = np.abs(y - c) / (np.maximum(s, 0) + eps)
        scores = np.maximum(scores, r.max(axis=1))
    q = finite_sample_quantile(scores, alpha)
    return {
        "mode": "all_tool_states",
        "alpha": alpha,
        "n_cal": int(n),
        "n_states": int(len(state_centers)),
        "n_outputs": int(j),
        "q": q,
        "score_mean": float(np.mean(scores)),
        "score_p90": float(np.quantile(scores, 0.9)) if n else float("nan"),
    }


def apply_all_state(center, scale, cal: dict[str, Any], lower_clip: float = 0.0):
    return conformal_interval(center, scale, float(cal["q"]), lower_clip=lower_clip)
