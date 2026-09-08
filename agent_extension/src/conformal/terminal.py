"""Frozen-policy terminal-trajectory conformal calibration."""
from __future__ import annotations

from typing import Any

import numpy as np

from .finite_sample import conformal_interval, finite_sample_quantile, residual_scores


def terminal_calibrate(
    y,
    center,
    scale,
    alpha: float = 0.10,
    extra: dict[str, Any] | None = None,
) -> dict[str, Any]:
    scores = residual_scores(y, center, scale)
    q = finite_sample_quantile(scores, alpha)
    out = {
        "mode": "terminal_policy",
        "alpha": alpha,
        "n_cal": int(len(scores)),
        "q": q,
        "score_mean": float(np.mean(scores)) if len(scores) else float("nan"),
        "score_p90": float(np.quantile(scores, 0.9)) if len(scores) else float("nan"),
    }
    if extra:
        out.update(extra)
    return out


def apply_terminal(center, scale, cal: dict[str, Any], lower_clip: float = 0.0):
    return conformal_interval(center, scale, float(cal["q"]), lower_clip=lower_clip)
