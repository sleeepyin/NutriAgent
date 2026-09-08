"""Path-conditional Mondrian calibration. Small groups fall back to global q."""
from __future__ import annotations

from collections import defaultdict
from typing import Any

import numpy as np

from .finite_sample import conformal_interval, finite_sample_quantile, residual_scores


def path_mondrian_calibrate(
    y,
    center,
    scale,
    paths: list[str],
    alpha: float = 0.10,
    min_group: int = 50,
) -> dict[str, Any]:
    scores = residual_scores(y, center, scale)
    q_global = finite_sample_quantile(scores, alpha)
    buckets: dict[str, list[float]] = defaultdict(list)
    for p, s in zip(paths, scores):
        buckets[str(p)].append(float(s))
    q_path: dict[str, float] = {}
    used_global: dict[str, bool] = {}
    for p, ss in buckets.items():
        if len(ss) >= min_group:
            q_path[p] = finite_sample_quantile(ss, alpha)
            used_global[p] = False
        else:
            q_path[p] = q_global
            used_global[p] = True
    return {
        "mode": "path_mondrian",
        "alpha": alpha,
        "q_global": q_global,
        "q_path": q_path,
        "n_path": {p: len(ss) for p, ss in buckets.items()},
        "fallback_global": used_global,
        "min_group": min_group,
    }


def apply_path_q(center, scale, path: str, cal: dict[str, Any], lower_clip: float = 0.0):
    q = cal.get("q_path", {}).get(str(path), cal.get("q_global"))
    return conformal_interval(center, scale, float(q), lower_clip=lower_clip)
