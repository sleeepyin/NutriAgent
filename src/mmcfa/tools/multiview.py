"""Simple multi-view fusion: quality-weighted robust median of per-view points."""
from __future__ import annotations

import numpy as np


def fuse_views(preds: list[float], weights: list[float] | None = None) -> dict:
    p = np.asarray(preds, dtype=float)
    p = p[np.isfinite(p)]
    if len(p) == 0:
        return {"success": False, "point": float("nan"), "spread": float("nan")}
    if weights is None:
        w = np.ones_like(p)
    else:
        w = np.asarray(weights, dtype=float)[: len(p)]
        w = np.maximum(w, 1e-6)
    order = np.argsort(p)
    p, w = p[order], w[order]
    cw = np.cumsum(w)
    cut = 0.5 * cw[-1]
    med = float(p[np.searchsorted(cw, cut, side="left")].clip(min=0))
    spread = float(np.std(p)) if len(p) > 1 else 0.0
    return {"success": True, "point": med, "spread": spread, "n_views": int(len(p))}
