"""Point-score helpers shared by the cache readers."""
from __future__ import annotations

import numpy as np

from mmcfa.metrics import mae, mape, wape


def clean(obj):
    if isinstance(obj, dict):
        return {str(k): clean(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [clean(v) for v in obj]
    if isinstance(obj, (np.floating, float)):
        value = float(obj)
        return None if not np.isfinite(value) else value
    if isinstance(obj, (np.integer, int)) and not isinstance(obj, bool):
        return int(obj)
    if isinstance(obj, np.ndarray):
        return clean(obj.tolist())
    return obj


def block(y, p) -> dict:
    y = np.asarray(y, dtype=float).reshape(-1)
    p = np.asarray(p, dtype=float).reshape(-1)
    mask = np.isfinite(y) & np.isfinite(p) & (np.abs(y) > 0)
    y, p = y[mask], p[mask]
    if y.size == 0:
        return {"n": 0, "mape": None, "wape": None, "mae": None}
    return {
        "n": int(y.size),
        "mape": mape(y, p),
        "wape": wape(y, p),
        "mae": mae(y, p),
    }


def wape_gap(y, p_a, p_b, n_boot: int = 2000, seed: int = 0) -> dict:
    """Bootstrap of WAPE(b) - WAPE(a) on the paired finite rows."""
    y = np.asarray(y, dtype=float)
    a = np.asarray(p_a, dtype=float)
    b = np.asarray(p_b, dtype=float)
    mask = np.isfinite(y) & np.isfinite(a) & np.isfinite(b) & (np.abs(y) > 0)
    y, a, b = y[mask], a[mask], b[mask]
    if y.size < 8:
        return {"n": int(y.size)}
    rng = np.random.default_rng(seed)
    gaps = np.empty(n_boot, dtype=float)
    n = y.size
    for i in range(n_boot):
        take = rng.integers(0, n, n)
        gaps[i] = wape(y[take], b[take]) - wape(y[take], a[take])
    return {
        "n": int(n),
        "wape_b_minus_a": float(wape(y, b) - wape(y, a)),
        "ci95": [float(np.quantile(gaps, 0.025)), float(np.quantile(gaps, 0.975))],
    }
