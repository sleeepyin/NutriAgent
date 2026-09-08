"""Point, interval, and Agent metrics. MAPE and WAPE are never aliased as PMAE."""
from __future__ import annotations

import math
from typing import Any

import numpy as np


def _arr(x) -> np.ndarray:
    return np.asarray(x, dtype=float).reshape(-1)


def mae(y, p) -> float:
    y, p = _arr(y), _arr(p)
    return float(np.mean(np.abs(y - p))) if len(y) else float("nan")


def medae(y, p) -> float:
    y, p = _arr(y), _arr(p)
    return float(np.median(np.abs(y - p))) if len(y) else float("nan")


def mape(y, p, eps: float = 1e-6) -> float:
    """mean(|e_i|/y_i). Do not call this PMAE."""
    y, p = _arr(y), _arr(p)
    if not len(y):
        return float("nan")
    return float(np.mean(np.abs(y - p) / np.maximum(np.abs(y), eps)))


def wape(y, p) -> float:
    """sum(|e|)/sum(y) = MAE/mean(y). Do not call this PMAE."""
    y, p = _arr(y), _arr(p)
    den = float(np.sum(np.abs(y)))
    if den <= 0:
        return float("nan")
    return float(np.sum(np.abs(y - p)) / den)


def smape(y, p, eps: float = 1e-6) -> float:
    y, p = _arr(y), _arr(p)
    den = np.maximum(np.abs(y) + np.abs(p), eps)
    return float(np.mean(2.0 * np.abs(y - p) / den))


def r2_score(y, p) -> float:
    y, p = _arr(y), _arr(p)
    if len(y) < 2:
        return float("nan")
    ss_res = float(np.sum((y - p) ** 2))
    ss_tot = float(np.sum((y - y.mean()) ** 2))
    if ss_tot <= 0:
        return float("nan")
    return 1.0 - ss_res / ss_tot


def spearman(y, p) -> float:
    y, p = _arr(y), _arr(p)
    if len(y) < 3:
        return float("nan")
    ry = y.argsort().argsort().astype(float)
    rp = p.argsort().argsort().astype(float)
    ry -= ry.mean()
    rp -= rp.mean()
    den = float(np.sqrt(np.sum(ry**2) * np.sum(rp**2)))
    if den <= 0:
        return float("nan")
    return float(np.sum(ry * rp) / den)


def point_metrics(y, p) -> dict[str, float]:
    return {
        "n": int(len(_arr(y))),
        "mae": mae(y, p),
        "medae": medae(y, p),
        "mape": mape(y, p),
        "wape": wape(y, p),
        "smape": smape(y, p),
        "r2": r2_score(y, p),
        "spearman": spearman(y, p),
    }


def winkler(y, lo, hi, alpha: float = 0.10) -> float:
    y, lo, hi = _arr(y), _arr(lo), _arr(hi)
    width = hi - lo
    under = np.maximum(0.0, lo - y)
    over = np.maximum(0.0, y - hi)
    return float(np.mean(width + (2.0 / alpha) * (under + over)))


def interval_metrics(y, lo, hi, alpha: float = 0.10) -> dict[str, float]:
    y, lo, hi = _arr(y), _arr(lo), _arr(hi)
    covered = (y >= lo) & (y <= hi)
    width = hi - lo
    mean_y = float(np.mean(np.abs(y))) if len(y) else float("nan")
    wis = winkler(y, lo, hi, alpha)
    return {
        "n": int(len(y)),
        "coverage": float(np.mean(covered)) if len(y) else float("nan"),
        "mean_width": float(np.mean(width)) if len(y) else float("nan"),
        "median_width": float(np.median(width)) if len(y) else float("nan"),
        "rel_width": float(np.mean(width) / mean_y) if mean_y and mean_y == mean_y else float("nan"),
        "winkler": wis,
        "nwinkler": float(wis / mean_y) if mean_y and mean_y == mean_y else float("nan"),
        "n_covered": int(np.sum(covered)),
    }


def clopper_pearson(k: int, n: int, alpha: float = 0.05) -> tuple[float, float]:
    """Exact Clopper-Pearson CI for a binomial proportion."""
    if n <= 0:
        return (float("nan"), float("nan"))
    try:
        from scipy.stats import beta
    except Exception:
        p = k / n
        z = 1.96
        se = math.sqrt(max(p * (1 - p) / n, 0.0))
        return (max(0.0, p - z * se), min(1.0, p + z * se))
    lo = 0.0 if k == 0 else float(beta.ppf(alpha / 2, k, n - k + 1))
    hi = 1.0 if k == n else float(beta.ppf(1 - alpha / 2, k + 1, n - k))
    return lo, hi


def paired_bootstrap_diff(
    a: np.ndarray,
    b: np.ndarray,
    n_boot: int = 10000,
    seed: int = 42,
    stat: str = "mean",
) -> dict[str, float]:
    """Episode-level paired bootstrap of mean(a-b)."""
    a, b = _arr(a), _arr(b)
    n = min(len(a), len(b))
    if n == 0:
        return {"n": 0, "diff": float("nan"), "lo": float("nan"), "hi": float("nan")}
    a, b = a[:n], b[:n]
    rng = np.random.default_rng(seed)
    diffs = np.empty(n_boot, dtype=float)
    idx = np.arange(n)
    for i in range(n_boot):
        take = rng.choice(idx, size=n, replace=True)
        d = a[take] - b[take]
        diffs[i] = float(np.mean(d) if stat == "mean" else np.median(d))
    point = float(np.mean(a - b) if stat == "mean" else np.median(a - b))
    return {
        "n": n,
        "diff": point,
        "lo": float(np.quantile(diffs, 0.025)),
        "hi": float(np.quantile(diffs, 0.975)),
        "n_boot": n_boot,
    }


def summarize_agent(traces: list[dict[str, Any]]) -> dict[str, float]:
    if not traces:
        return {"n": 0}
    n_act = [len(t.get("steps", [])) for t in traces]
    abstain = [1.0 if t.get("abstained") else 0.0 for t in traces]
    stop0 = [1.0 if t.get("path") == "STOP@0" else 0.0 for t in traces]
    costs = [float(t.get("total_cost", 0.0)) for t in traces]
    return {
        "n": len(traces),
        "mean_actions": float(np.mean(n_act)),
        "mean_cost": float(np.mean(costs)),
        "abstain_rate": float(np.mean(abstain)),
        "stop0_rate": float(np.mean(stop0)),
    }
