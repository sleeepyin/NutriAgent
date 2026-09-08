"""Per-action gain and failure models. HistGradientBoosting with grouped OOF."""
from __future__ import annotations

from typing import Any

import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier, HistGradientBoostingRegressor
from sklearn.model_selection import GroupKFold

try:
    from threadpoolctl import threadpool_limits
except Exception:  # pragma: no cover
    class threadpool_limits:  # type: ignore
        def __init__(self, *a, **k):
            pass
        def __enter__(self):
            return self
        def __exit__(self, *a):
            return False


def fit_gain_models(
    X: np.ndarray,
    gains: dict[str, np.ndarray],
    fails: dict[str, np.ndarray],
    groups: np.ndarray,
    max_depth: int = 3,
    seed: int = 42,
) -> dict[str, Any]:
    out: dict[str, Any] = {"gain": {}, "fail": {}, "oof_r2": {}, "oof_auc": {}}
    n = len(X)
    n_splits = min(5, max(2, len(np.unique(groups))))
    for action, y in gains.items():
        y = np.asarray(y, dtype=float)
        mask = np.isfinite(y)
        if mask.sum() < 20:
            out["gain"][action] = None
            out["oof_r2"][action] = float("nan")
            continue
        model = HistGradientBoostingRegressor(
            max_depth=max_depth, learning_rate=0.08, max_iter=80, random_state=seed
        )
        oof = np.full(n, np.nan)
        idx = np.where(mask)[0]
        with threadpool_limits(limits=4):
            try:
                gkf = GroupKFold(n_splits=min(3, n_splits))
                for tr, te in gkf.split(X[idx], y[idx], groups[idx]):
                    model.fit(X[idx[tr]], y[idx[tr]])
                    oof[idx[te]] = model.predict(X[idx[te]])
            except Exception:
                oof[idx] = float(np.mean(y[mask]))
            model.fit(X[mask], y[mask])
        resid = y[mask] - np.nan_to_num(oof[mask], nan=float(np.mean(y[mask])))
        ss_res = float(np.sum(resid**2))
        ss_tot = float(np.sum((y[mask] - y[mask].mean()) ** 2))
        r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else float("nan")
        sigma = float(np.std(resid)) if len(resid) else 0.2
        out["gain"][action] = {"predict": model.predict, "sigma": max(sigma, 0.05), "sklearn": model}
        out["oof_r2"][action] = r2

    for action, yb in fails.items():
        yb = np.asarray(yb, dtype=float)
        if yb.sum() < 3 or (1 - yb).sum() < 3:
            out["fail"][action] = None
            continue
        clf = HistGradientBoostingClassifier(
            max_depth=max_depth, learning_rate=0.08, max_iter=80, random_state=seed
        )
        clf.fit(X, yb)
        out["fail"][action] = {"predict": lambda x, m=clf: m.predict_proba(x)[:, 1], "sklearn": clf}
    return out
