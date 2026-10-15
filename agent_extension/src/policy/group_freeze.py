"""Group-conditional action freeze.

The previous agent regressed a per-item tool gain. Those regressions had
negative out-of-fold R², so the policy stopped. This rule does not predict
a continuous gain. On the policy split only, it freezes one action for each
(dataset, morphology) group: the available action with the lowest WAPE,
kept only when it beats STOP by at least `min_wape_drop`. Calibration and
test labels are not read.
"""
from __future__ import annotations

from collections import defaultdict

import numpy as np

ACTION_KEYS = {
    "STOP": "p_rgb",
    "REQUEST_REFERENCE": "p_ref",
    "REQUEST_VIEW": "p_view",
    "ASK_SEMANTIC": "p_semantic",
}

COSTS = {
    "STOP": 0.0,
    "ASK_SEMANTIC": 1.0,
    "REQUEST_VIEW": 2.0,
    "REQUEST_REFERENCE": 3.0,
}


def prediction(row: dict, action: str) -> float:
    key = ACTION_KEYS.get(action, "p_rgb")
    value = row.get(key, np.nan)
    if value is None or not np.isfinite(value):
        return float("nan")
    return float(value)


def wape(y, p) -> float:
    y = np.asarray(y, dtype=float)
    p = np.asarray(p, dtype=float)
    ok = np.isfinite(y) & np.isfinite(p) & (np.abs(y) > 0)
    if not np.any(ok):
        return float("nan")
    den = float(np.sum(np.abs(y[ok])))
    if den <= 0:
        return float("nan")
    return float(np.sum(np.abs(y[ok] - p[ok])) / den)


def mape(y, p) -> float:
    y = np.asarray(y, dtype=float)
    p = np.asarray(p, dtype=float)
    ok = np.isfinite(y) & np.isfinite(p) & (np.abs(y) > 0)
    if not np.any(ok):
        return float("nan")
    return float(np.mean(np.abs(y[ok] - p[ok]) / np.abs(y[ok])))


def _group_key(row: dict) -> tuple[str, str]:
    return (str(row.get("dataset")), str(row.get("morphology") or "unknown"))


def _best_action(rows: list[dict], min_wape_drop: float, min_n: int) -> tuple[str, dict]:
    y = np.array([r["y"] for r in rows], dtype=float)
    scores = {}
    for action in ACTION_KEYS:
        p = np.array([prediction(r, action) for r in rows], dtype=float)
        n = int(np.sum(np.isfinite(p) & np.isfinite(y)))
        scores[action] = {"n": n, "wape": wape(y, p) if n >= min_n else float("nan")}
    stop = scores["STOP"]["wape"]
    best, best_w = "STOP", stop
    for action, rec in scores.items():
        if action == "STOP" or not np.isfinite(rec["wape"]) or not np.isfinite(stop):
            continue
        if stop - rec["wape"] + 1e-12 < min_wape_drop:
            continue
        if rec["wape"] < best_w - 1e-12:
            best, best_w = action, rec["wape"]
    return best, scores


def freeze_table(policy_rows: list[dict], min_wape_drop: float = 0.05, min_n: int = 8) -> dict:
    """Return a JSON-ready table. Ignores any row whose split is not 'policy'."""
    rows = [r for r in policy_rows if r.get("split", "policy") == "policy"]
    by_group: dict[tuple[str, str], list[dict]] = defaultdict(list)
    by_dataset: dict[str, list[dict]] = defaultdict(list)
    for r in rows:
        by_group[_group_key(r)].append(r)
        by_dataset[str(r.get("dataset"))].append(r)

    dataset_choice = {}
    for ds, sub in by_dataset.items():
        action, scores = _best_action(sub, min_wape_drop, min_n)
        dataset_choice[ds] = {"action": action, "n": len(sub), "scores": scores}

    groups = {}
    for key, sub in sorted(by_group.items()):
        ds, morph = key
        if len(sub) >= min_n:
            action, scores = _best_action(sub, min_wape_drop, min_n)
            source = "group"
        else:
            action = dataset_choice[ds]["action"]
            scores = {}
            source = "dataset_fallback"
        groups[f"{ds}|{morph}"] = {
            "dataset": ds,
            "morphology": morph,
            "n_policy": len(sub),
            "action": action,
            "source": source,
            "scores": scores,
        }
    return {
        "min_wape_drop": min_wape_drop,
        "min_n": min_n,
        "groups": groups,
        "dataset_fallback": dataset_choice,
    }


def choose(row: dict, table: dict) -> str:
    key = f"{row.get('dataset')}|{row.get('morphology') or 'unknown'}"
    groups = table["groups"]
    if key in groups:
        action = groups[key]["action"]
    else:
        action = table["dataset_fallback"].get(str(row.get("dataset")), {}).get("action", "STOP")
    if np.isfinite(prediction(row, action)):
        return action
    return "STOP"


def apply_action(row: dict, action: str) -> tuple[str, float]:
    if np.isfinite(prediction(row, action)):
        return action, prediction(row, action)
    return "STOP", prediction(row, "STOP")
