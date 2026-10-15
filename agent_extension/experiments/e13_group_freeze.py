#!/usr/bin/env python3
"""E13: group-conditional action freeze on public ECUSTFD + MetaFood3D.

Policy rows choose the action. Calibration rows set one terminal quantile
per dataset. The test split is scored once.

Success (all required):
  1. ECUSTFD test MAPE of the frozen rule is at least 8 percentage points
     below the category-median action.
  2. MetaFood3D test WAPE of the frozen rule is below the stop baseline.
  3. Macro-MAPE of the frozen rule is below always-stop, always-reference,
     and always-semantic.
  4. Per-dataset terminal coverage on the test split is at least 0.80.
"""
from __future__ import annotations

import argparse
from collections import defaultdict

import numpy as np
import pandas as pd

from .common import add_common_args, out_dir, save_status
from agent_extension.src.conformal.finite_sample import conformal_interval, finite_sample_quantile
from agent_extension.src.data_build import build_ecustfd, build_mf3d
from agent_extension.src.metrics import interval_metrics, mape, wape
from agent_extension.src.paths import Paths
from agent_extension.src.policy.group_freeze import (
    COSTS,
    apply_action,
    choose,
    freeze_table,
    prediction,
)
from agent_extension.src.runtime import atomic_write_json, log


ALPHA = 0.10


def _jsonable(obj):
    if isinstance(obj, dict):
        return {str(k): _jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_jsonable(v) for v in obj]
    if isinstance(obj, (np.floating, float)):
        value = float(obj)
        return None if not np.isfinite(value) else value
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, np.ndarray):
        return _jsonable(obj.tolist())
    return obj


def _attach_ecustfd_semantic(rows: list[dict], raw: pd.DataFrame) -> None:
    """Category-median weight from the policy split only."""
    cat = raw.groupby("inst")["cat"].first().to_dict()
    for r in rows:
        inst = r["episode_id"].split("|", 1)[1]
        r["cat"] = str(cat[inst])
    buckets: dict[str, list[float]] = defaultdict(list)
    for r in rows:
        if r["split"] == "policy":
            buckets[r["cat"]].append(float(r["y"]))
    if not buckets:
        raise RuntimeError("ECUSTFD policy split is empty")
    global_med = float(np.median([y for ys in buckets.values() for y in ys]))
    med = {c: float(np.median(ys)) for c, ys in buckets.items()}
    for r in rows:
        r["p_semantic"] = med.get(r["cat"], global_med)


def _modes(row: dict, table: dict) -> dict[str, tuple[str, float]]:
    gcaf = choose(row, table)
    out = {
        "gcaf": apply_action(row, gcaf),
        "always_stop": apply_action(row, "STOP"),
        "always_reference": apply_action(row, "REQUEST_REFERENCE"),
        "always_semantic": apply_action(row, "ASK_SEMANTIC"),
        "always_view": apply_action(row, "REQUEST_VIEW"),
    }
    return out


def _point(rows, preds) -> dict:
    y = np.array([r["y"] for r in rows], dtype=float)
    p = np.array(preds, dtype=float)
    ok = np.isfinite(y) & np.isfinite(p)
    y, p = y[ok], p[ok]
    if len(y) == 0:
        return {"n": 0}
    return {
        "n": int(len(y)),
        "mape": mape(y, p),
        "wape": wape(y, p),
        "mae": float(np.mean(np.abs(y - p))),
    }


def _fit_q(rows, preds) -> dict[str, float]:
    by_y: dict[str, list[float]] = defaultdict(list)
    by_p: dict[str, list[float]] = defaultdict(list)
    for r, p in zip(rows, preds):
        if not np.isfinite(p):
            continue
        by_y[r["dataset"]].append(float(r["y"]))
        by_p[r["dataset"]].append(float(p))
    q = {}
    for ds in by_y:
        y = np.array(by_y[ds])
        p = np.array(by_p[ds])
        q[ds] = finite_sample_quantile(np.abs(y - p), ALPHA)
    return q


def _intervals(rows, preds, q_by_ds):
    y, p, lo, hi = [], [], [], []
    for r, pred in zip(rows, preds):
        if not np.isfinite(pred):
            continue
        q = q_by_ds.get(r["dataset"])
        if q is None:
            continue
        a, b = conformal_interval(np.array([pred]), np.ones(1), q)
        y.append(float(r["y"]))
        p.append(float(pred))
        lo.append(float(a[0]))
        hi.append(float(b[0]))
    return np.array(y), np.array(p), np.array(lo), np.array(hi)


def _macro(per_ds: dict[str, dict], key: str) -> float:
    vals = [rec[key] for rec in per_ds.values() if rec.get("n", 0) >= 5 and np.isfinite(rec.get(key, np.nan))]
    if not vals:
        return float("nan")
    return float(np.mean(vals))


def _bootstrap_mape_gap(rows, pred_a, pred_b, n_boot: int = 2000, seed: int = 0) -> dict:
    y = np.array([r["y"] for r in rows], dtype=float)
    a = np.array(pred_a, dtype=float)
    b = np.array(pred_b, dtype=float)
    rng = np.random.default_rng(seed)
    n = len(y)
    gaps = []
    for _ in range(n_boot):
        idx = rng.integers(0, n, n)
        gaps.append(mape(y[idx], b[idx]) - mape(y[idx], a[idx]))
    gaps = np.array(gaps)
    return {
        "mean_gap_b_minus_a": float(np.mean(gaps)),
        "ci95": [float(np.quantile(gaps, 0.025)), float(np.quantile(gaps, 0.975))],
    }


def _item_gain_regression(ec_rows: list[dict]) -> dict:
    """Negative control: predict the reference's error drop from the no-coin prediction.

    Fit on one half of the policy split, pick the decision threshold on the other
    half, score the test split. This is the old per-item gain idea on the only
    dataset where the baseline prediction actually varies.
    """
    pol = [r for r in ec_rows if r["split"] == "policy" and np.isfinite(r.get("p_rgb", np.nan)) and np.isfinite(r.get("p_ref", np.nan))]
    rng = np.random.default_rng(0)
    order = rng.permutation(len(pol))
    cut = max(8, len(pol) // 2)
    fit, val = [pol[i] for i in order[:cut]], [pol[i] for i in order[cut:]]

    def design(rows):
        x = np.array([np.log(max(r["p_rgb"], 1.0)) for r in rows], dtype=float)
        return np.column_stack([np.ones(len(rows)), x])

    y_drop = np.array([abs(r["y"] - r["p_rgb"]) - abs(r["y"] - r["p_ref"]) for r in fit], dtype=float)
    coef, *_ = np.linalg.lstsq(design(fit), y_drop, rcond=None)

    def pred_drop(rows):
        return design(rows) @ coef

    # threshold on the policy validation half
    best_t, best_w = 0.0, wape([r["y"] for r in val], [r["p_ref"] for r in val])
    grid = np.quantile(pred_drop(val), [0.0, 0.25, 0.5, 0.75, 1.0]) if len(val) else [0.0]
    for t in grid:
        preds = []
        for r, d in zip(val, pred_drop(val)):
            preds.append(r["p_ref"] if d >= t else r["p_rgb"])
        w = wape([r["y"] for r in val], preds)
        if np.isfinite(w) and w < best_w:
            best_t, best_w = float(t), w

    def apply_split(split):
        rows = [r for r in ec_rows if r["split"] == split]
        drops = pred_drop(rows)
        preds, use = [], []
        for r, d in zip(rows, drops):
            take = bool(d >= best_t and np.isfinite(r.get("p_ref", np.nan)))
            use.append(take)
            preds.append(r["p_ref"] if take else r["p_rgb"])
        return {
            "mape": mape([r["y"] for r in rows], preds),
            "wape": wape([r["y"] for r in rows], preds),
            "frac_reference": float(np.mean(use)) if use else float("nan"),
        }

    test_rows = [r for r in ec_rows if r["split"] == "test"]
    y = np.array([abs(r["y"] - r["p_rgb"]) - abs(r["y"] - r["p_ref"]) for r in test_rows])
    yhat = pred_drop(test_rows)
    ss_tot = float(np.sum((y - y.mean()) ** 2))
    r2 = float(1 - np.sum((y - yhat) ** 2) / ss_tot) if ss_tot > 0 else float("nan")
    always = {
        "mape": mape([r["y"] for r in test_rows], [r["p_ref"] for r in test_rows]),
        "wape": wape([r["y"] for r in test_rows], [r["p_ref"] for r in test_rows]),
    }
    learned = apply_split("test")
    return {
        "test_r2": r2,
        "threshold": best_t,
        "learned_on_test": learned,
        "always_reference_on_test": always,
        "learned_beats_always_reference": bool(learned["mape"] + 1e-9 < always["mape"]),
    }


def main(argv=None) -> int:
    ap = add_common_args(argparse.ArgumentParser())
    args = ap.parse_args(argv)
    paths = Paths.resolve()
    d = out_dir("e13_group_freeze", args)
    ec = build_ecustfd(paths)
    mf = build_mf3d(paths)
    if not ec or not mf:
        save_status(d, False, error="ECUSTFD or MetaFood3D table missing")
        return 2
    _attach_ecustfd_semantic(ec, pd.read_csv(paths.file("ecustfd_csv")))
    rows = ec + mf
    # Guard: the freeze function must not see calibration or test labels.
    policy = [r for r in rows if r["split"] == "policy"]
    table = freeze_table(policy)
    for r in rows:
        if r["split"] != "policy":
            assert "y" in r  # labels exist, but freeze_table was not given these rows

    modes = ["gcaf", "always_stop", "always_reference", "always_semantic", "always_view"]
    bundle = {m: {"cal_pred": [], "test_pred": [], "cal_rows": [], "test_rows": [], "action": []} for m in modes}
    cal_rows = [r for r in rows if r["split"] == "calibration"]
    test_rows = [r for r in rows if r["split"] == "test"]
    for split_name, split_rows in (("cal", cal_rows), ("test", test_rows)):
        for r in split_rows:
            chosen = _modes(r, table)
            for m in modes:
                action, pred = chosen[m]
                bundle[m][f"{split_name}_pred"].append(pred)
                bundle[m][f"{split_name}_rows"].append(r)
                if split_name == "test":
                    bundle[m]["action"].append(action)

    report = {"table": table, "modes": {}, "alpha": ALPHA}
    for m in modes:
        q = _fit_q(bundle[m]["cal_rows"], bundle[m]["cal_pred"])
        per_ds = {}
        coverage = {}
        for ds in sorted({r["dataset"] for r in bundle[m]["test_rows"]}):
            idx = [i for i, r in enumerate(bundle[m]["test_rows"]) if r["dataset"] == ds]
            sub_rows = [bundle[m]["test_rows"][i] for i in idx]
            sub_pred = [bundle[m]["test_pred"][i] for i in idx]
            per_ds[ds] = _point(sub_rows, sub_pred)
            y, p, lo, hi = _intervals(sub_rows, sub_pred, q)
            coverage[ds] = interval_metrics(y, lo, hi, ALPHA) if len(y) else {}
            per_ds[ds]["cost"] = float(np.mean([
                COSTS[bundle[m]["action"][i]] for i in idx
            ])) if idx else float("nan")
        report["modes"][m] = {
            "q_by_dataset": q,
            "test": per_ds,
            "coverage": coverage,
            "macro_mape": _macro(per_ds, "mape"),
            "macro_wape": _macro(per_ds, "wape"),
        }

    ec_test = [r for r in ec if r["split"] == "test"]
    # predictions under gcaf and always_semantic on ECUSTFD test, same order
    gcaf_ec, sem_ec = [], []
    for r in ec_test:
        gcaf_ec.append(_modes(r, table)["gcaf"][1])
        sem_ec.append(_modes(r, table)["always_semantic"][1])
    report["ecustfd_mape_gap_semantic_minus_gcaf"] = _bootstrap_mape_gap(ec_test, gcaf_ec, sem_ec)
    # Point metrics on every row the action table did not see. The quantile
    # still uses only the calibration split, so coverage stays test-only.
    ec_held = [r for r in ec if r["split"] != "policy"]
    report["ecustfd_heldout_point"] = {
        "n": len(ec_held),
        "reference_mape": mape([r["y"] for r in ec_held], [r["p_ref"] for r in ec_held]),
        "semantic_mape": mape([r["y"] for r in ec_held], [r["p_semantic"] for r in ec_held]),
        "stop_mape": mape([r["y"] for r in ec_held], [r["p_rgb"] for r in ec_held]),
        "gap_semantic_minus_reference": _bootstrap_mape_gap(
            ec_held, [r["p_ref"] for r in ec_held], [r["p_semantic"] for r in ec_held], n_boot=4000, seed=1
        ),
    }
    report["item_gain_regression"] = _item_gain_regression(ec)

    gcaf = report["modes"]["gcaf"]
    sem = report["modes"]["always_semantic"]
    stop = report["modes"]["always_stop"]
    ref = report["modes"]["always_reference"]
    ec_gap = sem["test"]["ecustfd"]["mape"] - gcaf["test"]["ecustfd"]["mape"]
    mf_wape_drop = stop["test"]["mf3d"]["wape"] - gcaf["test"]["mf3d"]["wape"]
    macro_beats = all(
        gcaf["macro_mape"] < report["modes"][m]["macro_mape"] - 1e-9
        for m in ("always_stop", "always_reference", "always_semantic")
    )
    cov_ok = all(
        gcaf["coverage"].get(ds, {}).get("coverage", 0.0) >= 0.80
        for ds in ("ecustfd", "mf3d")
    )
    success = bool(ec_gap >= 0.08 and mf_wape_drop > 0 and macro_beats and cov_ok)
    report["success"] = {
        "ecustfd_mape_gap": ec_gap,
        "mf3d_wape_drop_vs_stop": mf_wape_drop,
        "macro_beats_fixed": macro_beats,
        "coverage_at_least_0.80": cov_ok,
        "pass": success,
    }
    atomic_write_json(d / "report.json", _jsonable(report))
    save_status(d, success, success=report["success"], macro_mape=gcaf["macro_mape"])
    log(
        "E13 "
        f"pass={success} macro_mape={gcaf['macro_mape']:.3f} "
        f"ec_gap={ec_gap:.3f} mf_wape_drop={mf_wape_drop:.3f} cov_ok={cov_ok}"
    )
    for ds, rec in gcaf["test"].items():
        cov = gcaf["coverage"][ds]["coverage"]
        log(f"E13 gcaf {ds}: mape={rec['mape']:.3f} wape={rec['wape']:.3f} cov={cov:.3f}")
    return 0 if success else 1


if __name__ == "__main__":
    raise SystemExit(main())
