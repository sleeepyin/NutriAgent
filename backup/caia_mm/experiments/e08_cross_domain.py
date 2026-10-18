#!/usr/bin/env python3
"""N5K calibration -> NV-Real zero-shot and few-shot recovery. No 90% claim at n_cal=0."""
from __future__ import annotations

import argparse

import numpy as np

from .common import add_common_args, load_json, load_rows, out_dir, save_status
from agent_extension.src.conformal.finite_sample import conformal_interval, finite_sample_quantile, residual_scores
from agent_extension.src.metrics import clopper_pearson, interval_metrics, point_metrics
from agent_extension.src.paths import Paths
from agent_extension.src.runtime import atomic_write_json, log


def main(argv=None) -> int:
    ap = add_common_args(argparse.ArgumentParser())
    ap.add_argument("--target-calibration-sizes", default="0,25,50,100,200")
    ap.add_argument("--alpha", type=float, default=0.10)
    args = ap.parse_args(argv)
    d = out_dir("e08_cross_domain", args)
    rows = load_rows()
    n5k = [r for r in rows if r["dataset"] == "n5k" and r.get("split") == "calibration"]
    nv = [r for r in rows if r["dataset"] == "nvreal"]
    if len(n5k) < 8 or len(nv) < 10:
        save_status(d, False, error="need n5k cal + nvreal episodes")
        return 2
    y_s = np.array([r["y"] for r in n5k])
    p_s = np.array([r.get("p_rgb", np.nan) for r in n5k])
    h_s = np.maximum(0.15 * np.abs(p_s), 5.0)
    q_src = finite_sample_quantile(residual_scores(y_s, p_s, h_s), args.alpha)

    nv_te = [r for r in nv if r.get("split") == "test"] or nv
    y_t = np.array([r["y"] for r in nv_te])
    p_t = np.array([r.get("p_rgb", np.nan) for r in nv_te])
    h_t = np.maximum(0.15 * np.abs(p_t), 5.0)
    lo, hi = conformal_interval(p_t, h_t, q_src)
    zero = {
        "claim_90": False,
        "reason": "N5K and NV-Real are not exchangeable. Zero target cal => no finite-sample 90% claim.",
        "source_q": q_src,
        "point": point_metrics(y_t, p_t),
        "interval": interval_metrics(y_t, lo, hi, args.alpha),
        "coverage_ci": clopper_pearson(int(((y_t >= lo) & (y_t <= hi)).sum()), int(len(y_t))),
    }

    # OOD gate: abstain if |p-n5k_median|/n5k_mad large
    med = float(np.median(p_s))
    mad = float(np.median(np.abs(p_s - med)) + 1e-6)
    ood = np.abs(p_t - med) / (1.4826 * mad)
    accept = ood < 2.5
    if accept.sum() >= 5:
        lo_a, hi_a = lo[accept], hi[accept]
        sel = {
            "accept_rate": float(accept.mean()),
            "coverage_among_accepted_is_not_full_marginal": True,
            "accepted_interval": interval_metrics(y_t[accept], lo_a, hi_a, args.alpha),
            "catastrophic_recall": float(np.mean((np.abs(y_t - p_t) / np.maximum(y_t, 1)) > 1.0)),
            "catastrophic_caught": float(np.mean(((np.abs(y_t - p_t) / np.maximum(y_t, 1)) > 1.0) & (~accept))),
        }
    else:
        sel = {"accept_rate": float(accept.mean()), "note": "too few accepted"}

    sizes = [int(x) for x in str(args.target_calibration_sizes).replace(" ", "").split(",") if x]
    recovery = []
    nv_pool = [r for r in nv if r.get("split") in ("policy", "calibration")]
    if len(nv_pool) < 10:
        nv_pool = nv
    rng = np.random.default_rng(42)
    idx = np.arange(len(nv_pool))
    for n in sizes:
        rec = {"n": n, "claim_90": n > 0}
        if n == 0:
            rec["interval"] = zero["interval"]
            rec["q"] = q_src
            recovery.append(rec)
            continue
        if n > len(idx):
            rec["skipped"] = True
            recovery.append(rec)
            continue
        take = rng.choice(idx, size=n, replace=False)
        y_c = np.array([nv_pool[i]["y"] for i in take])
        p_c = np.array([nv_pool[i].get("p_rgb", np.nan) for i in take])
        h_c = np.maximum(0.15 * np.abs(p_c), 5.0)
        q = finite_sample_quantile(residual_scores(y_c, p_c, h_c), args.alpha)
        lo2, hi2 = conformal_interval(p_t, h_t, q)
        rec["q"] = q
        rec["interval"] = interval_metrics(y_t, lo2, hi2, args.alpha)
        rec["coverage_ci"] = clopper_pearson(int(((y_t >= lo2) & (y_t <= hi2)).sum()), int(len(y_t)))
        recovery.append(rec)

    # extra view on NV
    nv_view = [r for r in nv_te if r.get("p_view") is not None]
    view = None
    if len(nv_view) >= 8:
        yv = np.array([r["y"] for r in nv_view])
        pv = np.array([r["p_view"] for r in nv_view])
        view = {"point_view": point_metrics(yv, pv), "point_rgb": point_metrics(yv, np.array([r["p_rgb"] for r in nv_view]))}

    out = {"zero_target": zero, "selective": sel, "recovery": recovery, "view_action": view}
    atomic_write_json(d / "cross_domain.json", out)
    save_status(d, True, zero_cov=zero["interval"]["coverage"], recovery=recovery)
    log(f"E08 zero-shot coverage={zero['interval']['coverage']:.3f} (no 90% claim)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
