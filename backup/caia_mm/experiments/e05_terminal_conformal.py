#!/usr/bin/env python3
"""Freeze policy, run full traces on D_cal, compute terminal q. D_cal labels used only for q."""
from __future__ import annotations

import argparse
import pickle
from pathlib import Path

import numpy as np

from .common import add_common_args, load_json, load_rows, out_dir, save_status
from .e04_train_policy import featurize
from agent_extension.src.conformal.terminal import apply_terminal, terminal_calibrate
from agent_extension.src.controller import Controller
from agent_extension.src.metrics import interval_metrics, point_metrics
from agent_extension.src.paths import Paths
from agent_extension.src.runtime import atomic_write_json, log, sha256_file
from agent_extension.src.schemas import PRIOR_COST


def load_controller(paths: Paths, mode: str = "myopic_voi") -> Controller:
    p = paths.results / "e04_train_policy" / "gain_models.pkl"
    gain, fail = {}, {}
    if p.exists():
        blob = pickle.load(open(p, "rb"))
        for a, rec in (blob.get("gain") or {}).items():
            if rec and rec.get("sklearn") is not None:
                gain[a] = {"predict": rec["sklearn"].predict, "sigma": rec.get("sigma", 0.15)}
    return Controller(gain_models=gain, fail_models=fail, mode=mode, t_max=2)


def action_to_pred_key(action: str, r: dict) -> str:
    return {
        "STOP": "p_rgb",
        "REQUEST_DEPTH": "p_depth",
        "REQUEST_VIEW": "p_view",
        "REQUEST_REFERENCE": "p_ref",
        "ASK_SEMANTIC": "p_semantic",
        "ABSTAIN": "p_rgb",
    }.get(action, "p_rgb")


def run_episode(ctrl: Controller, r: dict, budget: float = 4.0) -> dict:
    avail = dict(r.get("available") or {})
    x, u, ood = featurize(r, 0, budget)
    dec0 = ctrl.decide(x, avail, budget, step=0, u=u, ood_score=ood)
    steps = [dec0.to_dict()]
    chosen = [dec0.action]
    cost = float(dec0.estimated_cost)
    remain = budget - cost
    key = action_to_pred_key(dec0.action, r)
    if dec0.action not in ("STOP", "ABSTAIN") and remain > 0:
        x2, u2, ood2 = featurize(r, 1, remain)
        # mask used action
        avail2 = dict(avail)
        avail2[dec0.action] = False
        dec1 = ctrl.decide(x2, avail2, remain, step=1, u=u2, ood_score=ood2)
        steps.append(dec1.to_dict())
        chosen.append(dec1.action)
        cost += float(dec1.estimated_cost)
        if dec1.action not in ("STOP", "ABSTAIN"):
            key = action_to_pred_key(dec1.action, r)
    pred = r.get(key, r.get("p_rgb"))
    if pred is None or not np.isfinite(pred):
        pred = r.get("p_rgb")
    path = "->".join(chosen)
    if chosen[0] == "STOP":
        path = "STOP@0"
    if "ABSTAIN" in chosen and chosen[-1] == "ABSTAIN":
        path = "ABSTAIN"
    return {
        "episode_id": r["episode_id"],
        "dataset": r["dataset"],
        "split": r["split"],
        "y": r["y"],
        "pred": float(pred) if pred is not None and np.isfinite(pred) else float("nan"),
        "scale": float(max(abs(r.get("disagree") or 0.0), 0.15 * abs(pred or 1.0), 5.0)),
        "path": path,
        "steps": steps,
        "total_cost": cost,
        "abstained": "ABSTAIN" in chosen,
        "group_id": r["group_id"],
        "morphology": r.get("morphology"),
    }


def main(argv=None) -> int:
    ap = add_common_args(argparse.ArgumentParser())
    ap.add_argument("--alpha", type=float, default=0.10)
    ap.add_argument("--split", default="calibration")
    ap.add_argument("--mode", default="myopic_voi")
    args = ap.parse_args(argv)
    paths = Paths.resolve()
    d = out_dir("e05_terminal_conformal", args)
    rows = load_rows()
    ctrl = load_controller(paths, args.mode)
    traces = [run_episode(ctrl, r) for r in rows if r.get("split") == args.split]
    y = np.array([t["y"] for t in traces if np.isfinite(t["pred"])], dtype=float)
    m = np.array([t["pred"] for t in traces if np.isfinite(t["pred"])], dtype=float)
    h = np.array([t["scale"] for t in traces if np.isfinite(t["pred"])], dtype=float)
    if len(y) < 8:
        save_status(d, False, error="too few cal traces", n=len(traces))
        return 2
    cal = terminal_calibrate(y, m, h, alpha=args.alpha)
    lo, hi = apply_terminal(m, h, cal)
    metrics = interval_metrics(y, lo, hi, args.alpha)
    # naive: reuse RGB-only q from policy (illegal if used as guarantee; we report as negative control)
    pol = [r for r in rows if r.get("split") == "policy" and np.isfinite(r.get("p_rgb") or np.nan)]
    if len(pol) >= 8:
        from agent_extension.src.conformal.finite_sample import finite_sample_quantile
        q_naive = finite_sample_quantile(np.abs([r["y"] - r["p_rgb"] for r in pol]), args.alpha)
        lo_n, hi_n = m - q_naive, m + q_naive
        lo_n = np.maximum(lo_n, 0)
        naive = interval_metrics(y, lo_n, hi_n, args.alpha)
    else:
        naive = None
        q_naive = None
    atomic_write_json(d / "calibrator.json", {**cal, "naive_q": q_naive, "metrics_on_cal_audit_only": metrics})
    atomic_write_json(d / "traces_cal.json", traces)
    save_status(d, True, q=cal["q"], n=len(traces), cal_audit=metrics, naive=naive)
    log(f"E05 terminal q={cal['q']} n={len(traces)} (cal audit is not a test result)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
