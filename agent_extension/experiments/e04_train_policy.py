#!/usr/bin/env python3
"""Fit myopic VOI gain models on D_policy only. Never touch D_cal / D_test labels for q."""
from __future__ import annotations

import os

os.environ.setdefault("OMP_NUM_THREADS", "4")
os.environ.setdefault("MKL_NUM_THREADS", "4")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "4")

import argparse
import pickle

import numpy as np

from .common import add_common_args, load_rows, out_dir, save_status
from agent_extension.src.policy.features import pack_features
from agent_extension.src.policy.gain_model import fit_gain_models
from agent_extension.src.runtime import atomic_write_json, log
from agent_extension.src.schemas import UncertaintyFactors
from agent_extension.src.tools.morphology import p_from_label


def factors(r: dict) -> UncertaintyFactors:
    yhat = float(r.get("p_rgb") or r.get("y") or 1.0)
    dis = float(r.get("disagree") or 0.0)
    rel_dis = dis / max(abs(yhat), 1.0)
    u_scale = 0.7 if r["dataset"] in ("n5k", "ecustfd") and r.get("p_ref") is None else (0.55 if r["dataset"] == "n5k" else 0.25)
    if r["dataset"] == "ecustfd":
        u_scale = 0.75
    u_geom = 0.65 if r["dataset"] in ("nvreal", "ecustfd") else 0.35
    u_dens = 0.75 if r.get("morphology") in ("prepared", "stacked", "liquid") else (0.55 if r["dataset"] == "mf3d" else 0.30)
    u_dom = 0.80 if r["dataset"] == "nvreal" else (0.20 if r["dataset"] == "n5k" else 0.40)
    u_qual = min(1.0, rel_dis)
    return UncertaintyFactors(u_scale, u_geom, u_dens, u_dom, u_qual)


def featurize(r: dict, step: int = 0, budget: float = 4.0) -> np.ndarray:
    u = factors(r)
    pr, pp = p_from_label(r.get("morphology", "unknown"))
    ood = u.u_domain
    return pack_features(
        float(r.get("p_rgb") or 1.0),
        float(r.get("disagree") or 0.0),
        u,
        ood,
        r.get("available") or {},
        pr,
        pp,
        budget,
        step,
    ), u, ood


def main(argv=None) -> int:
    import os
    os.environ.setdefault("OMP_NUM_THREADS", "4")
    os.environ.setdefault("MKL_NUM_THREADS", "4")
    ap = add_common_args(argparse.ArgumentParser())
    ap.add_argument("--policy", default="myopic_voi")
    ap.add_argument("--max-steps", type=int, default=2)
    args = ap.parse_args(argv)
    d = out_dir("e04_train_policy", args)
    rows = [r for r in load_rows() if r.get("split") == "policy"]
    if len(rows) < 20:
        save_status(d, False, error="not enough policy rows")
        return 2
    X, groups = [], []
    gains = {"REQUEST_DEPTH": [], "REQUEST_VIEW": [], "REQUEST_REFERENCE": [], "ASK_SEMANTIC": []}
    fails = {k: [] for k in gains}
    for r in rows:
        x, _, _ = featurize(r)
        X.append(x)
        groups.append(r["group_id"])
        gmap = r.get("gains") or {}
        for a in gains:
            g = gmap.get(a)
            if g is None or not np.isfinite(g):
                gains[a].append(np.nan)
                fails[a].append(0.0)
            else:
                gains[a].append(float(g))
                fails[a].append(0.0)
    X = np.vstack(X)
    groups = np.array(groups)
    gains_np = {k: np.array(v, dtype=float) for k, v in gains.items()}
    fails_np = {k: np.array(v, dtype=float) for k, v in fails.items()}
    fitted = fit_gain_models(X, gains_np, fails_np, groups)
    # persist sklearn models
    dump = {"gain": {}, "fail": {}, "oof_r2": fitted["oof_r2"], "max_steps": args.max_steps, "policy": args.policy}
    for a, rec in fitted["gain"].items():
        if rec is None:
            dump["gain"][a] = None
        else:
            dump["gain"][a] = {"sklearn": rec["sklearn"], "sigma": rec["sigma"]}
    with open(d / "gain_models.pkl", "wb") as f:
        pickle.dump(dump, f)
    atomic_write_json(d / "policy_fit.json", {"oof_r2": fitted["oof_r2"], "n": len(rows), "policy": args.policy})
    # gate: if all R2<=0, mark weak
    r2s = [v for v in fitted["oof_r2"].values() if v == v]
    weak = (not r2s) or max(r2s) <= 0
    save_status(d, True, oof_r2=fitted["oof_r2"], weak_gain_model=weak)
    log(f"E04 oof_r2={fitted['oof_r2']} weak={weak}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
