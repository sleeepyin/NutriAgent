#!/usr/bin/env python3
"""E1 tool-value / heterogeneity + Go-NoGo. Public data is per-tool, not cross-dataset causal."""
from __future__ import annotations

import argparse
from collections import defaultdict

import numpy as np

from .common import add_common_args, load_rows, out_dir, save_status
from agent_extension.src.metrics import mape, paired_bootstrap_diff, point_metrics, wape
from agent_extension.src.runtime import atomic_write_json, log

ACTION_KEYS = {
    "REQUEST_DEPTH": ("p_rgb", "p_depth", "n5k"),
    "REQUEST_VIEW": ("p_rgb", "p_view", None),
    "REQUEST_REFERENCE": ("p_rgb", "p_ref", "ecustfd"),
    "ASK_SEMANTIC": ("p_rgb", "p_semantic", "mf3d"),
}


def _sub(rows, ds=None, split="policy"):
    out = [r for r in rows if r.get("split") == split]
    if ds:
        out = [r for r in out if r["dataset"] == ds]
    return out


def eval_action(rows, before, after):
    y, p0, p1 = [], [], []
    for r in rows:
        a, b = r.get(before), r.get(after)
        if a is None or b is None:
            continue
        if not (np.isfinite(a) and np.isfinite(b) and np.isfinite(r["y"])):
            continue
        y.append(r["y"])
        p0.append(a)
        p1.append(b)
    if len(y) < 5:
        return None
    y, p0, p1 = np.array(y), np.array(p0), np.array(p1)
    e0, e1 = np.abs(y - p0), np.abs(y - p1)
    wins = e1 < e0 - 1e-9
    return {
        "n": int(len(y)),
        "before": point_metrics(y, p0),
        "after": point_metrics(y, p1),
        "d_mape": mape(y, p0) - mape(y, p1),
        "d_wape": wape(y, p0) - wape(y, p1),
        "win_rate": float(np.mean(wins)),
        "mean_abs_err_drop": float(np.mean(e0 - e1)),
        "boot_mae": paired_bootstrap_diff(e0, e1, n_boot=2000, seed=42),
    }


def main(argv=None) -> int:
    ap = add_common_args(argparse.ArgumentParser())
    ap.add_argument("--bootstrap", type=int, default=2000)
    args = ap.parse_args(argv)
    d = out_dir("e03_tool_value", args)
    rows = load_rows()
    if not rows:
        save_status(d, False, error="episodes.jsonl missing; run e02 first")
        return 2
    report = {"actions": {}, "by_morphology": {}, "notes": []}
    hetero_winners = defaultdict(int)
    oracle_vs_fixed = {}

    for act, (before, after, ds_lock) in ACTION_KEYS.items():
        datasets = [ds_lock] if ds_lock else sorted({r["dataset"] for r in rows if r.get(after) is not None})
        report["actions"][act] = {}
        for ds in datasets:
            sub = _sub(rows, ds)
            ev = eval_action(sub, before, after)
            if ev is None:
                continue
            report["actions"][act][ds] = ev
            # subgroup winners
            for r in sub:
                if r.get(before) is None or r.get(after) is None:
                    continue
                if not np.isfinite(r[before]) or not np.isfinite(r[after]):
                    continue
                morph = r.get("morphology", "unknown")
                if abs(r["y"] - r[after]) < abs(r["y"] - r[before]):
                    hetero_winners[(ds, morph, act)] += 1

    # Go/No-Go on policy rows, per dataset (honest)
    gates = {"unified_agent": False, "degrade": None, "reasons": []}
    useful = []
    for act, rec in report["actions"].items():
        for ds, ev in rec.items():
            if ev["d_mape"] >= 0.05 or ev["d_wape"] >= 0.05:
                useful.append((act, ds, ev["d_mape"], ev["win_rate"]))
            oracle_vs_fixed[(act, ds)] = ev["win_rate"]
    if len(useful) >= 2 and len({u[0] for u in useful}) >= 2:
        gates["unified_agent"] = True
        gates["reasons"].append(">=2 actions each beat RGB-only by >=5% MAPE or WAPE on their native dataset")
    else:
        # single best?
        best = None
        for act, rec in report["actions"].items():
            for ds, ev in rec.items():
                if best is None or ev["win_rate"] > best[2]:
                    best = (act, ds, ev["win_rate"])
        if best and best[2] >= 0.90:
            gates["degrade"] = "B_single_action"
            gates["single_action"] = best[0]
            gates["reasons"].append(f"{best[0]} wins on {best[2]:.1%} of {best[1]} policy episodes")
        elif not useful:
            gates["degrade"] = "A_conformal_triage"
            gates["reasons"].append("no action has >=5% MAPE/WAPE oracle gain; unified Agent not supported")
        else:
            gates["degrade"] = "A_conformal_triage"
            gates["reasons"].append("heterogeneity insufficient for a unified multi-tool Agent claim")
            gates["keep_as"] = "feasibility / single-tool adaptive trigger"

    # morphology table
    morph_tbl = defaultdict(lambda: defaultdict(list))
    for r in rows:
        if r.get("split") != "policy":
            continue
        morph = r.get("morphology", "unknown")
        for act, g in (r.get("gains") or {}).items():
            if g is not None and np.isfinite(g):
                morph_tbl[morph][act].append(float(g))
    report["by_morphology"] = {
        m: {a: {"n": len(v), "mean_gain": float(np.mean(v))} for a, v in rec.items()} for m, rec in morph_tbl.items()
    }
    report["useful"] = [{"action": a, "dataset": ds, "d_mape": dm, "win_rate": wr} for a, ds, dm, wr in useful]
    report["gates"] = gates
    report["caveat"] = (
        "Level A: each action is measured on its native public dataset. "
        "Do not interpret cross-dataset differences as causal action effects."
    )
    atomic_write_json(d / "tool_value.json", report)
    save_status(d, True, gates=gates, n=len(rows))
    log(f"E03 gates={gates}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
