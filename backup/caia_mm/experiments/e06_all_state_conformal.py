#!/usr/bin/env python3
"""All predefined reachable states x outputs joint envelope on D_cal."""
from __future__ import annotations

import argparse

import numpy as np

from .common import add_common_args, load_rows, out_dir, save_status
from agent_extension.src.conformal.all_tool_states import all_state_calibrate, apply_all_state
from agent_extension.src.conformal.multioutput import joint_calibrate
from agent_extension.src.conformal.path_mondrian import path_mondrian_calibrate
from agent_extension.src.metrics import interval_metrics
from agent_extension.src.runtime import atomic_write_json, log

STATE_KEYS = {
    "n5k": ["p_rgb", "p_depth"],
    "nvreal": ["p_rgb", "p_view"],
    "ecustfd": ["p_rgb", "p_ref", "p_view"],
    "mf3d": ["p_rgb", "p_semantic"],
}


def main(argv=None) -> int:
    ap = add_common_args(argparse.ArgumentParser())
    ap.add_argument("--alpha", type=float, default=0.10)
    ap.add_argument("--split", default="calibration")
    args = ap.parse_args(argv)
    d = out_dir("e06_all_state_conformal", args)
    rows = [r for r in load_rows() if r.get("split") == args.split]
    cals = {}
    for ds, keys in STATE_KEYS.items():
        sub = [r for r in rows if r["dataset"] == ds]
        if len(sub) < 8:
            continue
        y = np.array([r["y"] for r in sub], dtype=float)
        centers, scales = [], []
        for k in keys:
            p = np.array([r.get(k, np.nan) for r in sub], dtype=float)
            if not np.isfinite(p).all():
                med = np.nanmedian(p)
                p = np.where(np.isfinite(p), p, med)
            centers.append(p)
            scales.append(np.maximum(0.15 * np.abs(p), 5.0))
        cal = all_state_calibrate(y, centers, scales, alpha=args.alpha)
        # audit on cal (not test)
        lo, hi = apply_all_state(centers[0], scales[0], cal)
        cal["audit_state0"] = interval_metrics(y, lo, hi, args.alpha)
        cals[ds] = cal
    # joint five-output on N5K if nutrients exist
    n5k = [r for r in rows if r["dataset"] == "n5k" and r.get("y_cal") is not None]
    if len(n5k) >= 8:
        y5 = np.array([[r["y"], r["y_cal"], r["y_fat"], r["y_carb"], r["y_prot"]] for r in n5k], dtype=float)
        # only weight has action-specific center; nutrients stay rgb-proxy = missing -> skip joint if no preds
        m5 = np.column_stack([
            np.array([r.get("p_rgb", np.nan) for r in n5k]),
            y5[:, 1] * np.nan,  # placeholder
        ])
        # honest: joint over weight only if nutrients have no action-conditional model
        cals["n5k_weight_only_note"] = "five-output joint requires nutrient predictors per state; not available on Level A public pairing. Weight all-state envelope is reported."
    atomic_write_json(d / "all_state_calibrators.json", cals)
    save_status(d, True, datasets=list(cals.keys()))
    log(f"E06 all-state calibrators for {list(cals.keys())}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
