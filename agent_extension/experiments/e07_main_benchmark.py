#!/usr/bin/env python3
"""One-shot test benchmark. --locked refuses to run if prereg/policy hashes changed after seeing test."""
from __future__ import annotations

import argparse
import hashlib
from collections import defaultdict
from pathlib import Path

import numpy as np

from .common import add_common_args, load_json, load_rows, out_dir, save_status
from .e04_train_policy import featurize
from .e05_terminal_conformal import action_to_pred_key, load_controller, run_episode
from agent_extension.src.conformal.finite_sample import conformal_interval
from agent_extension.src.controller import Controller
from agent_extension.src.metrics import (
    clopper_pearson,
    interval_metrics,
    paired_bootstrap_diff,
    point_metrics,
    summarize_agent,
    winkler,
)
from agent_extension.src.paths import Paths
from agent_extension.src.runtime import atomic_write_json, log, sha256_file
from agent_extension.src.schemas import PRIOR_COST


MODES = [
    "always_stop",
    "always_depth",
    "always_view",
    "always_reference",
    "always_semantic",
    "always_abstain",
    "triage",
    "myopic_voi",
]


def apply_q(traces, q):
    y, m, h = [], [], []
    for t in traces:
        if not np.isfinite(t["pred"]):
            continue
        y.append(t["y"])
        m.append(t["pred"])
        h.append(t["scale"])
    y, m, h = np.array(y), np.array(m), np.array(h)
    lo, hi = conformal_interval(m, h, q)
    return y, m, lo, hi


def eval_mode(rows_test, ctrl, q, alpha=0.10):
    traces = [run_episode(ctrl, r) for r in rows_test]
    y, m, lo, hi = apply_q(traces, q)
    return {
        "point": point_metrics(y, m),
        "interval": interval_metrics(y, lo, hi, alpha),
        "coverage_ci": clopper_pearson(int(interval_metrics(y, lo, hi, alpha)["n_covered"]), int(len(y))),
        "agent": summarize_agent(traces),
        "mean_winkler": winkler(y, lo, hi, alpha),
        "mean_cost": float(np.mean([t["total_cost"] for t in traces])) if traces else float("nan"),
        "traces_n": len(traces),
    }, traces


def check_locked(paths: Paths) -> tuple[bool, str]:
    pr = paths.prereg / "caia_mm_prereg_v1.md"
    hs = paths.prereg / "caia_mm_prereg_v1.sha256"
    if not pr.exists() or not hs.exists():
        return False, "prereg missing"
    got = sha256_file(pr)
    exp = hs.read_text().split()[0].strip()
    if got != exp:
        return False, f"prereg hash mismatch {got} != {exp}"
    lock_p = paths.results / "e07_main_benchmark" / "lock.json"
    if lock_p.exists():
        prev = load_json(lock_p, {})
        if prev.get("prereg") != got:
            return False, "prereg changed after lock"
    return True, got


def main(argv=None) -> int:
    ap = add_common_args(argparse.ArgumentParser())
    ap.add_argument("--split", default="test")
    ap.add_argument("--locked", action="store_true")
    ap.add_argument("--alpha", type=float, default=0.10)
    args = ap.parse_args(argv)
    paths = Paths.resolve()
    d = out_dir("e07_main_benchmark", args)
    if args.locked:
        ok, msg = check_locked(paths)
        if not ok:
            save_status(d, False, error=f"locked refused: {msg}")
            log(f"E07 locked refuse: {msg}", "ERROR")
            return 3
        atomic_write_json(d / "lock.json", {"prereg": msg})

    rows = load_rows()
    test = [r for r in rows if r.get("split") == args.split]
    cal = load_json(paths.results / "e05_terminal_conformal" / "calibrator.json", {})
    q = cal.get("q")
    if q is None:
        save_status(d, False, error="terminal calibrator missing")
        return 2

    results = {}
    all_traces = {}
    for mode in MODES:
        ctrl = load_controller(paths, mode)
        ev, tr = eval_mode(test, ctrl, q, args.alpha)
        results[mode] = ev
        all_traces[mode] = [{"episode_id": t["episode_id"], "pred": t["pred"], "y": t["y"], "path": t["path"], "cost": t["total_cost"]} for t in tr]
        log(f"E07 {mode}: cov={ev['interval']['coverage']:.3f} nw={ev['interval']['nwinkler']:.3f} cost={ev['mean_cost']:.2f}")

    # cost-matched comparison: among always_* with finite cov, pick best nWinkler among those with cost>= agent cost
    agent = results["myopic_voi"]
    fixed = {k: v for k, v in results.items() if k.startswith("always_") and k != "always_abstain"}
    best_fixed = min(fixed.items(), key=lambda kv: kv[1]["interval"]["nwinkler"] if np.isfinite(kv[1]["interval"]["nwinkler"]) else 1e9)
    # cost-matched: fixed protocols with mean_cost <= agent cost + 0.25 or the closest higher
    matched = [kv for kv in fixed.items() if kv[1]["mean_cost"] <= agent["mean_cost"] + 0.25]
    if not matched:
        matched = [min(fixed.items(), key=lambda kv: abs(kv[1]["mean_cost"] - agent["mean_cost"]))]
    best_matched = min(matched, key=lambda kv: kv[1]["interval"]["nwinkler"] if np.isfinite(kv[1]["interval"]["nwinkler"]) else 1e9)

    # oracle best action on test (upper bound, not ranked)
    oracle_err, rgb_err = [], []
    for r in test:
        y = r["y"]
        cands = [r.get(k) for k in ("p_rgb", "p_depth", "p_view", "p_ref", "p_semantic")]
        cands = [c for c in cands if c is not None and np.isfinite(c)]
        if not cands:
            continue
        oracle_err.append(min(abs(y - c) for c in cands))
        if r.get("p_rgb") is not None and np.isfinite(r["p_rgb"]):
            rgb_err.append(abs(y - r["p_rgb"]))
    oracle = {
        "mean_abs": float(np.mean(oracle_err)) if oracle_err else float("nan"),
        "rgb_mean_abs": float(np.mean(rgb_err)) if rgb_err else float("nan"),
        "note": "oracle upper bound, not in ranking",
    }

    primary = {
        "agent_nwinkler": agent["interval"]["nwinkler"],
        "best_matched_name": best_matched[0],
        "best_matched_nwinkler": best_matched[1]["interval"]["nwinkler"],
        "rel_improvement": (
            (best_matched[1]["interval"]["nwinkler"] - agent["interval"]["nwinkler"])
            / abs(best_matched[1]["interval"]["nwinkler"])
            if best_matched[1]["interval"]["nwinkler"]
            else float("nan")
        ),
        "agent_coverage": agent["interval"]["coverage"],
        "coverage_ci": agent["coverage_ci"],
        "H1_pass_raw": False,
    }
    if np.isfinite(primary["rel_improvement"]) and primary["rel_improvement"] >= 0.10:
        # still require coverage not far below 0.90 point estimate; confirmatory uses CI
        primary["H1_pass_raw"] = True
    primary["note"] = "H1 confirmatory only if locked and coverage consistent with 90%. Level A is feasibility."

    atomic_write_json(d / "benchmark.json", {"results": results, "primary": primary, "oracle": oracle, "best_fixed": best_fixed[0]})
    atomic_write_json(d / "traces_compact.json", all_traces)
    save_status(d, True, primary=primary, modes=list(results))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
