#!/usr/bin/env python3
"""E16: offline domain cascade on existing NV-Real predictions.

Cascade the plan asks for: detect shift, ask for a canonical recapture,
route to a known-domain calibrator, otherwise abstain.

This script does not download images and does not retrain. It reads the
per-image table left by the NV-Real zero-shot run. Without that table the
abstention curve is not identified, and the script records the block instead
of substituting the published averages.

Expected columns in phase2b_per_image.csv:
  y_gt, p_ens
Optional:
  p_joint   prediction of the joint-training model on the same image
  dish_id   group id; otherwise parsed from the key column

The source-domain absolute residual quantile defaults to the published
Nutrition5K split quantile, 108.5 g. Pass --source-q when a paired
calibration file is available. Joint-training 23.9% MAPE is recorded as the
published point-error bound for the known-domain leg. It is not a coverage.
"""
from __future__ import annotations

import argparse

import numpy as np
import pandas as pd

from .common import add_common_args, out_dir, save_status
from agent_extension.src.conformal.finite_sample import conformal_interval, finite_sample_quantile
from agent_extension.src.metrics import mape, wape
from agent_extension.src.paths import Paths
from agent_extension.src.runtime import atomic_write_json, log


PUBLISHED_SOURCE_Q_G = 108.5
PUBLISHED_JOINT_MAPE = 0.239
ALPHA = 0.10


def _point(y, p) -> dict:
    y = np.asarray(y, dtype=float)
    p = np.asarray(p, dtype=float)
    ok = np.isfinite(y) & np.isfinite(p) & (np.abs(y) > 0)
    y, p = y[ok], p[ok]
    if len(y) == 0:
        return {"n": 0}
    return {
        "n": int(len(y)),
        "mape": mape(y, p),
        "wape": wape(y, p),
        "mae": float(np.mean(np.abs(y - p))),
    }


def _coverage(y, lo, hi) -> dict:
    y = np.asarray(y, dtype=float)
    inside = (y >= lo) & (y <= hi)
    return {
        "n": int(len(y)),
        "coverage": float(np.mean(inside)) if len(y) else float("nan"),
        "width": float(np.mean(hi - lo)) if len(y) else float("nan"),
    }


def _curve(y, p, q: float, rank: np.ndarray) -> list[dict]:
    """Drop the highest-ranked fraction, then score the frozen interval."""
    y = np.asarray(y, dtype=float)
    p = np.asarray(p, dtype=float)
    order = np.argsort(-np.asarray(rank, dtype=float))
    rows = []
    n = len(y)
    for rate in (0.0, 0.1, 0.2, 0.3, 0.4, 0.5):
        keep_n = int(round(n * (1.0 - rate)))
        if keep_n < 8:
            rows.append({"abstain_rate": rate, "n_kept": keep_n, "skipped": True})
            continue
        idx = order[n - keep_n :]
        lo, hi = conformal_interval(p[idx], np.ones(keep_n), q)
        rec = _coverage(y[idx], lo, hi)
        rec["abstain_rate"] = rate
        rec["point"] = _point(y[idx], p[idx])
        rows.append(rec)
    return rows


def _load(path) -> pd.DataFrame | None:
    if path is None or not path.exists():
        return None
    frame = pd.read_csv(path)
    if "y_gt" not in frame.columns and "y" in frame.columns:
        frame = frame.rename(columns={"y": "y_gt"})
    if "p_ens" not in frame.columns and "p_rgb" in frame.columns:
        frame = frame.rename(columns={"p_rgb": "p_ens"})
    if "y_gt" not in frame.columns or "p_ens" not in frame.columns:
        raise RuntimeError(f"{path} needs y_gt and p_ens")
    return frame


def main(argv=None) -> int:
    ap = add_common_args(argparse.ArgumentParser())
    ap.add_argument("--source-q", type=float, default=PUBLISHED_SOURCE_Q_G)
    args = ap.parse_args(argv)
    d = out_dir("e16_domain_cascade", args)
    paths = Paths.resolve()
    frame = _load(paths.file("nv_phase2b_csv"))
    if frame is None:
        payload = {
            "blocked": True,
            "reason": "nvreal/phase2b_per_image.csv is not on this machine. The abstention curve needs per-image predictions; published averages are not enough.",
            "published_bound": {
                "source_split_q_g": PUBLISHED_SOURCE_Q_G,
                "zero_shot_coverage": 0.328,
                "joint_training_mape_bound": PUBLISHED_JOINT_MAPE,
                "note": "23.9% is the published joint-training point MAPE, not a recapture coverage.",
            },
        }
        atomic_write_json(d / "report.json", payload)
        save_status(d, False, blocked=True)
        log("E16 blocked: NV-Real per-image table missing")
        return 2

    y = frame["y_gt"].to_numpy(dtype=float)
    p = frame["p_ens"].to_numpy(dtype=float)
    ok = np.isfinite(y) & np.isfinite(p)
    y, p = y[ok], p[ok]
    residual = np.abs(y - p)
    q = float(args.source_q)
    lo, hi = conformal_interval(p, np.ones(len(p)), q)
    detected = residual > q
    med = float(np.median(p))
    mad = float(np.median(np.abs(p - med)) + 1e-6)
    deployable_rank = np.abs(p - med) / (1.4826 * mad)

    report = {
        "blocked": False,
        "n": int(len(y)),
        "source_q_g": q,
        "source_q_is_published_default": abs(q - PUBLISHED_SOURCE_Q_G) < 1e-9,
        "zero_shot_point": _point(y, p),
        "frozen_interval": _coverage(y, lo, hi),
        "label_using_detection_rate": float(np.mean(detected)),
        "detection_note": "This rate uses the label residual. It is the complement of frozen-interval coverage, not a deployable detector.",
        "abstain_by_true_residual": _curve(y, p, q, residual),
        "abstain_by_prediction_deviation": _curve(y, p, q, deployable_rank),
        "known_domain_bound": {
            "published_joint_mape": PUBLISHED_JOINT_MAPE,
            "note": "Canonical recapture is not observed. Joint-training MAPE is an upper bound on point error after target-domain labels, not a coverage.",
        },
    }
    if "p_joint" in frame.columns:
        joint = frame["p_joint"].to_numpy(dtype=float)[ok]
        both = np.isfinite(joint)
        q_joint = finite_sample_quantile(np.abs(y[both] - joint[both]), ALPHA) if both.sum() >= 20 else float("inf")
        # The joint quantile above uses the same rows it evaluates. Keep it
        # labeled as an in-sample ceiling, not a held-out coverage claim.
        report["known_domain_bound"]["in_sample_joint_point"] = _point(y[both], joint[both])
        report["known_domain_bound"]["in_sample_joint_q_not_a_coverage_claim"] = q_joint
    atomic_write_json(d / "report.json", report)
    save_status(d, True, n=int(len(y)), coverage=report["frozen_interval"]["coverage"])
    log(
        f"E16 n={len(y)} frozen_coverage={report['frozen_interval']['coverage']:.3f} "
        f"label_detection={report['label_using_detection_rate']:.3f}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
