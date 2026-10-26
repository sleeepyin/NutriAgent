"""A4: domain-shift detection on the Nutrition5K eval half and NV-Real val."""
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
for entry in (str(ROOT), str(ROOT / "src")):
    if entry not in sys.path:
        sys.path.insert(0, entry)

import numpy as np
from sklearn.metrics import roc_auc_score

from experiments.vlm.scoring import block, clean
from experiments.vlm.store import load_record
from mmcfa.conformal.finite_sample import conformal_interval, finite_sample_quantile
from mmcfa.metrics import mae, wape
from mmcfa.paths import Paths
from mmcfa.runtime import atomic_write_json, log

ALPHA = 0.10


def _rows(paths: Paths) -> list[dict]:
    out = []
    for line in (paths.root / "results" / "vlm" / "points.jsonl").read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        if row["dataset"] == "n5k" or (row["dataset"] == "nvreal" and row["split"] == "val"):
            out.append(row)
    return out


def _auroc(labels, scores) -> float | None:
    labels = np.asarray(labels, dtype=int)
    scores = np.asarray(scores, dtype=float)
    if len(np.unique(labels)) < 2 or not np.isfinite(scores).all():
        return None
    return float(roc_auc_score(labels, scores))


def _curve(y, pred, lo, hi, score, ood) -> list[dict]:
    order = np.argsort(-np.asarray(score, dtype=float))
    n = len(order)
    rows = []
    for frac in (0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8):
        drop = int(round(n * frac))
        keep = order[drop:]
        if keep.size == 0:
            continue
        yy, pp = y[keep], pred[keep]
        covered = (yy >= lo[keep]) & (yy <= hi[keep])
        rows.append(
            {
                "reject_rate": frac,
                "n_kept": int(keep.size),
                "wape": wape(yy, pp),
                "mae": mae(yy, pp),
                "coverage": float(np.mean(covered)),
                "ood_fraction_kept": float(np.mean(ood[keep])),
            }
        )
    return rows


def main() -> int:
    paths = Paths.resolve()
    rows = _rows(paths)
    cal = [row for row in rows if row["dataset"] == "n5k" and row["split"] == "cal"]
    test = [row for row in rows if row["split"] in {"eval", "val"}]
    y_cal = np.array([row["y_g"] for row in cal], dtype=float)
    p_cal = np.array([row["p_ens3"] for row in cal], dtype=float)
    d_cal = np.abs(np.array([row["p_rn3"] - row["p_eff3"] for row in cal], dtype=float))
    q = finite_sample_quantile(np.abs(y_cal - p_cal), ALPHA)
    threshold = float(np.quantile(d_cal, 0.95))

    y, pred, disagree, ood_flag, labels, joint = [], [], [], [], [], []
    missing = 0
    for row in test:
        record = load_record(row["dataset"], row["sample_id"])
        if record is None:
            missing += 1
            continue
        y.append(float(row["y_g"]))
        pred.append(float(row["p_ens3"]))
        disagree.append(abs(float(row["p_rn3"]) - float(row["p_eff3"])))
        ood_flag.append(1.0 if record.get("out_of_domain") else 0.0)
        labels.append(1 if row["dataset"] == "nvreal" else 0)
        joint.append(float(row["p_joint"]) if row["dataset"] == "nvreal" else float(row["p_ens3"]))
    y = np.asarray(y)
    pred = np.asarray(pred)
    disagree = np.asarray(disagree)
    ood_flag = np.asarray(ood_flag)
    labels = np.asarray(labels)
    joint = np.asarray(joint)
    score_c = np.maximum(disagree / max(threshold, 1e-6), ood_flag)
    lo, hi = conformal_interval(pred, np.ones_like(pred), q)
    flagged = score_c >= 1.0
    mixed = pred.copy()
    mixed[flagged] = joint[flagged]
    report = {
        "n_cal": len(cal),
        "n_test": int(y.size),
        "n_missing_vlm": missing,
        "n_in_domain": int(np.sum(labels == 0)),
        "n_ood": int(np.sum(labels == 1)),
        "score_a": "absolute disagreement of the 3-channel ResNet and EfficientNet",
        "score_b": "VLM out_of_domain",
        "score_c": "max(disagreement / in-domain 95th percentile, out_of_domain)",
        "disagreement_threshold_g": threshold,
        "split_q_g": q,
        "auroc": {
            "disagreement": _auroc(labels, disagree),
            "vlm_out_of_domain": _auroc(labels, ood_flag),
            "or_score": _auroc(labels, score_c),
        },
        "curves": {
            "disagreement": _curve(y, pred, lo, hi, disagree, labels),
            "vlm_out_of_domain": _curve(y, pred, lo, hi, ood_flag, labels),
            "or_score": _curve(y, pred, lo, hi, score_c, labels),
        },
        "always_3ch": block(y, pred),
        "joint_when_flagged": {
            "flag_rate": float(np.mean(flagged)) if flagged.size else None,
            "metrics": block(y, mixed),
        },
    }
    dest = paths.results_action / "a4_ood_reject" / "report.json"
    atomic_write_json(dest, clean(report))
    log(f"A4 auroc={report['auroc']} n={y.size}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
