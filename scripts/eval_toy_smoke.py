#!/usr/bin/env python3
"""Eval smoke on data/toy + examples: MAE/MAPE, Fix/Worsen, Evidence Acc."""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from vdr.baselines.b1_rag_once import predict_from_hypotheses  # noqa: E402
from vdr.eval.metrics import (  # noqa: E402
    evidence_accuracy,
    fix_rate,
    mae,
    mape,
    worsen_rate,
)

TOY_DIR = ROOT / "data" / "toy" / "dishes"
EX_DIR = ROOT / "examples"
FIELDS = ("energy_kcal", "protein_g", "carb_g", "fat_g", "mass_g")


def load_json(p: Path) -> dict:
    with p.open("r", encoding="utf-8") as f:
        return json.load(f)


def main() -> int:
    dishes = sorted(TOY_DIR.glob("toy_*.json"))
    if not dishes:
        print("FAIL: no toy dishes")
        return 1

    y_true_e, y_pred_e = [], []
    preds = []
    for path in dishes:
        d = load_json(path)
        gt = d.get("n_gt") or {}
        # skip fully-null GT for MAE aggregation
        hyps = d.get("components") or []
        pred = predict_from_hypotheses(d["dish_id"], hyps)
        preds.append(pred)
        if gt.get("energy_kcal") is not None and pred["n"].get("energy_kcal") is not None:
            y_true_e.append(gt["energy_kcal"])
            y_pred_e.append(pred["n"]["energy_kcal"])

    energy_mae = mae(y_true_e, y_pred_e) if y_true_e else float("nan")
    energy_mape = mape(y_true_e, y_pred_e) if y_true_e else float("nan")

    # Synthetic before/after errors for Fix/Worsen smoke
    err_before = [80.0, 50.0, 30.0, 40.0]
    err_after = [40.0, 55.0, 10.0, 40.0]  # fix, worsen, fix, same
    fr = fix_rate(err_before, err_after)
    wr = worsen_rate(err_before, err_after)

    # Evidence Acc: examples + b1 preds
    records = []
    for name in ("pass.json", "corrected.json", "abstain.json"):
        records.append(load_json(EX_DIR / name))
    for p in preds:
        records.append(p)
    eacc = evidence_accuracy(records)

    # Per-field MAE on matched toy dishes (mean over dishes with both values)
    field_maes = {}
    for f in FIELDS:
        ts, ps = [], []
        for path, pred in zip(dishes, preds):
            d = load_json(path)
            gt = d.get("n_gt") or {}
            if gt.get(f) is not None and pred["n"].get(f) is not None:
                ts.append(gt[f])
                ps.append(pred["n"][f])
        if ts:
            field_maes[f] = mae(ts, ps)

    summary = (
        f"eval_toy_smoke: n_dishes={len(dishes)} n_mae={len(y_true_e)} "
        f"energy_MAE={energy_mae:.2f} energy_MAPE={energy_mape:.1f}% "
        f"FixRate={fr:.2f} WorsenRate={wr:.2f} EvidenceAcc={eacc:.2f} "
        f"field_MAEs={{{', '.join(f'{k}:{v:.1f}' for k,v in field_maes.items())}}}"
    )
    print(summary)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
