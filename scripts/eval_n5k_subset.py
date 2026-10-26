#!/usr/bin/env python3
"""B0/B1 on a fixed official Nutrition5K rgb_test subset that has overhead RGB.

Labels are used only after prediction, to score MAE/MAPE. This is not the
full 709-dish test table.
"""

from __future__ import annotations

import argparse
import json
import random
import statistics
import sys
import time
import traceback
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from vdr.baselines.b0_direct_vlm import predict as predict_b0  # noqa: E402
from vdr.baselines.b1_rag_once import predict as predict_b1  # noqa: E402
from vdr.data.n5k import dishes_with_overhead  # noqa: E402
from vdr.eval.metrics import mae, mape  # noqa: E402
from vdr.schemas import VDR_OUTPUT_SCHEMA_PATH, load_schema, validate_instance  # noqa: E402

FIELDS = ("energy_kcal", "protein_g", "carb_g", "fat_g", "mass_g")
JSONL = ROOT / "results" / "n5k_subset_b0_b1.jsonl"
SUMMARY = ROOT / "results" / "n5k_subset_b0_b1.json"


def choose_subset(rows: list[dict], n: int, seed: int) -> list[dict]:
    if n >= len(rows):
        chosen = list(rows)
    else:
        chosen = random.Random(seed).sample(rows, n)
    chosen.sort(key=lambda row: row["dish_id"])
    return chosen


def _score_block(rows: list[dict], key: str) -> dict:
    block = {}
    for field in FIELDS:
        paired = []
        for row in rows:
            pred = (row.get(key) or {}).get("n", {}).get(field) if row.get(key) else None
            if pred is None:
                continue
            paired.append((float(row["label"][field]), float(pred)))
        if not paired:
            block[field] = {"n": 0, "mae": None, "median_ae": None, "mape_pct": None}
            continue
        y_true, y_pred = zip(*paired)
        abs_err = [abs(a - b) for a, b in paired]
        block[field] = {
            "n": len(paired),
            "mae": round(mae(y_true, y_pred), 2),
            "median_ae": round(statistics.median(abs_err), 2),
            "mape_pct": round(mape(y_true, y_pred), 2),
        }
    return block


def mape_when(rows: list[dict], key: str, field: str, min_label: float) -> float | None:
    """MAPE on dishes whose label is at least min_label. Near-zero labels explode MAPE."""
    y_true = []
    y_pred = []
    for row in rows:
        true = float(row["label"][field])
        pred = (row.get(key) or {}).get("n", {}).get(field)
        if pred is None or true < min_label:
            continue
        y_true.append(true)
        y_pred.append(float(pred))
    if not y_true:
        return None
    return round(mape(y_true, y_pred), 2)


def _run_one(name: str, fn, image: Path, dish_id: str, schema: dict) -> dict:
    started = time.time()
    try:
        pred = fn(str(image), dish_id=dish_id)
        validate_instance(pred, schema)
    except Exception as exc:  # keep the subset going
        return {
            "ok": False,
            "error": f"{type(exc).__name__}: {exc}",
            "trace": traceback.format_exc(limit=8),
            "seconds": round(time.time() - started, 2),
        }
    vlm = (pred.get("meta") or {}).get("vlm") or {}
    hyps = vlm.get("raw")
    return {
        "ok": True,
        "s": pred.get("s"),
        "n": pred.get("n"),
        "verify_codes": (pred.get("meta") or {}).get("verify_codes"),
        "n_hypotheses": vlm.get("n_hypotheses"),
        "vlm_raw": hyps,
        "seconds": round(time.time() - started, 2),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--n", type=int, default=50)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--split", default="rgb_test")
    args = parser.parse_args()

    pool = dishes_with_overhead(args.split)
    subset = choose_subset(pool, args.n, args.seed)
    schema = load_schema(VDR_OUTPUT_SCHEMA_PATH)
    JSONL.parent.mkdir(parents=True, exist_ok=True)
    print(
        f"pool={len(pool)} subset={len(subset)} seed={args.seed} split={args.split}",
        flush=True,
    )
    rows = []
    with JSONL.open("w", encoding="utf-8") as handle:
        for index, row in enumerate(subset, start=1):
            dish_id = row["dish_id"]
            image = row["image"]
            record = {
                "dish_id": dish_id,
                "image": str(image.relative_to(ROOT)),
                "label": row["label"],
                "b1": _run_one("b1", predict_b1, image, dish_id, schema),
                "b0": _run_one("b0", predict_b0, image, dish_id, schema),
            }
            rows.append(record)
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
            handle.flush()
            b1 = record["b1"]
            b0 = record["b0"]
            print(
                f"[{index}/{len(subset)}] {dish_id} "
                f"label_kcal={row['label']['energy_kcal']:.0f} "
                f"b1={b1.get('n', {}).get('energy_kcal') if b1.get('ok') else b1.get('error')} "
                f"b0={b0.get('n', {}).get('energy_kcal') if b0.get('ok') else b0.get('error')} "
                f"({b1.get('seconds')}s+{b0.get('seconds')}s)",
                flush=True,
            )

    ok_b1 = [row for row in rows if row["b1"].get("ok")]
    ok_b0 = [row for row in rows if row["b0"].get("ok")]
    summary = {
        "split": args.split,
        "seed": args.seed,
        "n_requested": args.n,
        "n_pool_with_rgb_and_label": len(pool),
        "n_subset": len(subset),
        "dish_ids": [row["dish_id"] for row in subset],
        "n_b1_ok": len(ok_b1),
        "n_b0_ok": len(ok_b0),
        "n_b1_abstain": sum(row["b1"].get("s") == "abstain" for row in ok_b1),
        "b1": _score_block(ok_b1, "b1"),
        "b0": _score_block(ok_b0, "b0"),
        "b1_fat_mape_when_label_ge_1g": mape_when(ok_b1, "b1", "fat_g", 1.0),
        "b0_fat_mape_when_label_ge_1g": mape_when(ok_b0, "b0", "fat_g", 1.0),
        "note": (
            "Seeded subset of official rgb_test dishes that have overhead RGB. "
            "Not the full test table. MAPE blows up when the label is near zero; "
            "use MAE and median absolute error for fat."
        ),
    }
    SUMMARY.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"b1_energy": summary["b1"]["energy_kcal"], "b0_energy": summary["b0"]["energy_kcal"]}, indent=2))
    print(f"wrote {SUMMARY.relative_to(ROOT)}", flush=True)
    if len(ok_b1) == 0 or len(ok_b0) == 0:
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
