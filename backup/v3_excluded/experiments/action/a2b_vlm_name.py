"""A2b: replace the annotated category with the VLM food name. Tau stays at 4."""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
for entry in (str(ROOT), str(ROOT / "src")):
    if entry not in sys.path:
        sys.path.insert(0, entry)

import numpy as np

from experiments.vlm.prior import TAU, NamePrior
from experiments.vlm.scoring import block, clean, wape_gap
from experiments.vlm.store import load_record, read_manifest
from mmcfa.paths import Paths
from mmcfa.runtime import atomic_write_json, log, read_json


def _one(dataset: str, rows: list[dict], prior: NamePrior) -> dict:
    y, vlm_p, ann_p, emp_p = [], [], [], []
    wrong_y, wrong_p = [], []
    exact = synonym = wrong = missing = 0
    for row in rows:
        record = load_record(dataset, row["sample_id"])
        if record is None:
            missing += 1
            continue
        pred, matched, kind = prior.predict_vlm_name(dataset, record["food_name"], TAU)
        annotated = prior.predict_annotated(dataset, row["cat"], TAU)
        stats = prior.by_dataset[dataset]["stats"]
        empirical = float(stats["median"].get(row["cat"], stats["global"]))
        y.append(float(row["y_g"]))
        vlm_p.append(pred)
        ann_p.append(annotated)
        emp_p.append(empirical)
        if kind == "exact" and matched == row["cat"]:
            exact += 1
        elif matched == row["cat"]:
            synonym += 1
        else:
            wrong += 1
            wrong_y.append(float(row["y_g"]))
            wrong_p.append(pred)
    n = exact + synonym + wrong
    vlm_block = block(y, vlm_p)
    empirical = block(y, emp_p)
    gap = wape_gap(y, vlm_p, emp_p)
    clearly = False
    if gap.get("ci95") and gap["ci95"][0] > 0 and gap.get("wape_b_minus_a", 0) >= 0.02:
        clearly = True
    return {
        "n": n,
        "n_missing_cache": missing,
        "exact": exact,
        "synonym": synonym,
        "wrong_or_unmatched": wrong,
        "exact_rate": exact / n if n else None,
        "synonym_or_exact_rate": (exact + synonym) / n if n else None,
        "tau": TAU,
        "vlm_name_tau4": vlm_block,
        "annotated_name_tau4": block(y, ann_p),
        "category_median": empirical,
        "wrong_name_only": block(wrong_y, wrong_p),
        "wape_median_minus_vlm": gap,
        "clearly_below_category_median": clearly,
    }


def main() -> int:
    paths = Paths.resolve()
    cache = read_json(paths.results_action / "a2_name_prior" / "category_cache.json", {})
    prior = NamePrior(paths, cache)
    manifest = read_manifest()
    report = {}
    for dataset in ("ecustfd", "mf3d"):
        rows = [row for row in manifest if row["dataset"] == dataset]
        report[dataset] = _one(dataset, rows, prior)
        block_ = report[dataset]
        log(
            f"A2b {dataset} hit={block_['synonym_or_exact_rate']} "
            f"vlm_wape={block_['vlm_name_tau4']['wape']} "
            f"median_wape={block_['category_median']['wape']}"
        )
    report["note"] = (
        "Tau is fixed at 4. A name hit is an exact normalized match or a synonym "
        "that resolves to the annotated category. The median comparison uses the "
        "policy-split category median."
    )
    dest = paths.results_action / "a2b_vlm_name" / "report.json"
    atomic_write_json(dest, clean(report))
    log(f"wrote {dest}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
