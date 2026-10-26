"""D6: VLM weight_guess_g against the method used on each hold-out set."""
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
for entry in (str(ROOT), str(ROOT / "src")):
    if entry not in sys.path:
        sys.path.insert(0, entry)

from experiments.vlm.geometry import load_coin_table
from experiments.vlm.prior import NamePrior
from experiments.vlm.scoring import block, clean, wape_gap
from experiments.vlm.store import load_record, read_manifest
from mmcfa.paths import Paths
from mmcfa.runtime import atomic_write_json, log, read_json


def _points(paths: Paths) -> dict[tuple[str, str], dict]:
    path = paths.root / "results" / "vlm" / "points.jsonl"
    table = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            row = json.loads(line)
            table[(row["dataset"], row["sample_id"])] = row
    return table


def _method(dataset: str, row: dict, point: dict, coins, prior: NamePrior) -> float:
    if dataset == "n5k":
        return float(point["p_ens4"])
    if dataset == "nvreal":
        return float(point["p_ens3"])
    if dataset == "ecustfd":
        factors = coins.factors(row["sample_id"])
        if factors is None:
            return float("nan")
        return coins.predict(row["sample_id"], factors["coin_px"], use_ref=True)
    if dataset == "mf3d":
        return prior.predict_annotated("mf3d", row.get("cat") or "")
    return float("nan")


def main() -> int:
    paths = Paths.resolve()
    manifest = [row for row in read_manifest() if row.get("image")]
    points = _points(paths)
    coins = load_coin_table(paths)
    cache = read_json(paths.results_action / "a2_name_prior" / "category_cache.json", {})
    prior = NamePrior(paths, cache)
    names = {
        "n5k": "4ch ensemble",
        "nvreal": "3ch ensemble zero-shot",
        "ecustfd": "annotated coin geometry",
        "mf3d": "annotated name prior tau=4",
    }
    report = {"by_dataset": {}}
    for dataset in ("n5k", "ecustfd", "mf3d", "nvreal"):
        rows = [row for row in manifest if row["dataset"] == dataset]
        y, vlm, method = [], [], []
        missing = 0
        for row in rows:
            record = load_record(dataset, row["sample_id"])
            point = points.get((dataset, row["sample_id"]))
            if record is None or point is None:
                missing += 1
                continue
            y.append(float(row["y_g"]))
            vlm.append(float(record["weight_guess_g"]))
            method.append(_method(dataset, row, point, coins, prior))
        vlm_block = block(y, vlm)
        method_block = block(y, method)
        worse = None
        if vlm_block["n"] and method_block["wape"] is not None and vlm_block["wape"] is not None:
            worse = bool(vlm_block["wape"] > method_block["wape"])
        report["by_dataset"][dataset] = {
            "n_manifest": len(rows),
            "n_missing_cache": missing,
            "method": names[dataset],
            "vlm": vlm_block,
            "method_metrics": method_block,
            "wape_vlm_minus_method": wape_gap(y, method, vlm),
            "vlm_worse_than_method": worse,
        }
        log(
            f"D6-VLM {dataset} n={vlm_block['n']} "
            f"vlm_wape={vlm_block['wape']} method_wape={method_block['wape']}"
        )
    dest = paths.root / "backup" / "v3_excluded" / "results" / "diagnosis" / "d6_vlm" / "summary.json"
    atomic_write_json(dest, clean(report))
    log(f"wrote {dest}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
