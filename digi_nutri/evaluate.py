"""Train-set gate, then one official holdout evaluation if the gate passed."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import torch

from digi_nutri.bootstrap import contrast_ci
from digi_nutri.cohort import load_cohort
from digi_nutri.data import make_records
from digi_nutri.engine import predict
from digi_nutri.metadata import load_dish_table
from digi_nutri.metrics import summarize
from digi_nutri.protocol import (
    HEADLINE_METRICS,
    HOLDOUT_METRICS,
    META_DIR,
    NUTRIENTS,
    PLATE_GROUPS_PATH,
    TRAIN_FAIL,
    TRAIN_METRICS,
    TRAIN_PASS,
    VOCAB_PATH,
)
from digi_nutri.report import write_report
from digi_nutri.reshoot import plate_groups
from digi_nutri.train import load_config
from digi_nutri.usda import load_usda
from digi_nutri.verdict import holdout_block_reason, q0_verdict, tgeo_train_gate


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(payload, indent=2) + "\n")
    temporary.replace(path)


def _records(kind: str, dish_ids: list[str], dishes: dict, config: dict):
    groups = plate_groups(dish_ids, dishes)
    if kind == "t_ing":
        usda = load_usda(Path(META_DIR) / "ingredients_metadata.csv")
        vocab = json.loads(Path(VOCAB_PATH).read_text())
        return make_records(kind, dish_ids, dishes, groups, usda=usda, vocab=vocab)
    return make_records(kind, dish_ids, dishes, groups)


def _mae_block(summary: dict) -> dict:
    block = {name: summary[name]["mae"] for name in NUTRIENTS}
    block["mass"] = summary["mass"]["mae"]
    return block


def evaluate_train(config: dict, device: torch.device, root: Path) -> dict:
    train_ids, test_ids = load_cohort(config)
    dishes = load_dish_table(
        Path(META_DIR) / "dish_metadata_cafe1.csv",
        Path(META_DIR) / "dish_metadata_cafe2.csv",
    )
    rows = predict("t_geo", config, _records("t_geo", train_ids, dishes, config), device)
    if set(rows["dish_id"].tolist()) & set(test_ids):
        raise RuntimeError("train evaluation included an official test id")
    if len(rows["dish_id"]) != int(config["n_train"]):
        raise RuntimeError("train evaluation did not cover the kept training plates")
    summary = summarize(rows["yhat"], rows["nutrition"])
    summary["mass"] = summarize(
        rows["mass_hat"].reshape(-1, 1),
        rows["mass"].reshape(-1, 1),
        columns=("mass",),
    )["mass"]
    mae = {name: summary[name]["mae"] for name in NUTRIENTS}
    passed, failures = tgeo_train_gate(mae)
    payload = {
        "split": "official_depth_train",
        "n": int(len(rows["dish_id"])),
        "eval_mode": True,
        "augmentation": False,
        "model": "t_geo_plan_a",
        "batch_size": int(config["batch_size"]),
        "param_count": int(config["param_counts"]["t_geo"]),
        "apply_d_table_zero_exclusion": bool(config["apply_d_table_zero_exclusion"]),
        "metrics": summary,
        "mae": _mae_block(summary),
        "train_gate_pass": passed,
        "train_gate_failures": failures,
        "verdict": TRAIN_PASS if passed else TRAIN_FAIL,
        "holdout_evaluated": False,
    }
    _write_predictions(root / "results" / "plan_a" / "predictions_train.csv", {"t_geo": rows})
    _write_json(root / TRAIN_METRICS, payload)
    write_report(root)
    print(payload["verdict"], flush=True)
    return payload


def _write_predictions(path: Path, rows: dict[str, dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    models = list(rows)
    base = rows[models[0]]
    with path.open("w", newline="") as handle:
        writer = csv.writer(handle)
        header = ["dish_id", "group_id", "mass"] + [f"{name}_gt" for name in NUTRIENTS]
        for model in models:
            header.append(f"{model}_mass_hat")
            header.extend(f"{model}_{name}_hat" for name in NUTRIENTS)
        writer.writerow(header)
        for index, dish_id in enumerate(base["dish_id"]):
            row = [dish_id, base["group_id"][index], f"{float(base['mass'][index]):.8f}"]
            row.extend(f"{float(base['nutrition'][index, nutrient]):.8f}" for nutrient in range(4))
            for model in models:
                other = rows[model]
                if other["dish_id"][index] != dish_id:
                    raise RuntimeError("prediction rows are not aligned")
                row.append(f"{float(other['mass_hat'][index]):.8f}")
                row.extend(f"{float(other['yhat'][index, nutrient]):.8f}" for nutrient in range(4))
            writer.writerow(row)


def _column(rows: dict, model: str, nutrient: str):
    if nutrient == "mass":
        return rows[model]["mass_hat"], rows["b1"]["mass"]
    index = list(NUTRIENTS).index(nutrient)
    return rows[model]["yhat"][:, index], rows["b1"]["nutrition"][:, index]


def evaluate_holdout(config: dict, device: torch.device, root: Path) -> dict:
    holdout_path = root / HOLDOUT_METRICS
    train_path = root / TRAIN_METRICS
    reason = holdout_block_reason(
        train_gate_pass=bool(json.loads(train_path.read_text()).get("train_gate_pass"))
        if train_path.exists()
        else False,
        holdout_exists=holdout_path.exists(),
        train_metrics_exist=train_path.exists(),
    )
    if reason:
        raise SystemExit(reason)
    train_ids, test_ids = load_cohort(config)
    del train_ids
    dishes = load_dish_table(
        Path(META_DIR) / "dish_metadata_cafe1.csv",
        Path(META_DIR) / "dish_metadata_cafe2.csv",
    )
    groups = plate_groups(test_ids, dishes)
    saved = json.loads((root / PLATE_GROUPS_PATH).read_text())
    if saved != groups:
        raise RuntimeError("holdout plate groups do not match the frozen map")
    rows = {}
    for kind in ("b1", "t_ing", "t_geo"):
        rows[kind] = predict(kind, config, _records(kind, test_ids, dishes, config), device)
        if rows[kind]["dish_id"].tolist() != test_ids:
            raise RuntimeError(f"{kind} holdout rows are not in official test order")
        torch.cuda.empty_cache()
    summary = {}
    for kind in rows:
        summary[kind] = summarize(rows[kind]["yhat"], rows[kind]["nutrition"])
        summary[kind]["mass"] = summarize(
            rows[kind]["mass_hat"].reshape(-1, 1),
            rows[kind]["mass"].reshape(-1, 1),
            columns=("mass",),
        )["mass"]
    contrasts = {}
    group_ids = [str(group) for group in rows["b1"]["group_id"]]
    for teacher in ("t_geo", "t_ing"):
        contrasts[teacher] = {}
        for nutrient in list(NUTRIENTS) + ["mass"]:
            method_pred, target = _column(rows, teacher, nutrient)
            base_pred, _same = _column(rows, "b1", nutrient)
            contrasts[teacher][nutrient] = {
                metric: contrast_ci(method_pred, base_pred, target, group_ids, metric)
                for metric in HEADLINE_METRICS
            }
    verdict, failures = q0_verdict(contrasts)
    payload = {
        "split": "official_depth_test",
        "n": len(test_ids),
        "n_groups": len(set(groups.values())),
        "metrics": summary,
        "contrasts": contrasts,
        "verdict": verdict,
        "failures": failures,
        "holdout_evaluated": True,
    }
    _write_predictions(root / "results" / "plan_a" / "predictions_holdout.csv", rows)
    _write_json(holdout_path, payload)
    write_report(root)
    print(verdict, flush=True)
    return payload


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=Path("configs/plan_a_frozen.json"))
    parser.add_argument("--split", required=True, choices=("train", "holdout"))
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--root", type=Path, default=Path("."))
    args = parser.parse_args()
    config = load_config(args.config)
    if args.split == "holdout":
        evaluate_holdout(config, torch.device(args.device), args.root)
    else:
        evaluate_train(config, torch.device(args.device), args.root)


if __name__ == "__main__":
    main()
