"""Train-set gate, then one official holdout evaluation."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import torch

from digi_nutri.bootstrap import contrast_ci
from digi_nutri.data import make_records
from digi_nutri.engine import predict
from digi_nutri.metadata import load_dish_table
from digi_nutri.metrics import summarize
from digi_nutri.prepare import assert_config_matches_protocol
from digi_nutri.protocol import (
    HEADLINE_METRICS,
    HOLDOUT_METRICS,
    META_DIR,
    MODELS,
    NUTRIENTS,
    SPLIT_DIR,
    TEACHERS,
    TRAIN_METRICS,
)
from digi_nutri.reshoot import plate_groups
from digi_nutri.splits import read_id_file
from digi_nutri.usda import load_usda
from digi_nutri.verdict import q0_verdict, train_gate


def _ids(split: str, config: dict) -> list[str]:
    filename = "depth_test_ids.txt" if split == "holdout" else "depth_train_ids.txt"
    ids = read_id_file(Path(SPLIT_DIR) / filename)
    if split == "train":
        excluded = {row["dish_id"] for row in config["train_exclusions"]}
        ids = [dish_id for dish_id in ids if dish_id not in excluded]
        if len(ids) != int(config["n_train"]):
            raise RuntimeError("train eval count does not match the frozen config")
    elif len(ids) != int(config["n_test"]):
        raise RuntimeError("holdout count does not match the frozen config")
    return ids


def _records(kind: str, dish_ids: list[str], dishes: dict, usda, vocab) -> list[dict]:
    groups = plate_groups(dish_ids, dishes)
    if kind == "t_ing":
        return make_records(kind, dish_ids, dishes, groups, usda=usda, vocab=vocab)
    return make_records(kind, dish_ids, dishes, groups)


def _write_predictions(path: Path, rows: dict[str, dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    base = rows["b1"]
    with path.open("w", newline="") as handle:
        writer = csv.writer(handle)
        header = ["dish_id", "group_id", "mass"] + [f"{name}_gt" for name in NUTRIENTS]
        for model in MODELS:
            header.append(f"{model}_mass_hat")
            header.extend(f"{model}_{name}_hat" for name in NUTRIENTS)
        writer.writerow(header)
        for index, dish_id in enumerate(base["dish_id"]):
            row = [dish_id, base["group_id"][index], f"{float(base['mass'][index]):.8f}"]
            row.extend(f"{float(base['nutrition'][index, nutrient]):.8f}" for nutrient in range(4))
            for model in MODELS:
                other = rows[model]
                if other["dish_id"][index] != dish_id:
                    raise RuntimeError("prediction rows are not aligned on dish_id")
                row.append(f"{float(other['mass_hat'][index]):.8f}")
                row.extend(f"{float(other['yhat'][index, nutrient]):.8f}" for nutrient in range(4))
            writer.writerow(row)


def _predict_all(config: dict, split: str, device: torch.device) -> dict[str, dict]:
    meta = Path(META_DIR)
    dishes = load_dish_table(meta / "dish_metadata_cafe1.csv", meta / "dish_metadata_cafe2.csv")
    dish_ids = _ids(split, config)
    saved_groups = json.loads(Path("results/v2_q0/plate_groups_test.json").read_text())
    usda = None
    vocab = None
    rows = {}
    for kind in MODELS:
        if kind == "t_ing":
            usda = load_usda(meta / "ingredients_metadata.csv")
            vocab = json.loads(Path("configs/ingredient_vocab.json").read_text())
        checkpoint = Path(config["output_dir"]) / "checkpoints" / f"{kind}.pt"
        if not checkpoint.exists():
            raise RuntimeError(f"missing checkpoint for {kind}")
        prediction = predict(
            kind,
            config,
            _records(kind, dish_ids, dishes, usda, vocab),
            checkpoint,
            device,
        )
        if split == "holdout":
            for dish_id, group in zip(prediction["dish_id"], prediction["group_id"]):
                if saved_groups[str(dish_id)] != group:
                    raise RuntimeError("holdout plate groups do not match the frozen map")
        rows[kind] = prediction
        torch.cuda.empty_cache()
    return rows


def _column(rows: dict, model: str, nutrient: str):
    if nutrient == "mass":
        return rows[model]["mass_hat"], rows["b1"]["mass"]
    index = list(NUTRIENTS).index(nutrient)
    return rows[model]["yhat"][:, index], rows["b1"]["nutrition"][:, index]


def _contrasts(rows: dict) -> dict:
    groups = [str(group) for group in rows["b1"]["group_id"]]
    contrasts = {}
    for teacher in list(TEACHERS):
        contrasts[teacher] = {}
        for nutrient in list(NUTRIENTS) + ["mass"]:
            method_pred, target = _column(rows, teacher, nutrient)
            base_pred, _same_target = _column(rows, "b1", nutrient)
            contrasts[teacher][nutrient] = {
                metric: contrast_ci(method_pred, base_pred, target, groups, metric)
                for metric in HEADLINE_METRICS
            }
    return contrasts


def evaluate_split(config: dict, split: str, device: torch.device) -> dict:
    rows = _predict_all(config, split, device)
    summary = {model: summarize(rows[model]["yhat"], rows[model]["nutrition"]) for model in MODELS}
    for model in MODELS:
        summary[model]["mass"] = summarize(
            rows[model]["mass_hat"].reshape(-1, 1),
            rows[model]["mass"].reshape(-1, 1),
            columns=("mass",),
        )["mass"]
    payload = {
        "split": "official_depth_test" if split == "holdout" else "official_depth_train",
        "n": int(len(rows["b1"]["dish_id"])),
        "n_groups": len(set(str(group) for group in rows["b1"]["group_id"])),
        "metrics": summary,
    }
    _write_predictions(Path("results/v2_q0") / f"predictions_{split}.csv", rows)
    if split == "holdout":
        payload["contrasts"] = _contrasts(rows)
        verdict, failures = q0_verdict(payload["contrasts"])
        payload["verdict"] = verdict
        payload["failures"] = failures
        payload["holdout_evaluated"] = True
    else:
        mae = {
            model: {nutrient: summary[model][nutrient]["mae"] for nutrient in NUTRIENTS}
            for model in MODELS
        }
        passed, failures = train_gate(mae)
        payload["train_gate_pass"] = passed
        payload["train_gate_failures"] = failures
        payload["holdout_evaluated"] = False
    return payload


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=Path("configs/q0_frozen.json"))
    parser.add_argument("--split", required=True, choices=("train", "holdout"))
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()
    config = json.loads(args.config.read_text())
    assert_config_matches_protocol(config)
    holdout_path = Path(HOLDOUT_METRICS)
    if args.split == "holdout" and holdout_path.exists():
        raise SystemExit("holdout_metrics.json already exists; refusing a second eval")
    if args.split == "holdout" and not Path(TRAIN_METRICS).exists():
        raise SystemExit("train metrics are required before the holdout is opened")
    if args.split == "holdout":
        train_payload = json.loads(Path(TRAIN_METRICS).read_text())
        if not train_payload.get("train_gate_pass"):
            raise SystemExit("train gate did not pass; refusing the holdout")
    payload = evaluate_split(config, args.split, torch.device(args.device))
    target = holdout_path if args.split == "holdout" else Path(TRAIN_METRICS)
    temporary = target.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(payload, indent=2) + "\n")
    temporary.replace(target)
    if args.split == "holdout":
        Path("results/v2_q0/verdict.json").write_text(
            json.dumps(
                {
                    "verdict": payload["verdict"],
                    "failures": payload["failures"],
                    "holdout_evaluated": True,
                },
                indent=2,
            )
            + "\n"
        )
    print(payload.get("verdict", payload.get("train_gate_pass")), flush=True)


if __name__ == "__main__":
    main()
