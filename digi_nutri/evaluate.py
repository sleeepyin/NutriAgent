"""Train metrics for both models, then one official holdout against the new B1."""

from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np
import torch

from digi_nutri.bootstrap import contrast_ci
from digi_nutri.data import make_records
from digi_nutri.engine import config_digest, predict, sha256_file
from digi_nutri.metadata import load_dish_table
from digi_nutri.metrics import summarize
from digi_nutri.prepare import assert_config_matches_protocol
from digi_nutri.protocol import (
    B1_MODEL,
    DEPTH_ROOT,
    HEADLINE_METRICS,
    HOLDOUT_METRICS,
    META_DIR,
    MODEL,
    NUTRIENTS,
    N_HEADLINE_INTERVALS,
    PLATE_GROUPS_PATH,
    PREDICTIONS_HOLDOUT,
    PREDICTIONS_TRAIN,
    RGB_ROOT,
    SPLIT_DIR,
    TRAIN_METRICS,
    VOCAB_PATH,
)
from digi_nutri.report import write_report
from digi_nutri.reshoot import plate_groups
from digi_nutri.splits import audit_split, read_id_file
from digi_nutri.usda import load_usda
from digi_nutri.verdict import holdout_block_reason, q0_verdict


def load_config(path: Path) -> dict:
    config = json.loads(path.read_text())
    assert_config_matches_protocol(config)
    return config


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2) + "\n")
    temporary.replace(path)


def cohort(config: dict) -> tuple[list[str], list[str]]:
    split = Path(SPLIT_DIR)
    audit = audit_split(
        read_id_file(split / "depth_train_ids.txt"),
        read_id_file(split / "depth_test_ids.txt"),
        RGB_ROOT,
        DEPTH_ROOT,
    )
    if len(audit.train_ids) != int(config["n_train"]) or len(audit.test_ids) != int(config["n_test"]):
        raise RuntimeError("kept split sizes do not match the frozen config")
    if set(audit.train_ids) & set(audit.test_ids):
        raise RuntimeError("kept train ids overlap the official holdout")
    return audit.train_ids, audit.test_ids


def records_for(dish_ids: list[str], dishes: dict, config: dict, root: Path) -> list[dict]:
    del config
    groups = plate_groups(dish_ids, dishes)
    usda = load_usda(Path(META_DIR) / "ingredients_metadata.csv")
    vocab = json.loads((root / VOCAB_PATH).read_text())
    return make_records(dish_ids, dishes, groups, usda, vocab)


def _checkpoint(config: dict, kind: str) -> Path:
    path = Path(config["output_dir"]) / "checkpoints" / f"{kind}.pt"
    if not path.exists():
        raise RuntimeError(f"missing {kind} checkpoint")
    return path


def _mae_block(summary: dict) -> dict:
    block = {name: summary[name]["mae"] for name in NUTRIENTS}
    block["mass"] = summary["mass"]["mae"]
    return block


def _dataset_pmae_block(summary: dict) -> dict:
    block = {name: summary[name]["dataset_pmae"] for name in NUTRIENTS}
    block["mass"] = summary["mass"]["dataset_pmae"]
    return block


def _summarize_rows(rows: dict) -> dict:
    summary = summarize(rows["yhat"], rows["nutrition"])
    summary["mass"] = summarize(
        rows["mass_hat"].reshape(-1, 1),
        rows["mass"].reshape(-1, 1),
        columns=("mass",),
    )["mass"]
    return summary


def _align(rows: dict, dish_ids: list[str], groups: dict[str, str]) -> dict:
    order = {dish_id: index for index, dish_id in enumerate(rows["dish_id"].tolist())}
    if set(order) != set(dish_ids):
        raise RuntimeError("prediction ids do not match the requested plate list")
    index = [order[dish_id] for dish_id in dish_ids]
    aligned = {
        "dish_id": np.asarray(dish_ids),
        "group_id": np.asarray([groups[dish_id] for dish_id in dish_ids]),
        "mass": rows["mass"][index],
        "nutrition": rows["nutrition"][index],
        "mass_hat": rows["mass_hat"][index],
        "yhat": rows["yhat"][index],
        "density_hat": rows["density_hat"][index],
    }
    if "delta_cal" in rows:
        aligned["delta_cal"] = rows["delta_cal"][index]
    if rows["group_id"][index].tolist() != aligned["group_id"].tolist():
        raise RuntimeError("prediction group ids do not match the plate map")
    return aligned


def _write_predictions(path: Path, b1_rows: dict, teacher_rows: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.writer(handle)
        header = ["dish_id", "group_id", "mass"] + [f"{name}_gt" for name in NUTRIENTS]
        header.append("b1_mass_hat")
        header.extend(f"b1_{name}_hat" for name in NUTRIENTS)
        header.append(f"{MODEL}_mass_hat")
        header.extend(f"{MODEL}_{name}_hat" for name in NUTRIENTS)
        header.append(f"{MODEL}_delta_cal")
        writer.writerow(header)
        for index, dish_id in enumerate(teacher_rows["dish_id"]):
            if b1_rows["dish_id"][index] != dish_id:
                raise RuntimeError("prediction rows are not aligned")
            line = [dish_id, teacher_rows["group_id"][index], f"{float(teacher_rows['mass'][index]):.8f}"]
            line.extend(f"{float(teacher_rows['nutrition'][index, nutrient]):.8f}" for nutrient in range(4))
            line.append(f"{float(b1_rows['mass_hat'][index]):.8f}")
            line.extend(f"{float(b1_rows['yhat'][index, nutrient]):.8f}" for nutrient in range(4))
            line.append(f"{float(teacher_rows['mass_hat'][index]):.8f}")
            line.extend(f"{float(teacher_rows['yhat'][index, nutrient]):.8f}" for nutrient in range(4))
            line.append(f"{float(teacher_rows['delta_cal'][index]):.8f}")
            writer.writerow(line)


def _column(rows: dict, nutrient: str):
    if nutrient == "mass":
        return rows["mass_hat"], rows["mass"]
    index = list(NUTRIENTS).index(nutrient)
    return rows["yhat"][:, index], rows["nutrition"][:, index]


def _predict_pair(config: dict, records: list[dict], device: torch.device) -> tuple[dict, dict]:
    b1_rows = predict(B1_MODEL, config, records, _checkpoint(config, B1_MODEL), device)
    teacher_rows = predict(MODEL, config, records, _checkpoint(config, MODEL), device)
    return b1_rows, teacher_rows


def _labels_match(left: dict, right: dict) -> None:
    if not np.allclose(left["nutrition"], right["nutrition"], rtol=1e-4, atol=1e-4):
        raise RuntimeError("the two models do not share nutrition labels")
    if not np.allclose(left["mass"], right["mass"], rtol=1e-4, atol=1e-4):
        raise RuntimeError("the two models do not share mass labels")


def evaluate_train(config: dict, device: torch.device, root: Path, config_path: Path) -> dict:
    holdout_path = root / HOLDOUT_METRICS
    if holdout_path.exists():
        raise SystemExit("holdout metrics already exist; not recomputing train metrics")
    train_ids, test_ids = cohort(config)
    dishes = load_dish_table(
        Path(META_DIR) / "dish_metadata_cafe1.csv",
        Path(META_DIR) / "dish_metadata_cafe2.csv",
    )
    groups = plate_groups(train_ids, dishes)
    b1_raw, teacher_raw = _predict_pair(config, records_for(train_ids, dishes, config, root), device)
    b1_rows = _align(b1_raw, train_ids, groups)
    teacher_rows = _align(teacher_raw, train_ids, groups)
    _labels_match(b1_rows, teacher_rows)
    if set(teacher_rows["dish_id"].tolist()) & set(test_ids):
        raise RuntimeError("train evaluation included an official test id")
    if len(teacher_rows["dish_id"]) != int(config["n_train"]):
        raise RuntimeError("train evaluation did not cover the kept training plates")
    b1_summary = _summarize_rows(b1_rows)
    teacher_summary = _summarize_rows(teacher_rows)
    _write_predictions(root / PREDICTIONS_TRAIN, b1_rows, teacher_rows)
    payload = {
        "split": "official_depth_train",
        "n": int(config["n_train"]),
        "eval_mode": True,
        "augmentation": False,
        "role": "train metrics only; not a Q0 gate",
        "config_sha256": config_digest(config_path),
        "param_counts": config["param_counts"],
        "metrics": {B1_MODEL: b1_summary, MODEL: teacher_summary},
        "mae": {B1_MODEL: _mae_block(b1_summary), MODEL: _mae_block(teacher_summary)},
        "dataset_pmae": {
            B1_MODEL: _dataset_pmae_block(b1_summary),
            MODEL: _dataset_pmae_block(teacher_summary),
        },
        "b1_checkpoint_sha256": sha256_file(_checkpoint(config, B1_MODEL)),
        "t_multi_checkpoint_sha256": sha256_file(_checkpoint(config, MODEL)),
        "yhat_is_predicted_product": True,
        "distillation_started": False,
    }
    _write_json(root / TRAIN_METRICS, payload)
    write_report(root)
    print("train metrics recorded", flush=True)
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return payload


def evaluate_holdout(config: dict, device: torch.device, root: Path, config_path: Path) -> dict:
    holdout_path = root / HOLDOUT_METRICS
    train_path = root / TRAIN_METRICS
    reason = holdout_block_reason(holdout_path.exists(), train_path.exists())
    if reason:
        raise SystemExit(reason)
    train_ids, test_ids = cohort(config)
    if len(test_ids) != 507 or len(set(test_ids)) != 507:
        raise RuntimeError("the official holdout is not 507 unique plates")
    if set(train_ids) & set(test_ids):
        raise RuntimeError("train and holdout overlap")
    dishes = load_dish_table(
        Path(META_DIR) / "dish_metadata_cafe1.csv",
        Path(META_DIR) / "dish_metadata_cafe2.csv",
    )
    groups = plate_groups(test_ids, dishes)
    saved_groups = json.loads((root / PLATE_GROUPS_PATH).read_text())
    if saved_groups != groups:
        raise RuntimeError("holdout plate groups do not match the frozen map")
    if len(set(groups.values())) != int(config["n_test_groups"]):
        raise RuntimeError("holdout group count does not match the frozen config")
    b1_raw, teacher_raw = _predict_pair(config, records_for(test_ids, dishes, config, root), device)
    b1_rows = _align(b1_raw, test_ids, groups)
    teacher_rows = _align(teacher_raw, test_ids, groups)
    _labels_match(b1_rows, teacher_rows)
    if teacher_rows["dish_id"].tolist() != test_ids:
        raise RuntimeError("holdout rows are not in official test order")
    b1_summary = _summarize_rows(b1_rows)
    teacher_summary = _summarize_rows(teacher_rows)
    contrasts = {MODEL: {}}
    intervals = []
    group_ids = [str(group) for group in teacher_rows["group_id"]]
    for nutrient in list(NUTRIENTS) + ["mass"]:
        method_pred, target = _column(teacher_rows, nutrient)
        base_pred, base_target = _column(b1_rows, nutrient)
        if not np.allclose(target, base_target, rtol=1e-4, atol=1e-4):
            raise RuntimeError(f"{nutrient} labels differ between the two models")
        contrasts[MODEL][nutrient] = {
            metric: contrast_ci(method_pred, base_pred, target, group_ids, metric)
            for metric in HEADLINE_METRICS
        }
        if nutrient == "mass":
            continue
        for metric in HEADLINE_METRICS:
            row = contrasts[MODEL][nutrient][metric]
            intervals.append(
                {
                    "model": MODEL,
                    "nutrient": nutrient,
                    "metric": metric,
                    "lo": row["lo"],
                    "hi": row["hi"],
                    "below_zero": bool(float(row["hi"]) < 0.0),
                }
            )
    if len(intervals) != N_HEADLINE_INTERVALS:
        raise RuntimeError(f"expected {N_HEADLINE_INTERVALS} headline intervals, found {len(intervals)}")
    mass_intervals = []
    for metric in HEADLINE_METRICS:
        row = contrasts[MODEL]["mass"][metric]
        mass_intervals.append(
            {
                "model": MODEL,
                "nutrient": "mass",
                "metric": metric,
                "lo": row["lo"],
                "hi": row["hi"],
                "below_zero": bool(float(row["hi"]) < 0.0),
                "gating": False,
            }
        )
    dataset_pmae = _dataset_pmae_block(teacher_summary)
    verdict, details = q0_verdict(contrasts, dataset_pmae)
    payload = {
        "split": "official_depth_test",
        "n": len(test_ids),
        "n_groups": len(set(groups.values())),
        "train_holdout_overlap": 0,
        "eval_mode": True,
        "augmentation": False,
        "comparator": "this round's ConvNeXt-Small B1",
        "config_sha256": config_digest(config_path),
        "param_counts": config["param_counts"],
        "metrics": {B1_MODEL: b1_summary, MODEL: teacher_summary},
        "mae": {B1_MODEL: _mae_block(b1_summary), MODEL: _mae_block(teacher_summary)},
        "dataset_pmae": {B1_MODEL: _dataset_pmae_block(b1_summary), MODEL: dataset_pmae},
        "contrasts": contrasts,
        "intervals": intervals,
        "mass_intervals": mass_intervals,
        "n_intervals": N_HEADLINE_INTERVALS,
        "n_intervals_below_zero": sum(1 for row in intervals if row["below_zero"]),
        "absolute_gate": details["absolute"],
        "relative_pass": details["relative_pass"],
        "verdict": verdict,
        "failures": details,
        "holdout_evaluated_once": True,
        "distillation_started": False,
        "b1_checkpoint_sha256": sha256_file(_checkpoint(config, B1_MODEL)),
        "t_multi_checkpoint_sha256": sha256_file(_checkpoint(config, MODEL)),
    }
    _write_predictions(root / PREDICTIONS_HOLDOUT, b1_rows, teacher_rows)
    _write_json(holdout_path, payload)
    write_report(root)
    print(verdict, flush=True)
    return payload
