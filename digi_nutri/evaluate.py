"""Train metrics, then one official holdout against the frozen #91 B1."""

from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np
import torch

from digi_nutri.bootstrap import contrast_ci
from digi_nutri.data import make_records
from digi_nutri.engine import config_digest, predict
from digi_nutri.frozen_b1 import load_frozen_holdout_rows
from digi_nutri.metadata import load_dish_table
from digi_nutri.metrics import summarize
from digi_nutri.prepare import assert_config_matches_protocol
from digi_nutri.protocol import (
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
    return audit.train_ids, audit.test_ids


def records_for(dish_ids: list[str], dishes: dict, config: dict, root: Path) -> list[dict]:
    groups = plate_groups(dish_ids, dishes)
    usda = load_usda(Path(META_DIR) / "ingredients_metadata.csv")
    vocab = json.loads((root / VOCAB_PATH).read_text())
    return make_records(dish_ids, dishes, groups, usda, vocab)


def _checkpoint(config: dict) -> Path:
    path = Path(config["output_dir"]) / "checkpoints" / "t_multi.pt"
    if not path.exists():
        raise RuntimeError("missing T_multi checkpoint")
    return path


def _mae_block(summary: dict) -> dict:
    block = {name: summary[name]["mae"] for name in NUTRIENTS}
    block["mass"] = summary["mass"]["mae"]
    return block


def _summarize_rows(rows: dict) -> dict:
    summary = summarize(rows["yhat"], rows["nutrition"])
    summary["mass"] = summarize(
        rows["mass_hat"].reshape(-1, 1),
        rows["mass"].reshape(-1, 1),
        columns=("mass",),
    )["mass"]
    return summary


def _align_frozen(frozen: dict, dish_ids: list[str], groups: dict[str, str]) -> dict:
    order = {dish_id: index for index, dish_id in enumerate(frozen["dish_id"].tolist())}
    if set(order) != set(dish_ids):
        raise RuntimeError("frozen B1 holdout ids do not match the official test list")
    index = [order[dish_id] for dish_id in dish_ids]
    aligned = {
        "dish_id": np.asarray(dish_ids),
        "group_id": np.asarray([groups[dish_id] for dish_id in dish_ids]),
        "mass": frozen["mass"][index],
        "nutrition": frozen["nutrition"][index],
        "mass_hat": frozen["mass_hat"][index],
        "yhat": frozen["yhat"][index],
    }
    file_groups = frozen["group_id"][index].tolist()
    if file_groups != aligned["group_id"].tolist():
        raise RuntimeError("frozen B1 group ids do not match the frozen plate map")
    return aligned


def _write_predictions(path: Path, model_rows: dict, frozen_rows: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.writer(handle)
        header = ["dish_id", "group_id", "mass"] + [f"{name}_gt" for name in NUTRIENTS]
        header.append("b1_mass_hat")
        header.extend(f"b1_{name}_hat" for name in NUTRIENTS)
        header.append(f"{MODEL}_mass_hat")
        header.extend(f"{MODEL}_{name}_hat" for name in NUTRIENTS)
        writer.writerow(header)
        for index, dish_id in enumerate(model_rows["dish_id"]):
            if frozen_rows["dish_id"][index] != dish_id:
                raise RuntimeError("prediction rows are not aligned")
            line = [dish_id, model_rows["group_id"][index], f"{float(model_rows['mass'][index]):.8f}"]
            line.extend(f"{float(model_rows['nutrition'][index, nutrient]):.8f}" for nutrient in range(4))
            line.append(f"{float(frozen_rows['mass_hat'][index]):.8f}")
            line.extend(f"{float(frozen_rows['yhat'][index, nutrient]):.8f}" for nutrient in range(4))
            line.append(f"{float(model_rows['mass_hat'][index]):.8f}")
            line.extend(f"{float(model_rows['yhat'][index, nutrient]):.8f}" for nutrient in range(4))
            writer.writerow(line)


def _column(rows: dict, nutrient: str):
    if nutrient == "mass":
        return rows["mass_hat"], rows["mass"]
    index = list(NUTRIENTS).index(nutrient)
    return rows["yhat"][:, index], rows["nutrition"][:, index]


def _write_train_predictions(path: Path, rows: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.writer(handle)
        header = ["dish_id", "group_id", "mass"] + [f"{name}_gt" for name in NUTRIENTS]
        header.append(f"{MODEL}_mass_hat")
        header.extend(f"{MODEL}_{name}_hat" for name in NUTRIENTS)
        writer.writerow(header)
        for index, dish_id in enumerate(rows["dish_id"]):
            line = [dish_id, rows["group_id"][index], f"{float(rows['mass'][index]):.8f}"]
            line.extend(f"{float(rows['nutrition'][index, nutrient]):.8f}" for nutrient in range(4))
            line.append(f"{float(rows['mass_hat'][index]):.8f}")
            line.extend(f"{float(rows['yhat'][index, nutrient]):.8f}" for nutrient in range(4))
            writer.writerow(line)


def evaluate_train(config: dict, device: torch.device, root: Path, config_path: Path) -> dict:
    holdout_path = root / HOLDOUT_METRICS
    if holdout_path.exists():
        raise SystemExit("holdout metrics already exist; not recomputing train metrics")
    train_ids, test_ids = cohort(config)
    dishes = load_dish_table(
        Path(META_DIR) / "dish_metadata_cafe1.csv",
        Path(META_DIR) / "dish_metadata_cafe2.csv",
    )
    rows = predict(config, records_for(train_ids, dishes, config, root), _checkpoint(config), device)
    if set(rows["dish_id"].tolist()) & set(test_ids):
        raise RuntimeError("train evaluation included an official test id")
    if len(rows["dish_id"]) != int(config["n_train"]):
        raise RuntimeError("train evaluation did not cover the kept training plates")
    summary = _summarize_rows(rows)
    _write_train_predictions(root / PREDICTIONS_TRAIN, rows)
    payload = {
        "split": "official_depth_train",
        "n": int(config["n_train"]),
        "eval_mode": True,
        "augmentation": False,
        "role": "train metrics only; not a Q0 gate",
        "config_sha256": config_digest(config_path),
        "param_counts": config["param_counts"],
        "metrics": {MODEL: summary},
        "mae": {MODEL: _mae_block(summary)},
        "frozen_b1_mae": config["frozen_b1_train_mae"],
        "yhat_is_predicted_product": True,
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
    _train_ids, test_ids = cohort(config)
    dishes = load_dish_table(
        Path(META_DIR) / "dish_metadata_cafe1.csv",
        Path(META_DIR) / "dish_metadata_cafe2.csv",
    )
    groups = plate_groups(test_ids, dishes)
    saved = json.loads((root / PLATE_GROUPS_PATH).read_text())
    if saved != groups:
        raise RuntimeError("holdout plate groups do not match the frozen map")
    rows = predict(config, records_for(test_ids, dishes, config, root), _checkpoint(config), device)
    if rows["dish_id"].tolist() != test_ids:
        raise RuntimeError("holdout rows are not in official test order")
    if rows["group_id"].tolist() != [groups[dish_id] for dish_id in test_ids]:
        raise RuntimeError("holdout groups do not match the frozen map")
    frozen = _align_frozen(load_frozen_holdout_rows(), test_ids, groups)
    if not np.allclose(rows["nutrition"], frozen["nutrition"], rtol=1e-4, atol=1e-4):
        raise RuntimeError("holdout nutrition labels do not match the frozen B1 file")
    if not np.allclose(rows["mass"], frozen["mass"], rtol=1e-4, atol=1e-4):
        raise RuntimeError("holdout mass labels do not match the frozen B1 file")
    summary = _summarize_rows(rows)
    frozen_summary = _summarize_rows(frozen)
    contrasts = {MODEL: {}}
    intervals = []
    group_ids = [str(group) for group in rows["group_id"]]
    for nutrient in list(NUTRIENTS) + ["mass"]:
        method_pred, target = _column(rows, nutrient)
        base_pred, _same = _column(frozen, nutrient)
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
    verdict, failures = q0_verdict(contrasts)
    payload = {
        "split": "official_depth_test",
        "n": len(test_ids),
        "n_groups": len(set(groups.values())),
        "eval_mode": True,
        "augmentation": False,
        "comparator": "frozen #91 same-setting B1",
        "config_sha256": config_digest(config_path),
        "param_counts": config["param_counts"],
        "metrics": {MODEL: summary},
        "frozen_b1_metrics": frozen_summary,
        "mae": {MODEL: _mae_block(summary)},
        "frozen_b1_mae": _mae_block(frozen_summary),
        "contrasts": contrasts,
        "intervals": intervals,
        "mass_intervals": mass_intervals,
        "n_intervals": N_HEADLINE_INTERVALS,
        "n_intervals_below_zero": sum(1 for row in intervals if row["below_zero"]),
        "verdict": verdict,
        "failures": failures,
        "holdout_evaluated_once": True,
        "distillation_started": False,
    }
    _write_predictions(root / PREDICTIONS_HOLDOUT, rows, frozen)
    _write_json(holdout_path, payload)
    write_report(root)
    print(verdict, flush=True)
    return payload
