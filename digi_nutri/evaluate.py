"""Train metrics, then one official holdout against this run's B1."""

from __future__ import annotations

import csv
import json
from pathlib import Path

import torch

from digi_nutri.bootstrap import contrast_ci
from digi_nutri.data import make_records
from digi_nutri.engine import config_digest, predict
from digi_nutri.metadata import load_dish_table
from digi_nutri.metrics import summarize
from digi_nutri.prepare import assert_config_matches_protocol
from digi_nutri.protocol import (
    HEADLINE_METRICS,
    HOLDOUT_METRICS,
    META_DIR,
    MODELS,
    NUTRIENTS,
    PLATE_GROUPS_PATH,
    PREDICTIONS_HOLDOUT,
    PREDICTIONS_TRAIN,
    TRAIN_METRICS,
    VOCAB_PATH,
)
from digi_nutri.report import write_report
from digi_nutri.reshoot import plate_groups
from digi_nutri.splits import audit_split, read_id_file
from digi_nutri.protocol import DEPTH_ROOT, RGB_ROOT, SPLIT_DIR
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


def _cohort(config: dict) -> tuple[list[str], list[str]]:
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


def _records(kind: str, dish_ids: list[str], dishes: dict, config: dict, root: Path) -> list[dict]:
    groups = plate_groups(dish_ids, dishes)
    if kind == "t_ing":
        usda = load_usda(Path(META_DIR) / "ingredients_metadata.csv")
        vocab = json.loads((root / VOCAB_PATH).read_text())
        return make_records(kind, dish_ids, dishes, groups, usda=usda, vocab=vocab)
    return make_records(kind, dish_ids, dishes, groups)


def _checkpoint(config: dict, kind: str) -> Path:
    path = Path(config["output_dir"]) / "checkpoints" / f"{kind}.pt"
    if not path.exists():
        raise RuntimeError(f"missing checkpoint for {kind}")
    return path


def _summarize_rows(rows: dict) -> dict:
    summary = summarize(rows["yhat"], rows["nutrition"])
    summary["mass"] = summarize(
        rows["mass_hat"].reshape(-1, 1),
        rows["mass"].reshape(-1, 1),
        columns=("mass",),
    )["mass"]
    return summary


def _write_predictions(path: Path, rows: dict[str, dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    models = [kind for kind in MODELS if kind in rows]
    base = rows[models[0]]
    with path.open("w", newline="") as handle:
        writer = csv.writer(handle)
        header = ["dish_id", "group_id", "mass"] + [f"{name}_gt" for name in NUTRIENTS]
        for model in models:
            header.append(f"{model}_mass_hat")
            header.extend(f"{model}_{name}_hat" for name in NUTRIENTS)
        writer.writerow(header)
        for index, dish_id in enumerate(base["dish_id"]):
            line = [dish_id, base["group_id"][index], f"{float(base['mass'][index]):.8f}"]
            line.extend(f"{float(base['nutrition'][index, nutrient]):.8f}" for nutrient in range(4))
            for model in models:
                other = rows[model]
                if other["dish_id"][index] != dish_id:
                    raise RuntimeError("prediction rows are not aligned")
                line.append(f"{float(other['mass_hat'][index]):.8f}")
                line.extend(f"{float(other['yhat'][index, nutrient]):.8f}" for nutrient in range(4))
            writer.writerow(line)


def _column(rows: dict, model: str, nutrient: str):
    if nutrient == "mass":
        return rows[model]["mass_hat"], rows["b1"]["mass"]
    index = list(NUTRIENTS).index(nutrient)
    return rows[model]["yhat"][:, index], rows["b1"]["nutrition"][:, index]


def _mae_block(summary: dict) -> dict:
    block = {name: summary[name]["mae"] for name in NUTRIENTS}
    block["mass"] = summary["mass"]["mae"]
    return block


def evaluate_train(config: dict, device: torch.device, root: Path, config_path: Path) -> dict:
    holdout_path = root / HOLDOUT_METRICS
    if holdout_path.exists():
        raise SystemExit("holdout metrics already exist; not recomputing train metrics")
    train_ids, test_ids = _cohort(config)
    dishes = load_dish_table(
        Path(META_DIR) / "dish_metadata_cafe1.csv",
        Path(META_DIR) / "dish_metadata_cafe2.csv",
    )
    rows = {}
    summaries = {}
    for kind in MODELS:
        rows[kind] = predict(kind, config, _records(kind, train_ids, dishes, config, root), _checkpoint(config, kind), device)
        if set(rows[kind]["dish_id"].tolist()) & set(test_ids):
            raise RuntimeError(f"{kind} train evaluation included an official test id")
        if len(rows[kind]["dish_id"]) != int(config["n_train"]):
            raise RuntimeError(f"{kind} train evaluation did not cover the kept training plates")
        summaries[kind] = _summarize_rows(rows[kind])
        if device.type == "cuda":
            torch.cuda.empty_cache()
    payload = {
        "split": "official_depth_train",
        "n": int(config["n_train"]),
        "eval_mode": True,
        "augmentation": False,
        "role": "train metrics only; not a Q0 gate",
        "config_sha256": config_digest(config_path),
        "param_counts": config["param_counts"],
        "metrics": summaries,
        "mae": {kind: _mae_block(summaries[kind]) for kind in MODELS},
        "yhat_is_predicted_product": True,
    }
    _write_predictions(root / PREDICTIONS_TRAIN, rows)
    _write_json(root / TRAIN_METRICS, payload)
    write_report(root)
    print("train metrics recorded", flush=True)
    return payload


def evaluate_holdout(config: dict, device: torch.device, root: Path, config_path: Path) -> dict:
    holdout_path = root / HOLDOUT_METRICS
    train_path = root / TRAIN_METRICS
    reason = holdout_block_reason(holdout_path.exists(), train_path.exists())
    if reason:
        raise SystemExit(reason)
    _train_ids, test_ids = _cohort(config)
    dishes = load_dish_table(
        Path(META_DIR) / "dish_metadata_cafe1.csv",
        Path(META_DIR) / "dish_metadata_cafe2.csv",
    )
    groups = plate_groups(test_ids, dishes)
    saved = json.loads((root / PLATE_GROUPS_PATH).read_text())
    if saved != groups:
        raise RuntimeError("holdout plate groups do not match the frozen map")
    rows = {}
    summaries = {}
    for kind in MODELS:
        rows[kind] = predict(kind, config, _records(kind, test_ids, dishes, config, root), _checkpoint(config, kind), device)
        if rows[kind]["dish_id"].tolist() != test_ids:
            raise RuntimeError(f"{kind} holdout rows are not in official test order")
        if rows[kind]["group_id"].tolist() != [groups[dish_id] for dish_id in test_ids]:
            raise RuntimeError(f"{kind} holdout groups do not match the frozen map")
        summaries[kind] = _summarize_rows(rows[kind])
        if device.type == "cuda":
            torch.cuda.empty_cache()
    contrasts = {}
    group_ids = [str(group) for group in rows["b1"]["group_id"]]
    intervals = []
    for teacher in ("t_geo", "t_ing"):
        contrasts[teacher] = {}
        for nutrient in list(NUTRIENTS) + ["mass"]:
            method_pred, target = _column(rows, teacher, nutrient)
            base_pred, _same = _column(rows, "b1", nutrient)
            contrasts[teacher][nutrient] = {
                metric: contrast_ci(method_pred, base_pred, target, group_ids, metric)
                for metric in HEADLINE_METRICS
            }
            if nutrient == "mass":
                continue
            for metric in HEADLINE_METRICS:
                row = contrasts[teacher][nutrient][metric]
                intervals.append(
                    {
                        "teacher": teacher,
                        "nutrient": nutrient,
                        "metric": metric,
                        "lo": row["lo"],
                        "hi": row["hi"],
                        "below_zero": bool(float(row["hi"]) < 0.0),
                    }
                )
    if len(intervals) != 24:
        raise RuntimeError(f"expected 24 headline intervals, found {len(intervals)}")
    verdict, failures = q0_verdict(contrasts)
    payload = {
        "split": "official_depth_test",
        "n": len(test_ids),
        "n_groups": len(set(groups.values())),
        "eval_mode": True,
        "augmentation": False,
        "comparator": "b1 trained in this same-setting run",
        "config_sha256": config_digest(config_path),
        "param_counts": config["param_counts"],
        "metrics": summaries,
        "mae": {kind: _mae_block(summaries[kind]) for kind in MODELS},
        "contrasts": contrasts,
        "intervals": intervals,
        "n_intervals": 24,
        "n_intervals_below_zero": sum(1 for row in intervals if row["below_zero"]),
        "verdict": verdict,
        "failures": failures,
        "holdout_evaluated_once": True,
        "distillation_started": False,
    }
    _write_predictions(root / PREDICTIONS_HOLDOUT, rows)
    _write_json(holdout_path, payload)
    write_report(root)
    print(verdict, flush=True)
    return payload
