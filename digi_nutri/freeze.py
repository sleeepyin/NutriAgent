"""Write the Plan A config after the parameter count is measured."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from digi_nutri.audit_rule import use_zero_exclusion
from digi_nutri.cohort import load_cohort
from digi_nutri.frozen_teachers import FrozenB1, FrozenTIng
from digi_nutri.ingredients import build_vocab
from digi_nutri.metadata import load_dish_table
from digi_nutri.models import build_plan_a, count_parameters, load_rgb_from_frozen_b1
from digi_nutri.protocol import (
    AUDIT_PATH,
    B1_CHECKPOINT,
    CAB_REDUCTION,
    CACHE_DIR,
    CONFIG_PATH,
    DEPTH_ROOT,
    EPOCHS,
    ETA_MIN,
    FROZEN_B1_PARAMS,
    FROZEN_B1_TRAIN_MAE,
    FROZEN_B1_TRAIN_MASS_MAE,
    FROZEN_SIGMA_MASS,
    FROZEN_SIGMA_MEAN,
    FROZEN_TING_PARAMS,
    FROZEN_WEIGHT_COMMIT,
    GAMMA_INIT,
    HEIGHT_BACKBONE,
    IMAGE_SIZE,
    LEARNING_RATE,
    META_DIR,
    NUTRIENTS,
    PARAM_LIMIT,
    PLATE_GROUPS_PATH,
    RGB_BACKBONE,
    RGB_ROOT,
    SEED,
    STATUS_COMMIT,
    TING_CHECKPOINT,
    VOCAB_PATH,
    WEIGHT_DECAY,
)
from digi_nutri.reshoot import cross_split_pairs, plate_groups
from digi_nutri.usda import load_usda


def _locked() -> dict:
    return {
        "image_size": IMAGE_SIZE,
        "epochs": EPOCHS,
        "lr": LEARNING_RATE,
        "weight_decay": WEIGHT_DECAY,
        "seed": SEED,
        "gamma_init": GAMMA_INIT,
        "cab_reduction": CAB_REDUCTION,
        "rgb_backbone": RGB_BACKBONE,
        "height_backbone": HEIGHT_BACKBONE,
    }


def assert_config_matches_protocol(config: dict) -> None:
    for key, value in _locked().items():
        if config[key] != value:
            raise RuntimeError(f"frozen config {key}={config[key]!r} does not match protocol {value!r}")
    if int(config["param_counts"]["t_geo"]) >= PARAM_LIMIT:
        raise RuntimeError("frozen parameter count is not under 60M")
    if list(config["nutrients"]) != list(NUTRIENTS):
        raise RuntimeError("nutrient order does not match protocol.py")
    if int(config["n_train"]) != 2754 or int(config["n_test"]) != 507:
        raise RuntimeError("split counts do not match the frozen cohort")
    for nutrient, value in FROZEN_B1_TRAIN_MAE.items():
        if float(config["frozen_b1_train_mae"][nutrient]) != float(value):
            raise RuntimeError("frozen B1 train MAE was edited")
    if float(config["frozen_b1_train_mass_mae"]) != float(FROZEN_B1_TRAIN_MASS_MAE):
        raise RuntimeError("frozen B1 mass MAE was edited")


def _sigma(train_ids: list[str], dishes: dict[str, dict]) -> tuple[list[float], float]:
    kept = [dishes[dish_id] for dish_id in train_ids]
    sigma_mean = [float(np.mean([dish["nutrition"][name] for dish in kept])) for name in NUTRIENTS]
    sigma_mass = float(np.mean([dish["mass"] for dish in kept]))
    return sigma_mean, sigma_mass


def _norm_cross(rows: list[dict]) -> list[tuple]:
    return sorted((row["group"], tuple(row["train"]), tuple(row["test"])) for row in rows)


def build_frozen_config(batch_size: int, root: Path) -> dict:
    if batch_size < 1:
        raise RuntimeError("batch size must be positive")
    audit = json.loads((root / AUDIT_PATH).read_text())
    exclude_zeros = use_zero_exclusion(audit)
    draft = {
        "rgb_root": RGB_ROOT,
        "depth_root": DEPTH_ROOT,
        "n_train": 2754,
        "n_test": 507,
        "train_exclusions": audit["exclusions"],
    }
    train_ids, test_ids = load_cohort(draft)
    meta = Path(META_DIR)
    dishes = load_dish_table(meta / "dish_metadata_cafe1.csv", meta / "dish_metadata_cafe2.csv")
    sigma_mean, sigma_mass = _sigma(train_ids, dishes)
    if not np.allclose(sigma_mean, FROZEN_SIGMA_MEAN, rtol=0, atol=1e-6):
        raise RuntimeError(f"sigma_mean changed: {sigma_mean}")
    if abs(sigma_mass - FROZEN_SIGMA_MASS) > 1e-6:
        raise RuntimeError(f"sigma_mass changed: {sigma_mass}")
    sigma_mean = list(FROZEN_SIGMA_MEAN)
    sigma_mass = float(FROZEN_SIGMA_MASS)
    init_density = [value / sigma_mass for value in sigma_mean]
    usda = load_usda(meta / "ingredients_metadata.csv")
    vocab = json.loads((root / VOCAB_PATH).read_text())
    if build_vocab(usda)["tokens"] != vocab["tokens"]:
        raise RuntimeError("ingredient vocabulary does not match the frozen table")
    vocab_size = len(vocab["tokens"]) + 1
    groups = plate_groups(test_ids, dishes)
    saved_groups = json.loads((root / PLATE_GROUPS_PATH).read_text())
    if saved_groups != groups:
        raise RuntimeError("test plate groups do not match the frozen #86 map")
    if len(set(groups.values())) != 507:
        raise RuntimeError("test plate-group count is not 507")
    crosses = cross_split_pairs(train_ids, test_ids, dishes)
    reference = json.loads(Path("/home/sdnu/nutriagent-digi-v2/configs/q0_frozen.json").read_text())
    if _norm_cross(reference["cross_split_reshoots"]) != _norm_cross(crosses):
        raise RuntimeError("cross-split reshoot log does not match #86")

    geo = build_plan_a(sigma_mass, init_density, pretrained_height=False)
    geo_count = count_parameters(geo)
    if geo_count >= PARAM_LIMIT:
        raise RuntimeError(f"Plan A has {geo_count} parameters; shrink before training")
    b1 = FrozenB1(sigma_mass, init_density)
    if count_parameters(b1) != FROZEN_B1_PARAMS:
        raise RuntimeError(f"frozen B1 architecture counts as {count_parameters(b1)}")
    ting = FrozenTIng(sigma_mass, init_density, vocab_size)
    if count_parameters(ting) != FROZEN_TING_PARAMS:
        raise RuntimeError(f"frozen T_ing architecture counts as {count_parameters(ting)}")
    b1_state = torch.load(B1_CHECKPOINT, map_location="cpu", weights_only=False)["model"]
    load_rgb_from_frozen_b1(geo, b1_state)
    b1.load_state_dict(b1_state)
    del b1_state
    ting_state = torch.load(TING_CHECKPOINT, map_location="cpu", weights_only=False)["model"]
    ting.load_state_dict(ting_state)
    del ting_state, geo, b1, ting

    config = {
        "image_size": IMAGE_SIZE,
        "epochs": EPOCHS,
        "optimizer": "AdamW",
        "lr": LEARNING_RATE,
        "weight_decay": WEIGHT_DECAY,
        "betas": [0.9, 0.999],
        "scheduler": "cosine",
        "eta_min": ETA_MIN,
        "hflip_p": 0.5,
        "seed": SEED,
        "batch_size": int(batch_size),
        "amp": False,
        "rgb_backbone": RGB_BACKBONE,
        "height_backbone": HEIGHT_BACKBONE,
        "cab_reduction": CAB_REDUCTION,
        "gamma_init": GAMMA_INIT,
        "param_limit": PARAM_LIMIT,
        "param_counts": {"t_geo": geo_count, "b1": FROZEN_B1_PARAMS, "t_ing": FROZEN_TING_PARAMS},
        "nutrients": list(NUTRIENTS),
        "sigma_mean": sigma_mean,
        "sigma_mass": sigma_mass,
        "init_mass": sigma_mass,
        "init_density": init_density,
        "vocab_size": vocab_size,
        "n_train_file": int(audit["n_train_file"]),
        "n_train": 2754,
        "n_test": 507,
        "n_test_groups": 507,
        "train_exclusions": audit["exclusions"],
        "apply_d_table_zero_exclusion": exclude_zeros,
        "d_table_zero_count": int(audit["n_d_table_eq_0"]),
        "d_table_zero_proportion": float(audit["proportion_d_table_eq_0"]),
        "frozen_b1_train_mae": dict(FROZEN_B1_TRAIN_MAE),
        "frozen_b1_train_mass_mae": FROZEN_B1_TRAIN_MASS_MAE,
        "rgb_root": RGB_ROOT,
        "depth_root": DEPTH_ROOT,
        "cache_dir": CACHE_DIR,
        "output_dir": "outputs",
        "b1_checkpoint": B1_CHECKPOINT,
        "t_ing_checkpoint": TING_CHECKPOINT,
        "status_commit": STATUS_COMMIT,
        "frozen_weight_commit": FROZEN_WEIGHT_COMMIT,
        "mass_floor_g": 5.0,
        "smooth_l1_beta": 1.0,
        "loss_mass_weight": 0.5,
        "loss_density_weight": 0.5,
        "bootstrap_samples": 2000,
        "bootstrap_seed": 20261125,
        "reshoot_max_gap_seconds": 120,
        "reshoot_nutrition_atol": 0.001,
    }
    assert_config_matches_protocol(config)
    path = root / CONFIG_PATH
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(config, indent=2) + "\n")
    print(f"t_geo params {geo_count} batch {batch_size} exclude_zeros {exclude_zeros}", flush=True)
    return config


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch-size", type=int, required=True)
    parser.add_argument("--root", type=Path, default=Path("."))
    args = parser.parse_args()
    build_frozen_config(args.batch_size, args.root)


if __name__ == "__main__":
    main()
