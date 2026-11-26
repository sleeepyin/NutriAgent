"""Write the frozen T_multi config before any training."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from digi_nutri.data import assert_cache_ready
from digi_nutri.frozen_b1 import (
    load_frozen_holdout_rows,
    sha256_file,
    verify_checkpoint_file,
    verify_frozen_train_mae,
    verify_saved_plate_groups,
)
from digi_nutri.ingredients import assert_ingredients_resolve, build_vocab
from digi_nutri.metadata import load_dish_table
from digi_nutri.models import build_model, count_parameters, volume_feature_names
from digi_nutri.protocol import (
    CACHE_DIR,
    CONFIG_PATH,
    DEPTH_ROOT,
    EXPECTED_TEST_GROUPS,
    EXPECTED_TEST_N,
    EXPECTED_TRAIN_N,
    FROZEN_B1_PREDICTIONS,
    FROZEN_SIGMA_MASS,
    FROZEN_SIGMA_MEAN,
    HYGIENE_PATH,
    META_DIR,
    MODEL,
    NUTRIENTS,
    PARAM_LIMIT,
    PLATE_GROUPS_PATH,
    RGB_ROOT,
    SPLIT_DIR,
    T_MULTI_PARAMS,
    TRAIN_EXCLUSION_IDS,
    VOCAB_PATH,
    snapshot,
)
from digi_nutri.reshoot import cross_split_pairs, plate_groups
from digi_nutri.splits import audit_split, read_id_file
from digi_nutri.usda import load_usda


def assert_config_matches_protocol(config: dict) -> None:
    snap = snapshot()
    keys = (
        "image_size",
        "epochs",
        "batch_size",
        "head_lr",
        "backbone_lr",
        "fusion_lr",
        "weight_decay",
        "dropout_p",
        "hflip_p",
        "vflip_p",
        "seed",
        "mass_floor_g",
        "smooth_l1_beta",
        "loss_mass_weight",
        "loss_density_weight",
        "height_max_mm",
        "raw_unit_mm",
        "exclude_border_zeros",
        "param_limit",
        "volume_dim",
        "gamma_init",
        "bootstrap_samples",
        "bootstrap_seed",
        "reshoot_max_gap_seconds",
        "reshoot_nutrition_atol",
        "status_commit",
        "max_ingredients",
        "n_headline_intervals",
        "model",
        "comparator",
        "frozen_b1_sha256",
        "frozen_b1_config_sha256",
        "t_multi_params",
    )
    for key in keys:
        if config[key] != snap[key]:
            raise RuntimeError(f"frozen config {key}={config[key]!r} != protocol {snap[key]!r}")
    if list(config["nutrients"]) != list(snap["nutrients"]):
        raise RuntimeError("nutrient order does not match protocol.py")
    if list(config["rotation_degrees"]) != list(snap["rotation_degrees"]):
        raise RuntimeError("rotation set does not match protocol.py")
    if config["rgb_backbone"] != "resnet101" or config["height_backbone"] != "resnet18":
        raise RuntimeError("backbones do not match the frozen pair")
    if int(config["param_counts"][MODEL]) != T_MULTI_PARAMS:
        raise RuntimeError("frozen parameter count does not match the measured constant")
    if int(config["param_counts"][MODEL]) >= PARAM_LIMIT:
        raise RuntimeError("frozen parameter count is not strictly below 60M")


def measure_param_count(init_mass: float, init_density: list[float], vocab_size: int) -> int:
    model = build_model(init_mass, init_density, vocab_size, pretrained_height=False)
    count = count_parameters(model)
    del model
    if count != T_MULTI_PARAMS:
        raise RuntimeError(f"measured T_multi {count} != frozen constant {T_MULTI_PARAMS}")
    if count >= PARAM_LIMIT:
        raise RuntimeError(f"T_multi has {count} parameters, at or above {PARAM_LIMIT}; narrow and recount")
    return count


def build_frozen_config(root: Path) -> dict:
    meta = Path(META_DIR)
    split = Path(SPLIT_DIR)
    train_file = read_id_file(split / "depth_train_ids.txt")
    test_file = read_id_file(split / "depth_test_ids.txt")
    audit = audit_split(train_file, test_file, RGB_ROOT, DEPTH_ROOT)
    if len(audit.test_ids) != EXPECTED_TEST_N:
        raise RuntimeError(f"n_kept test is {len(audit.test_ids)}")
    if len(audit.train_ids) != EXPECTED_TRAIN_N:
        raise RuntimeError(f"n_kept train is {len(audit.train_ids)}")
    if len(train_file) != 2758:
        raise RuntimeError(f"depth_train_ids has {len(train_file)} rows, expected 2758")
    found_ids = {row["dish_id"] for row in audit.exclusions}
    if found_ids != set(TRAIN_EXCLUSION_IDS):
        raise RuntimeError(f"train exclusions changed: {sorted(found_ids)}")
    dishes = load_dish_table(meta / "dish_metadata_cafe1.csv", meta / "dish_metadata_cafe2.csv")
    missing = [dish_id for dish_id in audit.train_ids + audit.test_ids if dish_id not in dishes]
    if missing:
        raise RuntimeError(f"missing metadata for {missing[:8]}")
    usda = load_usda(meta / "ingredients_metadata.csv")
    if "deprecated" not in usda.ambiguous_names:
        raise RuntimeError("USDA name 'deprecated' is not marked ambiguous")
    maximum = assert_ingredients_resolve(audit.train_ids + audit.test_ids, dishes, usda)
    if maximum > snapshot()["max_ingredients"]:
        raise RuntimeError(f"a dish has {maximum} ingredients, above the frozen cap")
    vocab = build_vocab(usda)
    vocab_size = len(vocab["tokens"]) + 1
    if vocab_size != 556:
        raise RuntimeError(f"vocab size {vocab_size} != 556")
    kept = [dishes[dish_id] for dish_id in audit.train_ids]
    sigma_mean = [float(np.mean([dish["nutrition"][name] for dish in kept])) for name in NUTRIENTS]
    sigma_mass = float(np.mean([dish["mass"] for dish in kept]))
    if any(abs(got - exp) > 1e-6 for got, exp in zip(sigma_mean, FROZEN_SIGMA_MEAN)):
        raise RuntimeError(f"train nutrient means changed: {sigma_mean}")
    if abs(sigma_mass - FROZEN_SIGMA_MASS) > 1e-6:
        raise RuntimeError(f"train mass mean changed: {sigma_mass}")
    init_density = [value / sigma_mass for value in sigma_mean]
    count = measure_param_count(sigma_mass, init_density, vocab_size)
    assert_cache_ready(audit.train_ids + audit.test_ids, CACHE_DIR)
    verify_checkpoint_file()
    verify_frozen_train_mae()
    frozen_rows = load_frozen_holdout_rows()
    if frozen_rows["dish_id"].tolist() != audit.test_ids:
        raise RuntimeError("frozen B1 holdout order is not the official test order")
    test_groups = plate_groups(audit.test_ids, dishes)
    if len(set(test_groups.values())) != EXPECTED_TEST_GROUPS:
        raise RuntimeError(f"test groups {len(set(test_groups.values()))} != {EXPECTED_TEST_GROUPS}")
    verify_saved_plate_groups(test_groups)
    crosses = cross_split_pairs(audit.train_ids, audit.test_ids, dishes)
    config = snapshot()
    config.update(
        {
            "amp": False,
            "rgb_root": RGB_ROOT,
            "depth_root": DEPTH_ROOT,
            "cache_dir": CACHE_DIR,
            "output_dir": "outputs",
            "sigma_mean": sigma_mean,
            "sigma_mass": sigma_mass,
            "init_mass": sigma_mass,
            "init_density": init_density,
            "vocab_size": vocab_size,
            "param_counts": {MODEL: count},
            "volume_feature_names": volume_feature_names(),
            "train_exclusions": audit.exclusions,
            "n_train_file": len(train_file),
            "n_train": len(audit.train_ids),
            "n_test": len(audit.test_ids),
            "n_test_groups": len(set(test_groups.values())),
            "cross_split_reshoots": crosses,
            "usda_ambiguous_names": sorted(usda.ambiguous_names),
            "frozen_b1_predictions_sha256": sha256_file(FROZEN_B1_PREDICTIONS),
            "augmentation": {
                "train_only": True,
                "horizontal_flip_p": config["hflip_p"],
                "vertical_flip_p": config["vflip_p"],
                "rotation_degrees": config["rotation_degrees"],
                "shared_on_rgb_and_height": True,
                "labels": "unchanged",
                "eval": "identity",
            },
        }
    )
    assert_config_matches_protocol(config)
    for path in (root / CONFIG_PATH, root / VOCAB_PATH, root / HYGIENE_PATH, root / PLATE_GROUPS_PATH):
        path.parent.mkdir(parents=True, exist_ok=True)
    (root / CONFIG_PATH).write_text(json.dumps(config, indent=2) + "\n")
    (root / VOCAB_PATH).write_text(json.dumps(vocab, indent=2) + "\n")
    (root / HYGIENE_PATH).write_text(
        json.dumps(
            {
                "train_exclusions": audit.exclusions,
                "n_train_file": len(train_file),
                "n_train": len(audit.train_ids),
                "n_test": len(audit.test_ids),
                "n_test_groups": len(set(test_groups.values())),
                "max_unique_ingredients": maximum,
                "cross_split_reshoots": crosses,
                "param_counts": {MODEL: count},
                "status_commit": config["status_commit"],
            },
            indent=2,
        )
        + "\n"
    )
    (root / PLATE_GROUPS_PATH).write_text(json.dumps(test_groups, indent=2) + "\n")
    return config


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path("."))
    args = parser.parse_args()
    config = build_frozen_config(args.root)
    print(
        f"train {config['n_train']} test {config['n_test']} groups {config['n_test_groups']}",
        flush=True,
    )
    print("params", config["param_counts"], flush=True)


if __name__ == "__main__":
    main()
