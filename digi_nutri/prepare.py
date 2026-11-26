"""Write the frozen v3 step 1 config before any training."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from digi_nutri.data import assert_cache_ready
from digi_nutri.ingredients import assert_ingredients_resolve, build_vocab
from digi_nutri.metadata import load_dish_table
from digi_nutri.models import (
    build_b1,
    build_tmulti,
    count_parameters,
    parameter_breakdown,
    volume_feature_names,
)
from digi_nutri.protocol import (
    B1_PARAMS,
    CACHE_DIR,
    CONFIG_PATH,
    DEPTH_ROOT,
    EXPECTED_TEST_GROUPS,
    EXPECTED_TEST_N,
    EXPECTED_TRAIN_FILE_N,
    EXPECTED_TRAIN_N,
    FROZEN_SIGMA_MASS,
    FROZEN_SIGMA_MEAN,
    HEIGHT_ENCODER_PARAMS,
    HYGIENE_PATH,
    META_DIR,
    MODEL,
    NUTRIENTS,
    PARAM_LIMIT,
    PLATE_GROUPS_PATH,
    RGB_ENCODER_PARAMS,
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
        "residual_loss_weight",
        "atwater_fat",
        "atwater_carb",
        "atwater_pro",
        "hard_atwater",
        "height_max_mm",
        "raw_unit_mm",
        "exclude_border_zeros",
        "param_limit",
        "volume_dim",
        "gamma_init",
        "residual_hidden",
        "bootstrap_samples",
        "bootstrap_seed",
        "reshoot_max_gap_seconds",
        "reshoot_nutrition_atol",
        "status_commit",
        "max_ingredients",
        "n_headline_intervals",
        "model",
        "comparator",
        "t_multi_params",
        "b1_params",
        "energy_dataset_pmae_max",
        "mean5_dataset_pmae_max",
        "distillation_started",
        "train_gate",
        "rgb_backbone",
        "height_backbone",
        "convnext_weight_url",
        "convnext_weight_tag",
        "convnext_weight_sha256",
    )
    for key in keys:
        if config[key] != snap[key]:
            raise RuntimeError(f"frozen config {key}={config[key]!r} != protocol {snap[key]!r}")
    if list(config["nutrients"]) != list(snap["nutrients"]):
        raise RuntimeError("nutrient order does not match protocol.py")
    if list(config["absolute_pmae_keys"]) != list(snap["absolute_pmae_keys"]):
        raise RuntimeError("absolute PMAE keys do not match protocol.py")
    if list(config["rotation_degrees"]) != list(snap["rotation_degrees"]):
        raise RuntimeError("rotation set does not match protocol.py")
    if config["rgb_backbone"] != "convnext_small" or config["height_backbone"] != "resnet18":
        raise RuntimeError("backbones do not match the frozen pair")
    counts = config["param_counts"]
    if int(counts[MODEL]) != T_MULTI_PARAMS or int(counts["b1"]) != B1_PARAMS:
        raise RuntimeError("frozen parameter counts do not match the measured constants")
    if int(counts["rgb_encoder"]) != RGB_ENCODER_PARAMS:
        raise RuntimeError("frozen RGB encoder count does not match the measured constant")
    if int(counts["height_encoder"]) != HEIGHT_ENCODER_PARAMS:
        raise RuntimeError("frozen height encoder count does not match the measured constant")
    if int(counts[MODEL]) >= PARAM_LIMIT:
        raise RuntimeError("frozen T_multi count is not strictly below 70M")
    if config["hard_atwater"] is not False or config["distillation_started"] is not False:
        raise RuntimeError("hard Atwater and distillation stay off")


def measure_models(init_mass: float, init_density: list[float], vocab_size: int) -> dict:
    teacher = build_tmulti(init_mass, init_density, vocab_size, False, enforce_count=True)
    baseline = build_b1(init_mass, init_density, False, enforce_count=True)
    teacher_count = count_parameters(teacher)
    baseline_count = count_parameters(baseline)
    breakdown = parameter_breakdown(teacher)
    rgb_count = count_parameters(teacher.rgb_encoder)
    height_count = count_parameters(teacher.height_encoder)
    del teacher, baseline
    if teacher_count != T_MULTI_PARAMS or teacher_count >= PARAM_LIMIT:
        raise RuntimeError(f"measured T_multi {teacher_count} is not the frozen under-70M count")
    if baseline_count != B1_PARAMS:
        raise RuntimeError(f"measured B1 {baseline_count} != frozen constant {B1_PARAMS}")
    if rgb_count != RGB_ENCODER_PARAMS or height_count != HEIGHT_ENCODER_PARAMS:
        raise RuntimeError("measured encoder counts do not match the frozen constants")
    if sum(breakdown.values()) != teacher_count:
        raise RuntimeError("parameter breakdown does not sum to the T_multi count")
    return {
        "b1": baseline_count,
        MODEL: teacher_count,
        "rgb_encoder": rgb_count,
        "height_encoder": height_count,
        "t_multi_breakdown": breakdown,
    }


def build_frozen_config(root: Path) -> dict:
    if T_MULTI_PARAMS <= 0 or T_MULTI_PARAMS >= PARAM_LIMIT:
        raise RuntimeError("freeze a T_multi count that is strictly below 70M before writing the config")
    meta = Path(META_DIR)
    split = Path(SPLIT_DIR)
    train_file = read_id_file(split / "depth_train_ids.txt")
    test_file = read_id_file(split / "depth_test_ids.txt")
    audit = audit_split(train_file, test_file, RGB_ROOT, DEPTH_ROOT)
    if len(audit.test_ids) != EXPECTED_TEST_N:
        raise RuntimeError(f"n_kept test is {len(audit.test_ids)}")
    if len(audit.train_ids) != EXPECTED_TRAIN_N:
        raise RuntimeError(f"n_kept train is {len(audit.train_ids)}")
    if len(train_file) != EXPECTED_TRAIN_FILE_N:
        raise RuntimeError(f"depth_train_ids has {len(train_file)} rows, expected {EXPECTED_TRAIN_FILE_N}")
    if set(audit.train_ids) & set(audit.test_ids):
        raise RuntimeError("kept train ids overlap the official holdout")
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
    measured = measure_models(sigma_mass, init_density, vocab_size)
    assert_cache_ready(audit.train_ids + audit.test_ids, CACHE_DIR)
    test_groups = plate_groups(audit.test_ids, dishes)
    if len(set(test_groups.values())) != EXPECTED_TEST_GROUPS:
        raise RuntimeError(f"test groups {len(set(test_groups.values()))} != {EXPECTED_TEST_GROUPS}")
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
            "param_counts": {
                "b1": measured["b1"],
                MODEL: measured[MODEL],
                "rgb_encoder": measured["rgb_encoder"],
                "height_encoder": measured["height_encoder"],
            },
            "t_multi_breakdown": measured["t_multi_breakdown"],
            "volume_feature_names": volume_feature_names(),
            "train_exclusions": audit.exclusions,
            "n_train_file": len(train_file),
            "n_train": len(audit.train_ids),
            "n_test": len(audit.test_ids),
            "n_test_groups": len(set(test_groups.values())),
            "train_holdout_overlap": 0,
            "cross_split_reshoots": crosses,
            "usda_ambiguous_names": sorted(usda.ambiguous_names),
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
    hygiene = {
        "train_exclusions": audit.exclusions,
        "n_train_file": len(train_file),
        "n_train": len(audit.train_ids),
        "n_test": len(audit.test_ids),
        "n_test_groups": len(set(test_groups.values())),
        "train_holdout_overlap": 0,
        "max_unique_ingredients": maximum,
        "cross_split_reshoots": crosses,
        "param_counts": config["param_counts"],
        "t_multi_below_70m": True,
        "status_commit": config["status_commit"],
        "holdout_evaluated": False,
    }
    (root / HYGIENE_PATH).write_text(json.dumps(hygiene, indent=2) + "\n")
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
