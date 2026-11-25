"""Write the frozen rerun config before any training."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from digi_nutri.data import assert_cache_ready
from digi_nutri.ingredients import assert_ingredients_resolve, build_vocab
from digi_nutri.metadata import load_dish_table
from digi_nutri.models import build_model, count_parameters, volume_feature_names
from digi_nutri.protocol import (
    B1_STRUCTURE_PARAMS,
    CACHE_DIR,
    CONFIG_PATH,
    DEPTH_ROOT,
    EXPECTED_TRAIN_N,
    EXPECTED_TEST_N,
    HEAD_HIDDEN,
    HYGIENE_PATH,
    META_DIR,
    MODELS,
    NUTRIENTS,
    PARAM_LIMIT,
    PLAN_A_TGEO_WITHOUT_VOLUME,
    PLATE_GROUPS_PATH,
    RGB_ROOT,
    SPLIT_DIR,
    TING_STRUCTURE_PARAMS,
    TRAIN_EXCLUSION_IDS,
    VOCAB_PATH,
    VOLUME_DIM,
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
    )
    for key in keys:
        if config[key] != snap[key]:
            raise RuntimeError(f"frozen config {key}={config[key]!r} != protocol {snap[key]!r}")
    if list(config["nutrients"]) != list(snap["nutrients"]):
        raise RuntimeError("nutrient order does not match protocol.py")
    if list(config["rotation_degrees"]) != list(snap["rotation_degrees"]):
        raise RuntimeError("rotation set does not match protocol.py")
    if config["rgb_backbone"] != "resnet101" or config["height_backbone"] != "resnet18":
        raise RuntimeError("backbones do not match the frozen Plan A pair")


def measure_param_counts(init_mass: float, init_density: list[float], vocab_size: int) -> dict[str, int]:
    counts = {}
    for kind in MODELS:
        model = build_model(
            kind,
            init_mass=init_mass,
            init_density=init_density,
            vocab_size=vocab_size,
            pretrained=False,
        )
        counts[kind] = count_parameters(model)
        del model
    if counts["b1"] != B1_STRUCTURE_PARAMS:
        raise RuntimeError(f"B1 count {counts['b1']} != unchanged structure {B1_STRUCTURE_PARAMS}")
    if counts["t_ing"] != TING_STRUCTURE_PARAMS:
        raise RuntimeError(
            f"T_ing count {counts['t_ing']} != unchanged structure {TING_STRUCTURE_PARAMS}"
        )
    expected_geo = PLAN_A_TGEO_WITHOUT_VOLUME + VOLUME_DIM * HEAD_HIDDEN
    if counts["t_geo"] != expected_geo:
        raise RuntimeError(
            f"measured T_geo {counts['t_geo']} != Plan A plus volume readout {expected_geo}"
        )
    if counts["t_geo"] >= PARAM_LIMIT:
        raise RuntimeError(
            f"T_geo has {counts['t_geo']} parameters, at or above {PARAM_LIMIT}; narrow volume features"
        )
    return counts


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
    kept = [dishes[dish_id] for dish_id in audit.train_ids]
    sigma_mean = [float(np.mean([dish["nutrition"][name] for dish in kept])) for name in NUTRIENTS]
    sigma_mass = float(np.mean([dish["mass"] for dish in kept]))
    if sigma_mass <= 0 or any(value <= 0 for value in sigma_mean):
        raise RuntimeError("train-set means must be positive")
    init_density = [value / sigma_mass for value in sigma_mean]
    counts = measure_param_counts(sigma_mass, init_density, vocab_size)
    assert_cache_ready(audit.train_ids + audit.test_ids, CACHE_DIR, need_height=True)
    test_groups = plate_groups(audit.test_ids, dishes)
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
            "param_counts": counts,
            "volume_feature_names": volume_feature_names(),
            "train_exclusions": audit.exclusions,
            "n_train_file": len(train_file),
            "n_train": len(audit.train_ids),
            "n_test": len(audit.test_ids),
            "n_test_groups": len(set(test_groups.values())),
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
    config_path = root / CONFIG_PATH
    vocab_path = root / VOCAB_PATH
    hygiene_path = root / HYGIENE_PATH
    groups_path = root / PLATE_GROUPS_PATH
    for path in (config_path, vocab_path, hygiene_path, groups_path):
        path.parent.mkdir(parents=True, exist_ok=True)
    config_path.write_text(json.dumps(config, indent=2) + "\n")
    vocab_path.write_text(json.dumps(vocab, indent=2) + "\n")
    hygiene_path.write_text(
        json.dumps(
            {
                "train_exclusions": audit.exclusions,
                "n_train_file": len(train_file),
                "n_train": len(audit.train_ids),
                "n_test": len(audit.test_ids),
                "n_test_groups": len(set(test_groups.values())),
                "max_unique_ingredients": maximum,
                "cross_split_reshoots": crosses,
                "param_counts": counts,
                "status_commit": config["status_commit"],
            },
            indent=2,
        )
        + "\n"
    )
    groups_path.write_text(json.dumps(test_groups, indent=2) + "\n")
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
