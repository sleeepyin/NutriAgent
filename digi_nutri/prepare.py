"""Write the frozen Q0 config from the official split, before training."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from digi_nutri.ingredients import assert_ingredients_resolve, build_vocab
from digi_nutri.metadata import load_dish_table
from digi_nutri.models import build_model, count_parameters, select_backbone
from digi_nutri.protocol import (
    META_DIR,
    MODELS,
    NUTRIENTS,
    RGB_ROOT,
    DEPTH_ROOT,
    SPLIT_DIR,
    CACHE_DIR,
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
        "lr",
        "weight_decay",
        "mass_floor_g",
        "seed",
        "reshoot_max_gap_seconds",
        "reshoot_nutrition_atol",
        "bootstrap_samples",
        "bootstrap_seed",
        "smooth_l1_beta",
        "height_max_mm",
        "raw_unit_mm",
        "max_ingredients",
        "loss_mass_weight",
        "loss_density_weight",
    )
    for key in keys:
        if config[key] != snap[key]:
            raise RuntimeError(f"frozen config {key}={config[key]!r} does not match protocol {snap[key]!r}")
    if list(config["nutrients"]) != list(snap["nutrients"]):
        raise RuntimeError("nutrient order does not match protocol.py")
    if config["backbone"] not in {"resnet101", "resnet50"}:
        raise RuntimeError("backbone must be the shared ResNet chosen before training")


def build_frozen_config(batch_size: int, root: Path) -> dict:
    meta = Path(META_DIR)
    split = Path(SPLIT_DIR)
    train_file = read_id_file(split / "depth_train_ids.txt")
    test_file = read_id_file(split / "depth_test_ids.txt")
    audit = audit_split(train_file, test_file, RGB_ROOT, DEPTH_ROOT)
    dishes = load_dish_table(meta / "dish_metadata_cafe1.csv", meta / "dish_metadata_cafe2.csv")
    missing = [dish_id for dish_id in audit.train_ids + audit.test_ids if dish_id not in dishes]
    if missing:
        raise RuntimeError(f"missing metadata for {missing[:8]}")
    usda = load_usda(meta / "ingredients_metadata.csv")
    maximum = assert_ingredients_resolve(audit.train_ids + audit.test_ids, dishes, usda)
    if maximum > snapshot()["max_ingredients"]:
        raise RuntimeError(f"a dish has {maximum} ingredients, above the frozen cap")
    vocab = build_vocab(usda)
    kept = [dishes[dish_id] for dish_id in audit.train_ids]
    sigma_mean = [
        float(np.mean([dish["nutrition"][name] for dish in kept]))
        for name in NUTRIENTS
    ]
    sigma_mass = float(np.mean([dish["mass"] for dish in kept]))
    if sigma_mass <= 0 or any(value <= 0 for value in sigma_mean):
        raise RuntimeError("train-set means must be positive")
    backbone, _geo_count = select_backbone(init_mass=sigma_mass, init_density=[value / sigma_mass for value in sigma_mean])
    init_density = [value / sigma_mass for value in sigma_mean]
    counts = {}
    for kind in MODELS:
        model = build_model(
            kind,
            backbone,
            pretrained=False,
            init_mass=sigma_mass,
            init_density=init_density,
            vocab_size=len(vocab["tokens"]) + 1,
        )
        counts[kind] = count_parameters(model)
    if counts["t_geo"] >= snapshot()["param_limit"]:
        raise RuntimeError(f"T_geo has {counts['t_geo']} parameters")
    test_groups = plate_groups(audit.test_ids, dishes)
    n_groups = len(set(test_groups.values()))
    crosses = cross_split_pairs(audit.train_ids, audit.test_ids, dishes)
    config = snapshot()
    config.update(
        {
            "batch_size": int(batch_size),
            "amp": False,
            "backbone": backbone,
            "rgb_root": RGB_ROOT,
            "depth_root": DEPTH_ROOT,
            "cache_dir": CACHE_DIR,
            "output_dir": "outputs",
            "sigma_mean": sigma_mean,
            "sigma_mass": sigma_mass,
            "init_mass": sigma_mass,
            "init_density": init_density,
            "vocab_size": len(vocab["tokens"]) + 1,
            "param_counts": counts,
            "train_exclusions": audit.exclusions,
            "n_train_file": len(train_file),
            "n_train": len(audit.train_ids),
            "n_test": len(audit.test_ids),
            "n_test_groups": n_groups,
            "cross_split_reshoots": crosses,
            "usda_ambiguous_names": usda.ambiguous_names,
            "usda_identical_duplicate_names": usda.kept_duplicate_names,
            "usda_rule": (
                "Identical repeated names keep the first row. Names whose per-gram "
                "vectors differ are not looked up by name. A dish uses the unique USDA "
                "id from its ingredient field. A blank dish label keeps that row; a "
                "non-empty label that disagrees with the row fails. Plate grams and "
                "per-ingredient nutrition are not model inputs. B1 and T_geo do not "
                "read this table."
            ),
        }
    )
    assert_config_matches_protocol(config)
    config_path = root / "configs" / "q0_frozen.json"
    vocab_path = root / "configs" / "ingredient_vocab.json"
    hygiene_path = root / "results" / "v2_q0" / "split_hygiene.json"
    groups_path = root / "results" / "v2_q0" / "plate_groups_test.json"
    config_path.parent.mkdir(parents=True, exist_ok=True)
    hygiene_path.parent.mkdir(parents=True, exist_ok=True)
    config_path.write_text(json.dumps(config, indent=2) + "\n")
    vocab_path.write_text(json.dumps(vocab, indent=2) + "\n")
    hygiene_path.write_text(
        json.dumps(
            {
                "train_exclusions": audit.exclusions,
                "n_train_file": len(train_file),
                "n_train": len(audit.train_ids),
                "n_test": len(audit.test_ids),
                "n_test_groups": n_groups,
                "max_unique_ingredients": maximum,
                "cross_split_reshoots": crosses,
                "usda_ambiguous_names": usda.ambiguous_names,
                "backbone": backbone,
                "param_counts": counts,
            },
            indent=2,
        )
        + "\n"
    )
    groups_path.write_text(json.dumps(test_groups, indent=2) + "\n")
    return config


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch-size", type=int, required=True)
    parser.add_argument("--root", type=Path, default=Path("."))
    args = parser.parse_args()
    config = build_frozen_config(args.batch_size, args.root)
    print(
        f"backbone {config['backbone']} batch {config['batch_size']} "
        f"train {config['n_train']} test {config['n_test']} groups {config['n_test_groups']}",
        flush=True,
    )
    print("params", config["param_counts"], flush=True)
    print("exclusions", config["train_exclusions"], flush=True)


if __name__ == "__main__":
    main()
