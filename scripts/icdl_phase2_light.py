#!/usr/bin/env python3
"""ICDL phase 2 light experiment on the phase 1 Nutrition5k plate sequences.

Three models share one DINOv2 ViT-S/14 backbone and one training budget.
Inference is a single overhead RGB image. The losses follow the official
ICDL plan: absolute nutrition regression, composition-difference
supervision, difference-feature alignment, and path consistency. A random
cross-plate pairing control uses the same architecture and the same budget.

Nothing here is selected on the official test labels. The weight grid is
fixed before evaluation. Test numbers are written once, at the end.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import random
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
N5K = ROOT / "data" / "nutrition5k"
PAIRS_PATH = ROOT / "results" / "icdl_phase1_sequence_audit" / "valid_pairs.jsonl"
SEQUENCES_PATH = ROOT / "results" / "icdl_phase1_sequence_audit" / "valid_sequences.jsonl"
DEFAULT_OUTPUT = ROOT / "results" / "icdl_phase2_light_models"
DINOV2_WEIGHTS = ROOT / "backup" / "v3_excluded" / "weights" / "dinov2" / "dinov2_vits14_pretrain.pth"

PAIRS_SHA256 = "d05f66527e02fcc02c311d87a31d1bde138eeba40e8ca1dc984778f8539808a3"
SEQUENCES_SHA256 = "9d32ff8927b28ce514ac03531086a13a1e7390fce3de70873939bd7fed445dbc"
DINOV2_SHA256 = "b938bf1bc15cd2ec0feacfe3a1bb553fe8ea9ca46a7e1d8d00217f29aef60cd9"
STATUS_COMMIT = "b8d2976ea630bda7c5a123726489eafd5d415c8e"

# Budget. Shared by every run. Not searched on the test split.
EPOCHS = 12
BATCH_SIZE = 16
SEED = 0
LR_BACKBONE = 3e-5
LR_HEAD = 1e-3
WEIGHT_DECAY = 1e-2
GRAD_CLIP = 1.0
HEAD_HIDDEN = 256
IMAGE_SIZE = 224
RGB_MIN_BYTES = 10_000
BOOTSTRAP_DRAWS = 10_000
BOOTSTRAP_SEED = 0
IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)

TARGET_KEYS = ("mass_g", "kcal", "fat_g", "carb_g", "protein_g")
TARGET_LABELS = {
    "mass_g": "质量（克）",
    "kcal": "能量（千卡）",
    "fat_g": "脂肪（克）",
    "carb_g": "碳水（克）",
    "protein_g": "蛋白质（克）",
}
NUTRITION_KEYS = ("kcal", "fat_g", "carb_g", "protein_g")
LEAK_SEQUENCE = "cafe1-0168"
LEAK_DISHES = ("dish_1561662926", "dish_1561662948")
RETAKE_EXAMPLE = ("dish_1550705995", "dish_1550706037")

FROZEN_RESULT_NAMES = {
    "mass_estimator",
    "heldout_plan",
    "heldout_eval",
    "heldout_energy_audit",
    "heldout_massonly_energy_path",
    "food_id_density",
    "food_id_semantic_audit",
    "qphi",
    "qphi_counterfactual",
    "p0d",
    "audit",
    "coverage_early_stop",
    "phase1_cross_domain",
    "weak_steps",
    "icdl_phase1_sequence_audit",
    "experiment_a_n5k_dual_view",
    "diagnosis",
}


@dataclass(frozen=True)
class Spec:
    slug: str
    name: str
    pair_source: str
    feature: str
    lambda_delta: float
    lambda_cons: float
    lambda_path: float
    role: str

    def as_dict(self) -> dict:
        return {
            "slug": self.slug,
            "name": self.name,
            "pair_source": self.pair_source,
            "feature": self.feature,
            "lambda_delta": self.lambda_delta,
            "lambda_consistency": self.lambda_cons,
            "lambda_path": self.lambda_path,
            "role": self.role,
        }


SPECS = (
    Spec("single_image", "普通 DINOv2 加单图像营养回归", "none", "none", 0.0, 0.0, 0.0, "baseline"),
    Spec("composition_difference", "单图像基线加组成差异监督", "adjacent", "concat", 1.0, 0.0, 0.0, "primary"),
    Spec(
        "alignment_path",
        "差异特征对齐、营养差异预测与路径一致性",
        "adjacent",
        "difference",
        1.0,
        1.0,
        1.0,
        "primary",
    ),
    Spec("ablate_no_delta", "消融：去掉营养差异预测", "adjacent", "difference", 0.0, 1.0, 1.0, "ablation"),
    Spec("ablate_no_consistency", "消融：去掉单步预测一致", "adjacent", "difference", 1.0, 0.0, 1.0, "ablation"),
    Spec("ablate_no_path", "消融：去掉跨阶段路径一致", "adjacent", "difference", 1.0, 1.0, 0.0, "ablation"),
    Spec("ablate_no_alignment", "消融：用拼接特征代替差异特征", "adjacent", "concat", 1.0, 1.0, 1.0, "ablation"),
    Spec(
        "random_composition_difference",
        "负对照：组成差异监督，但跨盘随机配对",
        "random",
        "concat",
        1.0,
        0.0,
        0.0,
        "control",
    ),
    Spec(
        "random_alignment_path",
        "负对照：差异对齐、营养差异预测与路径一致性，但跨盘随机配对",
        "random",
        "difference",
        1.0,
        1.0,
        1.0,
        "control",
    ),
)
SPEC_BY_SLUG = {spec.slug: spec for spec in SPECS}
CONTROL_OF = {
    "composition_difference": "random_composition_difference",
    "alignment_path": "random_alignment_path",
}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_jsonl(path: Path) -> list[dict]:
    rows = []
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def assert_output_dir(path: Path) -> Path:
    resolved = path.resolve()
    hits = FROZEN_RESULT_NAMES.intersection(resolved.parts)
    if hits:
        raise SystemExit(f"refusing to write into frozen results {sorted(hits)} via {resolved}")
    resolved.mkdir(parents=True, exist_ok=True)
    return resolved


def sequence_split(sequence: dict) -> str:
    cafe = sequence["cafe"]
    if cafe == "cafe2":
        return "excluded_cafe2"
    if cafe != "cafe1":
        raise RuntimeError(f"unexpected cafe {cafe}")
    splits = set(sequence["rgb_splits"])
    leaked = bool(sequence.get("rgb_leak") or sequence.get("depth_leak") or splits == {"rgb_train", "rgb_test"})
    if leaked:
        return "quarantine"
    if splits == {"rgb_train"}:
        return "train"
    if splits == {"rgb_test"}:
        return "test"
    raise RuntimeError(f"unexpected rgb split for {sequence['sequence_id']}: {splits}")


def rgb_path(dish_id: str | None) -> Path | None:
    if not dish_id:
        return None
    path = N5K / "imagery" / "realsense_overhead" / dish_id / "rgb.png"
    folded = [part.casefold() for part in path.parts]
    if any("depth" in part or "side_angle" in part for part in folded):
        raise RuntimeError(f"refusing a non-overhead path: {path}")
    if path.is_file() and path.stat().st_size >= RGB_MIN_BYTES:
        return path
    return None


def member_targets(stage: dict, dish_id: str) -> np.ndarray:
    for member in stage["members"]:
        if member["dish_id"] == dish_id:
            return np.array(
                [
                    float(member["mass_g"]),
                    float(member["kcal"]),
                    float(member["fat_g"]),
                    float(member["carb_g"]),
                    float(member["protein_g"]),
                ],
                dtype=np.float64,
            )
    raise KeyError(dish_id)


def added_text(stage: dict) -> str:
    parts = []
    for item in stage.get("added") or []:
        parts.append(f"{item['name']} {float(item['grams']):.3f} g")
    return "、".join(parts)


def contiguous_paths(positions: list[int], indices: list[int], min_len: int = 3) -> list[list[int]]:
    if not indices:
        return []
    order = sorted(range(len(indices)), key=lambda i: positions[i])
    paths: list[list[int]] = []
    run = [indices[order[0]]]
    last = positions[order[0]]
    for cursor in order[1:]:
        if positions[cursor] == last + 1:
            run.append(indices[cursor])
        else:
            if len(run) >= min_len:
                paths.append(run)
            run = [indices[cursor]]
        last = positions[cursor]
    if len(run) >= min_len:
        paths.append(run)
    return paths


def sample_partners(anchor_sequences: np.ndarray, pool_sequences: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Pool indices whose physical plate differs from each anchor plate."""
    pool_size = int(pool_sequences.shape[0])
    if pool_size < 2 or len(set(pool_sequences.tolist())) < 2:
        raise RuntimeError("cross-plate sampling needs at least two training plates")
    choice = rng.integers(0, pool_size, size=anchor_sequences.shape[0])
    bad = pool_sequences[choice] == anchor_sequences
    guard = 0
    while np.any(bad):
        choice[bad] = rng.integers(0, pool_size, size=int(np.sum(bad)))
        bad = pool_sequences[choice] == anchor_sequences
        guard += 1
        if guard > 1000:
            raise RuntimeError("failed to sample a partner from a different plate")
    return choice


def sample_random_chains(
    lengths: list[int],
    pool_indices: np.ndarray,
    pool_sequences: np.ndarray,
    rng: np.random.Generator,
) -> list[list[int]]:
    by_sequence: dict[str, list[int]] = {}
    for index, sequence_id in zip(pool_indices.tolist(), pool_sequences.tolist()):
        by_sequence.setdefault(str(sequence_id), []).append(int(index))
    keys = np.array(sorted(by_sequence))
    chains = []
    for length in lengths:
        if length > keys.size:
            raise RuntimeError("not enough distinct plates to build a random chain")
        picked = rng.choice(keys, size=length, replace=False)
        chain = []
        for sequence_id in picked.tolist():
            dishes = by_sequence[str(sequence_id)]
            chain.append(int(dishes[int(rng.integers(0, len(dishes)))]))
        chains.append(chain)
    return chains


def interval_relation(low: float, high: float) -> str:
    if high < 0.0:
        return "entirely_below"
    if low > 0.0:
        return "entirely_above"
    return "includes_zero"


def per_sequence_mae(abs_err: np.ndarray, sequence_ids: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    uniq, inverse = np.unique(sequence_ids, return_inverse=True)
    sums = np.zeros((uniq.size, abs_err.shape[1]), dtype=np.float64)
    counts = np.zeros(uniq.size, dtype=np.float64)
    np.add.at(sums, inverse, abs_err)
    np.add.at(counts, inverse, 1.0)
    return sums / counts[:, None], uniq


def bootstrap_mean_difference(
    seq_mae_method: np.ndarray,
    seq_mae_reference: np.ndarray,
    *,
    draws: int,
    seed: int,
) -> list[dict]:
    """Sequence-equal bootstrap of mean(method sequence MAE - reference sequence MAE)."""
    if seq_mae_method.shape != seq_mae_reference.shape or seq_mae_method.ndim != 2:
        raise ValueError("sequence MAE arrays must align")
    n_seq, n_targets = seq_mae_method.shape
    rng = np.random.default_rng(seed)
    drawn = rng.integers(0, n_seq, size=(draws, n_seq))
    diff = seq_mae_method[drawn].mean(axis=1) - seq_mae_reference[drawn].mean(axis=1)
    point = seq_mae_method.mean(axis=0) - seq_mae_reference.mean(axis=0)
    low = np.quantile(diff, 0.025, axis=0)
    high = np.quantile(diff, 0.975, axis=0)
    rows = []
    for index, key in enumerate(TARGET_KEYS):
        rows.append(
            {
                "target": key,
                "mae_method": float(seq_mae_method[:, index].mean()),
                "mae_reference": float(seq_mae_reference[:, index].mean()),
                "difference": float(point[index]),
                "ci95_low": float(low[index]),
                "ci95_high": float(high[index]),
                "fraction_method_lower": float(np.mean(diff[:, index] < 0.0)),
                "relation": interval_relation(float(low[index]), float(high[index])),
                "n_sequences": int(n_seq),
            }
        )
    return rows


def dish_weighted_bootstrap(
    abs_method: np.ndarray,
    abs_reference: np.ndarray,
    sequence_ids: np.ndarray,
    *,
    draws: int,
    seed: int,
) -> list[dict]:
    """Sensitivity: resample plates, then average dishes inside the resampled multiset."""
    seq_codes, inverse = np.unique(sequence_ids, return_inverse=True)
    n_seq = seq_codes.size
    sums_m = np.zeros((n_seq, abs_method.shape[1]), dtype=np.float64)
    sums_r = np.zeros_like(sums_m)
    counts = np.zeros(n_seq, dtype=np.float64)
    np.add.at(sums_m, inverse, abs_method)
    np.add.at(sums_r, inverse, abs_reference)
    np.add.at(counts, inverse, 1.0)
    rng = np.random.default_rng(seed)
    drawn = rng.integers(0, n_seq, size=(draws, n_seq))
    denom = counts[drawn].sum(axis=1)
    diff = (sums_m[drawn].sum(axis=1) - sums_r[drawn].sum(axis=1)) / denom[:, None]
    point_m = abs_method.mean(axis=0)
    point_r = abs_reference.mean(axis=0)
    low = np.quantile(diff, 0.025, axis=0)
    high = np.quantile(diff, 0.975, axis=0)
    rows = []
    for index, key in enumerate(TARGET_KEYS):
        rows.append(
            {
                "target": key,
                "mae_method": float(point_m[index]),
                "mae_reference": float(point_r[index]),
                "difference": float(point_m[index] - point_r[index]),
                "ci95_low": float(low[index]),
                "ci95_high": float(high[index]),
                "relation": interval_relation(float(low[index]), float(high[index])),
                "n_dishes": int(abs_method.shape[0]),
                "n_sequences": int(n_seq),
            }
        )
    return rows


def nutrition_supported(rows: list[dict]) -> bool:
    by_target = {row["target"]: row for row in rows}
    return all(by_target[key]["relation"] == "entirely_below" for key in NUTRITION_KEYS)


def mass_only(rows: list[dict]) -> bool:
    by_target = {row["target"]: row for row in rows}
    mass_better = by_target["mass_g"]["relation"] == "entirely_below"
    nutrition_miss = any(by_target[key]["relation"] != "entirely_below" for key in NUTRITION_KEYS)
    return bool(mass_better and nutrition_miss)


class World:
    def __init__(self) -> None:
        self.stages: list[dict] = []
        self.pair_before: np.ndarray = np.zeros(0, dtype=np.int64)
        self.pair_after: np.ndarray = np.zeros(0, dtype=np.int64)
        self.pair_split: list[str] = []
        self.train_paths: list[list[int]] = []
        self.counts: dict = {}
        self.mean: np.ndarray = np.zeros(5, dtype=np.float64)
        self.std: np.ndarray = np.ones(5, dtype=np.float64)
        self.file_hashes: dict = {}


def build_world() -> World:
    if sha256_file(PAIRS_PATH) != PAIRS_SHA256 or sha256_file(SEQUENCES_PATH) != SEQUENCES_SHA256:
        raise SystemExit("phase 1 sequence files do not match the audited hashes; refusing to invent a new pairing")
    pairs = load_jsonl(PAIRS_PATH)
    sequences = load_jsonl(SEQUENCES_PATH)
    if len(pairs) != 1400 or len(sequences) != 692:
        raise SystemExit(f"expected 1400 pairs and 692 sequences, found {len(pairs)} and {len(sequences)}")

    stage_by_member: dict[tuple[str, str], dict] = {}
    split_of: dict[str, str] = {}
    for sequence in sequences:
        split = sequence_split(sequence)
        split_of[sequence["sequence_id"]] = split
        for stage in sequence["stages"]:
            for member in stage["members"]:
                stage_by_member[(sequence["sequence_id"], member["dish_id"])] = stage

    world = World()
    world.file_hashes = {"valid_pairs.jsonl": PAIRS_SHA256, "valid_sequences.jsonl": SEQUENCES_SHA256}
    index_of: dict[tuple[str, str], int] = {}
    split_counts = {"train": 0, "test": 0, "quarantine": 0, "excluded_cafe2": 0}
    dish_ids_by_split: dict[str, set[str]] = {name: set() for name in split_counts}
    quarantine_ids: list[str] = []
    for sequence in sequences:
        split = split_of[sequence["sequence_id"]]
        split_counts[split] += 1
        if split == "quarantine":
            quarantine_ids.append(sequence["sequence_id"])
        for stage in sequence["stages"]:
            overhead = stage.get("overhead_dish_id")
            image = rgb_path(overhead)
            for member in stage["members"]:
                dish_ids_by_split[split].add(member["dish_id"])
            if split not in {"train", "test"} or image is None or overhead is None:
                continue
            targets = member_targets(stage, overhead)
            record = {
                "dish_id": overhead,
                "sequence_id": sequence["sequence_id"],
                "split": split,
                "position": int(stage["position"]),
                "targets": targets,
                "added": added_text(stage),
                "image": str(image),
            }
            index_of[(sequence["sequence_id"], overhead)] = len(world.stages)
            world.stages.append(record)

    before_idx = []
    after_idx = []
    pair_split = []
    dropped_pairs = 0
    for pair in pairs:
        sequence_id = pair["sequence_id"]
        split = split_of[sequence_id]
        if split not in {"train", "test"}:
            continue
        before = stage_by_member.get((sequence_id, pair["before_dish_id"]))
        after = stage_by_member.get((sequence_id, pair["after_dish_id"]))
        if before is None or after is None:
            raise RuntimeError(f"pair dishes are not in the sequence: {sequence_id}")
        for side in ("before", "after"):
            # Phase 1 leaves overhead empty exactly when that stage has no RGB.
            if pair.get(f"{side}_overhead") != pair[f"{side}_dish_id"] and rgb_path(pair.get(f"{side}_overhead")):
                raise RuntimeError(f"usable image id disagrees with the dish id in {sequence_id}")
        key_before = (sequence_id, pair["before_dish_id"])
        key_after = (sequence_id, pair["after_dish_id"])
        if key_before not in index_of or key_after not in index_of:
            dropped_pairs += 1
            continue
        left = index_of[key_before]
        right = index_of[key_after]
        if world.stages[left]["split"] != split or world.stages[right]["split"] != split:
            raise RuntimeError("a pair crossed split sides")
        before_idx.append(left)
        after_idx.append(right)
        pair_split.append(split)

    world.pair_before = np.asarray(before_idx, dtype=np.int64)
    world.pair_after = np.asarray(after_idx, dtype=np.int64)
    world.pair_split = pair_split

    positions_by_sequence: dict[str, list[int]] = {}
    indices_by_sequence: dict[str, list[int]] = {}
    for index, stage in enumerate(world.stages):
        if stage["split"] != "train":
            continue
        positions_by_sequence.setdefault(stage["sequence_id"], []).append(stage["position"])
        indices_by_sequence.setdefault(stage["sequence_id"], []).append(index)
    for sequence_id, indices in indices_by_sequence.items():
        world.train_paths.extend(contiguous_paths(positions_by_sequence[sequence_id], indices))

    train_rows = [stage for stage in world.stages if stage["split"] == "train"]
    if not train_rows:
        raise RuntimeError("no training images")
    train_targets = np.stack([stage["targets"] for stage in train_rows])
    world.mean = train_targets.mean(axis=0)
    world.std = np.maximum(train_targets.std(axis=0), 1e-3)

    used_in_train_pairs = set()
    for left, right, split in zip(world.pair_before.tolist(), world.pair_after.tolist(), world.pair_split):
        if split == "train":
            used_in_train_pairs.add(left)
            used_in_train_pairs.add(right)
    orphans = [
        index
        for index, stage in enumerate(world.stages)
        if stage["split"] == "train" and index not in used_in_train_pairs
    ]
    world.counts = {
        "sequences_loaded": len(sequences),
        "pairs_loaded": len(pairs),
        "split_sequences": split_counts,
        "quarantine_sequence_ids": quarantine_ids,
        "imaged_train_dishes": sum(stage["split"] == "train" for stage in world.stages),
        "imaged_test_dishes": sum(stage["split"] == "test" for stage in world.stages),
        "imaged_train_sequences": len({stage["sequence_id"] for stage in world.stages if stage["split"] == "train"}),
        "imaged_test_sequences": len({stage["sequence_id"] for stage in world.stages if stage["split"] == "test"}),
        "train_pairs": sum(split == "train" for split in world.pair_split),
        "test_pairs": sum(split == "test" for split in world.pair_split),
        "dropped_pairs_missing_rgb": dropped_pairs,
        "train_paths": len(world.train_paths),
        "orphan_train_dishes": len(orphans),
        "train_target_mean": {key: float(world.mean[i]) for i, key in enumerate(TARGET_KEYS)},
        "train_target_std": {key: float(world.std[i]) for i, key in enumerate(TARGET_KEYS)},
        "dish_ids_by_split_including_unimaged_members": {name: len(values) for name, values in dish_ids_by_split.items()},
    }
    world.orphan_indices = np.asarray(orphans, dtype=np.int64)
    _assert_world(world, dish_ids_by_split)
    return world


def _assert_world(world: World, dish_ids_by_split: dict[str, set[str]]) -> None:
    counts = world.counts["split_sequences"]
    if counts["train"] != 548 or counts["test"] != 90 or counts["quarantine"] != 1 or counts["excluded_cafe2"] != 53:
        raise RuntimeError(f"unexpected sequence split counts: {counts}")
    if world.counts["quarantine_sequence_ids"] != [LEAK_SEQUENCE]:
        raise RuntimeError(f"unexpected quarantine sequences: {world.counts['quarantine_sequence_ids']}")
    if not set(LEAK_DISHES).issubset(dish_ids_by_split["quarantine"]):
        raise RuntimeError("the known cross-split plate is not quarantined")
    overlap = dish_ids_by_split["train"] & dish_ids_by_split["test"]
    if overlap:
        raise RuntimeError(f"a dish is on both split sides: {sorted(overlap)[:5]}")
    if dish_ids_by_split["excluded_cafe2"] & (dish_ids_by_split["train"] | dish_ids_by_split["test"]):
        raise RuntimeError("cafe2 entered the train or test split")
    retake_sides = []
    for dish_id in RETAKE_EXAMPLE:
        sides = [name for name, values in dish_ids_by_split.items() if dish_id in values]
        retake_sides.append(sides)
    if any(side in {"train", "test"} for sides in retake_sides for side in sides):
        raise RuntimeError("the published cross-split retake example entered train or test")
    if world.mean[1] <= 1.0 or np.any(world.std <= 1.0):
        raise RuntimeError("training nutrition statistics are not usable cafe1 labels")
    if any(not np.isfinite(stage["targets"]).all() for stage in world.stages):
        raise RuntimeError("non-finite nutrition target")
    if any(stage["targets"][1] == 0.0 for stage in world.stages):
        raise RuntimeError("a supervised dish has zero dish-level energy")


def self_check() -> None:
    positions = [0, 1, 2, 4, 5]
    indices = [10, 11, 12, 13, 14]
    paths = contiguous_paths(positions, indices, min_len=3)
    if paths != [[10, 11, 12]]:
        raise RuntimeError(f"contiguous path split failed: {paths}")
    rng = np.random.default_rng(0)
    anchors = np.array(["a", "a", "b", "c"])
    pool = np.array(["a", "b", "c", "a"])
    partners = sample_partners(anchors, pool, rng)
    if np.any(pool[partners] == anchors):
        raise RuntimeError("a random partner stayed on the same plate")
    chains = sample_random_chains([3, 4], np.arange(6), np.array(["s0", "s1", "s2", "s3", "s4", "s5"]), rng)
    if [len(set(np.array(["s0", "s1", "s2", "s3", "s4", "s5"])[chain])) for chain in chains] != [3, 4]:
        raise RuntimeError("a random chain reused a plate")
    method = np.zeros((3, len(TARGET_KEYS)))
    reference = np.ones((3, len(TARGET_KEYS)))
    rows = bootstrap_mean_difference(method, reference, draws=200, seed=0)
    if rows[0]["relation"] != "entirely_below" or abs(rows[0]["difference"] + 1.0) > 1e-9:
        raise RuntimeError("bootstrap did not detect a constant improvement")
    tied = bootstrap_mean_difference(reference, reference, draws=50, seed=0)
    if tied[0]["relation"] != "includes_zero":
        raise RuntimeError("identical errors should not count as a stable gain")
    fake_win = [
        {"target": key, "relation": "entirely_below", "difference": -1.0, "ci95_low": -1.2, "ci95_high": -0.2}
        for key in TARGET_KEYS
    ]
    fake_mass = []
    for key in TARGET_KEYS:
        relation = "entirely_below" if key == "mass_g" else "includes_zero"
        fake_mass.append({"target": key, "relation": relation, "difference": -0.1, "ci95_low": -0.4, "ci95_high": 0.2 if key != "mass_g" else -0.05})
    decision = decide_stop(
        {
            "composition_difference": {"versus_baseline": fake_mass, "versus_control": fake_win},
            "alignment_path": {"versus_baseline": fake_mass, "versus_control": fake_win},
        }
    )
    if not decision["stop_rule_fired"] or "mass_only" not in decision["reason_codes"]:
        raise RuntimeError("a mass-only gain must fire the stop rule")
    decision_ok = decide_stop(
        {
            "composition_difference": {"versus_baseline": fake_win, "versus_control": fake_win},
            "alignment_path": {"versus_baseline": fake_mass, "versus_control": fake_win},
        }
    )
    if decision_ok["stop_rule_fired"]:
        raise RuntimeError("a nutrition gain that also beats random pairing must not fire the stop rule")
    similar = []
    for key in TARGET_KEYS:
        similar.append({"target": key, "relation": "includes_zero", "difference": -0.1, "ci95_low": -0.4, "ci95_high": 0.2})
    decision_similar = decide_stop(
        {
            "composition_difference": {"versus_baseline": fake_win, "versus_control": similar},
            "alignment_path": {"versus_baseline": fake_mass, "versus_control": similar},
        }
    )
    if not decision_similar["stop_rule_fired"] or "random_pairing_similar" not in decision_similar["reason_codes"]:
        raise RuntimeError("a similar random-pairing control must fire the stop rule")
    world = build_world()
    if world.counts["train_pairs"] <= 0 or world.counts["imaged_test_dishes"] <= 0:
        raise RuntimeError("image filter removed the whole train or test split")
    try:
        assert_output_dir(ROOT / "results" / "icdl_phase1_sequence_audit")
    except SystemExit:
        pass
    else:
        raise RuntimeError("the output guard did not refuse the phase 1 directory")
    print(
        "self-check ok",
        f"train_dishes {world.counts['imaged_train_dishes']}",
        f"test_dishes {world.counts['imaged_test_dishes']}",
        f"train_pairs {world.counts['train_pairs']}",
        f"test_pairs {world.counts['test_pairs']}",
        f"paths {world.counts['train_paths']}",
        f"orphans {world.counts['orphan_train_dishes']}",
        flush=True,
    )


def decide_stop(comparisons: dict[str, dict]) -> dict:
    """Pre-specified stop rule. A win needs energy and every macro, plus a real pairing effect."""
    reason_codes = []
    per_model = {}
    any_win = False
    for slug in ("composition_difference", "alignment_path"):
        versus_baseline = comparisons[slug]["versus_baseline"]
        versus_control = comparisons[slug]["versus_control"]
        beats = nutrition_supported(versus_baseline)
        only_mass = mass_only(versus_baseline)
        control_relation = {row["target"]: row["relation"] for row in versus_control}["kcal"]
        control_supports = control_relation == "entirely_below"
        win = bool(beats and control_supports)
        any_win = any_win or win
        codes = []
        if not beats:
            codes.append("does_not_stably_beat_baseline_on_energy_and_macros")
        if only_mass:
            codes.append("mass_only")
        if beats and not control_supports:
            codes.append("random_pairing_similar" if control_relation == "includes_zero" else "random_pairing_better")
        if not beats:
            energy = next(row for row in versus_baseline if row["target"] == "kcal")
            if energy["difference"] < 0.0 and energy["relation"] != "entirely_below":
                codes.append("bootstrap_does_not_support")
        per_model[slug] = {"win": win, "reason_codes": codes}
        reason_codes.extend(codes)
    return {
        "stop_rule_fired": not any_win,
        "any_primary_model_wins": any_win,
        "reason_codes": sorted(set(reason_codes)),
        "per_model": per_model,
        "phase3_started": False,
        "rule": (
            "A primary difference model wins only when the sequence-equal 95% interval of "
            "(its MAE - the single-image MAE) lies entirely below 0 for energy, fat, carb, and protein, "
            "and the same interval against its random cross-plate control lies entirely below 0 for energy. "
            "Otherwise a stop rule fires. Ablations are not eligible to be selected."
        ),
    }


def _chunks(indices: np.ndarray, size: int) -> list[np.ndarray]:
    if indices.size == 0:
        return []
    return [indices[start : start + size] for start in range(0, indices.size, size)]


def _flip_batch(images, mask):
    import torch

    if not bool(mask.any()):
        return images
    flipped = images.clone()
    flipped[mask] = flipped[mask].flip(-1)
    return flipped


def load_image_tensor(world: World, device: str):
    import torch
    from PIL import Image

    arrays = []
    sizes = []
    for stage in world.stages:
        with Image.open(stage["image"]) as image:
            rgb = image.convert("RGB")
            sizes.append(rgb.size)
            resized = rgb.resize((IMAGE_SIZE, IMAGE_SIZE), Image.BILINEAR)
            array = np.asarray(resized, dtype=np.float32) / 255.0
        array = (array - IMAGENET_MEAN) / IMAGENET_STD
        arrays.append(array.transpose(2, 0, 1))
    images = torch.from_numpy(np.stack(arrays)).to(device)
    raw = torch.from_numpy(np.stack([stage["targets"] for stage in world.stages])).to(device=device, dtype=torch.float32)
    mean = torch.tensor(world.mean, dtype=torch.float32, device=device)
    std = torch.tensor(world.std, dtype=torch.float32, device=device)
    standardized = (raw - mean) / std
    widths = np.array([size[0] for size in sizes])
    heights = np.array([size[1] for size in sizes])
    world.counts["image_width_median"] = float(np.median(widths))
    world.counts["image_height_median"] = float(np.median(heights))
    return images, standardized, raw


def build_encoder():
    import torch
    import torch.nn as nn
    import torch.nn.functional as F

    class Attention(nn.Module):
        def __init__(self, dim: int, num_heads: int) -> None:
            super().__init__()
            self.num_heads = num_heads
            self.head_dim = dim // num_heads
            self.scale = self.head_dim**-0.5
            self.qkv = nn.Linear(dim, dim * 3, bias=True)
            self.proj = nn.Linear(dim, dim, bias=True)

        def forward(self, x):
            batch, length, channels = x.shape
            qkv = self.qkv(x).reshape(batch, length, 3, self.num_heads, self.head_dim)
            qkv = qkv.permute(2, 0, 3, 1, 4)
            query, key, value = qkv[0], qkv[1], qkv[2]
            scores = (query * self.scale) @ key.transpose(-2, -1)
            context = scores.softmax(dim=-1) @ value
            context = context.transpose(1, 2).reshape(batch, length, channels)
            return self.proj(context)

    class Mlp(nn.Module):
        def __init__(self, dim: int, hidden: int) -> None:
            super().__init__()
            self.fc1 = nn.Linear(dim, hidden)
            self.act = nn.GELU()
            self.fc2 = nn.Linear(hidden, dim)

        def forward(self, x):
            return self.fc2(self.act(self.fc1(x)))

    class LayerScale(nn.Module):
        def __init__(self, dim: int) -> None:
            super().__init__()
            self.gamma = nn.Parameter(torch.ones(dim) * 1e-5)

        def forward(self, x):
            return x * self.gamma

    class Block(nn.Module):
        def __init__(self, dim: int, num_heads: int) -> None:
            super().__init__()
            self.norm1 = nn.LayerNorm(dim, eps=1e-6)
            self.attn = Attention(dim, num_heads)
            self.ls1 = LayerScale(dim)
            self.norm2 = nn.LayerNorm(dim, eps=1e-6)
            self.mlp = Mlp(dim, dim * 4)
            self.ls2 = LayerScale(dim)

        def forward(self, x):
            x = x + self.ls1(self.attn(self.norm1(x)))
            x = x + self.ls2(self.mlp(self.norm2(x)))
            return x

    class Encoder(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.patch_size = 14
            self.cls_token = nn.Parameter(torch.zeros(1, 1, 384))
            self.pos_embed = nn.Parameter(torch.zeros(1, 1370, 384))
            self.mask_token = nn.Parameter(torch.zeros(1, 384))
            self.blocks = nn.ModuleList(Block(384, 6) for _ in range(12))
            self.norm = nn.LayerNorm(384, eps=1e-6)
            self.patch_embed = nn.Module()
            # The checkpoint names the patch projection patch_embed.proj.*.
            self.patch_embed.proj = nn.Conv2d(3, 384, kernel_size=14, stride=14)

        def interpolate_pos_encoding(self, tokens, height: int, width: int):
            patches = tokens.shape[1] - 1
            stored = self.pos_embed.shape[1] - 1
            if patches == stored and height == width:
                return self.pos_embed
            pos = self.pos_embed.float()
            class_pos = pos[:, :1]
            patch_pos = pos[:, 1:]
            dim = tokens.shape[-1]
            grid = int(math.sqrt(stored))
            target_w = width // self.patch_size + 0.1
            target_h = height // self.patch_size + 0.1
            patch_pos = patch_pos.reshape(1, grid, grid, dim).permute(0, 3, 1, 2)
            patch_pos = F.interpolate(
                patch_pos,
                scale_factor=(target_h / grid, target_w / grid),
                mode="bicubic",
                align_corners=False,
            )
            if patch_pos.shape[-2] != int(target_h) or patch_pos.shape[-1] != int(target_w):
                raise RuntimeError(f"position interpolation produced {tuple(patch_pos.shape)}")
            patch_pos = patch_pos.permute(0, 2, 3, 1).reshape(1, -1, dim)
            return torch.cat((class_pos, patch_pos), dim=1).to(dtype=tokens.dtype)

        def forward_features(self, images):
            batch, _channels, height, width = images.shape
            tokens = self.patch_embed.proj(images).flatten(2).transpose(1, 2)
            cls = self.cls_token.expand(batch, -1, -1)
            tokens = torch.cat((cls, tokens), dim=1)
            tokens = tokens + self.interpolate_pos_encoding(tokens, height, width)
            for block in self.blocks:
                tokens = block(tokens)
            return self.norm(tokens)[:, 0]

    encoder = Encoder()
    if sha256_file(DINOV2_WEIGHTS) != DINOV2_SHA256:
        raise SystemExit("DINOv2 ViT-S/14 weights do not match the local audited file")
    import torch

    state = torch.load(DINOV2_WEIGHTS, map_location="cpu", weights_only=False)
    encoder.load_state_dict(state, strict=True)
    return encoder


def build_model(spec: Spec):
    import torch.nn as nn

    def head(in_dim: int) -> nn.Sequential:
        network = nn.Sequential(
            nn.Linear(in_dim, HEAD_HIDDEN),
            nn.GELU(),
            nn.Linear(HEAD_HIDDEN, len(TARGET_KEYS)),
        )
        last = network[-1]
        nn.init.zeros_(last.weight)
        nn.init.zeros_(last.bias)
        return network

    class Regressor(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.feature = spec.feature
            self.encoder = build_encoder()
            self.absolute = head(384)
            if spec.feature == "difference":
                self.difference = head(384)
            elif spec.feature == "concat":
                self.difference = head(768)
            else:
                self.difference = None

        def encode(self, images):
            return self.encoder.forward_features(images)

        def predict_absolute(self, features):
            return self.absolute(features)

        def predict_delta(self, left, right):
            import torch

            if self.difference is None:
                raise RuntimeError("the single-image model has no difference head")
            if self.feature == "difference":
                features = right - left
            elif self.feature == "concat":
                features = torch.cat((left, right), dim=-1)
            else:
                raise RuntimeError(self.feature)
            return self.difference(features)

    import torch

    return Regressor()


def set_seed(seed: int) -> None:
    import torch

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _pair_views(spec: Spec, epoch: int, train_pair_pos: np.ndarray, world: World, n_train: int):
    flip_rng = np.random.default_rng(SEED + 30_000 + epoch)
    flips = flip_rng.random(train_pair_pos.size) < 0.5
    partners = None
    paths = None
    if spec.pair_source == "random":
        partner_rng = np.random.default_rng(SEED + 10_000 + epoch)
        pool = np.flatnonzero(np.array([stage["split"] == "train" for stage in world.stages]))
        pool_sequences = np.array([world.stages[index]["sequence_id"] for index in pool])
        anchors = np.array([world.stages[world.pair_before[pos]]["sequence_id"] for pos in train_pair_pos])
        # sample_partners indexes the pool; map back to stage indices.
        local = sample_partners(anchors, pool_sequences, partner_rng)
        partners = pool[local]
        if spec.lambda_path:
            path_rng = np.random.default_rng(SEED + 20_000 + epoch)
            lengths = [len(path) for path in world.train_paths]
            paths = sample_random_chains(lengths, pool, pool_sequences, path_rng)
    elif spec.pair_source == "adjacent" and spec.lambda_path:
        paths = world.train_paths
    return flips, partners, paths


def train_one(spec: Spec, world: World, images, targets, device: str, epochs: int) -> dict:
    import torch
    import torch.nn.functional as F

    set_seed(SEED)
    model = build_model(spec).to(device)
    backbone = [parameter for parameter in model.encoder.parameters() if parameter.requires_grad]
    heads = [parameter for name, parameter in model.named_parameters() if not name.startswith("encoder.")]
    optimizer = torch.optim.AdamW(
        [
            {"params": backbone, "lr": LR_BACKBONE},
            {"params": heads, "lr": LR_HEAD},
        ],
        weight_decay=WEIGHT_DECAY,
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    train_pair_pos = np.array([pos for pos, split in enumerate(world.pair_split) if split == "train"], dtype=np.int64)
    curve = []
    steps = 0
    for epoch in range(epochs):
        order_rng = np.random.default_rng(SEED + epoch)
        order = order_rng.permutation(train_pair_pos.size)
        flips, partners, paths = _pair_views(spec, epoch, train_pair_pos, world, train_pair_pos.size)
        batches = _chunks(order, BATCH_SIZE)
        path_groups = [[] for _ in batches]
        if paths:
            for index, path in enumerate(paths):
                path_groups[index % len(batches)].append(path)
        model.train()
        epoch_loss = 0.0
        seen = 0
        for batch, path_group in zip(batches, path_groups):
            loss, n_images = _batch_loss(
                model, spec, world, images, targets, train_pair_pos, batch, flips, partners, path_group, F
            )
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
            optimizer.step()
            steps += 1
            epoch_loss += float(loss.detach()) * n_images
            seen += n_images
        if world.orphan_indices.size:
            for orphan_batch in _chunks(world.orphan_indices, BATCH_SIZE * 2):
                orphan_flip = np.random.default_rng(SEED + 40_000 + epoch + int(orphan_batch[0])).random(orphan_batch.size) < 0.5
                orphan_index = torch.as_tensor(orphan_batch, device=device)
                view = _flip_batch(images[orphan_index], torch.as_tensor(orphan_flip, device=device))
                prediction = model.predict_absolute(model.encode(view))
                loss = F.smooth_l1_loss(prediction, targets[orphan_index], beta=1.0)
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
                optimizer.step()
                steps += 1
                epoch_loss += float(loss.detach()) * int(orphan_batch.size)
                seen += int(orphan_batch.size)
        scheduler.step()
        train_mae = _mae_raw(model, np.flatnonzero(np.array([s["split"] == "train" for s in world.stages])), images, world)
        row = {
            "epoch": epoch + 1,
            "mean_loss": epoch_loss / max(seen, 1),
            "train_mae": {key: float(train_mae[index]) for index, key in enumerate(TARGET_KEYS)},
        }
        curve.append(row)
        print(
            f"{spec.name} epoch {epoch + 1}/{epochs} loss {row['mean_loss']:.4f} "
            f"train_mass {train_mae[0]:.2f} train_kcal {train_mae[1]:.2f}",
            flush=True,
        )
    expected = epochs * (math.ceil(train_pair_pos.size / BATCH_SIZE) + (math.ceil(world.orphan_indices.size / (BATCH_SIZE * 2)) if world.orphan_indices.size else 0))
    if steps != expected:
        raise RuntimeError(f"{spec.slug} took {steps} steps, expected {expected}")
    return {"model": model, "curve": curve, "optimizer_steps": steps}


def _batch_loss(model, spec: Spec, world: World, images, targets, train_pair_pos, batch, flips, partners, path_group, functional):
    import torch

    pair_pos = train_pair_pos[batch]
    left = torch.as_tensor(world.pair_before[pair_pos], device=images.device)
    right = torch.as_tensor(world.pair_after[pair_pos], device=images.device)
    mask = torch.as_tensor(flips[batch], device=images.device)
    left_img = _flip_batch(images[left], mask)
    right_img = _flip_batch(images[right], mask)
    left_h = model.encode(left_img)
    right_h = model.encode(right_img)
    left_y = model.predict_absolute(left_h)
    right_y = model.predict_absolute(right_h)
    absolute = functional.smooth_l1_loss(
        torch.cat((left_y, right_y), dim=0),
        torch.cat((targets[left], targets[right]), dim=0),
        beta=1.0,
    )
    loss = absolute
    if spec.pair_source == "none":
        return loss, int(left.shape[0] * 2)
    if spec.pair_source == "random":
        partner_index = torch.as_tensor(partners[batch], device=images.device)
        partner_img = _flip_batch(images[partner_index], mask)
        partner_h = model.encode(partner_img)
        delta_right_h = partner_h
        delta_right_y = targets[partner_index]
        cons_right = model.predict_absolute(partner_h)
    else:
        delta_right_h = right_h
        delta_right_y = targets[right]
        cons_right = right_y
    delta_hat = model.predict_delta(left_h, delta_right_h)
    delta_true = delta_right_y - targets[left]
    if spec.lambda_delta:
        loss = loss + spec.lambda_delta * functional.smooth_l1_loss(delta_hat, delta_true, beta=1.0)
    if spec.lambda_cons:
        loss = loss + spec.lambda_cons * functional.smooth_l1_loss(cons_right - left_y, delta_hat, beta=1.0)
    if spec.lambda_path and path_group:
        loss = loss + spec.lambda_path * _path_loss(model, images, targets, path_group, functional)
    return loss, int(left.shape[0] * 2)


def _path_loss(model, images, targets, paths: list[list[int]], functional):
    import torch

    losses = []
    for path in paths:
        index = torch.as_tensor(path, device=images.device)
        features = model.encode(images[index])
        steps = [model.predict_delta(features[cursor], features[cursor + 1]) for cursor in range(len(path) - 1)]
        predicted = torch.stack(steps, dim=0).sum(dim=0)
        truth = targets[index[-1]] - targets[index[0]]
        losses.append(functional.smooth_l1_loss(predicted, truth, beta=1.0))
    return torch.stack(losses).mean()


def torch_inference(func):
    def wrapped(*args, **kwargs):
        import torch

        with torch.no_grad():
            return func(*args, **kwargs)

    return wrapped


@torch_inference
def _mae_raw(model, indices: np.ndarray, images, world: World) -> np.ndarray:
    prediction = predict_standardized(model, indices, images)
    raw = prediction * world.std + world.mean
    truth = np.stack([world.stages[int(index)]["targets"] for index in indices])
    return np.abs(raw - truth).mean(axis=0)


@torch_inference
def predict_standardized(model, indices: np.ndarray, images) -> np.ndarray:
    import torch

    model.eval()
    outputs = []
    for batch in _chunks(np.asarray(indices, dtype=np.int64), 64):
        features = model.encode(images[torch.as_tensor(batch, device=images.device)])
        outputs.append(model.predict_absolute(features).detach().cpu().numpy())
    return np.concatenate(outputs, axis=0)


@torch_inference
def predict_pair_deltas(model, left: np.ndarray, right: np.ndarray, images) -> np.ndarray:
    import torch

    model.eval()
    outputs = []
    order = np.arange(left.size)
    for batch in _chunks(order, 32):
        left_h = model.encode(images[torch.as_tensor(left[batch], device=images.device)])
        right_h = model.encode(images[torch.as_tensor(right[batch], device=images.device)])
        outputs.append(model.predict_delta(left_h, right_h).detach().cpu().numpy())
    return np.concatenate(outputs, axis=0)


def evaluate(world: World, predictions: dict[str, np.ndarray], draws: int) -> dict:
    test_idx = np.array([index for index, stage in enumerate(world.stages) if stage["split"] == "test"], dtype=np.int64)
    truth = np.stack([world.stages[int(index)]["targets"] for index in test_idx])
    sequence_ids = np.array([world.stages[int(index)]["sequence_id"] for index in test_idx])
    raw = {slug: pred * world.std + world.mean for slug, pred in predictions.items()}
    abs_err = {slug: np.abs(values - truth) for slug, values in raw.items()}
    seq_mae = {}
    for slug, err in abs_err.items():
        per_seq, _uniq = per_sequence_mae(err, sequence_ids)
        seq_mae[slug] = per_seq
    mean_raw = np.repeat(world.mean.reshape(1, -1), truth.shape[0], axis=0)
    mean_err = np.abs(mean_raw - truth)
    mean_seq, _uniq = per_sequence_mae(mean_err, sequence_ids)
    comparisons = {}
    for slug in predictions:
        if slug == "single_image":
            continue
        comparisons[slug] = {
            "versus_baseline_sequence_equal": bootstrap_mean_difference(
                seq_mae[slug], seq_mae["single_image"], draws=draws, seed=BOOTSTRAP_SEED
            ),
            "versus_baseline_dish_weighted": dish_weighted_bootstrap(
                abs_err[slug], abs_err["single_image"], sequence_ids, draws=draws, seed=BOOTSTRAP_SEED
            ),
        }
    for primary, control in CONTROL_OF.items():
        comparisons[primary]["versus_control_sequence_equal"] = bootstrap_mean_difference(
            seq_mae[primary], seq_mae[control], draws=draws, seed=BOOTSTRAP_SEED
        )
    decision_input = {}
    for slug in ("composition_difference", "alignment_path"):
        decision_input[slug] = {
            "versus_baseline": comparisons[slug]["versus_baseline_sequence_equal"],
            "versus_control": comparisons[slug]["versus_control_sequence_equal"],
        }
    test_pairs = np.array([pos for pos, split in enumerate(world.pair_split) if split == "test"], dtype=np.int64)
    return {
        "n_test_dishes": int(test_idx.size),
        "n_test_sequences": int(len(set(sequence_ids.tolist()))),
        "sequence_equal_mae": {
            slug: {key: float(seq_mae[slug][:, index].mean()) for index, key in enumerate(TARGET_KEYS)}
            for slug in predictions
        },
        "training_mean_sequence_equal_mae": {
            key: float(mean_seq[:, index].mean()) for index, key in enumerate(TARGET_KEYS)
        },
        "dish_micro_mae": {
            slug: {key: float(abs_err[slug][:, index].mean()) for index, key in enumerate(TARGET_KEYS)}
            for slug in predictions
        },
        "comparisons": comparisons,
        "decision": decide_stop(decision_input),
        "test_idx": test_idx,
        "truth": truth,
        "raw": raw,
        "abs_err": abs_err,
        "sequence_ids": sequence_ids,
        "test_pair_pos": test_pairs,
        "mean_abs_err": mean_err,
    }


def delta_diagnostic(world: World, models: dict, images, evaluation: dict) -> dict:
    positions = evaluation["test_pair_pos"]
    if positions.size == 0:
        return {"n_pairs": 0, "n_sequences": 0, "models": {}}
    left = world.pair_before[positions]
    right = world.pair_after[positions]
    truth = np.stack([world.stages[int(index)]["targets"] for index in right]) - np.stack(
        [world.stages[int(index)]["targets"] for index in left]
    )
    sequence_ids = np.array([world.stages[int(index)]["sequence_id"] for index in left])
    rows = {}
    implied = {}
    for slug, raw in evaluation["raw"].items():
        # Map stage index -> row in the test prediction matrix.
        lookup = {int(stage_index): row for row, stage_index in enumerate(evaluation["test_idx"].tolist())}
        left_row = np.array([lookup[int(index)] for index in left])
        right_row = np.array([lookup[int(index)] for index in right])
        implied[slug] = raw[right_row] - raw[left_row]
    for slug, model in models.items():
        if SPEC_BY_SLUG[slug].feature == "none":
            continue
        standardized = predict_pair_deltas(model, left, right, images)
        predicted = standardized * world.std
        rows[slug] = _delta_summary(np.abs(predicted - truth), sequence_ids)
    for slug, values in implied.items():
        rows[f"{slug}__absolute_implied"] = _delta_summary(np.abs(values - truth), sequence_ids)
    return {"n_pairs": int(positions.size), "n_sequences": int(len(set(sequence_ids.tolist()))), "models": rows}


def _delta_summary(abs_err: np.ndarray, sequence_ids: np.ndarray) -> dict:
    per_seq, _uniq = per_sequence_mae(abs_err, sequence_ids)
    return {
        "sequence_equal_mae": {key: float(per_seq[:, index].mean()) for index, key in enumerate(TARGET_KEYS)},
        "dish_micro_mae": {key: float(abs_err[:, index].mean()) for index, key in enumerate(TARGET_KEYS)},
    }


def failure_examples(world: World, evaluation: dict, slug: str, limit: int = 8) -> list[dict]:
    base = evaluation["abs_err"]["single_image"]
    method = evaluation["abs_err"][slug]
    energy_gap = method[:, 1] - base[:, 1]
    order = np.argsort(-energy_gap)
    examples = []
    for row in order[:limit]:
        stage_index = int(evaluation["test_idx"][row])
        stage = world.stages[stage_index]
        truth = evaluation["truth"][row]
        examples.append(
            {
                "sequence_id": stage["sequence_id"],
                "dish_id": stage["dish_id"],
                "position": stage["position"],
                "added": stage["added"],
                "energy_abs_error_increase_kcal": float(energy_gap[row]),
                "true": {key: float(truth[index]) for index, key in enumerate(TARGET_KEYS)},
                "baseline_abs_error": {key: float(base[row, index]) for index, key in enumerate(TARGET_KEYS)},
                "model_abs_error": {key: float(method[row, index]) for index, key in enumerate(TARGET_KEYS)},
                "model": SPEC_BY_SLUG[slug].name,
            }
        )
    mass_better_energy_worse = int(np.sum((method[:, 0] < base[:, 0]) & (method[:, 1] > base[:, 1])))
    return examples, {
        "n_test_dishes": int(method.shape[0]),
        "n_mass_better_and_energy_worse": mass_better_energy_worse,
    }


def write_predictions(path: Path, world: World, evaluation: dict) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for slug, raw in evaluation["raw"].items():
            for row, stage_index in enumerate(evaluation["test_idx"].tolist()):
                stage = world.stages[int(stage_index)]
                record = {
                    "model": SPEC_BY_SLUG[slug].name,
                    "slug": slug,
                    "split": "test",
                    "sequence_id": stage["sequence_id"],
                    "dish_id": stage["dish_id"],
                    "position": stage["position"],
                }
                for index, key in enumerate(TARGET_KEYS):
                    record[key] = float(stage["targets"][index])
                    record[f"pred_{key}"] = float(raw[row, index])
                handle.write(json.dumps(record, ensure_ascii=False) + "\n")


def _fmt(value: float) -> str:
    return f"{value:.4f}"


def _row_table(headers: list[str], rows: list[list[str]]) -> list[str]:
    lines = ["| " + " | ".join(headers) + " |", "| " + " | ".join("---" for _ in headers) + " |"]
    for row in rows:
        lines.append("| " + " | ".join(row) + " |")
    return lines


def render_report(payload: dict) -> str:
    counts = payload["counts"]
    decision = payload["decision"]
    mae = payload["evaluation"]["sequence_equal_mae"]
    mean_mae = payload["evaluation"]["training_mean_sequence_equal_mae"]
    official = payload["official_protocol"]
    lines = [
        "# ICDL 第二阶段轻量实验",
        "",
        "结论先说。三个轻量模型共用 DINOv2 ViT-S/14，训练预算相同，部署时只看一张俯视 RGB。"
        "评价按第一阶段恢复的物理餐盘序列等权，不把每个 dish_id 当成独立样本。",
        "",
    ]
    if not official:
        lines.append("这次运行改过预先规定的 epoch、批量或 bootstrap 次数，下面的停止判断不能当成正式第二阶段结论。")
        lines.append("")
    if decision["stop_rule_fired"]:
        lines.append(
            "**停止规则已触发。** 预先规定的差异模型没有同时满足：能量和三种宏量都稳定低于同骨干、同预算的单图像基线，"
            "并且这种优势相对跨盘随机配对也站得住。不能声称模型学到了组成变化。第三阶段没有开始，完整 ICDL 网络没有开始。"
        )
    else:
        lines.append(
            "**预先规定的停止规则这次没有触发。** 至少有一个预先规定的差异模型，在能量和三种宏量上稳定低于单图像基线，"
            "并且在能量上稳定低于它自己的跨盘随机配对对照。这仍不是论文贡献，也不解锁第三阶段。完整网络没有开始。"
        )
    lines.extend(
        [
            "",
            "## 做了哪三个模型",
            "",
            "方法按官方方案写，不用字母编号代替方法名。DINOv2 ViT-S/14 只是这次共用的骨干候选，不是已经选定的最终结构。",
            "",
            "目标是五个数：质量（克）、能量（千卡）、脂肪（克）、碳水（克）、蛋白质（克）。"
            "训练前只用 cafe1 训练集计算每个目标的均值和标准差，损失在标准化后的数上计算，报告时换回原单位。"
            "绝对营养头在部署时是唯一使用的头。它的最后一层从零开始，所以训练起点等于训练集均值。",
            "",
            "1. **普通 DINOv2 加单图像营养回归。** 一张俯视 RGB 进共享骨干，再进一个两层头，用平滑 L1 回归这五个数。"
            "这是同骨干、同预算的基线。它不读取第二张图。",
            "2. **单图像基线加组成差异监督。** 骨干和绝对头与基线相同。训练时另有一个头，输入相邻两阶段特征的拼接，"
            "预测营养差。这个头只在训练时使用。推理仍是单张 RGB。",
            "3. **差异特征对齐、营养差异预测与路径一致性。** 两阶段共用同一个骨干，差异特征是后一阶段特征减去前一阶段特征。"
            "营养差异头读这个差值。单步预测一致要求绝对头的预测差接近营养差异头的预测。"
            "跨阶段路径一致要求，一条不少于三个连续阶段的链上，相邻营养差预测之和接近整段营养差。"
            "三项权重预先都取 1，不做测试集搜索。消融是把其中一项设为 0，或把特征相减换成特征拼接。",
            "",
            "负对照把不同物理餐盘的扫描随机配成同样数量的对，架构、损失权重和优化步数与对应的真实相邻模型相同。"
            "绝对营养损失始终用真实标签，不使用 cafe2 的盘级零能量。随机链也不会落在同一个物理餐盘上。",
            "",
            "整图像素差和结构相似度没有进入损失。侧视和深度没有进入训练或推理。质量估计器、动作价值和阈值都没有重训。",
            "",
            "## 数据和划分",
            "",
            f"序列文件是第一阶段已经落盘的 `valid_pairs.jsonl` 与 `valid_sequences.jsonl`，对应开工提交 `{STATUS_COMMIT}`。"
            f"文件 sha256 分别是 `{PAIRS_SHA256}` 与 `{SEQUENCES_SHA256}`。有效相邻阶段对 {counts['pairs_loaded']}，"
            f"独立序列 {counts['sequences_loaded']}。",
            "",
            "划分规则在看测试误差之前就定死了。cafe1 里官方两侧都在训练的序列进入训练，两侧都在测试的序列进入测试。"
            f"跨官方 RGB 和深度划分的 `{LEAK_SEQUENCE}`（`{LEAK_DISHES[0]}` 在测试，`{LEAK_DISHES[1]}` 在训练，新增 green beans 17 克）"
            "整条移出训练和测试。cafe2 的 53 条序列不进入监督，因为盘级能量和三种宏量是 0，而且它们不在官方 RGB 清单上。"
            "分叉、组件拼盘和前驱不唯一的组不在这两个文件里，没有加回。92 组跨划分重复拍摄也没有加回；"
            f"报告里的例子 `{RETAKE_EXAMPLE[0]}` 与 `{RETAKE_EXAMPLE[1]}` 不在训练或测试中。",
            "",
        ]
    )
    split_rows = [
        ["cafe1 官方训练序列", str(counts["split_sequences"]["train"])],
        ["cafe1 官方测试序列", str(counts["split_sequences"]["test"])],
        ["整条隔离的跨划分序列", str(counts["split_sequences"]["quarantine"])],
        ["排除的 cafe2 序列", str(counts["split_sequences"]["excluded_cafe2"])],
        ["有俯视 RGB 的训练盘", str(counts["imaged_train_dishes"])],
        ["有俯视 RGB 的测试盘", str(counts["imaged_test_dishes"])],
        ["两侧都有图的训练阶段对", str(counts["train_pairs"])],
        ["两侧都有图的测试阶段对", str(counts["test_pairs"])],
        ["缺图而丢掉的阶段对", str(counts["dropped_pairs_missing_rgb"])],
        ["训练里长度不少于 3 的连续路径", str(counts["train_paths"])],
        ["没有完整阶段对的训练盘", str(counts["orphan_train_dishes"])],
    ]
    lines.extend(_row_table(["项目", "数量"], split_rows))
    lines.extend(
        [
            "",
            f"俯视图像中位尺寸是 {counts.get('image_width_median', 0):.0f}×{counts.get('image_height_median', 0):.0f}，"
            f"缩放到 {IMAGE_SIZE}×{IMAGE_SIZE}。训练集目标均值（质量、能量、脂肪、碳水、蛋白质）是 "
            + "、".join(_fmt(counts["train_target_mean"][key]) for key in TARGET_KEYS)
            + "。标准差是 "
            + "、".join(_fmt(counts["train_target_std"][key]) for key in TARGET_KEYS)
            + "。",
            "",
            "## 训练预算",
            "",
            f"每个模型 {payload['epochs']} 个 epoch，AdamW，骨干学习率 {LR_BACKBONE:g}，头学习率 {LR_HEAD:g}，"
            f"权重衰减 {WEIGHT_DECAY:g}，梯度裁剪 {GRAD_CLIP:g}，余弦学习率，种子 {payload['seed']}。"
            f"每个 step 看 {BATCH_SIZE} 个阶段对。没有完整阶段对的训练盘，所有模型都用同样的额外 step 做单图像回归。"
            f"实际优化步数是 {payload['optimizer_steps']}，九个运行彼此相同。"
            "水平翻转在一对图上使用同一个随机位，避免把镜像错当成组成变化。测试时不翻转。",
            "",
            "骨干权重是公开的 DINOv2 ViT-S/14 预训练文件，本地 sha256 "
            f"`{DINOV2_SHA256}`。营养头是新的，不加载历史上的质量头。",
            "",
            "## 主表",
            "",
            "主指标是测试集上按物理餐盘序列等权的平均绝对误差：先在每条序列内对盘子取平均，再对序列取平均。"
            f"测试盘 {payload['evaluation']['n_test_dishes']} 个，序列 {payload['evaluation']['n_test_sequences']} 条。"
            f"稳定低于的事先规则是：该方法与参照的序列等权 MAE 之差，做 {payload['bootstrap_draws']} 次、种子 {payload['seed']} 的序列有放回重抽样，"
            "2.5% 到 97.5% 分位区间整体落在 0 以下。要算赢，能量、脂肪、碳水、蛋白质四项都得稳定低于单图像基线；"
            "只在质量上稳定低于不算赢。区间跨过 0 就不算 bootstrap 支持。",
            "",
            "下面的差是「该方法 − 单图像基线」。负数表示该方法更低。",
            "",
        ]
    )
    headers = ["模型", "质量", "能量", "脂肪", "碳水", "蛋白质"]
    body = [["训练集均值参照", *[_fmt(mean_mae[key]) for key in TARGET_KEYS]]]
    for spec in SPECS:
        body.append([spec.name, *[_fmt(mae[spec.slug][key]) for key in TARGET_KEYS]])
    lines.extend(_row_table(headers, body))
    lines.extend(["", "### 相对单图像基线", ""])
    diff_headers = ["模型", "目标", "差", "95% 区间", "区间关系"]
    diff_rows = []
    for spec in SPECS:
        if spec.slug == "single_image":
            continue
        for row in payload["evaluation"]["comparisons"][spec.slug]["versus_baseline_sequence_equal"]:
            diff_rows.append(
                [
                    spec.name,
                    TARGET_LABELS[row["target"]],
                    _fmt(row["difference"]),
                    f"[{_fmt(row['ci95_low'])}, {_fmt(row['ci95_high'])}]",
                    {
                        "entirely_below": "整体低于 0",
                        "entirely_above": "整体高于 0",
                        "includes_zero": "跨过 0",
                    }[row["relation"]],
                ]
            )
    lines.extend(_row_table(diff_headers, diff_rows))
    lines.extend(
        [
            "",
            "按盘子等权、只在重抽样时按序列整组抽取的敏感性区间写在 `comparison.json` 的 `versus_baseline_dish_weighted`。"
            "停止规则不看这组数，避免事后改成比较好看的那一种。",
            "",
            "## 负对照",
            "",
            "相近的事先定义是：真实相邻配对与随机跨盘配对的序列等权能量 MAE 之差，其 95% 区间包含 0。"
            "区间整体高于 0 表示随机配对反而更低，同样不能把收益写成组成变化。",
            "",
        ]
    )
    control_rows = []
    for primary, control in CONTROL_OF.items():
        for row in payload["evaluation"]["comparisons"][primary]["versus_control_sequence_equal"]:
            control_rows.append(
                [
                    SPEC_BY_SLUG[primary].name,
                    TARGET_LABELS[row["target"]],
                    _fmt(row["difference"]),
                    f"[{_fmt(row['ci95_low'])}, {_fmt(row['ci95_high'])}]",
                    {
                        "entirely_below": "真实配对整体更低",
                        "entirely_above": "随机配对整体更低",
                        "includes_zero": "相近，区间跨过 0",
                    }[row["relation"]],
                ]
            )
    lines.extend(_row_table(["真实配对模型", "目标", "真实 − 随机", "95% 区间", "判断"], control_rows))
    lines.extend(["", "## 损失权重消融", ""])
    lines.append(
        "消融在同一个测试集上报告，因为划分只有官方测试一侧，没有再留验证集去做第二次选择。"
        "这些行是训练前写死的网格。脚本不会根据表里的最低误差再训一个模型，也不会把某一行改写成主结果。"
    )
    lines.append("")
    lines.extend(
        _row_table(
            ["消融", "营养差异预测", "单步一致", "路径一致", "特征", "能量 MAE", "能量相对基线的区间"],
            [
                [
                    spec.name,
                    _fmt(spec.lambda_delta),
                    _fmt(spec.lambda_cons),
                    _fmt(spec.lambda_path),
                    "特征相减" if spec.feature == "difference" else "特征拼接",
                    _fmt(mae[spec.slug]["kcal"]),
                    _interval_text(payload, spec.slug, "kcal"),
                ]
                for spec in SPECS
                if spec.role == "ablation"
            ],
        )
    )
    lines.extend(
        [
            "",
            "## 营养差诊断",
            "",
            "这一节不是部署指标。部署只看单张图的绝对头。这里把测试集上真实相邻阶段对的营养差，"
            "拿来看差异头有没有比随机配对更接近真实增量。绝对头隐含的预测差也一并写上。",
            "",
        ]
    )
    diagnostic = payload["delta_diagnostic"]
    lines.append(
        f"测试阶段对 {diagnostic['n_pairs']} 个，涉及 {diagnostic['n_sequences']} 条序列。表内是序列等权的平均绝对差。"
    )
    lines.append("")
    diag_rows = []
    for slug, summary in diagnostic["models"].items():
        label = slug
        if slug.endswith("__absolute_implied"):
            base = slug.replace("__absolute_implied", "")
            label = SPEC_BY_SLUG[base].name + "（绝对头隐含差）"
        elif slug in SPEC_BY_SLUG:
            label = SPEC_BY_SLUG[slug].name + "（差异头）"
        diag_rows.append([label, *[_fmt(summary["sequence_equal_mae"][key]) for key in TARGET_KEYS]])
    lines.extend(_row_table(["来源", * [TARGET_LABELS[key] for key in TARGET_KEYS]], diag_rows))
    lines.extend(["", "## 失败样例", ""])
    for slug in ("composition_difference", "alignment_path"):
        lines.append(f"### {SPEC_BY_SLUG[slug].name}")
        lines.append("")
        summary = payload["failure_summary"][slug]
        lines.append(
            f"测试盘里，质量绝对误差下降但能量绝对误差上升的有 {summary['n_mass_better_and_energy_worse']} / {summary['n_test_dishes']} 个。"
            "下表是能量绝对误差比单图像基线上升最多的盘子。"
        )
        lines.append("")
        example_rows = []
        for example in payload["failure_examples"][slug]:
            example_rows.append(
                [
                    example["sequence_id"],
                    example["dish_id"],
                    example["added"] or "起始阶段",
                    _fmt(example["true"]["kcal"]),
                    _fmt(example["baseline_abs_error"]["kcal"]),
                    _fmt(example["model_abs_error"]["kcal"]),
                    _fmt(example["energy_abs_error_increase_kcal"]),
                ]
            )
        lines.extend(
            _row_table(
                ["序列", "dish_id", "到达该阶段时新增", "能量真值", "基线绝对误差", "该方法绝对误差", "误差上升"],
                example_rows,
            )
        )
        lines.append("")
    lines.extend(
        [
            "## 停止规则",
            "",
            payload["decision_prose"],
            "",
            control_stop_prose(payload),
            "",
            "第三阶段没有解锁。OmniFood8K 没有下载。ACETADA 没有启动。505 和 Cafe2 没有拿来调这些模型。"
            "`main` 没有改。完整 ICDL 网络、Agent、检索增强、强化学习和不完整食材监督都没有开始。",
            "",
            "## 怎样复核",
            "",
            "```bash",
            "python3 scripts/icdl_phase2_light.py --self-check-only",
            "CUDA_VISIBLE_DEVICES=1 python3 scripts/icdl_phase2_light.py",
            "```",
            "",
            "自检不训练。正式命令使用脚本里的 12 个 epoch、批量 16、bootstrap 10000 次和种子 0。"
            "需要本机已有的 Nutrition5k 俯视 RGB，以及 DINOv2 ViT-S/14 预训练权重。"
            "脚本拒绝写入第一阶段目录、双视角实验目录和其他冻结结果目录。",
            "",
        ]
    )
    return "\n".join(lines) + "\n"


def _interval_text(payload: dict, slug: str, target: str) -> str:
    for row in payload["evaluation"]["comparisons"][slug]["versus_baseline_sequence_equal"]:
        if row["target"] == target:
            return f"[{_fmt(row['ci95_low'])}, {_fmt(row['ci95_high'])}]"
    return "—"


def control_stop_prose(payload: dict) -> str:
    """State the negative-control outcome even when the baseline comparison already fails."""
    relation_text = {
        "entirely_below": "真实相邻配对的误差整体更低",
        "entirely_above": "随机跨盘配对的误差整体更低",
        "includes_zero": "区间跨过 0，两者相近",
    }
    sentences = []
    comparisons = payload["evaluation"]["comparisons"]
    for slug in ("composition_difference", "alignment_path"):
        energy = next(row for row in comparisons[slug]["versus_control_sequence_equal"] if row["target"] == "kcal")
        sentences.append(
            f"{SPEC_BY_SLUG[slug].name}相对自己的随机跨盘配对，能量差（真实减随机）是 {_fmt(energy['difference'])}，"
            f"95% 区间 [{_fmt(energy['ci95_low'])}, {_fmt(energy['ci95_high'])}]，{relation_text[energy['relation']]}。"
        )
    closest = None
    for spec in SPECS:
        if spec.role != "ablation":
            continue
        energy = next(
            row
            for row in comparisons[spec.slug]["versus_baseline_sequence_equal"]
            if row["target"] == "kcal"
        )
        if closest is None or energy["difference"] < closest[1]["difference"]:
            closest = (spec, energy)
    if closest is not None:
        spec, energy = closest
        baseline_text = {
            "entirely_below": "区间整体低于 0",
            "entirely_above": "区间整体高于 0",
            "includes_zero": "区间跨过 0，不能算稳定低于单图像基线",
        }[energy["relation"]]
        sentences.append(
            f"预先写死的消融里，能量点估计最低的是「{spec.name}」，相对单图像基线的差是 {_fmt(energy['difference'])}，"
            f"区间 [{_fmt(energy['ci95_low'])}, {_fmt(energy['ci95_high'])}]，{baseline_text}。"
            "它没有被选成新的主结果，也没有据此再训。"
        )
    sentences.append("这些对照都不构成组成变化已经学成的证据。第三阶段仍然不开始。")
    return "".join(sentences)


def decision_prose(decision: dict) -> str:
    sentences = []
    for slug, info in decision["per_model"].items():
        name = SPEC_BY_SLUG[slug].name
        if info["win"]:
            sentences.append(f"{name}满足预先规定的能量、宏量和负对照条件。")
            continue
        bits = []
        codes = set(info["reason_codes"])
        if "does_not_stably_beat_baseline_on_energy_and_macros" in codes:
            bits.append("没有在能量和全部三种宏量上稳定低于单图像基线")
        if "mass_only" in codes:
            bits.append("稳定优势只出现在质量上")
        if "bootstrap_does_not_support" in codes:
            bits.append("能量点估计更低，但分组区间不支持")
        if "random_pairing_similar" in codes:
            bits.append("与跨盘随机配对在能量上相近")
        if "random_pairing_better" in codes:
            bits.append("跨盘随机配对的能量误差整体更低")
        sentences.append(f"{name}：" + "；".join(bits) + "。")
    if decision["stop_rule_fired"]:
        sentences.append("因此停止规则触发。到这里停下，不等待这些数字去改损失或开始第三阶段。")
    else:
        sentences.append("停止规则没有触发。即使如此，也不开始第三阶段，也不把第二阶段点估计写成投稿结论。")
    return "".join(sentences)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train the ICDL phase 2 light models.")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--epochs", type=int, default=EPOCHS)
    parser.add_argument("--bootstrap-draws", type=int, default=BOOTSTRAP_DRAWS)
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--self-check-only", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.seed != SEED:
        raise SystemExit("the phase 2 seed is fixed at 0")
    self_check()
    if args.self_check_only:
        return
    import torch

    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise SystemExit("CUDA is required for the phase 2 training run")
    output = assert_output_dir(args.output)
    world = build_world()
    print("loading overhead RGB", flush=True)
    images, targets, _raw = load_image_tensor(world, args.device)
    trained = {}
    curves = {}
    steps = None
    for spec in SPECS:
        print(f"train {spec.name}", flush=True)
        result = train_one(spec, world, images, targets, args.device, args.epochs)
        if steps is None:
            steps = result["optimizer_steps"]
        elif result["optimizer_steps"] != steps:
            raise RuntimeError("training budgets diverged")
        checkpoint = output / "checkpoints" / f"{spec.slug}.pt"
        checkpoint.parent.mkdir(parents=True, exist_ok=True)
        torch.save({"spec": spec.as_dict(), "state_dict": result["model"].state_dict(), "curve": result["curve"]}, checkpoint)
        trained[spec.slug] = result["model"]
        curves[spec.slug] = result["curve"]
    print("evaluate test split once", flush=True)
    test_idx = np.array([index for index, stage in enumerate(world.stages) if stage["split"] == "test"], dtype=np.int64)
    predictions = {slug: predict_standardized(model, test_idx, images) for slug, model in trained.items()}
    # The single-image head must not have been given a pair. predict_standardized encodes one batch of images.
    evaluation = evaluate(world, predictions, args.bootstrap_draws)
    diagnostic = delta_diagnostic(world, trained, images, evaluation)
    examples = {}
    example_summary = {}
    for slug in ("composition_difference", "alignment_path"):
        rows, summary = failure_examples(world, evaluation, slug)
        examples[slug] = rows
        example_summary[slug] = summary
    official = args.epochs == EPOCHS and args.bootstrap_draws == BOOTSTRAP_DRAWS and args.seed == SEED
    payload = {
        "status_commit": STATUS_COMMIT,
        "official_protocol": official,
        "epochs": args.epochs,
        "bootstrap_draws": args.bootstrap_draws,
        "seed": args.seed,
        "optimizer_steps": steps,
        "device": torch.cuda.get_device_name(0) if args.device.startswith("cuda") else args.device,
        "counts": world.counts,
        "specs": [spec.as_dict() for spec in SPECS],
        "curves": curves,
        "evaluation": {
            "n_test_dishes": evaluation["n_test_dishes"],
            "n_test_sequences": evaluation["n_test_sequences"],
            "sequence_equal_mae": evaluation["sequence_equal_mae"],
            "training_mean_sequence_equal_mae": evaluation["training_mean_sequence_equal_mae"],
            "dish_micro_mae": evaluation["dish_micro_mae"],
            "comparisons": evaluation["comparisons"],
        },
        "delta_diagnostic": diagnostic,
        "decision": evaluation["decision"],
        "decision_prose": decision_prose(evaluation["decision"]),
        "failure_examples": examples,
        "failure_summary": example_summary,
        "main_untouched": True,
        "phase3_started": False,
    }
    (output / "summary.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    (output / "comparison.json").write_text(
        json.dumps(
            {
                "decision": payload["decision"],
                "sequence_equal_mae": payload["evaluation"]["sequence_equal_mae"],
                "comparisons": payload["evaluation"]["comparisons"],
                "delta_diagnostic": diagnostic,
            },
            ensure_ascii=False,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    (output / "failures.json").write_text(
        json.dumps({"summary": example_summary, "examples": examples}, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    write_predictions(output / "predictions.jsonl", world, evaluation)
    (output / "report.md").write_text(render_report(payload), encoding="utf-8")
    print(payload["decision_prose"], flush=True)
    print(f"wrote {output / 'report.md'}", flush=True)


if __name__ == "__main__":
    main()
