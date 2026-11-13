#!/usr/bin/env python3
"""Shared rules for Experiment A (Nutrition5k top-view / side-view RGB).

The five settings are fixed before any test number is read:

- 单张俯视
- 单张侧视
- 双视角简单特征融合
- 双视角几何融合
- 双视角+食物语义约束

Geometric fusion is spatial cross-attention plus a cross-view mass
consistency term. It does not use depth, camera extrinsics, or a metric
stereo model. Nutrition5k does not publish side-view poses.
"""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
from typing import Iterable, Mapping, Sequence

ROOT = Path(__file__).resolve().parents[1]
ARCHIVE_PATH = Path("/var/tmp/nutri-downloads/nutrition5k/nutrition5k_dataset.tar.gz")
RESULT_DIR = ROOT / "results" / "experiment_a_n5k_dual_view"
LOCAL_N5K = ROOT / "data" / "nutrition5k"

# File-size floors. Chosen before the archive scan, not from a test error.
RGB_MIN_BYTES = 10_000
SIDE_MIN_BYTES = 100_000
DEPTH_MIN_BYTES = 1_000

# Side frame rule. Raw H.264 from this archive has no timestamps in PyAV,
# so a time offset is not available. Keep the middle decoded frame.
# A video longer than this cap stops early and keeps the middle of the cap.
FRAME_CAP = 300
SIDE_LONG_SIDE = 640
JPEG_QUALITY = 90
CAMERA_PRIORITY = ("A", "B", "C", "D")

# Training budget. Shared by all five settings. Not searched on rgb_test.
EPOCHS = 12
BATCH_SIZE = 16
SEED = 0
LR_BACKBONE = 3e-5
LR_HEAD = 1e-3
WEIGHT_DECAY = 1e-2
GRAD_CLIP = 1.0
MASS_UNIT_G = 100.0
# A balanced ingredient BCE of 1.0 is weighted like a 20 g mass error.
SEMANTIC_WEIGHT = 0.2
VIEW_LOSS_WEIGHT = 0.25
CONSISTENCY_WEIGHT = 0.25
HEAD_HIDDEN = 256
IMAGE_SIZE = 224
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)
BOOTSTRAP_DRAWS = 10_000
BOOTSTRAP_SEED = 0
POS_WEIGHT_CAP = 50.0

SETTINGS = (
    "单张俯视",
    "单张侧视",
    "双视角简单特征融合",
    "双视角几何融合",
    "双视角+食物语义约束",
)
SETTING_FILE = {
    "单张俯视": "overhead",
    "单张侧视": "side",
    "双视角简单特征融合": "simple_fusion",
    "双视角几何融合": "geometric_fusion",
    "双视角+食物语义约束": "semantic_fusion",
}


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def id_digest(ids: Iterable[str]) -> str:
    digest = hashlib.sha256()
    for dish_id in sorted(ids):
        digest.update(dish_id.encode("utf-8"))
        digest.update(b"\n")
    return digest.hexdigest()


def mass_bias(mean_g: float) -> float:
    """Inverse softplus so the initial head outputs the training-set mean mass."""
    mean = max(float(mean_g), 1.0)
    return math.log(math.expm1(mean))


def parse_dish_metadata_line(line: str) -> dict | None:
    """Parse one official Nutrition5k cafe metadata row.

    Columns are dish_id, total calories, total mass, total fat, total carb,
    total protein, then repeating ingredient blocks of
    (ingr_id, name, grams, calories, fat, carb, protein).
    """
    text = line.strip()
    if not text:
        return None
    parts = text.split(",")
    if len(parts) < 6 or not parts[0].startswith("dish_"):
        return None
    try:
        mass_g = float(parts[2])
    except ValueError:
        return None
    rest = parts[6:]
    n_groups = len(rest) // 7
    ingredients: list[dict[str, str]] = []
    for index in range(n_groups):
        block = rest[index * 7 : (index + 1) * 7]
        ingr_id, name = block[0], block[1]
        if ingr_id.startswith("ingr_"):
            ingredients.append({"id": ingr_id, "name": name})
    return {
        "dish_id": parts[0],
        "mass_g": mass_g,
        "ingredients": ingredients,
        "trailing_fields": len(rest) - n_groups * 7,
    }


def load_id_list(text: str) -> list[str]:
    return [line.strip() for line in text.splitlines() if line.strip()]


def stably_beats_top(diff_high: float) -> bool:
    """True when the 95% interval of (MAE_method - MAE_top) lies entirely below 0."""
    return diff_high < 0.0


def paired_mae_difference(
    method_abs: Sequence[float],
    top_abs: Sequence[float],
    *,
    draws: int = BOOTSTRAP_DRAWS,
    seed: int = BOOTSTRAP_SEED,
) -> dict[str, float]:
    """Paired bootstrap of mean(method absolute error - top absolute error)."""
    import numpy as np

    method = np.asarray(method_abs, dtype=np.float64)
    top = np.asarray(top_abs, dtype=np.float64)
    if method.shape != top.shape or method.ndim != 1 or method.size == 0:
        raise ValueError("absolute errors must be non-empty and aligned")
    delta = method - top
    rng = np.random.default_rng(seed)
    index = rng.integers(0, delta.size, size=(draws, delta.size))
    means = delta[index].mean(axis=1)
    low, high = np.quantile(means, [0.025, 0.975])
    return {
        "n": int(delta.size),
        "mae_difference_g": float(delta.mean()),
        "ci95_low_g": float(low),
        "ci95_high_g": float(high),
        "fraction_method_mae_lower": float(np.mean(means < 0.0)),
        "stably_beats_top": bool(stably_beats_top(float(high))),
    }


def write_json(path: Path, payload: Mapping) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
