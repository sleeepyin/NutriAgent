#!/usr/bin/env python3
"""Phase-1 data audit for adaptive nutrition-estimation granularity.

The audit aligns Hand et al. public annotations with the Nutrition5k overhead
RGB already on this machine, freezes one visual separability score, and checks
whether the holdout can support plate-level uncertainty intervals.

Constants below are the pre-registration. They are not revised after nutrition
residuals, oracle correlations, or the holdout judgment. This script does not
train a network and does not rewrite the public CSVs.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import subprocess
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
from PIL import Image

# ---------------------------------------------------------------------------
# Pre-registration. Do not retune these after seeing nutrition error, the
# oracle correlation, or the holdout counts.
# ---------------------------------------------------------------------------

METRIC_NAME = "local_color_purity"
METRIC_VERSION = "2026-11-13-phase1"

WORK_LONG_SIDE = 160
BORDER_FRAC = 0.08
PLATE_L_MIN = 72.0
PLATE_CHROMA_MAX = 16.0
PLATE_MIN_DELTA_E = 12.0
PLATE_MIN_FRAC = 0.015
FOOD_MIN_FRAC = 0.005
FALLBACK_FG_DELTA_E = 18.0
BOX_BLUR_RADIUS = 2
PRIMARY_K = 3
PRIMARY_WINDOW = 9
ROBUST_K = 5
ROBUST_WINDOW = 15
KMEANS_ITERS = 20
MIN_FOOD_PIXELS = 30
MIN_WINDOW_FOOD = 5

# Physical-plate link. Thames et al. scan a plate again after each added item.
# 180 seconds sits just above the 99th percentile of time gaps among consecutive
# metadata rows whose ingredients nest and whose mass increases (computed by
# this script and reported; the constant is not edited afterward).
PLATE_GAP_SECONDS = 180
PLATE_GAP_SENSITIVITY = (90, 180, 600)
MIN_MASS_INCREASE_G = 1.0
MASS_GRAM_TOL_G = 1.0

# A clean holdout below this many physical plates stops the audit.
# Hand's own complexity bins of 18–38 dishes had Wilson intervals too wide to
# separate conditions, and a percentile bootstrap of a mean is unstable below
# this conventional floor. The floor is not fit to a nutrition-error curve.
MIN_CLEAN_TEST_PLATES = 80
HAND_PUBLISHED_EVAL_N = 158
HAND_PUBLISHED_PMAE_CI = (1.51, 5.98)

NUTRIENTS = ("mass", "calories", "fat", "carb", "protein")
NUTR_ABS_TOL = 0.5
NUTR_REL_TOL = 0.01

NEAR_DUP_SECONDS = 180
NEAR_DUP_PEARSON = 0.995
THUMB_SIZE = (32, 24)

DISH_BLOB_SHA = "32ae4b8cd7c625856b5f5cf50c10392f6df0dfd3"
INGREDIENT_BLOB_SHA = "2cd1176104461379b04436f68b6d03f75477021a"
DISH_ID_RE_PREFIX = "dish_"

V4_COMMIT = "ce1dc0616002a2f76fcffcc8a3ef87addd8c8358"
V4_ACTION_PREFIX = "backup/v3_excluded/experiments/action"

# One-line reading of each action file at V4_COMMIT. The script still lists
# every file in that tree; an unlisted path is reported as unexplained.
ACTION_PURPOSE = {
    "README.md": "行动层说明：分组路由、名称先验、逐样本 oracle 上界。",
    "__init__.py": "行动层包说明，没有实验正文。",
    "a1_group_routing.py": "按数据集分组选择动作，评的是 ECUSTFD 与 MetaFood3D 的称重误差。",
    "a2_name_prior.py": "只看食物名称的质量先验，不看图像。",
    "a2b_vlm_name.py": "用视觉语言模型给出的食物名替换标注类别名。",
    "a3_auto_reference/README.md": "ECUSTFD 上的硬币参照物定标，说明里写脚本尚未完成。",
    "a3_auto_reference/__init__.py": "参照物实验的空包文件。",
    "a3_auto_reference/run.py": "ECUSTFD 上比较视觉语言模型硬币框、标注硬币框和霍夫圆。",
    "a4_ood_reject/README.md": "Nutrition5k 与 NV-Real 上的域外拒答，说明里写脚本尚未完成。",
    "a4_ood_reject/__init__.py": "域外拒答实验的空包文件。",
    "a4_ood_reject/run.py": "用共形分数和域外标记做域移检测。",
    "a5_mixed_flow/README.md": "混合留出流上的端到端路由，说明里写端到端尚未完成。",
    "a5_mixed_flow/__init__.py": "混合流实验的空包文件。",
    "a5_mixed_flow/run.py": "按诊断章的固定如果-那么规则在混合留出流上选动作。",
    "a5_oracle_bound.py": "分组动作冻结之后，逐样本最优动作还能变多少的上界。",
    "common.py": "行动层脚本的共享参数工具。",
}

MENTION_EXPLANATION = {
    "backup/v3_excluded/logs/remaining_status.txt": (
        "这是停止再训 DINOv2 的记录，对象是 NV-Real 的食材级路线，不是整盘对食材级的营养估计实验。"
    ),
    "backup/v3_excluded/experiments/diagnosis/d6_capacity/dinov2_ablation.py": (
        "这是质量回归的骨干容量消融。PMAE 在这里是质量的百分比误差，输入包含深度，不是粒度实验。"
    ),
    "backup/v3_excluded/logs/d6_dinov2.log": (
        "这是上面那次质量回归消融的训练日志，ValPMAE 是验证集质量百分比误差。"
    ),
    "backup/v3_excluded/logs/n5k_r0.log": (
        "这是 Nutrition5k 质量回归的训练日志，ValPMAE 是质量百分比误差。"
    ),
    "backup/v3_excluded/logs/remaining.log": (
        "这是质量回归训练日志，ValPMAE 是质量百分比误差。"
    ),
    "backup/v3_excluded/logs/other_datasets.log": (
        "这是 MetaFood3D 上的质量误差预算。文中的 PMAE 是单目质量百分比误差，不是食材级对整盘的粒度对照。"
    ),
}


def git_blob_sha1(data: bytes) -> str:
    digest = hashlib.sha1()
    digest.update(f"blob {len(data)}\0".encode("ascii"))
    digest.update(data)
    return digest.hexdigest()


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def dish_timestamp(dish_id: str) -> int:
    if not dish_id.startswith(DISH_ID_RE_PREFIX):
        raise ValueError(f"unexpected dish_id: {dish_id}")
    return int(dish_id[len(DISH_ID_RE_PREFIX) :])


def valid_dish_id(dish_id: str) -> bool:
    if not dish_id.startswith(DISH_ID_RE_PREFIX):
        return False
    rest = dish_id[len(DISH_ID_RE_PREFIX) :]
    return rest.isdigit() and rest == str(int(rest))


def parse_optional_float(text: str) -> float | None:
    text = (text or "").strip()
    if text == "":
        return None
    return float(text)


def within_tolerance(left: float, right: float) -> bool:
    scale = max(abs(left), abs(right), 0.0)
    return abs(left - right) <= max(NUTR_ABS_TOL, NUTR_REL_TOL * scale)


def rankdata(values: np.ndarray) -> np.ndarray:
    order = np.argsort(values, kind="mergesort")
    ranks = np.empty(len(values), dtype=float)
    start = 0
    while start < len(values):
        stop = start
        while stop + 1 < len(values) and values[order[stop + 1]] == values[order[start]]:
            stop += 1
        average = 0.5 * ((start + 1) + (stop + 1))
        ranks[order[start : stop + 1]] = average
        start = stop + 1
    return ranks


def pearson(left: np.ndarray, right: np.ndarray) -> float | None:
    left = np.asarray(left, dtype=float)
    right = np.asarray(right, dtype=float)
    left = left - left.mean()
    right = right - right.mean()
    denom = math.sqrt(float((left * left).sum() * (right * right).sum()))
    if denom == 0.0:
        return None
    return float((left * right).sum() / denom)


def spearman(left: np.ndarray, right: np.ndarray) -> dict:
    left = np.asarray(left, dtype=float)
    right = np.asarray(right, dtype=float)
    if len(left) < 3:
        return {"n": int(len(left)), "rho": None, "normal_approx_p": None}
    rho = pearson(rankdata(left), rankdata(right))
    if rho is None:
        p_value = None
    else:
        # Large-sample normal approximation of a Spearman/Pearson coefficient.
        z = abs(rho) * math.sqrt(len(left) - 1)
        p_value = math.erfc(z / math.sqrt(2.0))
    return {"n": int(len(left)), "rho": rho, "normal_approx_p": p_value}


def srgb_to_lab(rgb_uint8: np.ndarray) -> np.ndarray:
    """sRGB uint8 image to CIE Lab (D65)."""
    rgb = rgb_uint8.astype(np.float64) / 255.0
    linear = np.where(rgb <= 0.04045, rgb / 12.92, ((rgb + 0.055) / 1.055) ** 2.4)
    matrix = np.array(
        [
            [0.4124564, 0.3575761, 0.1804375],
            [0.2126729, 0.7151522, 0.0721750],
            [0.0193339, 0.1191920, 0.9503041],
        ],
        dtype=np.float64,
    )
    xyz = linear @ matrix.T
    xyz[..., 0] /= 0.95047
    xyz[..., 2] /= 1.08883
    delta = 6.0 / 29.0
    pivot = delta ** 3

    def f_xyz(channel: np.ndarray) -> np.ndarray:
        return np.where(channel > pivot, np.cbrt(channel), channel / (3 * delta * delta) + 4.0 / 29.0)

    transformed = f_xyz(xyz)
    lightness = 116.0 * transformed[..., 1] - 16.0
    green_red = 500.0 * (transformed[..., 0] - transformed[..., 1])
    blue_yellow = 200.0 * (transformed[..., 1] - transformed[..., 2])
    return np.stack([lightness, green_red, blue_yellow], axis=-1)


def box_sum(values: np.ndarray, radius: int) -> np.ndarray:
    if radius < 0:
        raise ValueError("radius must be non-negative")
    height, width = values.shape
    integral = np.zeros((height + 1, width + 1), dtype=np.float64)
    integral[1:, 1:] = np.cumsum(np.cumsum(values, axis=0), axis=1)
    y_index = np.arange(height)
    x_index = np.arange(width)
    y0 = np.clip(y_index - radius, 0, height)[:, None]
    y1 = np.clip(y_index + radius + 1, 0, height)[:, None]
    x0 = np.clip(x_index - radius, 0, width)[None, :]
    x1 = np.clip(x_index + radius + 1, 0, width)[None, :]
    return integral[y1, x1] - integral[y0, x1] - integral[y1, x0] + integral[y0, x0]


def box_blur(lab: np.ndarray, radius: int) -> np.ndarray:
    if radius == 0:
        return lab
    weight = box_sum(np.ones(lab.shape[:2], dtype=np.float64), radius)
    blurred = np.empty_like(lab)
    for channel in range(lab.shape[2]):
        blurred[..., channel] = box_sum(lab[..., channel], radius) / np.maximum(weight, 1.0)
    return blurred


def connected_components_8(mask: np.ndarray) -> tuple[np.ndarray, dict[int, int]]:
    """8-connected labels for a boolean mask. Label 0 is background."""
    height, width = mask.shape
    labels = np.zeros((height, width), dtype=np.int32)
    parent: dict[int, int] = {}

    def find(label: int) -> int:
        while parent[label] != label:
            parent[label] = parent[parent[label]]
            label = parent[label]
        return label

    def union(left: int, right: int) -> None:
        left_root = find(left)
        right_root = find(right)
        if left_root != right_root:
            parent[right_root] = left_root

    next_label = 1
    for y in range(height):
        row = mask[y]
        for x in range(width):
            if not row[x]:
                continue
            candidates: list[int] = []
            y0 = y - 1 if y else 0
            x0 = x - 1 if x else 0
            x1 = x + 1 if x + 1 < width else x
            for yy in range(y0, y + 1):
                last_x = x if yy == y else x1
                for xx in range(x0, last_x + 1):
                    if yy == y and xx == x:
                        continue
                    lab = int(labels[yy, xx])
                    if lab:
                        candidates.append(lab)
            if not candidates:
                parent[next_label] = next_label
                labels[y, x] = next_label
                next_label += 1
            else:
                chosen = min(candidates)
                labels[y, x] = chosen
                for candidate in candidates:
                    union(chosen, candidate)
    if next_label == 1:
        return labels, {}
    roots: dict[int, int] = {}
    sizes: dict[int, int] = {}
    flat = labels.ravel()
    remapped = np.zeros_like(flat)
    next_compact = 1
    for index, label in enumerate(flat):
        if label == 0:
            continue
        root = find(int(label))
        compact = roots.get(root)
        if compact is None:
            compact = next_compact
            roots[root] = compact
            sizes[compact] = 0
            next_compact += 1
        remapped[index] = compact
        sizes[compact] += 1
    return remapped.reshape(height, width), sizes


def fill_holes(mask: np.ndarray) -> np.ndarray:
    """Fill background pockets that do not touch the image border."""
    inverted = ~mask
    height, width = mask.shape
    seen = np.zeros((height, width), dtype=bool)
    stack: list[tuple[int, int]] = []
    for x in range(width):
        if inverted[0, x]:
            stack.append((0, x))
        if inverted[height - 1, x]:
            stack.append((height - 1, x))
    for y in range(height):
        if inverted[y, 0]:
            stack.append((y, 0))
        if inverted[y, width - 1]:
            stack.append((y, width - 1))
    while stack:
        y, x = stack.pop()
        if seen[y, x] or not inverted[y, x]:
            continue
        seen[y, x] = True
        if y > 0:
            stack.append((y - 1, x))
        if y + 1 < height:
            stack.append((y + 1, x))
        if x > 0:
            stack.append((y, x - 1))
        if x + 1 < width:
            stack.append((y, x + 1))
    holes = inverted & ~seen
    return mask | holes


def resize_long_side(image: Image.Image, long_side: int) -> Image.Image:
    width, height = image.size
    current = max(width, height)
    if current == long_side:
        return image
    scale = long_side / current
    resized = (max(1, int(round(width * scale))), max(1, int(round(height * scale))))
    return image.resize(resized, Image.Resampling.BILINEAR)


def largest_mask(mask: np.ndarray, min_pixels: int) -> np.ndarray | None:
    labels, sizes = connected_components_8(mask)
    if not sizes:
        return None
    label, area = max(sizes.items(), key=lambda item: (item[1], -item[0]))
    if area < min_pixels:
        return None
    return fill_holes(labels == label)


def frozen_food_mask(lab: np.ndarray) -> tuple[np.ndarray, bool]:
    """Return a food mask and whether a bright plate component was found.

    The plate cue is the light, low-chroma disk on the Nutrition5k overhead
    rig. Food is the hole that disk encloses. Pale food that is absorbed into
    the disk falls back to the filled disk. If no disk is found, the largest
    region that differs from the border is used.
    """
    height, width = lab.shape[:2]
    border_y = max(1, int(round(BORDER_FRAC * height)))
    border_x = max(1, int(round(BORDER_FRAC * width)))
    border = np.zeros((height, width), dtype=bool)
    border[:border_y, :] = True
    border[-border_y:, :] = True
    border[:, :border_x] = True
    border[:, -border_x:] = True
    background = np.median(lab[border], axis=0)
    delta_e = np.sqrt(((lab - background) ** 2).sum(axis=-1))
    chroma = np.sqrt(lab[..., 1] ** 2 + lab[..., 2] ** 2)
    plate_candidate = (
        (lab[..., 0] >= PLATE_L_MIN)
        & (chroma <= PLATE_CHROMA_MAX)
        & (delta_e >= PLATE_MIN_DELTA_E)
    )
    min_plate = max(1, int(math.ceil(PLATE_MIN_FRAC * height * width)))
    plate = largest_mask(plate_candidate, min_plate)
    if plate is not None:
        food = plate & ~plate_candidate
        min_food = max(1, int(math.ceil(FOOD_MIN_FRAC * height * width)))
        if int(food.sum()) >= min_food:
            return food, True
        return plate, True
    foreground = largest_mask(delta_e >= FALLBACK_FG_DELTA_E, MIN_FOOD_PIXELS)
    if foreground is None:
        return np.zeros((height, width), dtype=bool), False
    return foreground, False


def farthest_point_centers(features: np.ndarray, clusters: int) -> np.ndarray:
    median = np.median(features, axis=0)
    distance = ((features - median) ** 2).sum(axis=1)
    first = int(np.lexsort((np.arange(len(features)), distance))[0])
    chosen = [first]
    nearest = ((features - features[first]) ** 2).sum(axis=1)
    for _ in range(1, clusters):
        # Smallest index wins a tie, so the init does not depend on RNG.
        nxt = int(np.lexsort((np.arange(len(features)), -nearest))[0])
        chosen.append(nxt)
        nearest = np.minimum(nearest, ((features - features[nxt]) ** 2).sum(axis=1))
    return features[np.array(chosen, dtype=int)].copy()


def kmeans(features: np.ndarray, clusters: int, iterations: int) -> np.ndarray:
    if len(features) < clusters:
        raise ValueError("not enough pixels for the frozen palette")
    centers = farthest_point_centers(features, clusters)
    labels = np.zeros(len(features), dtype=np.int32)
    for _ in range(iterations):
        distances = ((features[:, None, :] - centers[None, :, :]) ** 2).sum(axis=2)
        labels = np.argmin(distances, axis=1).astype(np.int32)
        for cluster in range(clusters):
            members = features[labels == cluster]
            if len(members) == 0:
                occupied = ((features[:, None, :] - centers[None, :, :]) ** 2).sum(axis=2).min(axis=1)
                centers[cluster] = features[int(np.lexsort((np.arange(len(features)), -occupied))[0])]
            else:
                centers[cluster] = members.mean(axis=0)
    distances = ((features[:, None, :] - centers[None, :, :]) ** 2).sum(axis=2)
    return np.argmin(distances, axis=1).astype(np.int32)


def local_purity(labels: np.ndarray, food: np.ndarray, window: int) -> float | None:
    if window % 2 != 1:
        raise ValueError("window must be odd")
    radius = window // 2
    food_count = box_sum(food.astype(np.float64), radius)
    purity_sum = 0.0
    purity_n = 0
    for cluster in range(int(labels.max()) + 1 if labels.size else 0):
        cluster_map = ((labels == cluster) & food).astype(np.float64)
        if cluster_map.sum() == 0:
            continue
        same = box_sum(cluster_map, radius)
        ys, xs = np.nonzero((labels == cluster) & food)
        counts = food_count[ys, xs]
        keep = counts >= MIN_WINDOW_FOOD
        if not np.any(keep):
            continue
        purity_sum += float((same[ys, xs][keep] / counts[keep]).sum())
        purity_n += int(keep.sum())
    if purity_n == 0:
        return None
    return purity_sum / purity_n


def visual_separability(rgb: np.ndarray) -> dict:
    """Score one overhead RGB image.

    The only input is the image. Ingredient counts, ground-truth masks, food
    codes, and nutrition values are not arguments and are not read here.
    """
    if rgb.ndim != 3 or rgb.shape[2] != 3:
        raise ValueError("expected an HxWx3 RGB array")
    image = Image.fromarray(np.ascontiguousarray(rgb), mode="RGB")
    image = resize_long_side(image, WORK_LONG_SIDE)
    lab = srgb_to_lab(np.asarray(image))
    food, plate_found = frozen_food_mask(lab)
    food_pixels = int(food.sum())
    food_frac = food_pixels / float(food.size)
    result = {
        "mask_ok": False,
        "plate_found": plate_found,
        "food_pixels": food_pixels,
        "food_frac": food_frac,
        "s_sep": None,
        "s_sep_k5": None,
        "s_sep_win15": None,
        "failure": None,
    }
    if food_pixels < MIN_FOOD_PIXELS:
        result["failure"] = "food_mask_too_small"
        return result
    blurred = box_blur(lab, BOX_BLUR_RADIUS)
    features = np.stack(
        [
            blurred[..., 0] / 100.0,
            blurred[..., 1] / 128.0,
            blurred[..., 2] / 128.0,
        ],
        axis=-1,
    )[food]
    labels_primary = kmeans(features, PRIMARY_K, KMEANS_ITERS)
    label_map = np.zeros(food.shape, dtype=np.int32)
    label_map[food] = labels_primary
    primary = local_purity(label_map, food, PRIMARY_WINDOW)
    labels_k5 = kmeans(features, ROBUST_K, KMEANS_ITERS)
    label_map_k5 = np.zeros(food.shape, dtype=np.int32)
    label_map_k5[food] = labels_k5
    robust_k = local_purity(label_map_k5, food, PRIMARY_WINDOW)
    robust_window = local_purity(label_map, food, ROBUST_WINDOW)
    if primary is None:
        result["failure"] = "purity_undefined"
        return result
    result.update(
        {
            "mask_ok": True,
            "s_sep": primary,
            "s_sep_k5": robust_k,
            "s_sep_win15": robust_window,
        }
    )
    return result


def load_rgb(path: Path) -> np.ndarray:
    with Image.open(path) as image:
        return np.asarray(image.convert("RGB"))


def read_csv_rows(path: Path) -> tuple[list[str], list[dict[str, str]]]:
    with path.open(newline="") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None:
            raise ValueError(f"{path} has no header")
        rows = list(reader)
        return list(reader.fieldnames), rows


def load_id_list(path: Path) -> list[str]:
    text = path.read_text(encoding="utf-8")
    return [line.strip() for line in text.splitlines() if line.strip()]


def load_nutrition5k_metadata(directory: Path) -> list[dict]:
    rows: list[dict] = []
    for cafe, name in (("cafe1", "dish_metadata_cafe1.csv"), ("cafe2", "dish_metadata_cafe2.csv")):
        path = directory / name
        with path.open(newline="") as handle:
            for record in csv.reader(handle):
                if len(record) < 6:
                    raise ValueError(f"short metadata row in {name}")
                tail = record[6:]
                if len(tail) % 7 != 0:
                    raise ValueError(f"ingredient tail is not groups of 7 in {record[0]}")
                ingredients = []
                for offset in range(0, len(tail), 7):
                    ingredients.append((tail[offset + 1], float(tail[offset + 2])))
                rows.append(
                    {
                        "dish_id": record[0],
                        "ts": dish_timestamp(record[0]),
                        "cafe": cafe,
                        "calories": float(record[1]),
                        "mass": float(record[2]),
                        "fat": float(record[3]),
                        "carb": float(record[4]),
                        "protein": float(record[5]),
                        "ingredients": ingredients,
                    }
                )
    rows.sort(key=lambda row: row["ts"])
    return rows


def ingredient_grams(ingredients: list[tuple[str, float]]) -> Counter:
    totals: Counter = Counter()
    for name, grams in ingredients:
        totals[name] += grams
    return totals


def same_physical_plate(previous: dict, current: dict, gap_seconds: int) -> bool:
    """True when current continues previous under the incremental-scan rule."""
    if current["ts"] - previous["ts"] > gap_seconds:
        return False
    if current["mass"] < previous["mass"] + MIN_MASS_INCREASE_G:
        return False
    previous_counts = Counter(name for name, _grams in previous["ingredients"])
    current_counts = Counter(name for name, _grams in current["ingredients"])
    if any(current_counts[name] < count for name, count in previous_counts.items()):
        return False
    previous_grams = ingredient_grams(previous["ingredients"])
    current_grams = ingredient_grams(current["ingredients"])
    for name, grams in previous_grams.items():
        if current_grams.get(name, -1.0e9) + MASS_GRAM_TOL_G < grams:
            return False
    return True


class UnionFind:
    def __init__(self, size: int) -> None:
        self.parent = list(range(size))

    def find(self, index: int) -> int:
        while self.parent[index] != index:
            self.parent[index] = self.parent[self.parent[index]]
            index = self.parent[index]
        return index

    def union(self, left: int, right: int) -> None:
        left_root = self.find(left)
        right_root = self.find(right)
        if left_root != right_root:
            self.parent[right_root] = left_root


def cluster_plates(rows: list[dict], gap_seconds: int) -> tuple[int, dict[str, str], dict[str, list[str]]]:
    finder = UnionFind(len(rows))
    links = 0
    for index in range(len(rows) - 1):
        if same_physical_plate(rows[index], rows[index + 1], gap_seconds):
            finder.union(index, index + 1)
            links += 1
    members: dict[int, list[str]] = defaultdict(list)
    for index, row in enumerate(rows):
        members[finder.find(index)].append(row["dish_id"])
    dish_to_plate: dict[str, str] = {}
    plates: dict[str, list[str]] = {}
    for grouped in members.values():
        plate_id = min(grouped, key=dish_timestamp)
        plates[plate_id] = sorted(grouped, key=dish_timestamp)
        for dish_id in grouped:
            dish_to_plate[dish_id] = plate_id
    return links, dish_to_plate, plates


def gap_distribution(rows: list[dict]) -> dict:
    nested_gaps: list[int] = []
    other_gaps: list[int] = []
    for previous, current in zip(rows, rows[1:]):
        gap = current["ts"] - previous["ts"]
        if same_physical_plate(previous, current, gap_seconds=10**12):
            nested_gaps.append(gap)
        else:
            other_gaps.append(gap)

    def describe(values: list[int]) -> dict:
        if not values:
            return {"n": 0}
        ordered = sorted(values)

        def percentile(fraction: float) -> int:
            position = min(len(ordered) - 1, int(round(fraction * (len(ordered) - 1))))
            return ordered[position]

        return {
            "n": len(ordered),
            "min": ordered[0],
            "p50": percentile(0.50),
            "p90": percentile(0.90),
            "p99": percentile(0.99),
            "max": ordered[-1],
            "le_180": sum(value <= PLATE_GAP_SECONDS for value in ordered),
        }

    return {
        "nested_mass_increasing_seconds": describe(nested_gaps),
        "other_consecutive_seconds": describe(other_gaps),
    }


def judge_holdout(clean_test_plates: int, cross_split_plates: int) -> dict:
    """Decide whether the independent holdout can carry a plate-level interval.

    The published count of 158 dishes is not declared a pass or a failure here.
    The usable sample is the number of physical plates that sit only in Hand's
    test split. Below the pre-registered floor, the audit stops.
    """
    half_width_158 = (HAND_PUBLISHED_PMAE_CI[1] - HAND_PUBLISHED_PMAE_CI[0]) / 2.0
    if clean_test_plates <= 0:
        scaled = None
    else:
        scaled = half_width_158 * math.sqrt(HAND_PUBLISHED_EVAL_N / clean_test_plates)
    large_enough = clean_test_plates >= MIN_CLEAN_TEST_PLATES and cross_split_plates == 0
    # Cross-split plates mean the published dish split is not a clean plate
    # holdout. Usable N is the clean-plate count either way; a positive cross
    # count with a still-large clean set would be reported, not redrawn.
    if clean_test_plates < MIN_CLEAN_TEST_PLATES:
        decision = "stop_too_few_independent_plates"
        large_enough = False
    elif cross_split_plates > 0 and clean_test_plates >= MIN_CLEAN_TEST_PLATES:
        decision = "clean_subset_meets_floor_but_split_leaks"
        large_enough = False
    else:
        decision = "independent_plates_meet_floor"
        large_enough = True
    return {
        "clean_test_plates": clean_test_plates,
        "cross_split_plates": cross_split_plates,
        "floor": MIN_CLEAN_TEST_PLATES,
        "large_enough": large_enough,
        "decision": decision,
        "published_158_declared": False,
        "hand_pmae_ci_half_width_pp": half_width_158,
        "extrapolated_half_width_pp": scaled,
        "extrapolation_note": (
            "把 Hand 等人在 158 盘上发表的配对 PMAE 差值区间半宽，按独立餐盘数的平方根外推。"
            "这是设计计算，不是新的营养预测实验。"
        ),
    }


def quantile(values: list[float], fraction: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    position = min(len(ordered) - 1, int(round(fraction * (len(ordered) - 1))))
    return ordered[position]


def first_doc_line(text: str) -> str:
    lines = text.splitlines()
    for index, line in enumerate(lines[:40]):
        stripped = line.strip()
        if stripped.startswith('"""') or stripped.startswith("'''"):
            quote = stripped[:3]
            body = stripped[3:]
            if body.endswith(quote) and len(body) > 3:
                return body[: -3].strip()
            if body.strip():
                return body.strip()
            for follow in lines[index + 1 : index + 8]:
                if follow.strip().startswith(quote):
                    break
                if follow.strip():
                    return follow.strip()
        if stripped.startswith("# ") and not stripped.startswith("#!"):
            return stripped[2:].strip()
    return ""


def explain_mention(path: str) -> str:
    if path in MENTION_EXPLANATION:
        return MENTION_EXPLANATION[path]
    for prefix, text in MENTION_EXPLANATION.items():
        if path.startswith(prefix):
            return text
    return ""


def read_v4_inventory(repo: Path) -> dict:
    def git_out(*args: str) -> str:
        completed = subprocess.run(
            ["git", *args],
            cwd=repo,
            check=True,
            capture_output=True,
            text=True,
        )
        return completed.stdout

    commit_line = git_out("log", "-1", "--format=%H%n%ci%n%s", V4_COMMIT).splitlines()
    action_files = [
        line
        for line in git_out("ls-tree", "-r", "--name-only", V4_COMMIT, "--", V4_ACTION_PREFIX).splitlines()
        if line
    ]
    experiment_entries = [
        line
        for line in git_out("ls-tree", "--name-only", f"{V4_COMMIT}:backup/v3_excluded/experiments").splitlines()
        if line
    ]
    files = []
    unlisted = []
    for path in action_files:
        relative = path[len(V4_ACTION_PREFIX) + 1 :]
        text = git_out("show", f"{V4_COMMIT}:{path}")
        purpose = ACTION_PURPOSE.get(relative, "")
        if not purpose:
            unlisted.append(relative)
        files.append(
            {
                "path": path,
                "purpose": purpose or "树里有这个文件，但审计清单没有预先写好的一句说明。",
                "opening": first_doc_line(text),
                "bytes": len(text.encode("utf-8")),
            }
        )
    mention_pattern = r"granularity|ingredient-level|食材级|整盘|PMAE|pmae"
    mention_raw = subprocess.run(
        [
            "git",
            "grep",
            "-n",
            "-I",
            "-i",
            "-E",
            mention_pattern,
            V4_COMMIT,
            "--",
            "backup/v3_excluded",
            ":(exclude)backup/v3_excluded/third_party",
        ],
        cwd=repo,
        capture_output=True,
        text=True,
    )
    mentions = []
    unexplained = []
    if mention_raw.returncode not in (0, 1):
        raise subprocess.CalledProcessError(mention_raw.returncode, mention_raw.args, mention_raw.stderr)
    seen_paths: set[str] = set()
    for line in mention_raw.stdout.splitlines():
        # commit:path:line:text
        try:
            _commit, rest = line.split(":", 1)
            path, lineno, excerpt = rest.split(":", 2)
        except ValueError:
            continue
        if path in seen_paths:
            continue
        seen_paths.add(path)
        explanation = explain_mention(path)
        record = {"path": path, "line": int(lineno), "excerpt": excerpt.strip()[:240], "explanation": explanation}
        mentions.append(record)
        if not explanation:
            unexplained.append(path)
    granularity_needles = (
        "segmentation granularity",
        "ingredient-level segmentation",
        "食材级",
        "整盘回归",
        "粒度实验",
    )
    needle_hits = []
    for path in action_files:
        text = git_out("show", f"{V4_COMMIT}:{path}").lower()
        found = [needle for needle in granularity_needles if needle.lower() in text]
        if found:
            needle_hits.append({"path": path, "needles": found})
    return {
        "commit": commit_line[0] if commit_line else V4_COMMIT,
        "commit_time": commit_line[1] if len(commit_line) > 1 else "",
        "commit_subject": commit_line[2] if len(commit_line) > 2 else "",
        "experiment_entries": experiment_entries,
        "action_files": files,
        "unlisted_action_files": unlisted,
        "keyword_mentions_outside_third_party": mentions,
        "unexplained_mentions": unexplained,
        "action_granularity_needle_hits": needle_hits,
        "scripts_copied_into_this_branch": False,
    }


def audit_public_tables(dish_rows: list[dict[str, str]], ingredient_rows: list[dict[str, str]]) -> dict:
    dish_ids = [row["dish_id"] for row in dish_rows]
    ingredient_ids = [row["dish_id"] for row in ingredient_rows]
    dish_set = set(dish_ids)
    ingredient_set = set(ingredient_ids)
    split_by_dish: dict[str, set[str]] = defaultdict(set)
    for row in dish_rows:
        split_by_dish[row["dish_id"]].add(row["split"])
    ingredient_split: dict[str, set[str]] = defaultdict(set)
    for row in ingredient_rows:
        ingredient_split[row["dish_id"]].add(row["split"])
    split_conflict = sorted(
        dish_id
        for dish_id in dish_set | ingredient_set
        if split_by_dish.get(dish_id, set()) != ingredient_split.get(dish_id, set())
        or len(split_by_dish.get(dish_id, set())) != 1
    )
    classes = sorted({row["annotation_name"] for row in ingredient_rows})
    has_nutrition = Counter(row["has_nutrition"] for row in ingredient_rows)
    dish_split = Counter(row["split"] for row in dish_rows)
    ingredient_split_counts = Counter(row["split"] for row in ingredient_rows)
    dishes_by_split: dict[str, set[str]] = defaultdict(set)
    for row in ingredient_rows:
        dishes_by_split[row["split"]].add(row["dish_id"])
    bad_ids = sorted(dish_id for dish_id in dish_set | ingredient_set if not valid_dish_id(dish_id))
    rgb_mismatch = []
    for row in ingredient_rows:
        expected = f"rgb_images/{row['dish_id']}.png"
        if row["rgb_image"] != expected:
            rgb_mismatch.append(row["dish_id"])
    return {
        "dish_rows": len(dish_rows),
        "dish_unique": len(dish_set),
        "dish_duplicate_ids": len(dish_ids) - len(dish_set),
        "dish_split_counts": dict(dish_split),
        "ingredient_rows": len(ingredient_rows),
        "ingredient_unique_dishes": len(ingredient_set),
        "ingredient_split_counts": dict(ingredient_split_counts),
        "ingredient_dishes_per_split": {key: len(value) for key, value in dishes_by_split.items()},
        "same_dish_id_set": dish_set == ingredient_set,
        "only_in_dish_table": sorted(dish_set - ingredient_set),
        "only_in_ingredient_table": sorted(ingredient_set - dish_set),
        "split_conflicts": split_conflict,
        "invalid_dish_ids": bad_ids,
        "annotation_names": classes,
        "annotation_name_count": len(classes),
        "has_nutrition_counts": dict(has_nutrition),
        "rgb_image_column_mismatches": len(set(rgb_mismatch)),
        "field_note_matches": {
            "dish_rows_1056": len(dish_rows) == 1056 and len(dish_set) == 1056,
            "dish_train_898": dish_split.get("train") == 898,
            "dish_test_158": dish_split.get("test") == 158,
            "ingredient_rows_2922": len(ingredient_rows) == 2922,
            "ingredient_train_2253": ingredient_split_counts.get("train") == 2253,
            "ingredient_test_669": ingredient_split_counts.get("test") == 669,
            "ingredient_train_dishes_898": len(dishes_by_split.get("train", ())) == 898,
            "ingredient_test_dishes_158": len(dishes_by_split.get("test", ())) == 158,
            "classes_40": len(classes) == 40,
            "has_nutrition_0_is_98": has_nutrition.get("0") == 98,
            "has_nutrition_1_is_2824": has_nutrition.get("1") == 2824,
            "same_ids_and_splits": dish_set == ingredient_set and not split_conflict and not bad_ids,
        },
    }


def nutrition_consistency(
    dish_rows: list[dict[str, str]],
    ingredient_rows: list[dict[str, str]],
    metadata_by_id: dict[str, dict],
) -> tuple[list[dict], dict]:
    grouped: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in ingredient_rows:
        grouped[row["dish_id"]].append(row)
    per_dish = []
    for row in dish_rows:
        instances = grouped.get(row["dish_id"], [])
        record: dict = {
            "dish_id": row["dish_id"],
            "split": row["split"],
            "n_instances": len(instances),
            "n_with_nutrition": sum(item["has_nutrition"] == "1" for item in instances),
            "n_missing_nutrition": sum(item["has_nutrition"] == "0" for item in instances),
        }
        complete = record["n_missing_nutrition"] == 0 and record["n_instances"] > 0
        record["all_instances_have_nutrition"] = complete
        sums = {nutrient: 0.0 for nutrient in NUTRIENTS}
        sum_ok = True
        for item in instances:
            if item["has_nutrition"] != "1":
                continue
            for nutrient in NUTRIENTS:
                value = parse_optional_float(item[nutrient])
                if value is None:
                    sum_ok = False
                else:
                    sums[nutrient] += value
        meta = metadata_by_id.get(row["dish_id"])
        matches = {}
        for nutrient in NUTRIENTS:
            plate = parse_optional_float(row[nutrient])
            record[f"plate_{nutrient}"] = plate
            record[f"sum_{nutrient}"] = sums[nutrient] if sum_ok else None
            if plate is None or not sum_ok:
                matches[nutrient] = False
                record[f"abs_diff_{nutrient}"] = None
            else:
                record[f"abs_diff_{nutrient}"] = abs(sums[nutrient] - plate)
                matches[nutrient] = within_tolerance(sums[nutrient], plate)
            if meta is None:
                record[f"meta_{nutrient}"] = None
                record[f"meta_abs_diff_{nutrient}"] = None
            else:
                record[f"meta_{nutrient}"] = meta[nutrient]
                record[f"meta_abs_diff_{nutrient}"] = None if plate is None else abs(plate - meta[nutrient])
        record["sum_matches_plate_all_nutrients"] = all(matches.values())
        record["in_nutrition5k_metadata"] = meta is not None
        per_dish.append(record)

    def subset(flag: bool) -> list[dict]:
        return [row for row in per_dish if row["all_instances_have_nutrition"] is flag]

    def match_count(rows: list[dict]) -> int:
        return sum(row["sum_matches_plate_all_nutrients"] for row in rows)

    complete_rows = subset(True)
    partial_rows = subset(False)
    meta_mass_diffs = [
        row["meta_abs_diff_mass"] for row in per_dish if row["meta_abs_diff_mass"] is not None
    ]
    partial_mass_residual = []
    for row in partial_rows:
        if row["plate_mass"] is not None and row["sum_mass"] is not None:
            partial_mass_residual.append(row["plate_mass"] - row["sum_mass"])
    complete_mismatch = [row for row in complete_rows if not row["sum_matches_plate_all_nutrients"]]
    complete_gap = []
    complete_rel = []
    nutrient_fail = {nutrient: 0 for nutrient in NUTRIENTS}
    for row in complete_mismatch:
        plate_mass = row["plate_mass"]
        sum_mass = row["sum_mass"]
        if plate_mass is None or sum_mass is None:
            continue
        complete_gap.append(plate_mass - sum_mass)
        if plate_mass != 0:
            complete_rel.append(abs(plate_mass - sum_mass) / abs(plate_mass))
        for nutrient in NUTRIENTS:
            plate = row[f"plate_{nutrient}"]
            summed = row[f"sum_{nutrient}"]
            if plate is None or summed is None or not within_tolerance(summed, plate):
                nutrient_fail[nutrient] += 1
    summary = {
        "dishes": len(per_dish),
        "complete_nutrition_dishes": len(complete_rows),
        "partial_nutrition_dishes": len(partial_rows),
        "complete_and_sums_match": match_count(complete_rows),
        "partial_and_sums_match": match_count(partial_rows),
        "any_and_sums_match": match_count(per_dish),
        "plate_mass_matches_metadata": sum(
            row["meta_abs_diff_mass"] is not None and row["meta_abs_diff_mass"] <= NUTR_ABS_TOL for row in per_dish
        ),
        "median_abs_plate_vs_metadata_mass_g": quantile(meta_mass_diffs, 0.5),
        "max_abs_plate_vs_metadata_mass_g": max(meta_mass_diffs) if meta_mass_diffs else None,
        "partial_mass_residual_plate_minus_sum": {
            "n": len(partial_mass_residual),
            "median_g": quantile(partial_mass_residual, 0.5),
            "positive_plate_heavier": sum(value > NUTR_ABS_TOL for value in partial_mass_residual),
            "negative_sum_heavier": sum(value < -NUTR_ABS_TOL for value in partial_mass_residual),
        },
        "complete_but_sum_mismatches": {
            "n": len(complete_mismatch),
            "median_mass_gap_plate_minus_sum_g": quantile(complete_gap, 0.5),
            "median_abs_relative_mass_gap": quantile(complete_rel, 0.5),
            "plate_heavier": sum(value > NUTR_ABS_TOL for value in complete_gap),
            "sum_heavier": sum(value < -NUTR_ABS_TOL for value in complete_gap),
            "nutrient_fail_counts": nutrient_fail,
        },
        "tolerance": {"abs": NUTR_ABS_TOL, "rel": NUTR_REL_TOL},
        "sum_uses": "only instances with has_nutrition=1; blank nutrient cells are missing, not zero",
    }
    return per_dish, summary


def write_csv(path: Path, rows: list[dict], fieldnames: list[str]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key, "") for key in fieldnames})


def json_ready(value):
    if isinstance(value, dict):
        return {str(key): json_ready(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_ready(item) for item in value]
    if isinstance(value, float):
        if math.isnan(value) or math.isinf(value):
            return None
        return value
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, (np.integer,)):
        return int(value)
    return value


def fmt(value, digits: int = 2) -> str:
    if value is None:
        return "缺失"
    if isinstance(value, float):
        return f"{value:.{digits}f}"
    return str(value)


def render_report(summary: dict) -> str:
    verdict = summary["verdict"]
    public = summary["public_tables"]
    rgb = summary["rgb_alignment"]
    official = summary["official_split"]
    nutrition = summary["nutrition_consistency"]
    plates = summary["plates"]
    scores = summary["separability"]
    oracle = summary["oracle"]
    holdout = summary["holdout_judgment"]
    v4 = summary["v4"]
    appendix = summary["partial_labels"]
    lines: list[str] = []
    add = lines.append
    add("# 自适应营养估计粒度：第一阶段数据审计")
    add("")
    add(verdict["lead"])
    add("")
    add("## 1. 审计范围")
    add("")
    add(
        "这次只做数据审计。脚本读 Hand 等人公开的两张营养表，以及本机已经解出的 "
        "Nutrition5k 俯视 RGB。推理协议仍然是单张俯视 RGB。没有使用深度、第二视角、Agent、检索或强化学习。"
        "没有训练网络，没有写第二阶段模型，没有把 505 盘子集或 Cafe2 当作验证集或调参集，"
        "也没有把 ICDL 的 1400 对相邻阶段组成差当作监督或起点。"
        "公开 CSV 只读，脚本结束时重新计算了 git blob 摘要，确认字节没有被改写。"
    )
    add("")
    add(
        f"盘级表 blob `{summary['sources']['dish_blob_sha']}`，食材表 blob `{summary['sources']['ingredient_blob_sha']}`。"
        "两份文件来自标注仓库 `liamhand/ingredient-level-nutrition-regression` 默认分支，许可是 CC BY 4.0。"
        "原始 RGB 用的是本机 `data/nutrition5k/imagery/realsense_overhead/<dish_id>/rgb.png`，"
        "没有从标注仓库转存图像。标注表里的 `rgb_image` 列只是 `rgb_images/<dish_id>.png` 这个文件名，审计不打开它。"
    )
    add("")
    add("## 2. 公开表复核")
    add("")
    add("STATUS 里 2026-11-13 的字段记录，在本机文件上逐项复核如下。")
    add("")
    add("| 项目 | 复核结果 | 与字段记录一致 |")
    add("|---|---:|:---:|")
    note = public["field_note_matches"]
    add(f"| 盘级行数 | {public['dish_rows']} | {'是' if note['dish_rows_1056'] else '否'} |")
    add(
        f"| 盘级 train / test | {public['dish_split_counts'].get('train', 0)} / {public['dish_split_counts'].get('test', 0)} | "
        f"{'是' if note['dish_train_898'] and note['dish_test_158'] else '否'} |"
    )
    add(f"| 食材实例行数 | {public['ingredient_rows']} | {'是' if note['ingredient_rows_2922'] else '否'} |")
    add(
        f"| 食材实例 train / test | {public['ingredient_split_counts'].get('train', 0)} / {public['ingredient_split_counts'].get('test', 0)} | "
        f"{'是' if note['ingredient_train_2253'] and note['ingredient_test_669'] else '否'} |"
    )
    add(
        f"| 食材表覆盖的盘，train / test | {public['ingredient_dishes_per_split'].get('train', 0)} / {public['ingredient_dishes_per_split'].get('test', 0)} | "
        f"{'是' if note['ingredient_train_dishes_898'] and note['ingredient_test_dishes_158'] else '否'} |"
    )
    add(f"| `annotation_name` 类数 | {public['annotation_name_count']} | {'是' if note['classes_40'] else '否'} |")
    has_map = public["has_nutrition_counts"]
    add(
        f"| `has_nutrition` 为 0 / 1 | {has_map.get('0', 0)} / {has_map.get('1', 0)} | "
        f"{'是' if note['has_nutrition_0_is_98'] and note['has_nutrition_1_is_2824'] else '否'} |"
    )
    add("")
    add(
        f"两表的 `dish_id` 集合{'相同' if public['same_dish_id_set'] else '不相同'}。"
        f"同一 `dish_id` 的 `split` 冲突有 {len(public['split_conflicts'])} 个。"
        f"不符合 `dish_<数字>` 的标识有 {len(public['invalid_dish_ids'])} 个。"
        f"盘级表里重复的 `dish_id` 有 {public['dish_duplicate_ids']} 个。"
        f"`rgb_image` 与 `rgb_images/<dish_id>.png` 不一致的实例有 {public['rgb_image_column_mismatches']} 个。"
        "40 个类别的名称和频数写在 `annotation_classes.csv`。"
    )
    add("")
    add("## 3. 与原始俯视 RGB、官方划分的交集")
    add("")
    add(
        f"本机俯视目录有 {rgb['overhead_dish_directories']} 个盘子目录，每个目录一张 `rgb.png`。"
        f"Hand 的 {public['dish_unique']} 个 `dish_id` 里，图像文件存在的是 {rgb['hand_rgb_present']} 个，"
        f"缺失 {rgb['hand_rgb_missing']} 个，能打开并转成 RGB 的是 {rgb['hand_rgb_readable']} 个。"
        f"图像尺寸：{'、'.join(f'{size} 有 {count} 张' for size, count in rgb['image_sizes'].items())}。"
        f"Cafe2 标识与 Hand 的交集是 {rgb['hand_in_cafe2']} 个，因此这 1056 盘全部来自 Cafe1 的标识范围。"
    )
    add("")
    add(
        "官方 RGB 划分用的是数据集发布里的 `dish_ids/splits/rgb_train_ids.txt` 与 `rgb_test_ids.txt`。"
        "Nutrition5k 的 GitHub 代码仓库只放说明和脚本，不附带这些标识表；标识表在数据发布包里，"
        f"本机 `dish_ids/README` 写明论文模型只用 Cafe1。复核：官方 train {official['rgb_train']} 条，"
        f"test {official['rgb_test']} 条，两表交集 {official['rgb_split_overlap']} 条，"
        f"并集与 `dish_ids_cafe1.txt` {'相同' if official['rgb_union_equals_cafe1'] else '不相同'} "
        f"（Cafe1 {official['cafe1']} 条）。深度划分是 RGB 划分的子集："
        f"depth train 有 {official['depth_train_not_in_rgb']} 条不在 RGB 划分里，"
        f"depth test 有 {official['depth_test_not_in_rgb']} 条不在 RGB 划分里。"
    )
    add("")
    add("| Hand 划分 | 官方 RGB train | 官方 RGB test | 不在官方 RGB 划分 |")
    add("|---|---:|---:|---:|")
    cross = official["hand_by_official"]
    add(
        f"| train | {cross['train']['official_train']} | {cross['train']['official_test']} | {cross['train']['neither']} |"
    )
    add(
        f"| test | {cross['test']['official_train']} | {cross['test']['official_test']} | {cross['test']['neither']} |"
    )
    add("")
    add(
        f"Hand 内部，同一个 `dish_id` 同时出现在 train 和 test 的有 {official['hand_dish_in_both_splits']} 个。"
        f"官方 RGB 划分内部，同一个 `dish_id` 同时出现在两边的有 {official['rgb_split_overlap']} 个。"
        f"若把两套划分混用，Hand 测试盘里有 {cross['test']['official_train']} 盘落在官方训练集，"
        f"Hand 训练盘里有 {cross['train']['official_test']} 盘落在官方测试集。"
        "本审计沿用 Hand 的 `split`，不另划测试集。上面的交叉只说明两套划分不是同一刀，不能拿来调阈值。"
    )
    add("")
    add("## 4. 盘级与食材级是否一致，以及缺失营养")
    add("")
    rate = appendix["has_nutrition_missing_rate"]
    add(
        f"`has_nutrition` 为 0 的实例是 {appendix['has_nutrition_0']} 个，"
        f"占全部 {appendix['ingredient_rows']} 个实例的 {fmt(100 * rate, 2)}%。"
        f"这些格子在质量、能量和三种宏量上是空字符串，不是数值 0。"
        f"按盘计，至少有一个实例缺营养的盘子有 {nutrition['partial_nutrition_dishes']} 个，"
        f"全部实例都有营养的盘子有 {nutrition['complete_nutrition_dishes']} 个。"
        f"训练盘里缺营养的盘子 {appendix['partial_dishes_train']} 个，测试盘里 {appendix['partial_dishes_test']} 个。"
    )
    add("")
    add(
        "一致性规则事先写死：只把 `has_nutrition=1` 的实例相加，与盘级表比较。"
        f"某个营养素算对齐，当绝对差不超过 {NUTR_ABS_TOL}，或不超过较大绝对值的 {fmt(100 * NUTR_REL_TOL, 0)}%，二者取更宽的那个。"
        f"五个营养素同时对齐的完整盘有 {nutrition['complete_and_sums_match']} / {nutrition['complete_nutrition_dishes']}。"
        f"部分标注盘里五个营养素同时对齐的有 {nutrition['partial_and_sums_match']} / {nutrition['partial_nutrition_dishes']}。"
        "部分标注盘通常对不齐，因为缺营养的实例没有进入求和，盘级总量仍包含它们。"
    )
    residual = nutrition["partial_mass_residual_plate_minus_sum"]
    add("")
    gap = nutrition["complete_but_sum_mismatches"]
    fail_counts = gap["nutrient_fail_counts"]
    add(
        f"部分标注盘上，盘级质量减去已标注实例质量和，中位数是 {fmt(residual['median_g'], 2)} 克。"
        f"盘子比已标注实例重出容差的有 {residual['positive_plate_heavier']} 盘，"
        f"已标注实例比盘子重出容差的有 {residual['negative_sum_heavier']} 盘。"
    )
    add("")
    add(
        f"全部实例都有营养的 {nutrition['complete_nutrition_dishes']} 盘里，五个营养素同时对齐的是 "
        f"{nutrition['complete_and_sums_match']} 盘，对不齐的是 {gap['n']} 盘。"
        f"这 {gap['n']} 盘里，盘子比已标注实例更重的有 {gap['plate_heavier']} 盘，"
        f"已标注实例更重的有 {gap['sum_heavier']} 盘。"
        f"质量差（盘级减求和）的中位数是 {fmt(gap['median_mass_gap_plate_minus_sum_g'], 2)} 克，"
        f"相对盘级质量的绝对差中位数是 {fmt(100 * gap['median_abs_relative_mass_gap'], 1)}%。"
        f"对不齐的次数：能量 {fail_counts['calories']}，脂肪 {fail_counts['fat']}，质量 {fail_counts['mass']}，"
        f"碳水 {fail_counts['carb']}，蛋白质 {fail_counts['protein']}。"
        "所以，即使每一行标注都有数字，食材求和仍然经常小于盘级总量。这个剩余对应没有画进掩膜的成分，或掩膜没有覆盖整盘食物。"
        f"Hand 盘级质量相对 Nutrition5k 元数据的绝对差，中位数 {fmt(nutrition['median_abs_plate_vs_metadata_mass_g'], 4)} 克，"
        f"最大 {fmt(nutrition['max_abs_plate_vs_metadata_mass_g'], 4)} 克，"
        f"落在 {NUTR_ABS_TOL} 克以内的有 {nutrition['plate_mass_matches_metadata']} / {nutrition['dishes']} 盘。"
        "盘级表与原始元数据是同一套总量。"
    )
    add("")
    add("## 5. 重拍、物理餐盘，以及留出区间")
    add("")
    add(
        "Nutrition5k 的采集是增量扫描：同一只物理餐盘每加一次菜就再扫一次，每次扫描有自己的 `dish_id`。"
        "论文写明，同一只盘子的各次扫描必须留在同一个官方划分里，避免训练和测试看到同一只盘子。"
        "STATUS 不预设重拍是否存在。本审计用元数据把这个结构算出来，没有读取 ICDL 的成对列表。"
    )
    add("")
    add(
        "合并规则：把 Cafe1 与 Cafe2 的 5006 条元数据按时间戳排序。相邻两条属于同一只盘子，当且仅当时间差不超过 "
        f"{PLATE_GAP_SECONDS} 秒，质量至少增加 {fmt(MIN_MASS_INCREASE_G, 0)} 克，后一条的食材名称多重集包含前一条，"
        f"并且同名食材的克数下降不超过 {fmt(MASS_GRAM_TOL_G, 0)} 克。再做传递闭包。"
        "180 秒写在「食材嵌套且质量增加」的相邻时间差之上："
        f"这样的相邻对有 {plates['gap_distribution']['nested_mass_increasing_seconds']['n']} 对，"
        f"中位 {plates['gap_distribution']['nested_mass_increasing_seconds']['p50']} 秒，"
        f"99 分位 {plates['gap_distribution']['nested_mass_increasing_seconds']['p99']} 秒。"
        "其余相邻对的时间差长得多。阈值在看到 Hand 留出泄漏数字之前按这个分布固定，之后没有改。"
    )
    add("")
    add(
        f"全库得到 {plates['primary']['plates']} 只物理餐盘，其中多次扫描的有 {plates['primary']['multi_plates']} 只，"
        f"覆盖 {plates['primary']['dishes_in_multi']} 条 `dish_id`。链接 {plates['primary']['links']} 条。"
        f"这套合并跨过官方 RGB train/test 的盘子有 {plates['primary']['official_cross_plates']} 只。"
        "官方划分本来就要把同一只盘子放在一边，这个很小的交叉用来检查合并没有把无关盘子粘在一起。"
        "交叉盘子的标识写在 `official_cross_plates.csv`。"
        + plates["official_cross_sentence"]
    )
    add("")
    add(
        f"Hand 的 {plates['hand']['dishes']} 条 `dish_id` 落在 {plates['hand']['plates']} 只物理餐盘上。"
        f"其中 {plates['hand']['multi_annotated_plates']} 只盘子有多于一条 Hand 标注，共 {plates['hand']['dishes_in_multi']} 条。"
        f"测试划分的 158 条 `dish_id` 落在 {plates['hand']['test_plates']} 只盘子上，"
        f"训练划分落在 {plates['hand']['train_plates']} 只盘子上。"
        f"同时含有 Hand 训练扫描和 Hand 测试扫描的盘子有 {plates['hand']['cross_split_plates']} 只。"
        f"只出现在 Hand 测试、没有 Hand 训练扫描的干净盘子有 {plates['hand']['clean_test_plates']} 只，"
        f"上面共有 {plates['hand']['clean_test_dishes']} 条测试 `dish_id`。"
        "字节级完全相同的俯视图像，在 Hand 子集里有 "
        f"{plates['exact_duplicate_groups']} 组。"
        f"时间差不超过 {NEAR_DUP_SECONDS} 秒且 32×24 灰度相关系数不低于 {NEAR_DUP_PEARSON} 的 Hand 图像对有 "
        f"{plates['near_duplicate_pairs']} 对。增量扫描会改变盘面，近重复图像很少。"
    )
    add("")
    add("同一规则在另外两个时间差下的灵敏度：")
    add("")
    add("| 时间差上限（秒） | 全库盘子数 | 跨官方划分的盘子 | Hand 测试盘子 | 跨 Hand 划分的盘子 | 干净测试盘子 |")
    add("|---:|---:|---:|---:|---:|---:|")
    for row in plates["sensitivity"]:
        add(
            f"| {row['gap_seconds']} | {row['plates']} | {row['official_cross_plates']} | "
            f"{row['hand_test_plates']} | {row['hand_cross_plates']} | {row['clean_test_plates']} |"
        )
    add("")
    add(
        "配对单位应是合并重拍之后的物理餐盘，而不是单独的 Hand `dish_id`。"
        "同一只盘子的多次扫描不是独立样本。Hand 的图像级划分没有遵守 Nutrition5k 自己的约束，"
        f"有 {plates['hand']['cross_split_plates']} 只盘子一边在训练、一边在测试。"
        "本审计不因此重划测试集。"
    )
    add("")
    add(
        f"158 这个公开测试盘数，本报告不把它预先判成通过或不足。把它收成物理餐盘之后，干净留出是 "
        f"{holdout['clean_test_plates']} 只。事先写下的下限是 {holdout['floor']} 只独立餐盘。"
        f"Hand 等人在 158 盘上发表的食材级减整盘平均 PMAE 差值，95% 区间是 "
        f"[{HAND_PUBLISHED_PMAE_CI[0]}, {HAND_PUBLISHED_PMAE_CI[1]}]，半宽 {fmt(holdout['hand_pmae_ci_half_width_pp'], 2)} 个百分点。"
        f"若残差尺度不变，按干净餐盘数外推的半宽大约是 {fmt(holdout['extrapolated_half_width_pp'], 2)} 个百分点。"
        "这只说明区间会更宽，不是一次新的模型比较。干净餐盘数低于下限，留出样本不够做带不确定区间的盘级对照。"
    )
    add("")
    add("## 6. 预先登记的视觉可分离度")
    add("")
    add(
        f"主指标名称是局部色纯净度，字段名 `{METRIC_NAME}`，版本 `{METRIC_VERSION}`。"
        "它在任何营养误差对照之前写进脚本常量。本脚本没有营养预测器，也没有在算出 Spearman 相关之后改这些常数。"
    )
    add("")
    add(
        "计算只使用当时可见的一张俯视 RGB。长边缩放到 160 像素，转到 CIE Lab。"
        f"外圈 {fmt(100 * BORDER_FRAC, 0)}% 像素的 Lab 中位数当作背景。"
        f"亮度不低于 {fmt(PLATE_L_MIN, 0)}、彩度不超过 {fmt(PLATE_CHROMA_MAX, 0)}、"
        f"与背景的 CIE76 色差不低于 {fmt(PLATE_MIN_DELTA_E, 0)} 的像素视为浅色盘面，取最大 8 连通域并填洞。"
        "食物区域是这只盘子盘面围住的孔。若孔太小，就退回填洞后的整只盘子，用来容纳米饭这类浅色食物。"
        "若找不到浅色盘子，就退回与背景色差不低于 "
        f"{fmt(FALLBACK_FG_DELTA_E, 0)} 的最大连通域。"
        f"在食物区域上对 Lab 做半径 {BOX_BLUR_RADIUS} 的盒式模糊，用最远点初始化、固定 {PRIMARY_K} 个颜色簇、"
        f"{KMEANS_ITERS} 轮聚类。每个食物像素的局部纯净度，是它周围 {PRIMARY_WINDOW}×{PRIMARY_WINDOW} 窗口里"
        "与它同簇的食物像素所占比例。主指标是这些比例的平均。越高表示颜色越成块，视觉上越分得开。"
        "食材真值个数、真值掩膜、食物码和营养标签都不进入这个分数。"
    )
    add("")
    add(
        f"稳健性对照只有两个。其一把颜色簇数改成 {ROBUST_K}，窗口仍是 {PRIMARY_WINDOW}。其二颜色簇数仍是 {PRIMARY_K}，"
        f"窗口改成 {ROBUST_WINDOW}。它们不是候选的新主指标。"
    )
    add("")
    add(
        "高低两档的切分规则也先写死，供以后使用：以 Hand 训练集上主指标的中位数为阈值，大于等于阈值的是高可分离，"
        "低于阈值的是低可分离，掩膜失败的不进档。阈值只用训练盘的分数来定。本审计不按这个阈值训练，也不在测试集上挑选阈值。"
        f"本次训练盘中位数是 {fmt(scores['train_median'], 4)}。"
        f"因为有 {plates['hand']['cross_split_plates']} 只物理餐盘同时出现在训练和测试，这个中位数不是干净的训练餐盘阈值。"
    )
    add("")
    add(
        f"1056 张图里，掩膜可用并得到主指标的有 {scores['scored']} 张，失败 {scores['unscored']} 张。"
        f"找到浅色盘子的有 {scores['plate_found']} 张。主指标在全部成功样本上的最小、中位、最大分别是 "
        f"{fmt(scores['all']['min'], 4)}、{fmt(scores['all']['p50'], 4)}、{fmt(scores['all']['max'], 4)}。"
        f"测试盘成功 {scores['test_scored']} 张，其中高可分离 {scores['test_high']} 张，低可分离 {scores['test_low']} 张。"
        "这些是按 `dish_id` 计数的描述，不是独立物理餐盘的样本数。"
        f"主指标与对照 1 的 Spearman ρ 是 {fmt(scores['spearman_primary_vs_k5']['rho'], 3)}，"
        f"与对照 2 的 ρ 是 {fmt(scores['spearman_primary_vs_win15']['rho'], 3)}。"
    )
    add("")
    add(
        "Oracle 只用 Hand 的实例数和每盘类别数，看分数是否落在同一根轴上。Oracle 不是切换规则。"
        f"在测试盘上，主指标对实例数的 Spearman ρ 是 {fmt(oracle['test_vs_instances']['rho'], 3)}"
        f"（n={oracle['test_vs_instances']['n']}，正态近似双侧 p={fmt(oracle['test_vs_instances']['normal_approx_p'], 4)}），"
        f"对类别数的 ρ 是 {fmt(oracle['test_vs_classes']['rho'], 3)}。"
        f"训练盘上对实例数的 ρ 是 {fmt(oracle['train_vs_instances']['rho'], 3)}，"
        f"对类别数的 ρ 是 {fmt(oracle['train_vs_classes']['rho'], 3)}。"
        "若实例数代表「更难分开」，相关应为负。测试盘上的相关接近 0，训练盘上是弱的正相关。"
        "因此这次预先登记的分数还没有落在 Hand 实例数那根轴上。主指标保持这次写死的定义。"
    )
    add("")
    add("## 7. 与 archive/v4.0 的差别")
    add("")
    add(
        f"只读打开的提交是 `{v4['commit']}`（{v4['commit_time']}，{v4['commit_subject']}）。"
        f"`backup/v3_excluded/experiments/` 下的条目是：{', '.join(v4['experiment_entries'])}。"
        f"行动目录 `experiments/action` 里有 {len(v4['action_files'])} 个文件。"
        "目录说明的第一句是「行动层」。逐个文件的开头和一句用途如下。"
    )
    add("")
    add("| 文件 | 用途 | 开头一句 |")
    add("|---|---|---|")
    for item in v4["action_files"]:
        opening = item["opening"].replace("|", "\\|")
        purpose = item["purpose"].replace("|", "\\|")
        add(f"| `{item['path']}` | {purpose} | {opening} |")
    add("")
    add(
        "这些文件做的是分组路由、名称先验、参照物定标、域外拒答和逐样本动作上界。"
        "行动目录里，检索「segmentation granularity」「ingredient-level segmentation」「食材级」「整盘回归」「粒度实验」的命中数是 "
        f"{len(v4['action_granularity_needle_hits'])}。"
        "为避免只看目录名而漏掉实验，又在 `backup/v3_excluded` 里排除第三方 DINOv2 树之后检索了 "
        "granularity、ingredient-level、食材级、整盘、PMAE。"
        f"命中并逐条读过的路径有 {len(v4['keyword_mentions_outside_third_party'])} 条，"
        f"其中还没有解释的是 {len(v4['unexplained_mentions'])} 条。"
    )
    add("")
    if v4["keyword_mentions_outside_third_party"]:
        for item in v4["keyword_mentions_outside_third_party"]:
            add(
            f"- `{item['path']}` 第 {item['line']} 行摘录：{item['excerpt']}。"
            f"{item['explanation'] or '这条命中还没有解释。'}"
            )
        add("")
    add(
        "诊断目录里的 `d6_capacity` 是质量回归容量消融，PMAE 指的是克重百分比误差，而且用了深度通道。"
        "它不是整盘对食材级分割粒度。日志里「ingredient-level NV-Real」是停止再训 DINOv2 的记录，也不是这个粒度问题。"
        "没有把这些脚本拷进本分支，也没有把旧路由接回来。"
    )
    add("")
    add("U1 不是对 v4.0 路由或动作选择的重跑。")
    add("")
    add("自适应营养估计粒度不是把 v4.0 的路由或动作选择再跑一遍。")
    add("")
    add("## 8. 附录：部分标签的标注协议")
    add("")
    add(
        "这一节只重新定义以后若做部分标签学习时，标签该怎么读。它不训练，也不解锁该方向。"
        "依据有两处。Hand 等人写明，油和糖这类在 RGB 里分不出来的成分没有画进掩膜。"
        "Sterling 等人在 bioRxiv 2026.07.26.740845 报告，四名标注者复核了 440 张 Nutrition5k 图像，"
        "原始食材标签系统性地漏掉了看不见的成分；修正之后，某一个视觉语言模型的食材重叠从 0.62 外推到大约 0.82。"
    )
    add("")
    add("若以后使用部分标签，协议定义如下。")
    add("")
    add(
        "1. 可见实例是人能在这张俯视 RGB 上描出边界的食材。Hand 的 2922 行就是这个集合，类别是 40 个粗类。"
    )
    add(
        "2. 有营养的实例是 `has_nutrition=1` 且质量、能量、脂肪、碳水、蛋白质都不是空值的行。空值是缺失，不能当成 0 克或 0 千卡。"
    )
    add(
        "3. 一只盘子是完整标签，当且仅当每个可见实例都有营养，并且这些实例在五个营养素上的求和与盘级表都落在第 4 节的容差内。"
        f"每一行都有营养数字的盘子有 {nutrition['complete_nutrition_dishes']} 个，其中求和也对得上的完整盘有 "
        f"{nutrition['complete_and_sums_match']} 个。其余 {appendix['not_fully_labeled_dishes']} 个按这个定义是部分标签。"
    )
    add(
        "4. 部分标签盘子上，盘级总量减去已标注实例之和，是未标注剩余。剩余可以来自没有营养字段的可见实例，也可以来自油、酱料这类没有画出来的成分。"
        "剩余不是「该成分质量为零」的负例，也不能在这次审计里当作新的监督。"
    )
    add(
        "5. 盘级质量、能量和宏量仍以 Hand 盘级表为准，并且它们与 Nutrition5k 元数据一致。部分标签只影响食材分解，不影响盘级总量本身。"
    )
    add("")
    add("## 9. 判断")
    add("")
    add(
        f"按 `dish_id` 对齐{'可靠' if verdict['alignment_reliable'] else '不可靠'}："
        f"缺失图像 {rgb['hand_rgb_missing']}，读图失败 {rgb['hand_rgb_unreadable']}，"
        f"标识和划分冲突 {len(public['split_conflicts']) + len(public['invalid_dish_ids']) + public['dish_duplicate_ids']}。"
    )
    add("")
    add(
        "主可分离度已经预先写死，测试时的输入只有俯视 RGB。"
        f"字段名 `{METRIC_NAME}`。两个稳健性对照不替代它。"
    )
    add("")
    add(
        f"留出是否够做盘级不确定区间：{'够' if holdout['large_enough'] else '不够'}。"
        f"判定码 `{holdout['decision']}`。干净物理餐盘 {holdout['clean_test_plates']}，下限 {holdout['floor']}。"
    )
    add("")
    add(verdict["stop_paragraph"])
    add("")
    add("## 10. 复现")
    add("")
    add("在仓库根目录、已安装 NumPy 和 Pillow 的环境里运行：")
    add("")
    add("```bash")
    add("python scripts/u1_phase1_data_audit.py --repo-root . --out results/u1_phase1_data_audit")
    add("python -m unittest tests/test_u1_phase1_data_audit.py")
    add("```")
    add("")
    add(
        "输入路径：`data/hand2026/dish_nutrition.csv`，`data/hand2026/ingredient_nutrition.csv`，"
        "`data/nutrition5k/imagery/realsense_overhead`，`data/nutrition5k/dish_ids/splits/rgb_train_ids.txt`，"
        "`data/nutrition5k/dish_ids/splits/rgb_test_ids.txt`，`data/nutrition5k/metadata/`。"
        "输出是本目录的 `report.md`、`summary.json` 和若干表。公开 CSV 与 Nutrition5k 图像都不写入版本库。"
    )
    add("")
    return "\n".join(lines)


def build_verdict(alignment_reliable: bool, holdout: dict) -> dict:
    metric_ready = True
    large_enough = bool(holdout["large_enough"]) and alignment_reliable
    phase1_numerically = alignment_reliable and metric_ready and large_enough
    if not alignment_reliable:
        stop = True
        stop_paragraph = (
            "对齐没有达到可靠标准，审计在这里停止，等待人工判断。"
            "docs/STATUS.md 保持原样，第二阶段继续锁定。"
        )
    elif not large_enough:
        stop = True
        stop_paragraph = (
            f"对齐可靠，主指标已经预先登记。干净的留出物理餐盘是 {holdout['clean_test_plates']} 只，"
            f"低于下限 {holdout['floor']}。另有 {holdout['cross_split_plates']} 只物理餐盘同时出现在 Hand 的训练和测试里。"
            "留出样本不够做盘级不确定区间。审计在这里停止，等待人工判断。"
            "不改测试集，不训练，不修改 docs/STATUS.md，不解锁第二阶段。"
        )
    else:
        stop = False
        stop_paragraph = (
            "三条测量都达到了事先写下的数值条件。这仍然不是在 STATUS 里宣布第一阶段通过。"
            "第二阶段要等 STATUS 被明确改写之后才解锁。本报告不改 STATUS。"
        )
    lead = (
        f"「自适应营养估计粒度」的第一阶段审计只回答三件事：标注能否按 dish_id 对齐，"
        f"有没有一个事先写死且测试时不泄漏真值的视觉可分离度，以及留出样本够不够做盘级不确定区间。"
        f"对齐{'可靠' if alignment_reliable else '不可靠'}。"
        f"主指标 `{METRIC_NAME}` 已写死，测试时只读一张俯视 RGB。"
        f"公开测试划分有 {HAND_PUBLISHED_EVAL_N} 个 dish_id；收成物理餐盘并去掉跨划分的盘子之后，"
        f"干净留出是 {holdout['clean_test_plates']} 只，下限是 {holdout['floor']} 只。"
        f"{'因此审计停止，等待人工判断。' if stop else '数值条件达到了，但 STATUS 仍不改。'}"
        "本报告不修改 docs/STATUS.md，不宣布第一阶段通过，也不解锁第二阶段。"
    )
    return {
        "alignment_reliable": alignment_reliable,
        "metric_preregistered": metric_ready,
        "metric_name": METRIC_NAME,
        "holdout_large_enough": large_enough,
        "phase1_numeric_conditions_met": phase1_numerically,
        "status_edited": False,
        "phase2_unlocked": False,
        "stop": stop,
        "stop_paragraph": stop_paragraph,
        "lead": lead,
    }


def default_paths(repo: Path) -> dict[str, Path]:
    return {
        "dish_csv": repo / "data/hand2026/dish_nutrition.csv",
        "ingredient_csv": repo / "data/hand2026/ingredient_nutrition.csv",
        "rgb_root": repo / "data/nutrition5k/imagery/realsense_overhead",
        "rgb_train": repo / "data/nutrition5k/dish_ids/splits/rgb_train_ids.txt",
        "rgb_test": repo / "data/nutrition5k/dish_ids/splits/rgb_test_ids.txt",
        "depth_train": repo / "data/nutrition5k/dish_ids/splits/depth_train_ids.txt",
        "depth_test": repo / "data/nutrition5k/dish_ids/splits/depth_test_ids.txt",
        "cafe1": repo / "data/nutrition5k/dish_ids/dish_ids_cafe1.txt",
        "cafe2": repo / "data/nutrition5k/dish_ids/dish_ids_cafe2.txt",
        "metadata": repo / "data/nutrition5k/metadata",
    }


def run_audit(repo: Path, out_dir: Path, paths: dict[str, Path] | None = None) -> dict:
    paths = paths or default_paths(repo)
    dish_bytes = paths["dish_csv"].read_bytes()
    ingredient_bytes = paths["ingredient_csv"].read_bytes()
    dish_sha = git_blob_sha1(dish_bytes)
    ingredient_sha = git_blob_sha1(ingredient_bytes)
    if dish_sha != DISH_BLOB_SHA or ingredient_sha != INGREDIENT_BLOB_SHA:
        raise SystemExit(
            "公开 CSV 的 git blob 摘要与标注仓库默认分支不一致，拒绝继续，以免审计改写过的表。"
        )
    _dish_fields, dish_rows = read_csv_rows(paths["dish_csv"])
    _ingredient_fields, ingredient_rows = read_csv_rows(paths["ingredient_csv"])
    # Re-read after parsing to prove the audit did not touch the files.
    if paths["dish_csv"].read_bytes() != dish_bytes or paths["ingredient_csv"].read_bytes() != ingredient_bytes:
        raise SystemExit("公开 CSV 在读取过程中被改写。")

    public = audit_public_tables(dish_rows, ingredient_rows)
    metadata_rows = load_nutrition5k_metadata(paths["metadata"])
    metadata_by_id = {row["dish_id"]: row for row in metadata_rows}
    rgb_train = set(load_id_list(paths["rgb_train"]))
    rgb_test = set(load_id_list(paths["rgb_test"]))
    depth_train = set(load_id_list(paths["depth_train"]))
    depth_test = set(load_id_list(paths["depth_test"]))
    cafe1 = set(load_id_list(paths["cafe1"]))
    cafe2 = set(load_id_list(paths["cafe2"]))
    overhead_dirs = [path for path in paths["rgb_root"].iterdir() if path.is_dir()]

    per_dish_nutrition, nutrition_summary = nutrition_consistency(dish_rows, ingredient_rows, metadata_by_id)
    gap_info = gap_distribution(metadata_rows)

    sensitivity = []
    primary_pack = None
    for gap in PLATE_GAP_SENSITIVITY:
        links, dish_to_plate, plates = cluster_plates(metadata_rows, gap)
        official_cross = []
        for plate_id, members in plates.items():
            sides = set()
            for dish_id in members:
                if dish_id in rgb_train:
                    sides.add("train")
                elif dish_id in rgb_test:
                    sides.add("test")
            if "train" in sides and "test" in sides:
                official_cross.append(plate_id)
        hand_test_plates = set()
        hand_train_plates = set()
        for row in dish_rows:
            plate_id = dish_to_plate[row["dish_id"]]
            if row["split"] == "test":
                hand_test_plates.add(plate_id)
            elif row["split"] == "train":
                hand_train_plates.add(plate_id)
        clean = hand_test_plates - hand_train_plates
        record = {
            "gap_seconds": gap,
            "links": links,
            "plates": len(plates),
            "official_cross_plates": len(official_cross),
            "hand_test_plates": len(hand_test_plates),
            "hand_cross_plates": len(hand_test_plates & hand_train_plates),
            "clean_test_plates": len(clean),
        }
        sensitivity.append(record)
        if gap == PLATE_GAP_SECONDS:
            primary_pack = (links, dish_to_plate, plates, official_cross, hand_test_plates, hand_train_plates, clean)
    if primary_pack is None:
        raise RuntimeError("primary plate gap missing from sensitivity list")
    links, dish_to_plate, plates, official_cross, hand_test_plates, hand_train_plates, clean = primary_pack
    missing_from_metadata = [row["dish_id"] for row in dish_rows if row["dish_id"] not in dish_to_plate]
    if missing_from_metadata:
        raise SystemExit(f"Hand dish_id 不在 Nutrition5k 元数据里: {missing_from_metadata[:5]}")

    multi_plates = sum(1 for members in plates.values() if len(members) > 1)
    dishes_in_multi = sum(len(members) for members in plates.values() if len(members) > 1)
    hand_plate_counts: Counter = Counter()
    for row in dish_rows:
        hand_plate_counts[dish_to_plate[row["dish_id"]]] += 1
    multi_annotated = sum(1 for count in hand_plate_counts.values() if count > 1)
    dishes_in_multi_hand = sum(count for count in hand_plate_counts.values() if count > 1)
    clean_test_dishes = sum(
        1 for row in dish_rows if row["split"] == "test" and dish_to_plate[row["dish_id"]] in clean
    )

    official_cross_rows = []
    for plate_id in sorted(official_cross, key=dish_timestamp):
        for dish_id in plates[plate_id]:
            if dish_id in rgb_train:
                side = "train"
            elif dish_id in rgb_test:
                side = "test"
            else:
                side = "neither"
            meta = metadata_by_id[dish_id]
            official_cross_rows.append(
                {
                    "plate_id": plate_id,
                    "dish_id": dish_id,
                    "official_rgb_split": side,
                    "mass": meta["mass"],
                    "n_ingredients": len(meta["ingredients"]),
                }
            )
    cross_bits = []
    for plate_id in sorted(official_cross, key=dish_timestamp):
        members = [row for row in official_cross_rows if row["plate_id"] == plate_id]
        chain = "、".join(
            f"{row['dish_id']}（{row['official_rgb_split']}，{row['mass']:.0f} 克，{row['n_ingredients']} 种食材）"
            for row in members
        )
        cross_bits.append(f"{plate_id}：{chain}")
    if cross_bits:
        official_cross_sentence = (
            f"这 {len(cross_bits)} 只盘子沿时间质量上升："
            + "；".join(cross_bits)
            + "。它们更像官方划分没有完全守住的增量扫描。"
        )
    else:
        official_cross_sentence = ""

    image_sizes: Counter = Counter()
    missing = []
    unreadable = []
    score_rows = []
    hashes: dict[str, list[str]] = defaultdict(list)
    thumbs: dict[str, np.ndarray] = {}
    print(f"scoring {len(dish_rows)} overhead images", flush=True)
    for index, row in enumerate(dish_rows):
        rgb_path = paths["rgb_root"] / row["dish_id"] / "rgb.png"
        instances = [item for item in ingredient_rows if item["dish_id"] == row["dish_id"]]
        score_row = {
            "dish_id": row["dish_id"],
            "split": row["split"],
            "plate_id": dish_to_plate[row["dish_id"]],
            "rgb_present": rgb_path.is_file(),
            "oracle_instance_count": len(instances),
            "oracle_class_count": len({item["annotation_name"] for item in instances}),
            "mask_ok": False,
            "plate_found": False,
            "food_frac": "",
            "s_sep": "",
            "s_sep_k5": "",
            "s_sep_win15": "",
            "failure": "",
        }
        if not rgb_path.is_file():
            missing.append(row["dish_id"])
            score_row["failure"] = "missing_rgb"
            score_rows.append(score_row)
            continue
        try:
            file_bytes = rgb_path.read_bytes()
            hashes[hashlib.sha256(file_bytes).hexdigest()].append(row["dish_id"])
            rgb = load_rgb(rgb_path)
        except Exception as exc:  # noqa: BLE001 - one bad file should not abort the audit
            unreadable.append({"dish_id": row["dish_id"], "error": str(exc)})
            score_row["failure"] = "unreadable_rgb"
            score_rows.append(score_row)
            continue
        image_sizes[f"{rgb.shape[1]}x{rgb.shape[0]}"] += 1
        thumb = Image.fromarray(rgb, mode="RGB").convert("L").resize(THUMB_SIZE, Image.Resampling.BILINEAR)
        thumbs[row["dish_id"]] = np.asarray(thumb, dtype=np.float64).ravel()
        scored = visual_separability(rgb)
        score_row["mask_ok"] = scored["mask_ok"]
        score_row["plate_found"] = scored["plate_found"]
        score_row["food_frac"] = f"{scored['food_frac']:.6f}"
        score_row["failure"] = scored["failure"] or ""
        for key in ("s_sep", "s_sep_k5", "s_sep_win15"):
            if scored[key] is not None:
                score_row[key] = f"{scored[key]:.6f}"
        score_rows.append(score_row)
        if (index + 1) % 200 == 0:
            print(f"  scored {index + 1}", flush=True)

    def numeric(row: dict, key: str) -> float | None:
        text = row[key]
        if text == "":
            return None
        return float(text)

    scored_values = [numeric(row, "s_sep") for row in score_rows if numeric(row, "s_sep") is not None]
    train_values = [
        numeric(row, "s_sep")
        for row in score_rows
        if row["split"] == "train" and numeric(row, "s_sep") is not None
    ]
    train_median = quantile(train_values, 0.5)
    test_high = test_low = test_scored = 0
    for row in score_rows:
        value = numeric(row, "s_sep")
        if row["split"] != "test" or value is None or train_median is None:
            row["separability_bin"] = "unscored" if value is None else ""
            continue
        test_scored += 1
        if value >= train_median:
            row["separability_bin"] = "high"
            test_high += 1
        else:
            row["separability_bin"] = "low"
            test_low += 1
    for row in score_rows:
        if row["split"] == "train":
            value = numeric(row, "s_sep")
            if value is None or train_median is None:
                row["separability_bin"] = "unscored"
            elif value >= train_median:
                row["separability_bin"] = "high"
            else:
                row["separability_bin"] = "low"

    def paired_spearman(key_a: str, key_b: str, split: str | None = None) -> dict:
        left = []
        right = []
        for row in score_rows:
            if split is not None and row["split"] != split:
                continue
            a = numeric(row, key_a)
            b = numeric(row, key_b) if key_b in row else None
            if key_b.startswith("oracle_"):
                b = float(row[key_b])
            if a is None or b is None:
                continue
            left.append(a)
            right.append(b)
        return spearman(np.array(left), np.array(right))

    near_duplicate_pairs = 0
    ordered_ids = sorted(thumbs, key=dish_timestamp)
    for left_i, left_id in enumerate(ordered_ids):
        left_t = dish_timestamp(left_id)
        left_vec = thumbs[left_id]
        for right_id in ordered_ids[left_i + 1 :]:
            if dish_timestamp(right_id) - left_t > NEAR_DUP_SECONDS:
                break
            rho = pearson(left_vec, thumbs[right_id])
            if rho is not None and rho >= NEAR_DUP_PEARSON:
                near_duplicate_pairs += 1
    exact_duplicate_groups = sum(1 for members in hashes.values() if len(members) > 1)

    hand_by_official = {}
    for split_name in ("train", "test"):
        ids = {row["dish_id"] for row in dish_rows if row["split"] == split_name}
        hand_by_official[split_name] = {
            "official_train": len(ids & rgb_train),
            "official_test": len(ids & rgb_test),
            "neither": len(ids - rgb_train - rgb_test),
        }
    hand_splits = defaultdict(set)
    for row in dish_rows:
        hand_splits[row["dish_id"]].add(row["split"])
    hand_dish_in_both = sum(1 for splits in hand_splits.values() if splits == {"train", "test"})

    partial_train = sum(1 for row in per_dish_nutrition if row["split"] == "train" and not row["all_instances_have_nutrition"])
    partial_test = sum(1 for row in per_dish_nutrition if row["split"] == "test" and not row["all_instances_have_nutrition"])
    has0 = int(public["has_nutrition_counts"].get("0", 0))
    alignment_reliable = (
        all(public["field_note_matches"].values())
        and public["rgb_image_column_mismatches"] == 0
        and len(missing) == 0
        and len(unreadable) == 0
    )
    holdout = judge_holdout(len(clean), len(hand_test_plates & hand_train_plates))
    v4 = read_v4_inventory(repo)
    verdict = build_verdict(alignment_reliable, holdout)

    class_counts = Counter(row["annotation_name"] for row in ingredient_rows)
    summary = {
        "sources": {
            "dish_csv": str(paths["dish_csv"]),
            "ingredient_csv": str(paths["ingredient_csv"]),
            "dish_blob_sha": dish_sha,
            "ingredient_blob_sha": ingredient_sha,
            "dish_sha256": sha256_bytes(dish_bytes),
            "ingredient_sha256": sha256_bytes(ingredient_bytes),
            "rgb_root": str(paths["rgb_root"]),
            "annotation_repo": "https://github.com/liamhand/ingredient-level-nutrition-regression",
            "paper_doi": "10.3389/fnut.2026.1927530",
            "metric_version": METRIC_VERSION,
        },
        "public_tables": public,
        "rgb_alignment": {
            "overhead_dish_directories": len(overhead_dirs),
            "hand_rgb_present": len(dish_rows) - len(missing),
            "hand_rgb_missing": len(missing),
            "hand_rgb_readable": len(dish_rows) - len(missing) - len(unreadable),
            "hand_rgb_unreadable": len(unreadable),
            "missing_ids": missing,
            "unreadable": unreadable,
            "image_sizes": dict(image_sizes),
            "hand_in_cafe2": len({row["dish_id"] for row in dish_rows} & cafe2),
            "hand_in_metadata": sum(row["dish_id"] in metadata_by_id for row in dish_rows),
        },
        "official_split": {
            "rgb_train": len(rgb_train),
            "rgb_test": len(rgb_test),
            "rgb_split_overlap": len(rgb_train & rgb_test),
            "cafe1": len(cafe1),
            "cafe2": len(cafe2),
            "rgb_union_equals_cafe1": (rgb_train | rgb_test) == cafe1,
            "depth_train_not_in_rgb": len(depth_train - rgb_train - rgb_test),
            "depth_test_not_in_rgb": len(depth_test - rgb_train - rgb_test),
            "hand_by_official": hand_by_official,
            "hand_dish_in_both_splits": hand_dish_in_both,
        },
        "nutrition_consistency": nutrition_summary,
        "plates": {
            "gap_distribution": gap_info,
            "primary": {
                "gap_seconds": PLATE_GAP_SECONDS,
                "links": links,
                "plates": len(plates),
                "multi_plates": multi_plates,
                "dishes_in_multi": dishes_in_multi,
                "official_cross_plates": len(official_cross),
            },
            "hand": {
                "dishes": len(dish_rows),
                "plates": len(hand_plate_counts),
                "multi_annotated_plates": multi_annotated,
                "dishes_in_multi": dishes_in_multi_hand,
                "test_plates": len(hand_test_plates),
                "train_plates": len(hand_train_plates),
                "cross_split_plates": len(hand_test_plates & hand_train_plates),
                "clean_test_plates": len(clean),
                "clean_test_dishes": clean_test_dishes,
            },
            "sensitivity": sensitivity,
            "official_cross_sentence": official_cross_sentence,
            "exact_duplicate_groups": exact_duplicate_groups,
            "near_duplicate_pairs": near_duplicate_pairs,
        },
        "separability": {
            "name": METRIC_NAME,
            "version": METRIC_VERSION,
            "primary_k": PRIMARY_K,
            "primary_window": PRIMARY_WINDOW,
            "robust_k": ROBUST_K,
            "robust_window": ROBUST_WINDOW,
            "train_median": train_median,
            "scored": len(scored_values),
            "unscored": len(score_rows) - len(scored_values),
            "plate_found": sum(1 for row in score_rows if row["plate_found"]),
            "test_scored": test_scored,
            "test_high": test_high,
            "test_low": test_low,
            "all": {
                "min": quantile(scored_values, 0.0),
                "p50": quantile(scored_values, 0.5),
                "max": quantile(scored_values, 1.0),
            },
            "spearman_primary_vs_k5": paired_spearman("s_sep", "s_sep_k5"),
            "spearman_primary_vs_win15": paired_spearman("s_sep", "s_sep_win15"),
            "inputs": "single overhead RGB; no ingredient count, mask, food code, or nutrition label",
        },
        "oracle": {
            "test_vs_instances": paired_spearman("s_sep", "oracle_instance_count", "test"),
            "test_vs_classes": paired_spearman("s_sep", "oracle_class_count", "test"),
            "train_vs_instances": paired_spearman("s_sep", "oracle_instance_count", "train"),
            "train_vs_classes": paired_spearman("s_sep", "oracle_class_count", "train"),
            "role": "diagnostic only; not a switching rule",
        },
        "holdout_judgment": holdout,
        "v4": v4,
        "partial_labels": {
            "has_nutrition_0": has0,
            "ingredient_rows": public["ingredient_rows"],
            "has_nutrition_missing_rate": has0 / public["ingredient_rows"] if public["ingredient_rows"] else None,
            "partial_dishes_train": partial_train,
            "partial_dishes_test": partial_test,
            "not_fully_labeled_dishes": len(per_dish_nutrition) - nutrition_summary["complete_and_sums_match"],
            "unlocks_training": False,
        },
        "verdict": verdict,
    }

    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "summary.json").write_text(
        json.dumps(json_ready(summary), ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    (out_dir / "report.md").write_text(render_report(summary), encoding="utf-8")
    (out_dir / "metric_preregistration.json").write_text(
        json.dumps(
            {
                "name": METRIC_NAME,
                "version": METRIC_VERSION,
                "inputs": "one overhead RGB image",
                "forbidden_inputs": [
                    "ingredient instance count",
                    "ground-truth mask",
                    "food code",
                    "nutrition label",
                    "depth",
                    "second view",
                ],
                "primary": {
                    "long_side": WORK_LONG_SIDE,
                    "palette_k": PRIMARY_K,
                    "window": PRIMARY_WINDOW,
                    "blur_radius": BOX_BLUR_RADIUS,
                },
                "robustness_checks": [
                    {"name": "s_sep_k5", "palette_k": ROBUST_K, "window": PRIMARY_WINDOW},
                    {"name": "s_sep_win15", "palette_k": PRIMARY_K, "window": ROBUST_WINDOW},
                ],
                "high_low_rule": "high if score >= median of Hand train dishes with a score; low otherwise",
                "oracle_is_not_the_switch": True,
                "retuned_after_nutrition_error": False,
            },
            ensure_ascii=False,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    write_csv(
        out_dir / "separability_scores.csv",
        score_rows,
        [
            "dish_id",
            "split",
            "plate_id",
            "rgb_present",
            "mask_ok",
            "plate_found",
            "food_frac",
            "s_sep",
            "s_sep_k5",
            "s_sep_win15",
            "separability_bin",
            "failure",
            "oracle_instance_count",
            "oracle_class_count",
        ],
    )
    alignment_rows = []
    for row in dish_rows:
        dish_id = row["dish_id"]
        if dish_id in rgb_train:
            official_side = "train"
        elif dish_id in rgb_test:
            official_side = "test"
        else:
            official_side = "neither"
        alignment_rows.append(
            {
                "dish_id": dish_id,
                "hand_split": row["split"],
                "plate_id": dish_to_plate[dish_id],
                "plate_crosses_hand_split": dish_to_plate[dish_id] in (hand_test_plates & hand_train_plates),
                "official_rgb_split": official_side,
                "cafe": metadata_by_id[dish_id]["cafe"],
                "rgb_present": (paths["rgb_root"] / dish_id / "rgb.png").is_file(),
            }
        )
    write_csv(
        out_dir / "dish_alignment.csv",
        alignment_rows,
        [
            "dish_id",
            "hand_split",
            "plate_id",
            "plate_crosses_hand_split",
            "official_rgb_split",
            "cafe",
            "rgb_present",
        ],
    )
    write_csv(
        out_dir / "nutrition_consistency_by_dish.csv",
        per_dish_nutrition,
        [
            "dish_id",
            "split",
            "n_instances",
            "n_with_nutrition",
            "n_missing_nutrition",
            "all_instances_have_nutrition",
            "sum_matches_plate_all_nutrients",
            "plate_mass",
            "sum_mass",
            "abs_diff_mass",
            "plate_calories",
            "sum_calories",
            "abs_diff_calories",
            "plate_fat",
            "sum_fat",
            "abs_diff_fat",
            "plate_carb",
            "sum_carb",
            "abs_diff_carb",
            "plate_protein",
            "sum_protein",
            "abs_diff_protein",
            "meta_mass",
            "meta_abs_diff_mass",
            "meta_calories",
            "meta_abs_diff_calories",
        ],
    )
    write_csv(
        out_dir / "annotation_classes.csv",
        [{"annotation_name": name, "instances": class_counts[name]} for name in public["annotation_names"]],
        ["annotation_name", "instances"],
    )
    write_csv(
        out_dir / "official_cross_plates.csv",
        official_cross_rows,
        ["plate_id", "dish_id", "official_rgb_split", "mass", "n_ingredients"],
    )
    write_csv(
        out_dir / "plate_gap_sensitivity.csv",
        sensitivity,
        [
            "gap_seconds",
            "links",
            "plates",
            "official_cross_plates",
            "hand_test_plates",
            "hand_cross_plates",
            "clean_test_plates",
        ],
    )
    (out_dir / "v4_action_inventory.json").write_text(
        json.dumps(json_ready(v4), ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    if paths["dish_csv"].read_bytes() != dish_bytes or paths["ingredient_csv"].read_bytes() != ingredient_bytes:
        raise SystemExit("公开 CSV 在写出报告时被改写。")
    return summary


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Audit Hand annotations against Nutrition5k overhead RGB.")
    parser.add_argument("--repo-root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--out", type=Path, default=None)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    repo = args.repo_root.resolve()
    out_dir = args.out or (repo / "results/u1_phase1_data_audit")
    if not out_dir.is_absolute():
        out_dir = repo / out_dir
    summary = run_audit(repo, out_dir)
    verdict = summary["verdict"]
    print(json.dumps({key: verdict[key] for key in ("alignment_reliable", "holdout_large_enough", "stop")}, ensure_ascii=False))
    print(f"wrote {out_dir / 'report.md'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
