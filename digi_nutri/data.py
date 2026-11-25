"""RGB batches, shared geometric augmentation, and the height cache."""

from __future__ import annotations

import json
import random
from pathlib import Path

import numpy as np
import torch

from digi_nutri.ingredients import encode_ingredients
from digi_nutri.metadata import nutrition_vector
from digi_nutri.protocol import (
    HFLIP_P,
    IMAGE_SIZE,
    IMAGENET_MEAN,
    IMAGENET_STD,
    MAX_INGREDIENTS,
    VFLIP_P,
)
from digi_nutri.usda import USDADictionary


def sample_geometry(train: bool, rng: random.Random | None = None) -> tuple[int, bool, bool]:
    """Train-only dihedral draw. Evaluation returns the identity."""
    if not train:
        return 0, False, False
    draw = random if rng is None else rng
    rot_k = draw.randrange(4)
    flip_h = draw.random() < HFLIP_P
    flip_v = draw.random() < VFLIP_P
    return rot_k, flip_h, flip_v


def apply_geometry(array: np.ndarray, rot_k: int, flip_h: bool, flip_v: bool) -> np.ndarray:
    """Rotate by rot_k * 90 degrees, then flip. Labels are not an input."""
    if rot_k not in (0, 1, 2, 3):
        raise ValueError(f"rotation k must be 0..3, got {rot_k}")
    out = np.rot90(np.asarray(array), k=rot_k, axes=(0, 1))
    if flip_h:
        out = np.flip(out, axis=1)
    if flip_v:
        out = np.flip(out, axis=0)
    return np.ascontiguousarray(out)


def make_records(
    kind: str,
    dish_ids: list[str],
    dishes: dict[str, dict],
    groups: dict[str, str],
    usda: USDADictionary | None = None,
    vocab: dict | None = None,
) -> list[dict]:
    if kind in {"b1", "t_geo"} and (usda is not None or vocab is not None):
        raise ValueError(f"{kind} does not read the USDA dictionary")
    if kind == "t_ing" and (usda is None or vocab is None):
        raise ValueError("t_ing requires the static USDA dictionary and vocabulary")
    if kind not in {"b1", "t_geo", "t_ing"}:
        raise ValueError("records are only b1, t_geo, and t_ing")
    records = []
    for dish_id in dish_ids:
        dish = dishes[dish_id]
        record = {
            "dish_id": dish_id,
            "group_id": groups[dish_id],
            "nutrition": nutrition_vector(dish),
            "mass": float(dish["mass"]),
        }
        if kind == "t_ing":
            name_ids, priors, mask = encode_ingredients(dish, usda, vocab, MAX_INGREDIENTS)
            record["name_ids"] = name_ids
            record["priors"] = priors
            record["ing_mask"] = mask
        records.append(record)
    return records


def _imagenet(array_hwc: np.ndarray) -> torch.Tensor:
    mean = np.asarray(IMAGENET_MEAN, dtype=np.float32)
    std = np.asarray(IMAGENET_STD, dtype=np.float32)
    normalized = (array_hwc.astype(np.float32) - mean) / std
    return torch.from_numpy(np.ascontiguousarray(normalized.transpose(2, 0, 1)))


def rgb_uint8_to_model(array_hwc: np.ndarray, rot_k: int, flip_h: bool, flip_v: bool) -> torch.Tensor:
    array = apply_geometry(array_hwc, rot_k, flip_h, flip_v)
    return _imagenet(array.astype(np.float32) / 255.0)


def height01_to_model(height_hw: np.ndarray, rot_k: int, flip_h: bool, flip_v: bool) -> tuple[torch.Tensor, torch.Tensor]:
    height = apply_geometry(np.asarray(height_hw, dtype=np.float32), rot_k, flip_h, flip_v)
    if float(height.min()) < -1e-4 or float(height.max()) > 1.0 + 1e-3:
        raise RuntimeError("height map left the fixed [0, 1] scale")
    stacked = np.repeat(height[:, :, None], 3, axis=2)
    height01 = torch.from_numpy(height[None, ...].astype(np.float32))
    return _imagenet(stacked), height01


def cache_paths(cache_dir: str | Path, dish_id: str) -> tuple[Path, Path]:
    root = Path(cache_dir)
    return root / f"{dish_id}.rgb.npy", root / f"{dish_id}.height.npy"


def assert_cache_ready(dish_ids: list[str], cache_dir: str | Path, need_height: bool) -> None:
    """Read the existing height cache. Do not rewrite it."""
    root = Path(cache_dir)
    marker = root / "height_formula.json"
    if not marker.exists():
        raise RuntimeError("height cache has no formula marker")
    found = json.loads(marker.read_text())
    if found.get("exclude_border_zeros") is not True:
        raise RuntimeError("height cache was not built with missing-depth zeros excluded")
    missing = []
    for dish_id in dish_ids:
        rgb_path, height_path = cache_paths(root, dish_id)
        if not rgb_path.exists():
            missing.append(dish_id)
        if need_height and not height_path.exists():
            missing.append(f"{dish_id}:height")
        if len(missing) > 8:
            break
    if missing:
        raise RuntimeError(f"height cache is missing files: {missing}")


def _load_rgb(cache_dir: Path, dish_id: str) -> np.ndarray:
    path, _height = cache_paths(cache_dir, dish_id)
    array = np.load(path)
    if array.shape != (IMAGE_SIZE, IMAGE_SIZE, 3):
        raise RuntimeError(f"{dish_id} rgb cache shape is {array.shape}")
    return array


def _load_height(cache_dir: Path, dish_id: str) -> np.ndarray:
    _rgb, path = cache_paths(cache_dir, dish_id)
    array = np.load(path).astype(np.float32)
    if array.shape != (IMAGE_SIZE, IMAGE_SIZE):
        raise RuntimeError(f"{dish_id} height cache shape is {array.shape}")
    return array


class PlateDataset(torch.utils.data.Dataset):
    def __init__(self, records: list[dict], kind: str, train: bool, cache_dir: str | Path):
        if kind not in {"b1", "t_geo", "t_ing"}:
            raise ValueError(kind)
        self.records = records
        self.kind = kind
        self.train = train
        self.cache_dir = Path(cache_dir)
        assert_cache_ready(
            [record["dish_id"] for record in records],
            self.cache_dir,
            need_height=(kind == "t_geo"),
        )

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int) -> dict:
        record = self.records[index]
        rot_k, flip_h, flip_v = sample_geometry(self.train)
        item = {
            "dish_id": record["dish_id"],
            "group_id": record["group_id"],
            "rgb": rgb_uint8_to_model(
                _load_rgb(self.cache_dir, record["dish_id"]), rot_k, flip_h, flip_v
            ),
            "nutrition": torch.tensor(record["nutrition"], dtype=torch.float32),
            "mass": torch.tensor(record["mass"], dtype=torch.float32),
        }
        if self.kind == "t_geo":
            height, height01 = height01_to_model(
                _load_height(self.cache_dir, record["dish_id"]), rot_k, flip_h, flip_v
            )
            item["height"] = height
            item["height01"] = height01
        elif self.kind == "t_ing":
            item["name_ids"] = torch.tensor(record["name_ids"], dtype=torch.long)
            item["priors"] = torch.tensor(record["priors"], dtype=torch.float32)
            item["ing_mask"] = torch.tensor(record["ing_mask"], dtype=torch.bool)
        return item


def collate_plates(batch: list[dict]) -> dict:
    out = {
        "dish_id": [item["dish_id"] for item in batch],
        "group_id": [item["group_id"] for item in batch],
        "rgb": torch.stack([item["rgb"] for item in batch]),
        "nutrition": torch.stack([item["nutrition"] for item in batch]),
        "mass": torch.stack([item["mass"] for item in batch]),
    }
    if "height" in batch[0]:
        out["height"] = torch.stack([item["height"] for item in batch])
        out["height01"] = torch.stack([item["height01"] for item in batch])
    if "name_ids" in batch[0]:
        out["name_ids"] = torch.stack([item["name_ids"] for item in batch])
        out["priors"] = torch.stack([item["priors"] for item in batch])
        out["ing_mask"] = torch.stack([item["ing_mask"] for item in batch])
    return out
