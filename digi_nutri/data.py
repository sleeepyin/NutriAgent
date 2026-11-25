"""RGB, relative-height, and ingredient batches."""

from __future__ import annotations

import random
from pathlib import Path

import cv2
import numpy as np
import torch
from PIL import Image

from digi_nutri.depth_to_height import depth_raw_to_height01
from digi_nutri.ingredients import encode_ingredients
from digi_nutri.metadata import nutrition_vector
from digi_nutri.protocol import HFLIP_P, IMAGE_SIZE, IMAGENET_MEAN, IMAGENET_STD, MAX_INGREDIENTS
from digi_nutri.usda import USDADictionary


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
        raise ValueError("Q0 records are only b1, t_geo, and t_ing")
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


def rgb_uint8_to_model(array_hwc: np.ndarray, flip: bool) -> torch.Tensor:
    array = np.asarray(array_hwc)
    if flip:
        array = np.flip(array, axis=1)
    return _imagenet(array.astype(np.float32) / 255.0)


def height01_to_model(height_hw: np.ndarray, flip: bool) -> torch.Tensor:
    height = np.asarray(height_hw, dtype=np.float32)
    if flip:
        height = np.flip(height, axis=1)
    stacked = np.repeat(height[:, :, None], 3, axis=2)
    return _imagenet(stacked)


def read_rgb_uint8(path: Path) -> np.ndarray:
    image = Image.open(path).convert("RGB")
    image = image.resize((IMAGE_SIZE, IMAGE_SIZE), Image.BILINEAR)
    return np.asarray(image, dtype=np.uint8)


def read_height01(path: Path) -> np.ndarray:
    raw = np.asarray(Image.open(path))
    if raw.dtype != np.uint16 or raw.ndim != 2:
        raise ValueError(f"{path} is not a 16-bit depth map")
    height = depth_raw_to_height01(raw)
    resized = cv2.resize(height, (IMAGE_SIZE, IMAGE_SIZE), interpolation=cv2.INTER_LINEAR)
    return np.asarray(resized, dtype=np.float32)


def cache_paths(cache_dir: str | Path, dish_id: str) -> tuple[Path, Path]:
    root = Path(cache_dir)
    return root / f"{dish_id}.rgb.npy", root / f"{dish_id}.height.npy"


def ensure_cache(
    dish_ids: list[str],
    rgb_root: str | Path,
    depth_root: str | Path,
    cache_dir: str | Path,
) -> None:
    root = Path(cache_dir)
    root.mkdir(parents=True, exist_ok=True)
    rgb_root = Path(rgb_root)
    depth_root = Path(depth_root)
    for index, dish_id in enumerate(dish_ids, start=1):
        rgb_out, height_out = cache_paths(root, dish_id)
        if rgb_out.exists() and height_out.exists():
            continue
        rgb = read_rgb_uint8(rgb_root / dish_id / "rgb.png")
        height = read_height01(depth_root / dish_id / "depth_raw.png")
        np.save(rgb_out, rgb)
        np.save(height_out, height.astype(np.float16))
        if index % 250 == 0 or index == len(dish_ids):
            print(f"cache {index}/{len(dish_ids)}", flush=True)


class PlateDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        records: list[dict],
        kind: str,
        train: bool,
        rgb_root: str | Path,
        depth_root: str | Path,
        cache_dir: str | Path | None = None,
    ):
        self.records = records
        self.kind = kind
        self.train = train
        self.rgb_root = Path(rgb_root)
        self.depth_root = Path(depth_root)
        self.cache_dir = None if cache_dir is None else Path(cache_dir)

    def __len__(self) -> int:
        return len(self.records)

    def _load_rgb(self, dish_id: str) -> np.ndarray:
        if self.cache_dir is not None:
            path, _height = cache_paths(self.cache_dir, dish_id)
            if path.exists():
                return np.load(path)
        return read_rgb_uint8(self.rgb_root / dish_id / "rgb.png")

    def _load_height(self, dish_id: str) -> np.ndarray:
        if self.cache_dir is not None:
            _rgb, path = cache_paths(self.cache_dir, dish_id)
            if path.exists():
                return np.load(path).astype(np.float32)
        return read_height01(self.depth_root / dish_id / "depth_raw.png")

    def __getitem__(self, index: int) -> dict:
        record = self.records[index]
        flip = self.train and random.random() < HFLIP_P
        item = {
            "dish_id": record["dish_id"],
            "group_id": record["group_id"],
            "rgb": rgb_uint8_to_model(self._load_rgb(record["dish_id"]), flip),
            "nutrition": torch.tensor(record["nutrition"], dtype=torch.float32),
            "mass": torch.tensor(record["mass"], dtype=torch.float32),
        }
        if self.kind == "t_geo":
            item["height"] = height01_to_model(self._load_height(record["dish_id"]), flip)
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
    if "name_ids" in batch[0]:
        out["name_ids"] = torch.stack([item["name_ids"] for item in batch])
        out["priors"] = torch.stack([item["priors"] for item in batch])
        out["ing_mask"] = torch.stack([item["ing_mask"] for item in batch])
    return out
