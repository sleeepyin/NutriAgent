"""Overhead RGB and depth readers. Public Nutrition5k files are not rewritten."""

from __future__ import annotations

import random
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset

from repro.geometry import (
    hflip,
    imagenet_norm,
    long_side_resize,
    pad_square,
    resize_depth_content,
    rgb_to_chw,
    _resize_rgb,
)
from repro.metadata import label_vector


def read_rgb(path: Path) -> np.ndarray:
    return np.asarray(Image.open(path).convert("RGB"), dtype=np.uint8)


def read_depth(path: Path) -> tuple[np.ndarray, np.ndarray]:
    raw = np.asarray(Image.open(path))
    meters = raw.astype(np.float32) * 1e-4
    return meters, raw > 0


def depth_to_resnet(meters: np.ndarray, max_meters: float, mean: list[float], std: list[float]) -> torch.Tensor:
    scaled = np.clip(meters / max_meters, 0.0, 1.0).astype(np.float32)
    chw = torch.from_numpy(np.repeat(scaled[None, ...], 3, axis=0))
    return imagenet_norm(chw, mean, std)


class SwinPlates(Dataset):
    def __init__(self, dish_ids: list[str], dishes: dict, config: dict, train: bool):
        self.ids = list(dish_ids)
        self.labels = [label_vector(dishes[dish_id], tuple(config["label_order"])) for dish_id in self.ids]
        self.rgb_root = Path(config["paths"]["rgb_root"])
        self.train = train
        self.epoch = 0
        self.seed = int(config["seed"])
        spec = config["swin"]
        self.height = int(spec["height"])
        self.width = int(spec["width"])
        self.flip_p = float(spec["horizontal_flip_p"])
        self.mean = spec["imagenet_mean"]
        self.std = spec["imagenet_std"]

    def set_epoch(self, epoch: int) -> None:
        self.epoch = epoch

    def __len__(self) -> int:
        return len(self.ids)

    def __getitem__(self, index: int):
        dish_id = self.ids[index]
        rgb = _resize_rgb(read_rgb(self.rgb_root / dish_id / "rgb.png"), self.width, self.height)
        rng = random.Random(self.seed + self.epoch * 1000003 + index)
        if self.train and rng.random() < self.flip_p:
            rgb = np.ascontiguousarray(rgb[:, ::-1])
        image = imagenet_norm(rgb_to_chw(rgb), self.mean, self.std)
        target = torch.tensor(self.labels[index], dtype=torch.float32)
        return image, target, dish_id


class DepthPlates(Dataset):
    def __init__(self, dish_ids: list[str], config: dict, train: bool):
        self.ids = list(dish_ids)
        self.rgb_root = Path(config["paths"]["rgb_root"])
        self.depth_root = Path(config["paths"]["depth_root"])
        self.train = train
        self.epoch = 0
        self.seed = int(config["seed"])
        spec = config["dpf"]["depth_module"]
        self.long_side = int(spec["long_side"])
        self.crop = int(spec["crop"])
        self.mean = spec["normalize_mean"]
        self.std = spec["normalize_std"]

    def set_epoch(self, epoch: int) -> None:
        self.epoch = epoch

    def __len__(self) -> int:
        return len(self.ids)

    def __getitem__(self, index: int):
        dish_id = self.ids[index]
        rgb = read_rgb(self.rgb_root / dish_id / "rgb.png")
        depth, valid = read_depth(self.depth_root / dish_id / "depth_raw.png")
        rng = random.Random(self.seed + self.epoch * 1000003 + index)
        rgb, depth, valid = long_side_resize(rgb, depth, valid, self.long_side)
        rgb, depth, valid, _top, _left = pad_square(rgb, depth, valid, self.crop, self.train, rng)
        image = imagenet_norm(rgb_to_chw(rgb), self.mean, self.std)
        return (
            image,
            torch.from_numpy(depth),
            torch.from_numpy(valid),
            dish_id,
        )


class FusionPlates(Dataset):
    def __init__(self, dish_ids: list[str], dishes: dict, config: dict, train: bool, cache_dir: Path):
        self.ids = list(dish_ids)
        self.labels = [label_vector(dishes[dish_id], tuple(config["label_order"])) for dish_id in self.ids]
        self.rgb_root = Path(config["paths"]["rgb_root"])
        self.cache_dir = Path(cache_dir)
        self.train = train
        self.epoch = 0
        self.seed = int(config["seed"])
        spec = config["dpf"]["fusion_module"]
        self.height = int(spec["height"])
        self.width = int(spec["width"])
        self.flip_p = float(spec["horizontal_flip_p"])
        self.mean = spec["imagenet_mean"]
        self.std = spec["imagenet_std"]
        self.max_meters = float(spec["depth_vis_max_meters"])

    def set_epoch(self, epoch: int) -> None:
        self.epoch = epoch

    def __len__(self) -> int:
        return len(self.ids)

    def __getitem__(self, index: int):
        dish_id = self.ids[index]
        rgb = _resize_rgb(read_rgb(self.rgb_root / dish_id / "rgb.png"), self.width, self.height)
        depth = np.load(self.cache_dir / f"{dish_id}.npy")
        if depth.shape != (self.height, self.width):
            depth = resize_depth_content(depth, self.width, self.height)
        rng = random.Random(self.seed + self.epoch * 1000003 + index)
        if self.train and rng.random() < self.flip_p:
            rgb, depth = hflip(rgb, depth)
        image = imagenet_norm(rgb_to_chw(rgb), self.mean, self.std)
        depth_image = depth_to_resnet(depth, self.max_meters, self.mean, self.std)
        target = torch.tensor(self.labels[index], dtype=torch.float32)
        return image, depth_image, target, dish_id


def predict_fusion_depth(model, rgb: np.ndarray, config: dict) -> np.ndarray:
    """Deterministic DPT depth resampled to the fusion resolution."""
    spec = config["dpf"]["depth_module"]
    fusion = config["dpf"]["fusion_module"]
    dummy = np.zeros(rgb.shape[:2], dtype=np.float32)
    valid = np.ones(rgb.shape[:2], dtype=bool)
    rgb_r, dummy_r, valid_r = long_side_resize(rgb, dummy, valid, int(spec["long_side"]))
    content_h, content_w = rgb_r.shape[:2]
    rgb_p, _depth_p, _valid_p, top, left = pad_square(
        rgb_r, dummy_r, valid_r, int(spec["crop"]), False, random.Random(0)
    )
    image = imagenet_norm(rgb_to_chw(rgb_p), spec["normalize_mean"], spec["normalize_std"])
    pred = model(image.unsqueeze(0).to(next(model.parameters()).device))[0].detach().float().cpu().numpy()
    content = pred[top : top + content_h, left : left + content_w]
    return resize_depth_content(content, int(fusion["width"]), int(fusion["height"]))
