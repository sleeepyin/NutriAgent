"""Resize, pad, and normalization. No Mixup or ColorJitter."""

from __future__ import annotations

import random

import numpy as np
import torch
from PIL import Image


def imagenet_norm(chw: torch.Tensor, mean: list[float], std: list[float]) -> torch.Tensor:
    mean_t = torch.tensor(mean, dtype=chw.dtype).view(3, 1, 1)
    std_t = torch.tensor(std, dtype=chw.dtype).view(3, 1, 1)
    return (chw - mean_t) / std_t


def _resize_rgb(rgb: np.ndarray, width: int, height: int) -> np.ndarray:
    image = Image.fromarray(rgb)
    return np.asarray(image.resize((width, height), Image.BICUBIC), dtype=np.uint8)


def _resize_float(array: np.ndarray, width: int, height: int, nearest: bool) -> np.ndarray:
    mode = Image.NEAREST if nearest else Image.BILINEAR
    if array.dtype == np.bool_ or nearest:
        image = Image.fromarray(array.astype(np.uint8) * (255 if array.dtype == np.bool_ else 1))
        out = np.asarray(image.resize((width, height), Image.NEAREST))
        if array.dtype == np.bool_:
            return out > 0
        return out
    image = Image.fromarray(array.astype(np.float32), mode="F")
    return np.asarray(image.resize((width, height), mode), dtype=np.float32)


def long_side_resize(
    rgb: np.ndarray, depth_m: np.ndarray, valid: np.ndarray, long_side: int
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    height, width = rgb.shape[:2]
    if height >= width:
        new_h = long_side
        new_w = max(1, int(round(width * long_side / height)))
    else:
        new_w = long_side
        new_h = max(1, int(round(height * long_side / width)))
    return (
        _resize_rgb(rgb, new_w, new_h),
        _resize_float(depth_m, new_w, new_h, nearest=False),
        _resize_float(valid.astype(bool), new_w, new_h, nearest=True).astype(bool),
    )


def pad_square(
    rgb: np.ndarray,
    depth_m: np.ndarray,
    valid: np.ndarray,
    size: int,
    train: bool,
    rng: random.Random,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, int, int]:
    height, width = rgb.shape[:2]
    if height > size or width > size:
        raise RuntimeError(f"frame {height}x{width} exceeds pad size {size}")
    pad_h = size - height
    pad_w = size - width
    if train:
        top = rng.randrange(pad_h + 1) if pad_h else 0
        left = rng.randrange(pad_w + 1) if pad_w else 0
    else:
        top = pad_h // 2
        left = pad_w // 2
    canvas = np.zeros((size, size, 3), dtype=np.uint8)
    depth_canvas = np.zeros((size, size), dtype=np.float32)
    valid_canvas = np.zeros((size, size), dtype=bool)
    canvas[top : top + height, left : left + width] = rgb
    depth_canvas[top : top + height, left : left + width] = depth_m
    valid_canvas[top : top + height, left : left + width] = valid
    return canvas, depth_canvas, valid_canvas, top, left


def hflip(rgb: np.ndarray, *arrays: np.ndarray) -> tuple[np.ndarray, ...]:
    flipped = [np.ascontiguousarray(rgb[:, ::-1])]
    for array in arrays:
        flipped.append(np.ascontiguousarray(array[:, ::-1]))
    return tuple(flipped)


def rgb_to_chw(rgb: np.ndarray) -> torch.Tensor:
    array = rgb.astype(np.float32) / 255.0
    return torch.from_numpy(np.ascontiguousarray(array.transpose(2, 0, 1)))


def resize_depth_content(depth: np.ndarray, width: int, height: int) -> np.ndarray:
    return _resize_float(depth.astype(np.float32), width, height, nearest=False)
