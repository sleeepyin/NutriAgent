"""GPU guard, seeding, and checkpoints."""

from __future__ import annotations

import os
import random
from pathlib import Path

import numpy as np
import torch

from repro.protocol import ROOT


def require_gpu() -> torch.device:
    visible = os.environ.get("CUDA_VISIBLE_DEVICES")
    if visible != "1":
        raise SystemExit("refusing to run unless CUDA_VISIBLE_DEVICES=1")
    if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise SystemExit("expected exactly one visible GPU")
    return torch.device("cuda:0")


def seed_all(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def save_checkpoint(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".tmp")
    torch.save(payload, temporary)
    temporary.replace(path)


def load_checkpoint(path: Path) -> dict:
    return torch.load(path, map_location="cpu")


def output_dir(name: str) -> Path:
    path = ROOT / "outputs" / name
    path.mkdir(parents=True, exist_ok=True)
    return path
