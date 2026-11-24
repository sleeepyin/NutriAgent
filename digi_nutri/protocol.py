"""Frozen Phase 1 protocol constants.

Hyperparameters here are the pre-registered starting point. They are not a
stop rule. Stop rules live in verdict.py and are applied once, after the
official holdout read.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path

PROTOCOL_VERSION = "digi-nutri-phase1-2026-11-24"

NUTRIENTS = ("energy", "mass", "fat", "carb", "protein")
HEADLINE = ("energy", "fat", "carb", "protein")
UNITS = {
    "energy": "kcal",
    "mass": "g",
    "fat": "g",
    "carb": "g",
    "protein": "g",
}

# Per-dish PMAE divides by max(|y|, PMAE_EPS). 1 kcal or 1 g.
PMAE_EPS = 1.0

RGB_MEAN = (0.485, 0.456, 0.406)
RGB_STD = (0.229, 0.224, 0.225)

VARIANTS = ("b1", "b2a", "b2b", "b3")

OFFICIAL_SPLIT_FILES = {
    "train": "depth_train_ids.txt",
    "test": "depth_test_ids.txt",
}
OFFICIAL_SPLIT_COUNTS = {"train": 2758, "test": 507}

LIVE_RGB_ROOT = Path("/home/sdnu/nutriagent/data/nutrition5k/imagery/realsense_overhead")
LIVE_SPLITS = Path("/home/sdnu/nutriagent/data/nutrition5k/dish_ids/splits")
LIVE_METADATA = (
    Path("/home/sdnu/nutriagent/data/nutrition5k/metadata/dish_metadata_cafe1.csv"),
    Path("/home/sdnu/nutriagent/data/nutrition5k/metadata/dish_metadata_cafe2.csv"),
)
WORK_ROOT = Path("/home/sdnu/nutriagent-digi")
DEPTH_ROOT = WORK_ROOT / "data" / "official_depth"
OUTPUTS = WORK_ROOT / "outputs" / "digi_nutri_phase1"
RESULTS = WORK_ROOT / "results" / "digi_nutri_phase1"
HOLDOUT_METRICS = RESULTS / "holdout_metrics.json"
FROZEN_CONFIG = RESULTS / "frozen_config.json"


def needs_geo(variant: str) -> bool:
    return variant in {"b2a", "b3"}


def needs_sem(variant: str) -> bool:
    return variant in {"b2b", "b3"}


def needs_aux(variant: str) -> bool:
    return variant in {"b2b", "b3"}


@dataclass
class Phase1Config:
    protocol_version: str = PROTOCOL_VERSION
    backbone: str = "resnet101"
    image_size: int = 256
    batch_size: int = 32
    epochs_teachers: int = 30
    epochs_students: int = 30
    lr: float = 1e-4
    weight_decay: float = 1e-4
    warmup_epochs: int = 5
    ramp_epochs: int = 5
    eta_min: float = 1e-6
    seed: int = 0
    hidden: int = 512
    fusion: str = "add"
    depth_fill: str = "median"
    lambda_geo: float = 1.0
    lambda_sem: float = 1.0
    lambda_aux: float = 0.2
    pmae_eps: float = PMAE_EPS
    bootstrap_samples: int = 2000
    bootstrap_seed: int = 0
    ci_alpha: float = 0.05
    num_workers: int = 4
    text_model: str = "openai/clip-vit-base-patch32"
    gram_weight: bool = False
    optimizer: str = "adamw"
    scheduler: str = "cosine"
    target_mean: list | None = None
    target_std: list | None = None
    vocab: list | None = None
    split_train_sha256: str = ""
    split_test_sha256: str = ""
    code_fingerprint: str = ""
    batch_size_note: str = ""
    n_train: int = 0
    n_test: int = 0


def config_dict(cfg: Phase1Config) -> dict:
    return asdict(cfg)
