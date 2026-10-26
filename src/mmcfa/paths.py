"""All project locations, resolved from this repository.

Scripts should call `n5k()` or `Paths.resolve()` instead of hard-coding a
machine path. A missing file stays missing: nothing here downloads data or
creates weight files.

Assets that v4.0 does not use resolve under ``backup/v3_excluded``.
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path


def repo_root() -> Path:
    here = Path(__file__).resolve()
    for parent in here.parents:
        marker = parent / "src" / "mmcfa" / "paths.py"
        if marker.is_file() and (parent / "docs").is_dir():
            return parent
    raise RuntimeError("找不到项目根目录（缺少 src/mmcfa/paths.py）")


def _env_path(name: str, default: Path) -> Path:
    raw = os.environ.get(name)
    if not raw:
        return default
    return Path(raw).expanduser().resolve()


_CACHE_NAMES = {
    "deep_crops_cache.npz",
    "aec_v2_pred_cache.npz",
    "nutrient_labels.npz",
    "nutrient_density.npz",
    "nutrient_conformal_cache.npz",
    "cqr_pred_cache.npz",
    "features_cache_v3.pkl",
}


@dataclass
class Paths:
    root: Path
    data_n5k: Path
    n5k_raw: Path
    n5k_cache: Path
    data_mf3d: Path
    data_ecustfd: Path
    data_nvreal: Path
    weights_n5k: Path
    weights_nvreal: Path
    weights_mf3d: Path
    weights_sam3d: Path
    weights_dinov2: Path
    dinov2_hub: Path
    sam3d_code: Path
    d1: Path
    d3: Path
    d4_csv: Path
    results_action: Path
    results_estimation: Path
    results_diagnosis: Path
    results_appendix: Path
    vlm_cache: Path
    extras: dict = field(default_factory=dict)

    @property
    def results(self) -> Path:
        """Action-layer result root. Kept under this name for the runners."""
        return self.results_action

    @classmethod
    def resolve(cls) -> "Paths":
        root = repo_root()
        excluded = root / "backup" / "v3_excluded"
        data_n5k = root / "data" / "nutrition5k"
        n5k_cache = data_n5k / "cache"
        data_mf3d = root / "data" / "metafood3d"
        data_ecustfd = root / "data" / "ecustfd"
        data_nvreal = root / "data" / "nvreal"
        weights_n5k = root / "weights" / "nutrition5k"
        weights_dinov2 = excluded / "weights" / "dinov2"
        d1 = root / "experiments" / "diagnosis" / "d1_mf3d_oracle"
        d3 = excluded / "experiments" / "diagnosis" / "d3_mf3d_sam3d"
        derived = data_nvreal / "derived"
        paths = cls(
            root=root,
            data_n5k=data_n5k,
            n5k_raw=data_n5k / "raw",
            n5k_cache=n5k_cache,
            data_mf3d=data_mf3d,
            data_ecustfd=data_ecustfd,
            data_nvreal=data_nvreal,
            weights_n5k=weights_n5k,
            weights_nvreal=root / "weights" / "nvreal",
            weights_mf3d=root / "weights" / "metafood3d",
            weights_sam3d=excluded / "weights" / "sam3d",
            weights_dinov2=weights_dinov2,
            dinov2_hub=excluded / "third_party" / "dinov2",
            sam3d_code=_env_path("SAM3D_CODE", excluded / "third_party" / "sam3d"),
            d1=d1,
            d3=d3,
            d4_csv=root / "experiments" / "diagnosis" / "d4_ecustfd_scale" / "reference_results.csv",
            results_action=excluded / "results" / "action",
            results_estimation=root / "results" / "estimation",
            results_diagnosis=root / "results" / "diagnosis",
            results_appendix=excluded / "results" / "appendix",
            vlm_cache=excluded / "cache" / "vlm",
        )
        paths.extras = {
            "deep_crops": n5k_cache / "deep_crops_cache.npz",
            "aec_cache": n5k_cache / "aec_v2_pred_cache.npz",
            "nutrient_labels": n5k_cache / "nutrient_labels.npz",
            "nutrient_density": n5k_cache / "nutrient_density.npz",
            "nutrient_cache": n5k_cache / "nutrient_conformal_cache.npz",
            "cqr_cache": n5k_cache / "cqr_pred_cache.npz",
            "features_cache": n5k_cache / "features_cache_v3.pkl",
            "calibrator": excluded / "weights" / "nutrition5k" / "nutrient_conformal_calibrator.pkl",
            "results_v3": root / "results" / "v3",
            "vlm_cache_v3": root / "cache" / "vlm_v3",
            "w_rn4": weights_n5k / "deep_weight_resnet18_final.pth",
            "w_eff4": weights_n5k / "deep_weight_efficientnet_final.pth",
            "w_rn3": paths.weights_nvreal / "deep_weight_resnet18_3ch.pth",
            "w_eff3": paths.weights_nvreal / "deep_weight_efficientnet_3ch.pth",
            "nv_dish_cache": data_nvreal / "cache" / "nvreal_dish_cache.npz",
            "nv_phase2b_csv": derived / "phase2b_per_image.csv",
            "nv_phase2b_json": derived / "phase2b_results.json",
            "nv_phase3_json": derived / "phase3_conformal_results.json",
            "nv_phase4_json": derived / "phase4_nutrient_results.json",
            "ecustfd_csv": paths.d4_csv,
            "ecustfd_xls": data_ecustfd / "density.xls",
            "mf3d_labels": d1 / "metafood3d_labels_clean.csv",
            "mf3d_phase1": d1 / "phase1_results.csv",
            "mf3d_oracle6": d3 / "oracle6_results.csv",
            "mf3d_render": data_mf3d / "data" / "Processed" / "Processed" / "Blender_render_images",
            "dino_weights": weights_dinov2 / "dinov2_vits14_pretrain.pth",
        }
        return paths

    def file(self, key: str) -> Path | None:
        value = self.extras.get(key)
        if value is None:
            return None
        path = Path(value)
        return path if path.exists() else None

    def expect(self, key: str) -> Path:
        value = self.extras.get(key)
        if value is None:
            raise KeyError(key)
        return Path(value)


def n5k_file(name: str) -> Path:
    """Nutrition5K asset. Caches stay under data/; checkpoints stay under weights/."""
    paths = Paths.resolve()
    if name in _CACHE_NAMES or name.endswith(".npz"):
        return paths.n5k_cache / name
    return paths.weights_n5k / name


def n5k(name: str) -> str:
    return str(n5k_file(name))
