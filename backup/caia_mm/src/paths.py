"""Resolve all project paths from MM_CFA_ROOT with fallbacks. Never hardcode a single absolute path as the only option."""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path


def _first_existing(*cands: str | Path) -> Path | None:
    for c in cands:
        if c and Path(c).exists():
            return Path(c)
    return None


def mm_cfa_root() -> Path:
    env = os.environ.get("MM_CFA_ROOT")
    if env and Path(env).exists():
        return Path(env).resolve()
    here = Path(__file__).resolve()
    for p in [here.parents[2], Path("/home/mm-cfa-repro"), Path.cwd()]:
        if (p / "Nutri_0811").exists() or (p / "Agent创新点_详细完整实施方案.md").exists():
            return p.resolve()
    return Path("/home/mm-cfa-repro")


@dataclass
class Paths:
    root: Path
    nutri: Path
    models: Path
    nvreal: Path
    ecustfd: Path
    metafood3d: Path
    n5k_raw: Path | None
    agent: Path
    results: Path
    traces: Path
    logs: Path
    configs: Path
    prereg: Path
    extras: dict = field(default_factory=dict)

    @classmethod
    def resolve(cls) -> "Paths":
        root = mm_cfa_root()
        os.environ["MM_CFA_ROOT"] = str(root)
        nutri = _first_existing(root / "Nutri_0811", Path("/home/Nutri_0811")) or (root / "Nutri_0811")
        models = _first_existing(nutri / "models", Path("/home/Nutri_0811/models")) or (nutri / "models")
        nvreal = _first_existing(root / "nvreal", Path("/home/nvreal")) or (root / "nvreal")
        ecustfd = _first_existing(Path("/home/ecustfd"), root / "ecustfd") or (root / "ecustfd")
        metafood3d = _first_existing(Path("/home/metafood3d"), root / "metafood3d") or (root / "metafood3d")
        n5k_raw = _first_existing(
            Path("/home/nutrition5k/nutri_dataset/nutrition5k_dataset"),
            Path("/home/nutrition5k"),
        )
        agent = root / "agent_extension"
        p = cls(
            root=root,
            nutri=nutri,
            models=models,
            nvreal=nvreal,
            ecustfd=ecustfd,
            metafood3d=metafood3d,
            n5k_raw=n5k_raw,
            agent=agent,
            results=agent / "results",
            traces=agent / "traces",
            logs=agent / "logs",
            configs=agent / "configs",
            prereg=agent / "preregistration",
        )
        for d in (p.results, p.traces, p.logs):
            d.mkdir(parents=True, exist_ok=True)
        p.extras = {
            "deep_crops": models / "deep_crops_cache.npz",
            "aec_cache": models / "aec_v2_pred_cache.npz",
            "nutrient_labels": models / "nutrient_labels.npz",
            "nutrient_density": models / "nutrient_density.npz",
            "nutrient_cache": models / "nutrient_conformal_cache.npz",
            "w_rn4": models / "deep_weight_resnet18_final.pth",
            "w_eff4": models / "deep_weight_efficientnet_final.pth",
            "w_rn3": nvreal / "deep_weight_resnet18_3ch.pth",
            "w_eff3": nvreal / "deep_weight_efficientnet_3ch.pth",
            "nv_dish_cache": nvreal / "nvreal_dish_cache.npz",
            "nv_phase2b_csv": nvreal / "phase2b_per_image.csv",
            "nv_phase2b_json": nvreal / "phase2b_results.json",
            "nv_phase3_json": nvreal / "phase3_conformal_results.json",
            "nv_phase4_json": nvreal / "phase4_nutrient_results.json",
            "ecustfd_csv": _first_existing(ecustfd / "ecustfd_reference_results.csv", root / "ecustfd" / "ecustfd_reference_results.csv"),
            "ecustfd_xls": _first_existing(ecustfd / "density.xls"),
            "mf3d_labels": _first_existing(
                root / "metafood3d" / "metafood3d_labels_clean.csv",
                metafood3d / "metafood3d_labels_clean.csv",
            ),
            "mf3d_phase1": _first_existing(
                root / "metafood3d" / "mf3d_phase1_results.csv",
                metafood3d / "mf3d_phase1_results.csv",
            ),
            "sha256sum": root / "sha256sum.txt",
            "n5k_meta1": (n5k_raw / "metadata" / "dish_metadata_cafe1.csv") if n5k_raw else None,
            "n5k_meta2": (n5k_raw / "metadata" / "dish_metadata_cafe2.csv") if n5k_raw else None,
            "n5k_ids": (n5k_raw / "dish_ids" / "dish_ids_all.txt") if n5k_raw else None,
        }
        return p

    def file(self, key: str) -> Path | None:
        v = self.extras.get(key)
        if v is None:
            return None
        return Path(v) if Path(v).exists() else None
