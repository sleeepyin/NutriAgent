"""CPU point predictions for the hold-out images.

Nutrition5K and NV-Real use the cached 224 tensors. ECUSTFD and MetaFood3D
are resized the same way. The process must not see a GPU: the 27B server
already occupies GPU 0.
"""
from __future__ import annotations

import os

os.environ["CUDA_VISIBLE_DEVICES"] = ""

import sys
from pathlib import Path

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[2]
for entry in (str(ROOT), str(ROOT / "src")):
    if entry not in sys.path:
        sys.path.insert(0, entry)

import torch

from experiments.estimation.train_weight_4ch import WeightNet
from experiments.vlm.holdout import n5k_cal_indices, n5k_eval_indices
from experiments.vlm.store import read_manifest
from mmcfa.paths import Paths
from mmcfa.runtime import atomic_write_json, log


def _load(path: Path, backbone: str, channels: int) -> WeightNet:
    model = WeightNet(backbone, 0.3, in_channels=channels, pretrained=False)
    model.load_state_dict(torch.load(path, map_location="cpu", weights_only=True))
    model.eval()
    return model


@torch.no_grad()
def _predict(model: WeightNet, images: np.ndarray, bs: int = 32) -> np.ndarray:
    out = []
    for start in range(0, len(images), bs):
        batch = torch.tensor(images[start : start + bs], dtype=torch.float32)
        out.append(model(batch).numpy())
    return np.clip(np.concatenate(out), 0, None)


def _rgb224(paths: list[str]) -> np.ndarray:
    frames = []
    for path in paths:
        image = Image.open(path).convert("RGB").resize((224, 224), Image.BILINEAR)
        arr = np.transpose(np.asarray(image, dtype=np.float32) / 255.0, (2, 0, 1))
        frames.append(arr)
    return np.stack(frames).astype(np.float32)


def _write(path: Path, rows: list[dict]) -> None:
    import json

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")


def main() -> int:
    torch.set_num_threads(int(os.environ.get("PRED_THREADS", "8")))
    paths = Paths.resolve()
    vlm_out = paths.root / "backup" / "v3_excluded" / "results" / "vlm"
    dest = vlm_out / "points.jsonl"
    mark = vlm_out / "points_complete.json"
    if mark.is_file() and dest.is_file():
        log(f"point predictions already present: {dest}")
        return 0
    manifest = read_manifest()
    if not manifest:
        raise SystemExit("cache/vlm/manifest.jsonl is missing")

    crops = np.load(paths.expect("deep_crops"))
    x_te = crops["X_test"]
    y_te = np.asarray(crops["y_test"], dtype=float)
    dish = np.asarray(crops["dish_id_test"]).astype(str)
    y_train_median = float(np.median(crops["y_train"]))
    ev = set(int(i) for i in n5k_eval_indices(len(y_te)))
    cal = set(int(i) for i in n5k_cal_indices(len(y_te)))

    rn3 = _load(paths.weights_nvreal / "deep_weight_resnet18_3ch.pth", "resnet18", 3)
    eff3 = _load(paths.weights_nvreal / "deep_weight_efficientnet_3ch.pth", "efficientnet_b0", 3)
    p_rn3 = _predict(rn3, x_te[:, :3])
    p_eff3 = _predict(eff3, x_te[:, :3])
    rn4 = _load(paths.weights_n5k / "deep_weight_resnet18_final.pth", "resnet18", 4)
    eff4 = _load(paths.weights_n5k / "deep_weight_efficientnet_final.pth", "efficientnet_b0", 4)
    p_rn4 = _predict(rn4, x_te)
    p_eff4 = _predict(eff4, x_te)
    del rn4, eff4

    rows = []
    for i in range(len(y_te)):
        rows.append(
            {
                "dataset": "n5k",
                "sample_id": dish[i],
                "split": "eval" if i in ev else "cal",
                "y_g": float(y_te[i]),
                "p_rn3": float(p_rn3[i]),
                "p_eff3": float(p_eff3[i]),
                "p_ens3": float((p_rn3[i] + p_eff3[i]) / 2),
                "p_rn4": float(p_rn4[i]),
                "p_eff4": float(p_eff4[i]),
                "p_ens4": float((p_rn4[i] + p_eff4[i]) / 2),
            }
        )
    log(f"N5K point predictions n={len(y_te)} cal={len(cal)} eval={len(ev)}")

    with np.load(paths.expect("nv_dish_cache"), allow_pickle=True) as nv:
        x_nv = np.asarray(nv["X"])
        y_nv = np.asarray(nv["y"], dtype=float)
        files = np.asarray(nv["file_name"]).astype(str)
        split = np.asarray(nv["split"]).astype(str)
    p_nv_rn = _predict(rn3, x_nv)
    p_nv_eff = _predict(eff3, x_nv)
    joint_dir = paths.root / "backup" / "v3_excluded" / "weights" / "nvreal"
    jrn = _load(joint_dir / "joint_resnet18_3ch.pth", "resnet18", 3)
    jeff = _load(joint_dir / "joint_efficientnet_3ch.pth", "efficientnet_b0", 3)
    p_jrn = _predict(jrn, x_nv)
    p_jeff = _predict(jeff, x_nv)
    del rn3, eff3, jrn, jeff
    for i, name in enumerate(files):
        rows.append(
            {
                "dataset": "nvreal",
                "sample_id": name,
                "split": split[i],
                "y_g": float(y_nv[i]),
                "p_rn3": float(p_nv_rn[i]),
                "p_eff3": float(p_nv_eff[i]),
                "p_ens3": float((p_nv_rn[i] + p_nv_eff[i]) / 2),
                "p_joint": float((p_jrn[i] + p_jeff[i]) / 2),
            }
        )
    log(f"NV-Real point predictions n={len(files)}")

    for dataset in ("ecustfd", "mf3d"):
        chosen = [row for row in manifest if row["dataset"] == dataset and row.get("image")]
        if not chosen:
            continue
        rn = _load(paths.weights_nvreal / "deep_weight_resnet18_3ch.pth", "resnet18", 3)
        eff = _load(paths.weights_nvreal / "deep_weight_efficientnet_3ch.pth", "efficientnet_b0", 3)
        images = _rgb224([row["image"] for row in chosen])
        pr = _predict(rn, images)
        pe = _predict(eff, images)
        del rn, eff
        for row, a, b in zip(chosen, pr, pe):
            rows.append(
                {
                    "dataset": dataset,
                    "sample_id": row["sample_id"],
                    "split": row["split"],
                    "y_g": float(row["y_g"]),
                    "p_ens3": float((a + b) / 2),
                }
            )
        log(f"{dataset} point predictions n={len(chosen)}")

    _write(dest, rows)
    atomic_write_json(
        mark,
        {"n": len(rows), "n5k_train_median_g": y_train_median, "protocol": "cpu 224 bilinear [0,1]"},
    )
    log(f"wrote {dest}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
