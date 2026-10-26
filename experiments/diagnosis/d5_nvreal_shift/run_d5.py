#!/usr/bin/env python3
"""NV-Real domain shift on the R0 3-channel Nutrition5K weights.

Zero-shot uses every NV-Real image. The frozen calibrator is fit on one half
of the Nutrition5K dish hold-out and applied unchanged. Joint training adds
only the official Train dishes and is scored on the official Val dishes.
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from PIL import Image, ImageDraw
from scipy.ndimage import maximum_filter
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader


def _boot() -> Path:
    here = Path(__file__).resolve()
    for parent in here.parents:
        if (parent / "src" / "mmcfa" / "paths.py").is_file():
            for entry in (str(parent), str(parent / "src")):
                if entry not in sys.path:
                    sys.path.insert(0, entry)
            return parent
    raise RuntimeError("找不到项目根目录")


ROOT = _boot()
from experiments.estimation.run_r0 import (  # noqa: E402
    mondrian_intervals,
    norm_intervals,
    split_intervals,
)
from experiments.estimation.train_weight_4ch import (  # noqa: E402
    AugDS,
    WeightNet,
    predict,
    train_model,
)
from mmcfa.metrics import mae, mape, wape  # noqa: E402
from mmcfa.paths import Paths  # noqa: E402

SIZE = 224
DILATE = 41  # 20 px radius, odd kernel
PROTOCOL = "coco-union dilate-20 zero-background bilinear-224 rgb-[0,1]"
DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def status(msg: str) -> None:
    print(time.strftime("%F %T ") + msg, flush=True)


def point_block(y, p) -> dict:
    y = np.asarray(y, dtype=float).reshape(-1)
    p = np.asarray(p, dtype=float).reshape(-1)
    if y.size == 0:
        return {"n": 0}
    spearman = float(pd.Series(y).corr(pd.Series(p), method="spearman"))
    return {
        "n": int(y.size),
        "mape_pct": float(100.0 * mape(y, p)),
        "wape": float(wape(y, p)),
        "mae_g": float(mae(y, p)),
        "r2": float(r2_score(y, p)),
        "spearman": spearman,
        "shrink": float(np.std(p) / max(np.std(y), 1e-8)),
    }


def load_model(path: Path, backbone: str, channels: int) -> WeightNet:
    model = WeightNet(backbone, 0.3, in_channels=channels, pretrained=False).to(DEVICE)
    model.load_state_dict(torch.load(path, map_location=DEVICE, weights_only=True))
    model.eval()
    return model


def ensemble(rn, eff, x: np.ndarray) -> np.ndarray:
    return (predict(rn, x) + predict(eff, x)) / 2.0


def interval_report(name: str, y, lo, hi) -> dict:
    y = np.asarray(y, dtype=float)
    cover = (y >= lo) & (y <= hi) & np.isfinite(hi)
    width = np.asarray(hi, dtype=float) - np.asarray(lo, dtype=float)
    block = {
        "method": name,
        "n": int(y.size),
        "coverage": float(np.mean(cover)) if y.size else float("nan"),
        "width_g": float(np.mean(width)) if y.size else float("nan"),
    }
    status(f"  {name} cov={100 * block['coverage']:.1f}% width={block['width_g']:.1f}g n={block['n']}")
    return block


def draw_union(width: int, height: int, segments: list) -> np.ndarray:
    mask = Image.new("L", (width, height), 0)
    draw = ImageDraw.Draw(mask)
    for seg in segments:
        if not isinstance(seg, list) or len(seg) < 6:
            continue
        points = list(zip(seg[0::2], seg[1::2]))
        draw.polygon(points, fill=255)
    arr = np.asarray(mask)
    if min(arr.shape) < DILATE:
        return arr
    return maximum_filter(arr, size=DILATE)


def build_cache(paths: Paths) -> dict:
    cache = paths.expect("nv_dish_cache")
    mark = cache.with_suffix(".json")
    if cache.is_file() and mark.is_file():
        info = json.loads(mark.read_text(encoding="utf-8"))
        if info.get("protocol") == PROTOCOL:
            status(f"cache exists, skip rebuild: {cache}")
            with np.load(cache, allow_pickle=True) as data:
                return {key: data[key] for key in data.files}

    image_dir = paths.data_nvreal / "raw" / "nutritionverse-manual" / "nutritionverse-manual" / "images"
    coco_path = image_dir / "_annotations.coco.json"
    split_path = image_dir.parent / "updated-manual-dataset-splits.csv"
    meta_path = paths.data_nvreal / "raw" / "nutritionverse_dish_metadata3.csv"
    coco = json.loads(coco_path.read_text(encoding="utf-8"))
    splits = pd.read_csv(split_path)
    splits["file_name"] = splits["file_name"].astype(str).str.strip()
    splits["category"] = splits["category"].astype(str).str.strip().str.lower()
    split_of = dict(zip(splits["file_name"], splits["category"]))
    meta = pd.read_csv(meta_path)
    by_dish = meta.set_index(meta["dish_id"].astype(int))
    by_image = {}
    for ann in coco["annotations"]:
        by_image.setdefault(ann["image_id"], []).append(ann)

    xs, ys, dishes, split_names, files = [], [], [], [], []
    nutr = []
    skipped = {"no_file": 0, "no_meta": 0, "empty_mask": 0}
    t0 = time.time()
    for i, image in enumerate(coco["images"]):
        name = image["file_name"]
        path = image_dir / name
        dish = int(name.split("_")[1])
        if not path.is_file():
            skipped["no_file"] += 1
            continue
        if dish not in by_dish.index:
            skipped["no_meta"] += 1
            continue
        segments = []
        for ann in by_image.get(image["id"], []):
            segments.extend(ann["segmentation"])
        rgb = Image.open(path).convert("RGB")
        mask_arr = draw_union(rgb.width, rgb.height, segments)
        ys_px, xs_px = np.nonzero(mask_arr)
        if xs_px.size == 0:
            skipped["empty_mask"] += 1
            continue
        box = (int(xs_px.min()), int(ys_px.min()), int(xs_px.max()) + 1, int(ys_px.max()) + 1)
        crop = np.asarray(rgb.crop(box), dtype=np.float32)
        crop_mask = mask_arr[box[1]:box[3], box[0]:box[2]] > 0
        crop[~crop_mask] = 0.0
        resized = Image.fromarray(crop.astype(np.uint8), mode="RGB").resize((SIZE, SIZE), Image.BILINEAR)
        arr = np.transpose(np.asarray(resized, dtype=np.float32) / 255.0, (2, 0, 1))
        row = by_dish.loc[dish]
        xs.append(arr)
        ys.append(float(row["total_food_weight"]))
        dishes.append(dish)
        split_names.append(split_of.get(name, "unlisted"))
        files.append(name)
        nutr.append([
            float(row["total_calories"]),
            float(row["total_fats"]),
            float(row["total_carbohydrates"]),
            float(row["total_protein"]),
        ])
        if (i + 1) % 50 == 0:
            status(f"crops {i + 1}/{len(coco['images'])} kept={len(xs)} elapsed={time.time() - t0:.0f}s")

    payload = {
        "X": np.stack(xs).astype(np.float32),
        "y": np.asarray(ys, dtype=np.float32),
        "dish_id": np.asarray(dishes, dtype=np.int32),
        "split": np.asarray(split_names),
        "file_name": np.asarray(files),
        "nutrients": np.asarray(nutr, dtype=np.float32),
    }
    cache.parent.mkdir(parents=True, exist_ok=True)
    tmp = cache.with_suffix(".tmp.npz")
    np.savez(str(tmp), **payload)
    tmp.replace(cache)
    info = {
        "protocol": PROTOCOL,
        "n": int(len(ys)),
        "skipped": skipped,
        "seconds": round(time.time() - t0, 1),
    }
    mark.write_text(json.dumps(info, indent=2) + "\n", encoding="utf-8")
    status(f"wrote {cache} n={info['n']} skipped={skipped}")
    return payload


def fit_models(x, y, rn_path: Path, eff_path: Path) -> None:
    if rn_path.is_file() and eff_path.is_file():
        status(f"skip existing joint weights {rn_path.name}")
        return
    xtr, xv, ytr, yv = train_test_split(x, y, test_size=0.15, random_state=42)
    train_loader = DataLoader(AugDS(xtr, ytr, aug=True), 32, shuffle=True, num_workers=0, pin_memory=True)
    val_loader = DataLoader(AugDS(xv, yv, aug=False), 64, shuffle=False, num_workers=0, pin_memory=True)
    for path, backbone in ((rn_path, "resnet18"), (eff_path, "efficientnet_b0")):
        if path.is_file():
            continue
        status(f"train {path.name} n={len(y)}")
        model = train_model(path.name, WeightNet(backbone, 0.3, in_channels=3, pretrained=True), train_loader, val_loader)
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(model.state_dict(), path)
        del model
        torch.cuda.empty_cache()


def main() -> None:
    if DEVICE.type != "cuda":
        raise SystemExit("D5 需要 GPU")
    paths = Paths.resolve()
    out_dir = paths.results_diagnosis / "d5_nvreal"
    out_dir.mkdir(parents=True, exist_ok=True)
    derived = paths.data_nvreal / "derived"
    derived.mkdir(parents=True, exist_ok=True)
    status(f"D5 start device={DEVICE}")
    nv = build_cache(paths)
    x_nv = nv["X"]
    y_nv = nv["y"].astype(np.float32)
    split = np.asarray(nv["split"]).astype(str)
    val = split == "val"
    status(f"NV images={len(y_nv)} val={int(val.sum())} train={int((split == 'train').sum())}")

    n5k = np.load(paths.n5k_cache / "deep_crops_cache.npz")
    x_te = n5k["X_test"][:, :3]
    y_te = n5k["y_test"].astype(np.float32)
    x_tr = n5k["X_train"][:, :3]
    y_tr = n5k["y_train"].astype(np.float32)

    rn3 = load_model(paths.weights_nvreal / "deep_weight_resnet18_3ch.pth", "resnet18", 3)
    eff3 = load_model(paths.weights_nvreal / "deep_weight_efficientnet_3ch.pth", "efficientnet_b0", 3)
    p_n5k = ensemble(rn3, eff3, x_te)
    p_nv = ensemble(rn3, eff3, x_nv)
    del rn3, eff3
    torch.cuda.empty_cache()

    rn4 = load_model(paths.weights_n5k / "deep_weight_resnet18_final.pth", "resnet18", 4)
    eff4 = load_model(paths.weights_n5k / "deep_weight_efficientnet_final.pth", "efficientnet_b0", 4)
    x_nv4 = np.concatenate([x_nv, np.zeros((len(x_nv), 1, SIZE, SIZE), dtype=np.float32)], axis=1)
    p_depth0 = ensemble(rn4, eff4, x_nv4)
    del rn4, eff4, x_nv4
    torch.cuda.empty_cache()

    median = float(np.median(y_tr))
    blocks = {
        "n5k_test_3ch": point_block(y_te, p_n5k),
        "nv_all_3ch": point_block(y_nv, p_nv),
        "nv_val_3ch": point_block(y_nv[val], p_nv[val]),
        "nv_all_4ch_depth0": point_block(y_nv, p_depth0),
        "nv_all_n5k_median": point_block(y_nv, np.full_like(y_nv, median)),
    }
    for name, block in blocks.items():
        status(
            f"{name} n={block['n']} MAPE={block['mape_pct']:.1f}% "
            f"WAPE={block['wape']:.3f} MAE={block['mae_g']:.1f}g"
        )

    idx = np.arange(len(y_te))
    cal_idx, ev_idx = train_test_split(idx, test_size=0.5, random_state=42)
    frozen = []
    for name, fn in (("Split", split_intervals), ("Normalized", norm_intervals), ("Mondrian", mondrian_intervals)):
        _cl, _cu, lo_in, hi_in = fn(y_te[cal_idx], p_n5k[cal_idx], p_n5k[ev_idx])
        _cl, _cu, lo_nv, hi_nv = fn(y_te[cal_idx], p_n5k[cal_idx], p_nv)
        frozen.append({
            "method": name,
            "in_domain_heldout": interval_report(name + " N5K held-out", y_te[ev_idx], lo_in, hi_in),
            "nv_all": interval_report(name + " NV-Real", y_nv, lo_nv, hi_nv),
            "nv_val": interval_report(name + " NV-Real val", y_nv[val], lo_nv[val], hi_nv[val]),
        })

    joint_dir = paths.root / "backup" / "v3_excluded" / "weights" / "nvreal"
    joint_rn = joint_dir / "joint_resnet18_3ch.pth"
    joint_eff = joint_dir / "joint_efficientnet_3ch.pth"
    train_nv = split == "train"
    fit_models(
        np.concatenate([x_tr, x_nv[train_nv]], axis=0),
        np.concatenate([y_tr, y_nv[train_nv]], axis=0),
        joint_rn,
        joint_eff,
    )
    jrn = load_model(joint_rn, "resnet18", 3)
    jeff = load_model(joint_eff, "efficientnet_b0", 3)
    p_joint_nv = ensemble(jrn, jeff, x_nv)
    p_joint_n5k = ensemble(jrn, jeff, x_te)
    del jrn, jeff
    torch.cuda.empty_cache()
    joint = {
        "nv_val": point_block(y_nv[val], p_joint_nv[val]),
        "n5k_test": point_block(y_te, p_joint_n5k),
    }
    for name, block in joint.items():
        status(
            f"joint {name} n={block['n']} MAPE={block['mape_pct']:.1f}% "
            f"WAPE={block['wape']:.3f} MAE={block['mae_g']:.1f}g"
        )

    p_joint_out = p_joint_nv.astype(np.float32).copy()
    p_joint_out[train_nv] = np.nan
    table = pd.DataFrame({
        "file_name": nv["file_name"],
        "dish_id": nv["dish_id"],
        "split": split,
        "y_gt": y_nv,
        "p_ens": p_nv,
        "p_depth0": p_depth0,
        "p_joint": p_joint_out,
        "calories": nv["nutrients"][:, 0],
        "fat_g": nv["nutrients"][:, 1],
        "carb_g": nv["nutrients"][:, 2],
        "protein_g": nv["nutrients"][:, 3],
    })
    csv_path = derived / "phase2b_per_image.csv"
    table.to_csv(csv_path, index=False)
    summary = {
        "protocol": PROTOCOL,
        "calibrator": "fit on half of Nutrition5K test dishes, seed 42; other half is the in-domain check",
        "joint_train": "Nutrition5K train dishes plus official NV-Real Train dishes; scored on official Val",
        "point": blocks,
        "frozen_intervals": frozen,
        "joint": joint,
        "per_image_csv": str(csv_path),
    }
    dest = out_dir / "summary.json"
    dest.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    (derived / "phase2b_results.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    status(f"D5 complete {dest}")


if __name__ == "__main__":
    main()
