#!/usr/bin/env python3
"""Finish the NV-Real checks that the first rerun did not write.

NV-only training, the split-quantile cost of restoring 90% coverage,
ingredient-level rigid versus prepared error, and frozen nutrient intervals.
Existing dish-level zero-shot and joint-training numbers are left in place
and this script adds the missing blocks to the same summary.
"""
from __future__ import annotations

import importlib.util
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from PIL import Image
from sklearn.model_selection import train_test_split
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler


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
from experiments.estimation.nutrient_intervals import (  # noqa: E402
    CANDS5,
    DensityEffNet,
    DensityResNet,
    NutrientConformalCalibrator,
    WeightEffNet,
    WeightResNet,
    _features,
    _predict,
    get_test_pipeline_cache,
    hybrid_nutrients,
    nutrient_disagreement,
)
from mmcfa.conformal.finite_sample import finite_sample_quantile  # noqa: E402
from mmcfa.metrics import mape  # noqa: E402
from mmcfa.paths import Paths  # noqa: E402
from mmcfa.tools.morphology import classify_nv  # noqa: E402


def _load_d5():
    path = ROOT / "experiments" / "diagnosis" / "d5_nvreal_shift" / "run_d5.py"
    spec = importlib.util.spec_from_file_location("nvreal_run_d5", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


D5 = _load_d5()
NUTR_KEYS = ("calories", "fat_g", "carb_g", "protein_g")
DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def status(msg: str) -> None:
    print(time.strftime("%F %T ") + msg, flush=True)


def clean(obj):
    if isinstance(obj, dict):
        return {str(k): clean(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [clean(v) for v in obj]
    if isinstance(obj, (np.floating, float)):
        value = float(obj)
        return None if not np.isfinite(value) else value
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, np.ndarray):
        return clean(obj.tolist())
    return obj


def write_json(path: Path, obj) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(clean(obj), indent=2) + "\n", encoding="utf-8")


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"), parse_constant=lambda _value: None)


def norm_name(value) -> str:
    return str(value).strip().lower().replace("_", "-").replace(" ", "-")


# COCO names and the dish metadata disagree on these labels.
ALIASES = {
    "costco-california-sushi-roll": "costco-california-sushi-roll-1",
    "costco-cucumber-sushi-roll": "costco-cucumber-sushi-roll-1",
    "costco-salad-sushi-roll": "costco-salad-sushi-roll-1",
    "costco-shrimp-sushi-roll": "costco-shrimp-sushi-roll-1",
    "tuna-rice-ball": "costco-tuna-rice-ball",
    "full-bread-loaf": "half-bread-loaf",
}


def resolve_name(key: str, weights: dict) -> str | None:
    if key in weights:
        return key
    alt = ALIASES.get(key)
    if alt in weights:
        return alt
    return None


def nutrient_block(y, p, lo=None, hi=None) -> dict:
    out = {}
    y = np.asarray(y, dtype=float)
    p = np.asarray(p, dtype=float)
    for i, name in enumerate(NUTR_KEYS):
        mask = np.isfinite(y[:, i]) & np.isfinite(p[:, i]) & (y[:, i] >= 1.0)
        row = {"n_ge_1": int(mask.sum())}
        row["mape_pct"] = float(100.0 * mape(y[mask, i], p[mask, i])) if mask.any() else None
        if lo is not None:
            valid = np.isfinite(y[:, i]) & np.isfinite(lo[:, i]) & np.isfinite(hi[:, i])
            covered = valid & (y[:, i] >= lo[:, i]) & (y[:, i] <= hi[:, i])
            row["n"] = int(valid.sum())
            row["coverage"] = float(covered.sum() / valid.sum()) if valid.any() else None
        out[name] = row
    return out


def ingredient_table(paths: Paths) -> pd.DataFrame:
    image_dir = paths.data_nvreal / "raw" / "nutritionverse-manual" / "nutritionverse-manual" / "images"
    coco = json.loads((image_dir / "_annotations.coco.json").read_text(encoding="utf-8"))
    meta = pd.read_csv(paths.data_nvreal / "raw" / "nutritionverse_dish_metadata3.csv")
    meta = meta.set_index(meta["dish_id"].astype(int))
    names = {int(cat["id"]): norm_name(cat["name"]) for cat in coco["categories"]}
    type_cols = [col for col in meta.columns if str(col).startswith("food_item_type_")]
    by_image = {}
    for ann in coco["annotations"]:
        by_image.setdefault(ann["image_id"], []).append(ann)

    crops, labels, kinds, files, dishes = [], [], [], [], []
    skipped = {"no_meta": 0, "no_weight": 0, "empty": 0, "no_file": 0}
    t0 = time.time()
    for i, image in enumerate(coco["images"]):
        path = image_dir / image["file_name"]
        dish = int(str(image["file_name"]).split("_")[1])
        if not path.is_file():
            skipped["no_file"] += 1
            continue
        if dish not in meta.index:
            skipped["no_meta"] += 1
            continue
        row = meta.loc[dish]
        if isinstance(row, pd.DataFrame):
            row = row.iloc[0]
        weights = {}
        for col in type_cols:
            raw = row[col]
            if pd.isna(raw) or not str(raw).strip():
                continue
            weight_col = str(col).replace("food_item_type_", "food_weight_g_")
            weight = row[weight_col]
            if pd.isna(weight):
                continue
            key = norm_name(raw)
            weights[key] = weights.get(key, 0.0) + float(weight)
        grouped = {}
        for ann in by_image.get(image["id"], []):
            key = names.get(int(ann["category_id"]))
            if not key:
                continue
            grouped.setdefault(key, []).extend(ann.get("segmentation") or [])
        rgb = Image.open(path).convert("RGB")
        for key, segments in grouped.items():
            resolved = resolve_name(key, weights)
            if resolved is None or weights[resolved] <= 0:
                skipped["no_weight"] += 1
                continue
            key = resolved
            mask = D5.draw_union(rgb.width, rgb.height, segments)
            ys_px, xs_px = np.nonzero(mask)
            if xs_px.size == 0:
                skipped["empty"] += 1
                continue
            box = (int(xs_px.min()), int(ys_px.min()), int(xs_px.max()) + 1, int(ys_px.max()) + 1)
            crop = np.asarray(rgb.crop(box), dtype=np.float32)
            crop_mask = mask[box[1]:box[3], box[0]:box[2]] > 0
            crop[~crop_mask] = 0.0
            resized = Image.fromarray(crop.astype(np.uint8), mode="RGB").resize((D5.SIZE, D5.SIZE), Image.BILINEAR)
            arr = np.transpose(np.asarray(resized, dtype=np.float32) / 255.0, (2, 0, 1))
            crops.append(arr)
            labels.append(weights[key])
            kinds.append(classify_nv(key))
            files.append(image["file_name"])
            dishes.append(dish)
        if (i + 1) % 100 == 0:
            status(f"ingredient crops {i + 1}/{len(coco['images'])} kept={len(crops)} elapsed={time.time() - t0:.0f}s")
    status(f"ingredient crops kept={len(crops)} skipped={skipped}")
    return pd.DataFrame({
        "file_name": files,
        "dish_id": dishes,
        "morphology": kinds,
        "y_g": np.asarray(labels, dtype=np.float32),
        "x_index": np.arange(len(crops)),
    }), np.stack(crops).astype(np.float32) if crops else np.zeros((0, 3, D5.SIZE, D5.SIZE), np.float32), skipped


def main() -> None:
    if DEVICE.type != "cuda":
        raise SystemExit("D5 remaining 需要 GPU")
    paths = Paths.resolve()
    derived = paths.data_nvreal / "derived"
    out_dir = paths.results_diagnosis / "d5_nvreal"
    status(f"D5 remaining start device={DEVICE}")
    nv = D5.build_cache(paths)
    x_nv = nv["X"]
    y_nv = nv["y"].astype(np.float32)
    split = np.asarray(nv["split"]).astype(str)
    nutrients = np.asarray(nv["nutrients"], dtype=np.float32)
    n5k = np.load(paths.n5k_cache / "deep_crops_cache.npz")
    x_te = n5k["X_test"][:, :3]
    y_te = n5k["y_test"].astype(np.float32)
    x_tr = n5k["X_train"]

    rn3 = D5.load_model(paths.weights_nvreal / "deep_weight_resnet18_3ch.pth", "resnet18", 3)
    eff3 = D5.load_model(paths.weights_nvreal / "deep_weight_efficientnet_3ch.pth", "efficientnet_b0", 3)
    p_n5k = D5.ensemble(rn3, eff3, x_te)
    p_nv = D5.ensemble(rn3, eff3, x_nv)

    cal_idx, _ev_idx = train_test_split(np.arange(len(y_te)), test_size=0.5, random_state=42)
    q_in = float(finite_sample_quantile(np.abs(y_te[cal_idx] - p_n5k[cal_idx]), 0.10))
    q_nv = float(finite_sample_quantile(np.abs(y_nv - p_nv), 0.10))
    phase3 = {
        "protocol": "order-statistic absolute residual; in-domain q fit on half of Nutrition5K test, seed 42",
        "split_q_n5k_g": q_in,
        "split_q_nv_to_cover_90_g": q_nv,
        "inflation": q_nv / q_in if q_in else None,
        "note": "The NV quantile is the width that would cover 90% of these NV residuals. It uses target labels and is a cost, not a deployed calibrator.",
    }
    status(f"split q in-domain={q_in:.1f}g nv-90%={q_nv:.1f}g inflation={phase3['inflation']:.2f}")
    write_json(derived / "phase3_conformal_results.json", phase3)

    train_nv = split == "train"
    val = split == "val"
    only_dir = paths.root / "backup" / "v3_excluded" / "weights" / "nvreal"
    only_rn = only_dir / "nvonly_resnet18_3ch.pth"
    only_eff = only_dir / "nvonly_efficientnet_3ch.pth"
    D5.fit_models(x_nv[train_nv], y_nv[train_nv], only_rn, only_eff)
    orn = D5.load_model(only_rn, "resnet18", 3)
    oeff = D5.load_model(only_eff, "efficientnet_b0", 3)
    nv_only = {
        "nv_val": D5.point_block(y_nv[val], D5.ensemble(orn, oeff, x_nv[val])),
        "n5k_test": D5.point_block(y_te, D5.ensemble(orn, oeff, x_te)),
    }
    for name, block in nv_only.items():
        status(f"nv-only {name} n={block['n']} MAPE={block['mape_pct']:.1f}% MAE={block['mae_g']:.1f}g")
    del orn, oeff
    torch.cuda.empty_cache()

    cache = get_test_pipeline_cache()
    n_hybrid = hybrid_nutrients(cache)
    n_disag = nutrient_disagreement(cache)
    y_nutr = cache["y_nutr"]
    w_pred = cache["w_pred"]
    y_w = cache["y_w"]
    fit_idx, held_idx = train_test_split(np.arange(len(y_w)), test_size=0.5, random_state=42)
    calibrator = NutrientConformalCalibrator(cands=CANDS5, strat="weight").fit(
        w_pred[fit_idx], y_w[fit_idx], n_hybrid[fit_idx], y_nutr[fit_idx],
        n_disag=n_disag[fit_idx], seed=42,
    )
    lo_in, hi_in = calibrator.predict_intervals(w_pred[held_idx], n_hybrid[held_idx], n_disag[held_idx], "m")

    x_nv4 = np.concatenate([x_nv, np.zeros((len(x_nv), 1, D5.SIZE, D5.SIZE), np.float32)], axis=1)
    d_rn = DensityResNet().to(DEVICE)
    d_rn.load_state_dict(torch.load(paths.weights_n5k / "density_resnet18_final.pth", map_location=DEVICE, weights_only=True))
    d_eff = DensityEffNet().to(DEVICE)
    d_eff.load_state_dict(torch.load(paths.weights_n5k / "density_efficientnet_final.pth", map_location=DEVICE, weights_only=True))
    d_direct = np.maximum(np.expm1((_predict(d_rn, x_nv4) + _predict(d_eff, x_nv4)) / 2.0), 0)
    del d_rn, d_eff
    w_rn = WeightResNet().to(DEVICE)
    w_rn.load_state_dict(torch.load(paths.weights_n5k / "deep_weight_resnet18_final.pth", map_location=DEVICE, weights_only=True))
    w_eff = WeightEffNet().to(DEVICE)
    w_eff.load_state_dict(torch.load(paths.weights_n5k / "deep_weight_efficientnet_final.pth", map_location=DEVICE, weights_only=True))
    feat_tr = np.concatenate([_features(w_rn, x_tr), _features(w_eff, x_tr)], axis=1)
    feat_nv = np.concatenate([_features(w_rn, x_nv4), _features(w_eff, x_nv4)], axis=1)
    del w_rn, w_eff, x_nv4
    torch.cuda.empty_cache()
    train_density = np.load(paths.n5k_cache / "nutrient_density.npz")["y_train_density"]
    scaler = StandardScaler().fit(feat_tr)
    neighbors = NearestNeighbors(n_neighbors=5, metric="cosine").fit(scaler.transform(feat_tr))
    _dist, ind = neighbors.kneighbors(scaler.transform(feat_nv))
    d_knn = np.stack([train_density[ind[i]].mean(axis=0) for i in range(len(feat_nv))])
    d_hybrid = np.maximum(0.60 * d_direct + 0.40 * d_knn, 0)
    n_nv = np.maximum(p_nv, 0).reshape(-1, 1) * d_hybrid
    n_nv_direct = np.maximum(p_nv, 0).reshape(-1, 1) * d_direct
    disag_nv = np.abs(n_nv_direct - np.maximum(p_nv, 0).reshape(-1, 1) * d_knn)
    lo_nv, hi_nv = calibrator.predict_intervals(p_nv, n_nv, disag_nv, "m")
    phase4 = {
        "protocol": (
            "3-channel dish weight times hybrid density. Density direct is the 4-channel "
            "ensemble with NV depth set to 0; kNN uses those depth-zero features against "
            "Nutrition5K train features. AEC-S2m-5W is fit on half of the Nutrition5K test "
            "pipeline, seed 42, and applied unchanged. MAPE uses labels of at least 1."
        ),
        "in_domain_heldout": nutrient_block(y_nutr[held_idx], n_hybrid[held_idx], lo_in, hi_in),
        "nv_all": nutrient_block(nutrients, n_nv, lo_nv, hi_nv),
        "nv_val": nutrient_block(nutrients[val], n_nv[val], lo_nv[val], hi_nv[val]),
        "nv_all_direct_density": nutrient_block(nutrients, n_nv_direct),
    }
    for name in NUTR_KEYS:
        block = phase4["nv_all"][name]
        status(f"nutrient {name} NV MAPE={block['mape_pct']:.1f}% coverage={100 * (block['coverage'] or 0):.1f}%")
    write_json(derived / "phase4_nutrient_results.json", phase4)
    del x_tr
    torch.cuda.empty_cache()

    frame, crops, skipped = ingredient_table(paths)
    if len(frame):
        pred = D5.ensemble(rn3, eff3, crops)
        frame["p_g"] = pred
        groups = {}
        for kind, part in frame.groupby("morphology"):
            groups[str(kind)] = {
                "n": int(len(part)),
                "mape_pct": float(100.0 * mape(part["y_g"].to_numpy(), part["p_g"].to_numpy())),
            }
            status(f"ingredient {kind} n={groups[str(kind)]['n']} MAPE={groups[str(kind)]['mape_pct']:.1f}%")
        frame.drop(columns=["x_index"]).to_csv(derived / "ingredient_morphology.csv", index=False)
    else:
        groups = {}
    morphology = {
        "level": "ingredient crop, dish-trained 3-channel ensemble",
        "skipped": skipped,
        "groups": groups,
    }
    del rn3, eff3, crops
    torch.cuda.empty_cache()

    summary_path = out_dir / "summary.json"
    summary = read_json(summary_path) if summary_path.is_file() else {}
    summary["quantile_inflation"] = phase3
    summary["nv_only"] = nv_only
    summary["ingredient_morphology"] = morphology
    summary["frozen_nutrients"] = phase4
    write_json(summary_path, summary)
    write_json(derived / "phase2b_results.json", summary)
    status(f"D5 remaining complete {summary_path}")


if __name__ == "__main__":
    main()
