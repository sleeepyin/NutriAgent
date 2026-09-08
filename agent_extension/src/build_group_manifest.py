"""Rebuild plate/scan/dish/object manifests. Cache-only N5K is flagged, not claimed leak-free."""
from __future__ import annotations

import csv
from pathlib import Path

import numpy as np
import pandas as pd

from .paths import Paths
from .runtime import atomic_write_json, log, sha256_file


def _load_n5k_meta(paths: Paths) -> pd.DataFrame | None:
    frames = []
    for key in ("n5k_meta1", "n5k_meta2"):
        p = paths.extras.get(key)
        if p and Path(p).exists():
            rows = []
            with open(p, newline="") as f:
                for line in f:
                    parts = line.strip().split(",")
                    if len(parts) < 6:
                        continue
                    try:
                        rows.append(
                            {
                                "dish_id": parts[0],
                                "total_mass": float(parts[1]),
                                "calories": float(parts[2]),
                                "fat": float(parts[3]),
                                "carb": float(parts[4]),
                                "protein": float(parts[5]),
                            }
                        )
                    except ValueError:
                        continue
            if rows:
                frames.append(pd.DataFrame(rows))
    if not frames:
        return None
    return pd.concat(frames, ignore_index=True).drop_duplicates("dish_id")


def match_n5k_weights(y: np.ndarray, meta: pd.DataFrame, atol: float = 0.05) -> list[dict]:
    """Best-effort weight match. Collisions are marked unmatched_unique."""
    by_w: dict[float, list[str]] = {}
    for _, r in meta.iterrows():
        by_w.setdefault(round(float(r["total_mass"]), 3), []).append(str(r["dish_id"]))
    out = []
    for i, w in enumerate(y.tolist()):
        key = round(float(w), 3)
        cands = by_w.get(key, [])
        if len(cands) == 1:
            out.append({"idx": i, "dish_id": cands[0], "match": "unique_weight"})
        elif len(cands) > 1:
            out.append({"idx": i, "dish_id": None, "match": "weight_collision", "n_cands": len(cands)})
        else:
            # nearest
            diffs = (meta["total_mass"] - float(w)).abs()
            j = int(diffs.idxmin())
            if float(diffs.loc[j]) <= atol:
                out.append({"idx": i, "dish_id": str(meta.loc[j, "dish_id"]), "match": "nearest_atol"})
            else:
                out.append({"idx": i, "dish_id": None, "match": "unmatched"})
    return out


def build_manifests(paths: Paths | None = None) -> dict:
    paths = paths or Paths.resolve()
    out_dir = paths.results / "manifests"
    out_dir.mkdir(parents=True, exist_ok=True)
    report: dict = {"ok": True, "notes": []}

    crops = paths.file("deep_crops")
    n5k = {
        "source": str(crops) if crops else None,
        "has_dish_id_in_cache": False,
        "n_train": 0,
        "n_test": 0,
        "protocol": "cache_index_plus_optional_weight_match",
        "leak_proof": False,
    }
    if crops:
        z = np.load(crops)
        n5k["n_train"] = int(len(z["y_train"]))
        n5k["n_test"] = int(len(z["y_test"]))
        n5k["cache_sha256"] = sha256_file(crops)
        meta = _load_n5k_meta(paths)
        if meta is not None:
            m_tr = match_n5k_weights(z["y_train"], meta)
            m_te = match_n5k_weights(z["y_test"], meta)
            n_ok = sum(1 for r in m_te if r["match"] == "unique_weight")
            n5k["test_unique_weight_matches"] = n_ok
            n5k["test_match_rate"] = n_ok / max(len(m_te), 1)
            pd.DataFrame(m_tr).to_csv(out_dir / "n5k_train_weight_match.csv", index=False)
            pd.DataFrame(m_te).to_csv(out_dir / "n5k_test_weight_match.csv", index=False)
            report["notes"].append(
                "N5K cache has no dish/plate/scan id. Weight match is not a leak-proof group key."
            )
        else:
            report["notes"].append("N5K raw metadata missing; using cache index as episode id.")
        # membership: all train indices are D_base
        mem = pd.DataFrame(
            {
                "role": ["base"] * n5k["n_train"] + ["holdout_pool"] * n5k["n_test"],
                "idx": list(range(n5k["n_train"])) + list(range(n5k["n_test"])),
                "split_origin": ["X_train"] * n5k["n_train"] + ["X_test"] * n5k["n_test"],
            }
        )
        mem.to_csv(out_dir / "n5k_base_membership.csv", index=False)
    else:
        report["ok"] = False
        report["notes"].append("deep_crops_cache.npz missing")

    nv = {"n_dishes_cache": 0, "n_images_phase2b": 0, "group": "dish_id"}
    nv_c = paths.file("nv_dish_cache")
    if nv_c:
        z = np.load(nv_c, allow_pickle=True)
        dishes = np.concatenate([z["dish_train"], z["dish_test"]])
        nv["n_dishes_cache"] = int(len(np.unique(dishes)))
        nv["n_train"] = int(len(z["y_train"]))
        nv["n_test"] = int(len(z["y_test"]))
        pd.DataFrame(
            {
                "dish_id": dishes,
                "split": ["cache_train"] * len(z["dish_train"]) + ["cache_test"] * len(z["dish_test"]),
                "y": np.concatenate([z["y_train"], z["y_test"]]),
            }
        ).to_csv(out_dir / "nvreal_cache_manifest.csv", index=False)
    p2 = paths.file("nv_phase2b_csv")
    if p2:
        df = pd.read_csv(p2)
        dish = df["key"].astype(str).str.extract(r"^(\d+)", expand=False)
        nv["n_images_phase2b"] = int(len(df))
        nv["n_dishes_phase2b"] = int(dish.nunique())
        nv["imgs_per_dish_median"] = float(dish.value_counts().median())
        df.assign(dish_id=dish).to_csv(out_dir / "nvreal_phase2b_manifest.csv", index=False)

    mf = {"n_objects": 0, "group": "object_id"}
    mlp = paths.file("mf3d_labels")
    if mlp:
        df = pd.read_csv(mlp)
        mf["n_objects"] = int(df["obj"].nunique())
        mf["n_cats"] = int(df["cat"].nunique())
        df[["cat", "obj", "weight_g"]].to_csv(out_dir / "mf3d_object_manifest.csv", index=False)

    ec = {"n_instances": 0, "group": "food_instance_id"}
    ecp = paths.file("ecustfd_csv")
    if ecp:
        df = pd.read_csv(ecp)
        ec["n_instances"] = int(df["inst"].nunique())
        ec["n_images"] = int(len(df))
        df.groupby("inst").agg(cat=("cat", "first"), n=("view", "size"), wt=("wt_g", "first")).reset_index().to_csv(
            out_dir / "ecustfd_instance_manifest.csv", index=False
        )

    report.update({"n5k": n5k, "nvreal": nv, "mf3d": mf, "ecustfd": ec, "out_dir": str(out_dir)})
    atomic_write_json(out_dir / "manifest_report.json", report)
    log(f"manifests written -> {out_dir}")
    return report
