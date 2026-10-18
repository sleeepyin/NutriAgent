"""Build Level-A paired action tables from public datasets. Honest isolation."""
from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from sklearn.model_selection import GroupShuffleSplit

from .conformal.finite_sample import conformal_interval, finite_sample_quantile
from .metrics import mape, point_metrics, wape, winkler
from .paths import Paths
from .runtime import atomic_write_json, log, sha256_obj
from .schemas import PRIOR_COST, default_available
from .tools.morphology import classify_name, classify_nv, p_from_label
from .tools.multiview import fuse_views
from .tools.reference_scale import ReferenceScaleTool, instance_table
from .tools.semantic_density import SemanticDensityTool


SEED = 42
ALPHA = 0.10
POLICY_FRAC = 0.40
CAL_FRAC = 0.35
# remainder = test


def grouped_three_way(groups: np.ndarray, seed: int = SEED) -> np.ndarray:
    """Return split labels: policy / calibration / test. One group -> one role."""
    groups = np.asarray(groups)
    n = len(groups)
    labels = np.array(["policy"] * n, dtype=object)
    uniq = np.unique(groups)
    if len(uniq) < 6:
        rng = np.random.default_rng(seed)
        perm = rng.permutation(len(uniq))
        n_te = max(1, int(round(len(uniq) * (1 - POLICY_FRAC - CAL_FRAC))))
        n_cal = max(1, int(round(len(uniq) * CAL_FRAC)))
        te = set(uniq[perm[:n_te]])
        cal = set(uniq[perm[n_te:n_te + n_cal]])
        for i, g in enumerate(groups):
            if g in te:
                labels[i] = "test"
            elif g in cal:
                labels[i] = "calibration"
        return labels
    rng = np.random.RandomState(seed)
    gss = GroupShuffleSplit(n_splits=1, test_size=1 - POLICY_FRAC, random_state=seed)
    idx = np.arange(n)
    pol, rest = next(gss.split(idx, groups=groups))
    rest_g = groups[rest]
    frac_cal_of_rest = CAL_FRAC / (CAL_FRAC + (1 - POLICY_FRAC - CAL_FRAC))
    gss2 = GroupShuffleSplit(n_splits=1, test_size=1 - frac_cal_of_rest, random_state=seed + 1)
    cal_rel, te_rel = next(gss2.split(rest, groups=rest_g))
    labels[:] = "policy"
    labels[rest[cal_rel]] = "calibration"
    labels[rest[te_rel]] = "test"
    return labels


def _interval_from_cal(y_cal, p_cal, p_eval, scale_cal=None, scale_eval=None):
    if scale_cal is None:
        scores = np.abs(np.asarray(y_cal) - np.asarray(p_cal))
        q = finite_sample_quantile(scores, ALPHA)
        lo, hi = conformal_interval(p_eval, np.ones_like(np.asarray(p_eval), dtype=float), q)
        return lo, hi, q
    scores = np.abs(np.asarray(y_cal) - np.asarray(p_cal)) / np.maximum(np.asarray(scale_cal), 1e-6)
    q = finite_sample_quantile(scores, ALPHA)
    lo, hi = conformal_interval(p_eval, scale_eval, q)
    return lo, hi, q


def build_n5k_depth(paths: Paths, pred_cache: dict) -> list[dict]:
    crops = np.load(paths.file("deep_crops"))
    nutr = np.load(paths.file("nutrient_labels"))
    y = crops["y_test"]
    y_n = nutr["y_test_nutr"]
    n = len(y)
    p3 = pred_cache["n5k_p3"]
    p4 = pred_cache["n5k_p4"]
    p4_d0 = pred_cache.get("n5k_p4_d0", p3)
    dis3 = pred_cache.get("n5k_dis3", np.zeros(n))
    groups = np.arange(n)  # cache has no dish id
    split = grouped_three_way(groups)
    rows = []
    for i in range(n):
        rows.append(
            {
                "episode_id": f"n5k|xtest|{i}",
                "dataset": "n5k",
                "group_id": f"n5k-{i}",
                "split": str(split[i]),
                "y": float(y[i]),
                "y_cal": float(y_n[i, 0]),
                "y_fat": float(y_n[i, 1]),
                "y_carb": float(y_n[i, 2]),
                "y_prot": float(y_n[i, 3]),
                "morphology": "unknown",
                "domain": "n5k",
                "p_rgb": float(p3[i]),
                "p_depth": float(p4[i]),
                "p_depth0": float(p4_d0[i]),
                "disagree": float(dis3[i]),
                "available": default_available("n5k"),
                "base_membership": "X_test_only",
            }
        )
    return rows


def build_nv_view(paths: Paths) -> list[dict]:
    p2 = paths.file("nv_phase2b_csv")
    if p2 is None:
        return []
    df = pd.read_csv(p2)
    df["dish_id"] = df["key"].astype(str).str.extract(r"^(\d+)", expand=False)
    df["fname"] = df["key"].astype(str).str.split("|").str[-1]
    df = df.sort_values(["dish_id", "fname"])
    dishes = df["dish_id"].unique()
    split_map = dict(zip(dishes, grouped_three_way(dishes)))
    rows = []
    for did, sub in df.groupby("dish_id"):
        sub = sub.reset_index(drop=True)
        y = float(sub["y_gt"].iloc[0])
        preds = sub["p_ens"].astype(float).tolist()
        p0 = preds[0]
        p_view = fuse_views(preds[:2])["point"] if len(preds) >= 2 else p0
        p_all = fuse_views(preds)["point"]
        rows.append(
            {
                "episode_id": f"nvreal|{did}",
                "dataset": "nvreal",
                "group_id": f"nv-{did}",
                "split": str(split_map[did]),
                "y": y,
                "morphology": "unknown",
                "domain": "nvreal",
                "p_rgb": float(p0),
                "p_view": float(p_view),
                "p_all_views": float(p_all),
                "n_views": int(len(preds)),
                "disagree": float(np.std(preds)) if len(preds) > 1 else 0.0,
                "available": default_available("nvreal"),
            }
        )
    return rows


def _stats_from_policy(inst_df: pd.DataFrame, policy_ids: set[str]) -> dict:
    pol = inst_df[inst_df["inst"].astype(str).isin(policy_ids)]
    global_mmpp = float(np.median(25.0 / pol["coin_px"].clip(lower=1)))
    cat_stats = {}
    for cat, sub in pol.groupby("cat"):
        area = sub["food_px"] * (25.0 / sub["coin_px"].clip(lower=1)) ** 2
        k = float(np.median(area / sub["vol_ml"].replace(0, np.nan)))
        rho = float(np.median(sub["wt_g"] / sub["vol_ml"].replace(0, np.nan)))
        cat_stats[str(cat)] = {
            "k": k if np.isfinite(k) and k > 0 else float("nan"),
            "rho": rho if np.isfinite(rho) and rho > 0 else float("nan"),
        }
    glob_k = float(np.nanmedian([v["k"] for v in cat_stats.values()]))
    glob_rho = float(np.nanmedian([v["rho"] for v in cat_stats.values()]))
    return {"mmpp": global_mmpp, "cat": cat_stats, "k": glob_k, "rho": glob_rho}


def build_ecustfd(paths: Paths) -> list[dict]:
    p = paths.file("ecustfd_csv")
    if p is None:
        return []
    raw = pd.read_csv(p)
    inst = instance_table(raw)
    groups = inst["inst"].astype(str).to_numpy()
    split = grouped_three_way(groups)
    inst = inst.assign(split=split)
    policy_ids = set(inst.loc[inst["split"] == "policy", "inst"].astype(str))
    stats = _stats_from_policy(inst, policy_ids)
    tool = ReferenceScaleTool()
    rows = []
    for _, r in inst.iterrows():
        cat = str(r["cat"])
        st = stats["cat"].get(cat, {})
        k = st.get("k", stats["k"])
        rho = st.get("rho", stats["rho"])
        if not np.isfinite(k):
            k = stats["k"]
        if not np.isfinite(rho):
            rho = stats["rho"]
        ref = tool.predict_instance(r["food_px"], r["coin_px"], k, rho, use_ref=True)
        noref = tool.predict_instance(r["food_px"], r["coin_px"], k, rho, global_mmpp=stats["mmpp"], use_ref=False)
        p_view = noref.w_hat
        if pd.notna(r.get("food_px_s")) and pd.notna(r.get("coin_px_s")):
            side = tool.predict_instance(r["food_px_s"], r["coin_px_s"], k, rho, use_ref=True)
            if ref.success and side.success:
                p_view = float(np.median([ref.w_hat, side.w_hat]))
        rows.append(
            {
                "episode_id": f"ecustfd|{r['inst']}",
                "dataset": "ecustfd",
                "group_id": f"ec-{r['inst']}",
                "split": str(r["split"]),
                "y": float(r["wt_g"]),
                "morphology": "rigid",
                "domain": "ecustfd",
                "p_rgb": float(noref.w_hat) if noref.success else float("nan"),
                "p_ref": float(ref.w_hat) if ref.success else float("nan"),
                "p_view": float(p_view) if np.isfinite(p_view) else float("nan"),
                "tool_ref_ok": bool(ref.success),
                "available": default_available("ecustfd"),
                "note": "annotation-assisted coin box; deployable detector not used",
            }
        )
    return rows


def build_mf3d(paths: Paths) -> list[dict]:
    p = paths.file("mf3d_labels")
    if p is None:
        return []
    df = pd.read_csv(p)
    groups = df["obj"].astype(str).to_numpy()
    split = grouped_three_way(groups)
    df = df.assign(split=split)
    pol = df[df["split"] == "policy"]
    sem = SemanticDensityTool(pol)
    glob_d = float(pol["density"].median()) if "density" in pol.columns else 0.7
    if "density" not in df.columns:
        df["density"] = df["weight_g"] / df["volume_cm3"].replace(0, np.nan)
        glob_d = float(np.nanmedian(df.loc[df["split"] == "policy", "density"]))
    rows = []
    for _, r in df.iterrows():
        morph = classify_name(str(r["cat"]))
        look = sem.lookup(str(r["cat"]), glob_d)
        # deployable semantic: category prior density * GT volume is still oracle volume.
        # Honest deployable proxy: category mean weight (policy-only).
        cat = str(r["cat"])
        cat_w = float(pol.loc[pol["cat"] == cat, "weight_g"].median()) if (pol["cat"] == cat).any() else float(pol["weight_g"].median())
        # visual-only weak prior
        p_rgb = float(pol["weight_g"].median())
        p_sem = cat_w
        # oracle geometry (NOT deployable)
        d_loo = look.density_prior if look.success else glob_d
        p_oracle_vol = float(r["volume_cm3"]) * d_loo
        rows.append(
            {
                "episode_id": f"mf3d|{r['obj']}",
                "dataset": "mf3d",
                "group_id": f"mf-{r['obj']}",
                "split": str(r["split"]),
                "y": float(r["weight_g"]),
                "y_cal": float(r["energy"]) if pd.notna(r.get("energy")) else None,
                "y_fat": float(r["fat"]) if pd.notna(r.get("fat")) else None,
                "y_carb": float(r["carb"]) if pd.notna(r.get("carb")) else None,
                "y_prot": float(r["prot"]) if pd.notna(r.get("prot")) else None,
                "morphology": morph,
                "domain": "mf3d",
                "p_rgb": p_rgb,
                "p_semantic": p_sem,
                "p_oracle_volume": p_oracle_vol,
                "available": default_available("mf3d"),
                "note": "p_oracle_volume uses GT volume; not a deployable tool",
            }
        )
    return rows


def attach_provisional_intervals(rows: list[dict]) -> list[dict]:
    """Route intervals from policy-internal cal only. Same alpha for all actions."""
    by_ds: dict[str, list[int]] = {}
    for i, r in enumerate(rows):
        by_ds.setdefault(r["dataset"], []).append(i)

    action_pred = {
        "n5k": [("rgb", "p_rgb"), ("depth", "p_depth"), ("depth0", "p_depth0")],
        "nvreal": [("rgb", "p_rgb"), ("view", "p_view"), ("all_views", "p_all_views")],
        "ecustfd": [("rgb", "p_rgb"), ("ref", "p_ref"), ("view", "p_view")],
        "mf3d": [("rgb", "p_rgb"), ("semantic", "p_semantic")],
    }
    for ds, idxs in by_ds.items():
        pol = [i for i in idxs if rows[i]["split"] == "policy"]
        if len(pol) < 8:
            continue
        # split policy into route-fit / route-val
        rng = np.random.default_rng(SEED)
        perm = rng.permutation(pol)
        k = max(4, len(perm) // 2)
        fit, val_plus = perm[:k], perm[k:]
        y_fit = np.array([rows[i]["y"] for i in fit], dtype=float)
        for name, key in action_pred.get(ds, []):
            p_fit = np.array([rows[i].get(key, np.nan) for i in fit], dtype=float)
            ok = np.isfinite(p_fit) & np.isfinite(y_fit)
            if ok.sum() < 4:
                continue
            scores = np.abs(y_fit[ok] - p_fit[ok])
            q = finite_sample_quantile(scores, ALPHA)
            for i in idxs:
                p = rows[i].get(key, np.nan)
                if not np.isfinite(p):
                    continue
                lo, hi = conformal_interval(np.array([p]), np.ones(1), q)
                rows[i][f"lo_{name}"] = float(lo[0])
                rows[i][f"hi_{name}"] = float(hi[0])
                rows[i][f"q_{name}"] = float(q)
    return rows


def compute_gains(rows: list[dict]) -> list[dict]:
    mapping = {
        "n5k": {"REQUEST_DEPTH": ("rgb", "depth")},
        "nvreal": {"REQUEST_VIEW": ("rgb", "view")},
        "ecustfd": {"REQUEST_REFERENCE": ("rgb", "ref"), "REQUEST_VIEW": ("rgb", "view")},
        "mf3d": {"ASK_SEMANTIC": ("rgb", "semantic")},
    }
    for r in rows:
        y = r["y"]
        acts = mapping.get(r["dataset"], {})
        r["gains"] = {}
        r["is_before"] = None
        r["errors"] = {}
        p0 = r.get("p_rgb", np.nan)
        lo0, hi0 = r.get("lo_rgb"), r.get("hi_rgb")
        if np.isfinite(p0):
            r["errors"]["rgb"] = abs(y - p0)
        if lo0 is not None and hi0 is not None:
            r["is_before"] = winkler([y], [lo0], [hi0], ALPHA)
        for act, (before, after) in acts.items():
            pa = r.get(f"p_{after}", np.nan)
            loa, hia = r.get(f"lo_{after}"), r.get(f"hi_{after}")
            if not np.isfinite(pa):
                r["gains"][act] = None
                continue
            r["errors"][after] = abs(y - pa)
            if r["is_before"] is not None and loa is not None:
                isa = winkler([y], [loa], [hia], ALPHA)
                r["gains"][act] = float(r["is_before"] - isa)
            else:
                # fallback: error reduction as proxy, flagged
                eb = r["errors"].get(before, np.nan)
                ea = r["errors"].get(after, np.nan)
                r["gains"][act] = float(eb - ea) if np.isfinite(eb) and np.isfinite(ea) else None
                r.setdefault("gain_proxy", True)
    return rows


def summarize_rows(rows: list[dict]) -> dict:
    out: dict[str, Any] = {"n": len(rows), "by_dataset": {}, "hash": sha256_obj({"n": len(rows), "ids": [r["episode_id"] for r in rows[:20]]})}
    for ds in sorted({r["dataset"] for r in rows}):
        sub = [r for r in rows if r["dataset"] == ds]
        d: dict[str, Any] = {"n": len(sub), "splits": {}}
        for sp in ("policy", "calibration", "test"):
            d["splits"][sp] = int(sum(1 for r in sub if r["split"] == sp))
        out["by_dataset"][ds] = d
    return out
