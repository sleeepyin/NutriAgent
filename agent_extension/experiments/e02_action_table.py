#!/usr/bin/env python3
"""E2: counterfactual action table on public paired tools. Policy rows only for gains."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from .common import add_common_args, np_save, out_dir, save_status
from agent_extension.src.data_build import (
    attach_provisional_intervals,
    build_ecustfd,
    build_mf3d,
    build_n5k_depth,
    build_nv_view,
    compute_gains,
    summarize_rows,
)
from agent_extension.src.episode_store import EpisodeStore
from agent_extension.src.models import WeightEffNet4, WeightNet3ch, WeightResNet4, load_state, predict_np
from agent_extension.src.paths import Paths
from agent_extension.src.runtime import log


def _infer_or_cache(paths: Paths, d: Path) -> dict:
    cache_p = d / "n5k_pred_cache.npz"
    if cache_p.exists():
        z = np.load(cache_p)
        log("loaded n5k_pred_cache.npz")
        return {k: z[k] for k in z.files}
    crops = np.load(paths.file("deep_crops"))
    Xte = crops["X_test"]
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    log(f"infer 3ch/4ch on X_test n={len(Xte)} device={device}")
    rn3 = load_state(WeightNet3ch("resnet18"), paths.file("w_rn3"), device)
    ef3 = load_state(WeightNet3ch("efficientnet_b0"), paths.file("w_eff3"), device)
    p3r = predict_np(rn3, Xte[:, :3], device)
    p3e = predict_np(ef3, Xte[:, :3], device)
    p3 = 0.5 * (p3r + p3e)
    # 4ch from AEC cache (already computed on concat train+test)
    aec = np.load(paths.file("aec_cache"))
    n_tr = len(crops["y_train"])
    p4 = 0.5 * (aec["final_rn"][n_tr:] + aec["final_eff"][n_tr:])
    # same-weight depth-zero ablation
    rn4 = load_state(WeightResNet4(), paths.file("w_rn4"), device)
    ef4 = load_state(WeightEffNet4(), paths.file("w_eff4"), device)
    X0 = np.array(Xte, copy=True)
    X0[:, 3] = 0
    p40 = 0.5 * (predict_np(rn4, X0, device) + predict_np(ef4, X0, device))
    out = {
        "n5k_p3": p3.astype(np.float32),
        "n5k_p4": p4.astype(np.float32),
        "n5k_p4_d0": p40.astype(np.float32),
        "n5k_dis3": np.abs(p3r - p3e).astype(np.float32),
        "n5k_y": crops["y_test"].astype(np.float32),
    }
    np_save(cache_p, **out)
    return out


def main(argv=None) -> int:
    ap = add_common_args(argparse.ArgumentParser())
    args = ap.parse_args(argv)
    paths = Paths.resolve()
    d = out_dir("e02_action_table", args)
    pred = _infer_or_cache(paths, d)
    rows = []
    rows += build_n5k_depth(paths, pred)
    rows += build_nv_view(paths)
    rows += build_ecustfd(paths)
    rows += build_mf3d(paths)
    rows = attach_provisional_intervals(rows)
    rows = compute_gains(rows)
    store = EpisodeStore(d / "episodes.jsonl")
    store.write_all(rows)
    summary = summarize_rows(rows)
    # also a flat CSV of key fields
    import pandas as pd

    flat = []
    for r in rows:
        flat.append(
            {
                "episode_id": r["episode_id"],
                "dataset": r["dataset"],
                "split": r["split"],
                "y": r["y"],
                "morphology": r.get("morphology"),
                "p_rgb": r.get("p_rgb"),
                "p_depth": r.get("p_depth"),
                "p_view": r.get("p_view"),
                "p_ref": r.get("p_ref"),
                "p_semantic": r.get("p_semantic"),
                "gain_depth": (r.get("gains") or {}).get("REQUEST_DEPTH"),
                "gain_view": (r.get("gains") or {}).get("REQUEST_VIEW"),
                "gain_ref": (r.get("gains") or {}).get("REQUEST_REFERENCE"),
                "gain_sem": (r.get("gains") or {}).get("ASK_SEMANTIC"),
            }
        )
    pd.DataFrame(flat).to_csv(d / "episodes.csv", index=False)
    save_status(d, True, summary=summary, n=len(rows))
    log(f"E02 wrote {len(rows)} episodes: {summary['by_dataset']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
