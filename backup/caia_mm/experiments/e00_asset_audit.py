#!/usr/bin/env python3
"""E0: asset + protocol audit. Gate: missing critical assets stop main experiments."""
from __future__ import annotations

import argparse
import hashlib
import sys
from pathlib import Path

import numpy as np

from .common import ROOT, add_common_args, out_dir, save_status

from agent_extension.src.paths import Paths
from agent_extension.src.runtime import log, sha256_file


CRITICAL = [
    "deep_crops",
    "aec_cache",
    "nutrient_labels",
    "w_rn4",
    "w_eff4",
    "w_rn3",
    "w_eff3",
]
IMPORTANT = [
    "nv_dish_cache",
    "nv_phase2b_csv",
    "nv_phase3_json",
    "ecustfd_csv",
    "mf3d_labels",
    "sha256sum",
]


def parse_sha256sum(path: Path) -> dict[str, str]:
    out = {}
    if not path.exists():
        return out
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        parts = line.strip().split()
        if len(parts) >= 2:
            out[parts[-1].lstrip("./")] = parts[0]
    return out


def main(argv=None) -> int:
    ap = add_common_args(argparse.ArgumentParser())
    args = ap.parse_args(argv)
    paths = Paths.resolve()
    d = out_dir("e00_asset_audit", args)
    report = {
        "root": str(paths.root),
        "nutri": str(paths.nutri),
        "models": str(paths.models),
        "nvreal": str(paths.nvreal),
        "ecustfd": str(paths.ecustfd),
        "metafood3d": str(paths.metafood3d),
        "n5k_raw": str(paths.n5k_raw) if paths.n5k_raw else None,
        "files": {},
        "checksums": {},
        "cache_keys": {},
        "gates": {},
        "notes": [],
    }
    missing_crit = []
    for key in CRITICAL + IMPORTANT:
        p = paths.file(key) if key != "sha256sum" else (paths.extras.get("sha256sum") if Path(paths.extras.get("sha256sum") or "").exists() else None)
        if key == "sha256sum":
            p = Path(paths.extras["sha256sum"]) if paths.extras.get("sha256sum") and Path(paths.extras["sha256sum"]).exists() else None
        exists = p is not None and Path(p).exists()
        rec = {"path": str(p) if p else None, "exists": exists}
        if exists:
            rec["bytes"] = Path(p).stat().st_size
            rec["sha256"] = sha256_file(p)
        report["files"][key] = rec
        if key in CRITICAL and not exists:
            missing_crit.append(key)

    expected = parse_sha256sum(Path(paths.extras["sha256sum"])) if paths.extras.get("sha256sum") else {}
    mismatches = []
    if expected:
        for key, rec in report["files"].items():
            if not rec.get("exists"):
                continue
            rel = str(Path(rec["path"]).relative_to(paths.root)) if rec["path"] and str(rec["path"]).startswith(str(paths.root)) else None
            # try several key forms
            cands = []
            if rel:
                cands += [rel, "./" + rel, Path(rel).as_posix()]
            cands.append(Path(rec["path"]).name)
            exp = None
            for c in cands:
                if c in expected:
                    exp = expected[c]
                    break
                for ek, ev in expected.items():
                    if ek.endswith(Path(rec["path"]).name) and Path(rec["path"]).name in (
                        "deep_crops_cache.npz",
                        "deep_weight_resnet18_final.pth",
                        "deep_weight_efficientnet_final.pth",
                        "aec_v2_pred_cache.npz",
                    ):
                        exp = ev
                        break
                if exp:
                    break
            rec["expected_sha256"] = exp
            rec["checksum_ok"] = (exp is None) or (exp == rec.get("sha256"))
            if exp and exp != rec.get("sha256"):
                mismatches.append(key)
    report["checksum_mismatches"] = mismatches

    # cache inspection
    for key in ("deep_crops", "aec_cache", "nutrient_labels", "nv_dish_cache"):
        p = paths.file(key)
        if not p:
            continue
        z = np.load(p, allow_pickle=True)
        report["cache_keys"][key] = {k: {"shape": list(np.shape(z[k])), "dtype": str(getattr(z[k], "dtype", ""))} for k in z.files}

    crops = paths.file("deep_crops")
    if crops:
        z = np.load(crops)
        report["n5k"] = {
            "n_train": int(len(z["y_train"])),
            "n_test": int(len(z["y_test"])),
            "has_dish_id": "dish_id" in z.files or "dish_ids" in z.files,
            "y_train_unique": int(len(np.unique(z["y_train"]))),
            "y_test_unique": int(len(np.unique(z["y_test"]))),
        }
        if not report["n5k"]["has_dish_id"]:
            report["notes"].append(
                "N5K cache has arrays only. Cannot prove plate/scan isolation from cache. Agent test uses X_test only (Protocol A)."
            )

    # historical PMAE naming
    report["metric_glossary"] = {
        "MAPE": "mean(|e_i|/y_i)",
        "WAPE": "sum(|e|)/sum(y) = MAE/mean(y)",
        "PMAE_forbidden": "do not use; historical 9.44% is MAPE on contaminated resplit; 21.0% WAPE / 31.2% MAPE on 652 hold-out",
        "holdout_652_wape_paper": 0.210,
        "holdout_652_mape_paper": 0.312,
        "resplit_944_is_contaminated": True,
    }

    report["gates"]["critical_assets"] = len(missing_crit) == 0
    report["gates"]["checksum"] = len(mismatches) == 0
    report["missing_critical"] = missing_crit
    ok = len(missing_crit) == 0
    if mismatches:
        report["notes"].append("Checksum mismatches recorded; continue with warning, do not hide.")
    save_status(d, ok, report=report, missing_critical=missing_crit)
    (d / "audit.json").write_text(__import__("json").dumps(report, indent=2, default=str), encoding="utf-8")
    log(f"E00 missing_critical={missing_crit} mismatches={mismatches}")
    return 0 if ok else 2


if __name__ == "__main__":
    raise SystemExit(main())
