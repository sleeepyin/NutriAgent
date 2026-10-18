#!/usr/bin/env python3
"""A2a: Qwen3.8-27B category-mass prior, frozen on the policy split.

The model sees only the food name. It does not see the weight label, the
split, or any image. A single shrinkage constant is chosen by policy-split
WAPE. Calibration labels are used only for the terminal quantile. The test
split is scored after that choice is frozen.

The LLM prior replaces the category median only when the policy split says
so. On ECUSTFD the coin reference remains a candidate, because the diagnosis
says scale, not the name, is the missing factor there.
"""
from __future__ import annotations

import argparse
from collections import defaultdict

import numpy as np
import pandas as pd

from .common import add_common_args, out_dir, save_status
from mmcfa.conformal.finite_sample import conformal_interval, finite_sample_quantile
from mmcfa.data_build import build_ecustfd, build_mf3d
from mmcfa.llm.qwen_local import chat_json
from mmcfa.metrics import interval_metrics, mape, wape
from mmcfa.paths import Paths
from mmcfa.runtime import atomic_write_json, log

ALPHA = 0.10
TAUS = (0, 1, 2, 4, 8, 16, 32)
PROMPT = """You estimate one typical weighed instance of a food in a laboratory portion dataset. The name may mean one piece, one bowl, or one loose pile. Use the name only.

Food name: {name}

Return a JSON object with exactly these keys:
- typical_mass_g: positive number in grams
- form: one of rigid, stacked, prepared, liquid
"""


def _jsonable(obj):
    if isinstance(obj, dict):
        return {str(k): _jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_jsonable(v) for v in obj]
    if isinstance(obj, (np.floating, float)):
        value = float(obj)
        return None if not np.isfinite(value) else value
    if isinstance(obj, (np.integer,)):
        return int(obj)
    return obj


def _attach_categories(ec, mf, paths: Paths) -> None:
    raw = pd.read_csv(paths.file("ecustfd_csv"))
    cat = raw.groupby("inst")["cat"].first().to_dict()
    for r in ec:
        r["cat"] = str(cat[r["episode_id"].split("|", 1)[1]])
    df = pd.read_csv(paths.file("mf3d_labels"))
    objcat = dict(zip(df["obj"].astype(str), df["cat"].astype(str)))
    for r in mf:
        r["cat"] = objcat[r["episode_id"].split("|", 1)[1]]


def _load_cache(path) -> dict:
    if path.exists():
        return json_load(path)
    return {}


def json_load(path) -> dict:
    import json
    return json.loads(path.read_text(encoding="utf-8"))


def _query_categories(names: list[str], cache: dict) -> dict:
    for i, name in enumerate(names):
        if name in cache and cache[name].get("typical_mass_g"):
            continue
        last = None
        for _ in range(2):
            try:
                obj = chat_json(PROMPT.format(name=name))
                mass = float(obj["typical_mass_g"])
                form = str(obj.get("form", "")).lower()
                if mass <= 0 or not np.isfinite(mass):
                    raise ValueError("nonpositive mass")
                if form not in {"rigid", "stacked", "prepared", "liquid"}:
                    form = "unknown"
                cache[name] = {"typical_mass_g": mass, "form": form}
                last = None
                break
            except Exception as exc:  # noqa: BLE001 — keep the batch going
                last = str(exc)
        if last is not None:
            cache[name] = {"error": last}
        if (i + 1) % 20 == 0:
            log(f"E14 queried {i+1}/{len(names)}")
    return cache


def _policy_stats(rows: list[dict]) -> tuple[dict, float]:
    buckets: dict[str, list[float]] = defaultdict(list)
    for r in rows:
        if r["split"] == "policy":
            buckets[r["cat"]].append(float(r["y"]))
    med = {c: float(np.median(v)) for c, v in buckets.items()}
    counts = {c: len(v) for c, v in buckets.items()}
    glob = float(np.median([y for ys in buckets.values() for y in ys]))
    return {"median": med, "count": counts, "global": glob, "values": buckets}, glob


def _predict_row(row, stats, cache, kind: str, tau: float = 0.0) -> float:
    cat = row["cat"]
    n = int(stats["count"].get(cat, 0))
    med = stats["median"].get(cat, stats["global"])
    llm = cache.get(cat, {}).get("typical_mass_g")
    llm = float(llm) if llm is not None and np.isfinite(llm) else float("nan")
    if kind == "stop":
        return float(stats["global"])
    if kind == "empirical":
        return float(med if n > 0 else stats["global"])
    if kind == "llm":
        return llm if np.isfinite(llm) else float(stats["global"])
    if kind == "shrink":
        if n <= 0 or not np.isfinite(llm):
            return llm if np.isfinite(llm) else float(stats["global"])
        return float((n * med + tau * llm) / (n + tau))
    if kind == "rare_llm":
        if n >= 3:
            return float(med)
        return llm if np.isfinite(llm) else float(stats["global"])
    if kind == "reference":
        value = row.get("p_ref", np.nan)
        return float(value) if value is not None and np.isfinite(value) else float("nan")
    raise KeyError(kind)


def _loo_stats(row, stats) -> dict:
    """Category median excluding this policy row. Held-out rows keep the full policy median."""
    if row.get("split") != "policy":
        return stats
    cat = row["cat"]
    values = list(stats["values"][cat])
    y = float(row["y"])
    # Remove one matching label so a singleton category cannot score a zero error.
    for i, value in enumerate(values):
        if value == y:
            del values[i]
            break
    med = dict(stats["median"])
    count = dict(stats["count"])
    count[cat] = len(values)
    if values:
        med[cat] = float(np.median(values))
    else:
        med.pop(cat, None)
    return {"median": med, "count": count, "global": stats["global"], "values": stats["values"]}


def _score(rows, stats, cache, kind, tau=0.0, loo: bool = False) -> dict:
    y, p = [], []
    for r in rows:
        use = _loo_stats(r, stats) if loo else stats
        pred = _predict_row(r, use, cache, kind, tau)
        if np.isfinite(pred):
            y.append(r["y"])
            p.append(pred)
    y, p = np.array(y, dtype=float), np.array(p, dtype=float)
    if len(y) == 0:
        return {"n": 0, "mape": float("nan"), "wape": float("nan")}
    return {"n": int(len(y)), "mape": mape(y, p), "wape": wape(y, p), "mae": float(np.mean(np.abs(y - p)))}


def _choose(policy_rows, stats, cache, candidates) -> dict:
    best = None
    table = []
    for kind, tau in candidates:
        sc = _score(policy_rows, stats, cache, kind, tau, loo=True)
        rec = {"kind": kind, "tau": tau, **sc}
        table.append(rec)
        if sc["n"] < 8 or not np.isfinite(sc["wape"]):
            continue
        if best is None or sc["wape"] < best["wape"] - 1e-12:
            best = rec
    return {"chosen": best, "policy_scores": table}


def _coverage(cal_rows, test_rows, stats, cache, kind, tau) -> dict:
    y_cal, p_cal = [], []
    for r in cal_rows:
        pred = _predict_row(r, stats, cache, kind, tau)
        if np.isfinite(pred):
            y_cal.append(r["y"])
            p_cal.append(pred)
    if len(y_cal) < 8:
        return {"q": None}
    q = finite_sample_quantile(np.abs(np.array(y_cal) - np.array(p_cal)), ALPHA)
    y, lo, hi = [], [], []
    for r in test_rows:
        pred = _predict_row(r, stats, cache, kind, tau)
        if not np.isfinite(pred):
            continue
        a, b = conformal_interval(np.array([pred]), np.ones(1), q)
        y.append(r["y"])
        lo.append(float(a[0]))
        hi.append(float(b[0]))
    return {"q": q, "test": interval_metrics(y, lo, hi, ALPHA)}


def _bootstrap_gap(rows, stats, cache, kind_a, tau_a, kind_b, tau_b, n_boot=2000, seed=0) -> dict:
    y = np.array([r["y"] for r in rows], dtype=float)
    a = np.array([_predict_row(r, stats, cache, kind_a, tau_a) for r in rows], dtype=float)
    b = np.array([_predict_row(r, stats, cache, kind_b, tau_b) for r in rows], dtype=float)
    rng = np.random.default_rng(seed)
    gaps = []
    n = len(rows)
    for _ in range(n_boot):
        idx = rng.integers(0, n, n)
        gaps.append(wape(y[idx], b[idx]) - wape(y[idx], a[idx]))
    gaps = np.array(gaps)
    return {
        "mean_wape_b_minus_a": float(np.mean(gaps)),
        "ci95": [float(np.quantile(gaps, 0.025)), float(np.quantile(gaps, 0.975))],
    }


def main(argv=None) -> int:
    ap = add_common_args(argparse.ArgumentParser())
    ap.add_argument("--skip-query", action="store_true")
    args = ap.parse_args(argv)
    paths = Paths.resolve()
    d = out_dir("a2_name_prior", args)
    ec = build_ecustfd(paths)
    mf = build_mf3d(paths)
    _attach_categories(ec, mf, paths)
    names = sorted({r["cat"] for r in ec + mf})
    cache_path = d / "category_cache.json"
    cache = _load_cache(cache_path)
    if not args.skip_query:
        cache = _query_categories(names, cache)
        atomic_write_json(cache_path, _jsonable(cache))
    failed = [k for k, v in cache.items() if not v.get("typical_mass_g")]
    if failed:
        save_status(d, False, error=f"{len(failed)} categories failed", examples=failed[:5])
        return 2

    report = {"n_categories": len(names), "datasets": {}}
    datasets = {"ecustfd": ec, "mf3d": mf}
    menus = {
        "ecustfd": [("reference", 0)] + [("shrink", t) for t in TAUS] + [("llm", 0), ("empirical", 0), ("rare_llm", 0), ("stop", 0)],
        "mf3d": [("shrink", t) for t in TAUS] + [("llm", 0), ("empirical", 0), ("rare_llm", 0), ("stop", 0)],
    }
    for ds, rows in datasets.items():
        stats, _ = _policy_stats(rows)
        policy = [r for r in rows if r["split"] == "policy"]
        cal = [r for r in rows if r["split"] == "calibration"]
        test = [r for r in rows if r["split"] == "test"]
        held = [r for r in rows if r["split"] != "policy"]
        rare = [r for r in held if stats["count"].get(r["cat"], 0) < 3]
        choice = _choose(policy, stats, cache, menus[ds])
        kind = choice["chosen"]["kind"]
        tau = choice["chosen"]["tau"]
        block = {
            "choice": choice["chosen"],
            "policy_menu": choice["policy_scores"],
            "heldout": _score(held, stats, cache, kind, tau),
            "heldout_empirical": _score(held, stats, cache, "empirical"),
            "heldout_llm": _score(held, stats, cache, "llm"),
            "heldout_stop": _score(held, stats, cache, "stop"),
            "test": _score(test, stats, cache, kind, tau),
            "test_empirical": _score(test, stats, cache, "empirical"),
            "rare_heldout_n": len(rare),
            "rare_heldout_chosen": _score(rare, stats, cache, kind, tau) if rare else {},
            "rare_heldout_empirical": _score(rare, stats, cache, "empirical") if rare else {},
            "rare_heldout_llm": _score(rare, stats, cache, "llm") if rare else {},
            "coverage": _coverage(cal, test, stats, cache, kind, tau),
        }
        if rare and kind != "empirical":
            block["rare_gap_empirical_minus_chosen"] = _bootstrap_gap(
                rare, stats, cache, kind, tau, "empirical", 0, seed=1
            )
        if ds == "ecustfd":
            block["heldout_reference"] = _score(held, stats, cache, "reference")
        report["datasets"][ds] = block
        log(
            f"E14 {ds} chose {kind} tau={tau} "
            f"held_wape={block['heldout']['wape']:.3f} "
            f"empirical_wape={block['heldout_empirical']['wape']:.3f}"
        )

    mf = report["datasets"]["mf3d"]
    ec = report["datasets"]["ecustfd"]
    # Success: the frozen rule beats the category median on MetaFood3D held-out
    # WAPE, and it does not give up the ECUSTFD coin result.
    mf_beats = mf["heldout"]["wape"] < mf["heldout_empirical"]["wape"] - 1e-4
    ec_keeps_coin = ec["choice"]["kind"] == "reference" and ec["heldout"]["mape"] <= ec["heldout_empirical"]["mape"]
    report["success"] = {"mf3d_heldout_wape_beats_empirical": mf_beats, "ecustfd_keeps_reference": ec_keeps_coin, "pass": bool(mf_beats and ec_keeps_coin)}
    atomic_write_json(d / "report.json", _jsonable(report))
    save_status(d, report["success"]["pass"], success=report["success"])
    log(f"E14 pass={report['success']['pass']}")
    return 0 if report["success"]["pass"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
