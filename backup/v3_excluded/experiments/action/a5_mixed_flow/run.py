"""A5: fixed diagnosis-to-action routing on a mixed hold-out stream.

The rule is the chapter-5 if-else. It is not fit on the policy split.
The policy split only chooses which always-on action is the single-action baseline.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
for entry in (str(ROOT), str(ROOT / "src")):
    if entry not in sys.path:
        sys.path.insert(0, entry)

import numpy as np

from experiments.vlm.geometry import load_coin_table
from experiments.vlm.prior import NamePrior
from experiments.vlm.scoring import block, clean, wape_gap
from experiments.vlm.store import load_record, read_manifest
from mmcfa.conformal.finite_sample import conformal_interval, finite_sample_quantile
from mmcfa.data_build import grouped_three_way
from mmcfa.metrics import interval_metrics
from mmcfa.paths import Paths
from mmcfa.runtime import atomic_write_json, log, read_json

ALPHA = 0.10


def _coin_px(record: dict):
    best = None
    for ref in (record or {}).get("references") or []:
        if "coin" not in str(ref.get("object", "")):
            continue
        box = ref.get("bbox") or []
        if len(box) != 4 or float(ref.get("conf") or 0) < 0.3:
            continue
        diam = max(float(box[2]) - float(box[0]), float(box[3]) - float(box[1]))
        if diam <= 1:
            continue
        if best is None or float(ref["conf"]) > best[0]:
            best = (float(ref["conf"]), diam)
    return None if best is None else best[1]


def _finite(value) -> bool:
    try:
        return bool(np.isfinite(float(value)))
    except (TypeError, ValueError):
        return False


def _best(y, candidates: list[float]) -> float:
    usable = [float(v) for v in candidates if _finite(v)]
    if not usable:
        return float("nan")
    return min(usable, key=lambda pred: abs(float(y) - pred))


def _build(paths: Paths) -> tuple[list[dict], float]:
    manifest = [row for row in read_manifest() if row.get("image")]
    points = {}
    for line in (paths.root / "results" / "vlm" / "points.jsonl").read_text(encoding="utf-8").splitlines():
        if line.strip():
            row = json.loads(line)
            points[(row["dataset"], row["sample_id"])] = row
    mark = read_json(paths.root / "results" / "vlm" / "points_complete.json", {})
    median = float(mark.get("n5k_train_median_g"))
    cache = read_json(paths.results_action / "a2_name_prior" / "category_cache.json", {})
    prior = NamePrior(paths, cache)
    coins = load_coin_table(paths)
    samples = []
    for row in manifest:
        point = points.get((row["dataset"], row["sample_id"]))
        record = load_record(row["dataset"], row["sample_id"])
        if point is None:
            continue
        dataset = row["dataset"]
        has_depth = int(row.get("has_depth") or 0) == 1
        mono = float(point["p_ens4"]) if has_depth else float(point["p_ens3"])
        if dataset in {"ecustfd", "mf3d"}:
            name_ann = prior.predict_annotated(dataset, row.get("cat") or "")
            name_vlm, matched, kind = prior.predict_vlm_name(dataset, (record or {}).get("food_name") or "")
        else:
            name_ann = median
            name_vlm, matched, kind = median, None, "none"
        ref_ann = float("nan")
        ref_vlm = float("nan")
        if dataset == "ecustfd":
            factors = coins.factors(row["sample_id"])
            if factors is not None:
                ref_ann = coins.predict(row["sample_id"], factors["coin_px"], True)
            diam = _coin_px(record)
            if diam is not None:
                ref_vlm = coins.predict(row["sample_id"], diam, True)
        ood = bool(record and record.get("out_of_domain"))
        if record is None:
            route = "name_prior"
        elif ood:
            route = "reject"
        elif has_depth:
            route = "depth"
        elif dataset == "ecustfd" and _finite(ref_vlm):
            route = "reference"
        else:
            route = "name_prior"
        pred = {
            "reject": float("nan"),
            "depth": mono,
            "reference": ref_vlm if _finite(ref_vlm) else name_vlm,
            "name_prior": name_vlm,
        }[route]
        if dataset == "nvreal":
            source_route, source_pred = "reject", float("nan")
        elif dataset == "n5k":
            source_route, source_pred = "depth", mono
        elif dataset == "ecustfd":
            source_route, source_pred = "reference", ref_ann
        else:
            source_route, source_pred = "name_prior", name_ann
        samples.append(
            {
                **row,
                "mono": mono,
                "name_ann": name_ann,
                "name_vlm": name_vlm,
                "ref_ann": ref_ann,
                "ref_vlm": ref_vlm,
                "route": route,
                "pred": pred,
                "source_route": source_route,
                "source_pred": source_pred,
                "oracle_auto": _best(row["y_g"], [mono, name_vlm, ref_vlm]),
                "oracle_labeled": _best(row["y_g"], [mono, name_ann, ref_ann]),
                "matched": matched,
                "match_kind": kind,
                "vlm_missing": record is None,
            }
        )
    return samples, median


def _take(rows: list[dict], key: str):
    y, p = [], []
    for row in rows:
        if _finite(row.get(key)):
            y.append(float(row["y_g"]))
            p.append(float(row[key]))
    return np.asarray(y, dtype=float), np.asarray(p, dtype=float)


def _conformal(cal: list[dict], test: list[dict]) -> dict:
    pooled = [abs(float(row["y_g"]) - float(row["pred"])) for row in cal if _finite(row["pred"])]
    by_route = {}
    routes = sorted({row["route"] for row in cal + test})
    quantiles = {}
    for route in routes:
        if route == "reject":
            continue
        scores = [
            abs(float(row["y_g"]) - float(row["pred"]))
            for row in cal
            if row["route"] == route and _finite(row["pred"])
        ]
        if len(scores) >= 8:
            quantiles[route] = finite_sample_quantile(scores, ALPHA)
            by_route[route] = {"n_cal": len(scores), "q": quantiles[route], "fallback": False}
        else:
            quantiles[route] = finite_sample_quantile(pooled, ALPHA) if pooled else float("inf")
            by_route[route] = {"n_cal": len(scores), "q": quantiles[route], "fallback": True}
    groups = {}
    answered = []
    for row in test:
        if row["route"] == "reject" or not _finite(row["pred"]):
            continue
        q = quantiles.get(row["route"], float("inf"))
        lo, hi = conformal_interval(np.array([row["pred"]]), np.ones(1), q)
        item = (float(row["y_g"]), float(lo[0]), float(hi[0]), row["route"])
        answered.append(item)
        groups.setdefault(row["route"], []).append(item)
    def pack(items):
        if not items:
            return {"n": 0}
        y, lo, hi = zip(*[(a, b, c) for a, b, c, _ in items])
        return interval_metrics(y, lo, hi, ALPHA)
    return {
        "overall_answered": pack(answered),
        "by_route": {route: pack(items) for route, items in groups.items()},
        "quantiles": by_route,
        "n_test": len(test),
        "n_answered": len(answered),
        "reject_rate": 1.0 - (len(answered) / len(test) if test else float("nan")),
        "coverage_counting_reject_as_miss": (
            pack(answered).get("n_covered", 0) / len(test) if test else None
        ),
    }


def _rate(rows: list[dict], dataset: str, route: str) -> dict:
    chosen = [row for row in rows if row["dataset"] == dataset]
    if not chosen:
        return {"n": 0}
    hit = sum(row["route"] == route for row in chosen)
    return {"n": len(chosen), "rate": hit / len(chosen), "route": route}


def main() -> int:
    paths = Paths.resolve()
    samples, median = _build(paths)
    groups = np.array([row["group_id"] for row in samples])
    labels = grouped_three_way(groups)
    for row, label in zip(samples, labels):
        row["flow_split"] = str(label)
    policy = [row for row in samples if row["flow_split"] == "policy"]
    cal = [row for row in samples if row["flow_split"] == "calibration"]
    test = [row for row in samples if row["flow_split"] == "test"]
    y_pol, p_mono = _take(policy, "mono")
    _, p_name = _take(policy, "name_ann")
    # The two arrays can drop different rows. Score each action on the rows where it is finite,
    # which for these two actions is every policy row.
    mono_w = block([row["y_g"] for row in policy], [row["mono"] for row in policy])
    name_w = block([row["y_g"] for row in policy], [row["name_ann"] for row in policy])
    best_name = "mono" if (mono_w["wape"] or 1) <= (name_w["wape"] or 1) else "name_ann"
    best_key = "mono" if best_name == "mono" else "name_ann"
    answered = [row for row in test if _finite(row["pred"])]
    report = {
        "n": {"all": len(samples), "policy": len(policy), "calibration": len(cal), "test": len(test)},
        "n5k_train_median_g": median,
        "rule": "out_of_domain -> reject; depth channel -> 4ch ensemble; ECUSTFD coin -> scale; else tau=4 name prior",
        "single_action_choice": {
            "policy_mono": mono_w,
            "policy_name_annotated": name_w,
            "chosen": best_name,
        },
        "test": {
            "vlm_route": block([row["y_g"] for row in answered], [row["pred"] for row in answered]),
            "best_single_on_vlm_answered": block(
                [row["y_g"] for row in answered], [row[best_key] for row in answered]
            ),
            "best_single_on_all_test": block([row["y_g"] for row in test], [row[best_key] for row in test]),
            "mono": block([row["y_g"] for row in test], [row["mono"] for row in test]),
            "name_annotated": block([row["y_g"] for row in test], [row["name_ann"] for row in test]),
            "source_route": block(
                [row["y_g"] for row in test if _finite(row["source_pred"])],
                [row["source_pred"] for row in test if _finite(row["source_pred"])],
            ),
            "oracle_automatic": block([row["y_g"] for row in test], [row["oracle_auto"] for row in test]),
            "oracle_labeled": block([row["y_g"] for row in test], [row["oracle_labeled"] for row in test]),
            "reject_rate": 1.0 - (len(answered) / len(test) if test else float("nan")),
            "wape_gap_best_single_minus_vlm_on_answered": wape_gap(
                [row["y_g"] for row in answered],
                [row["pred"] for row in answered],
                [row[best_key] for row in answered],
            ),
        },
        "conformal": _conformal(cal, test),
        "routing": {
            "n5k_depth": _rate(test, "n5k", "depth"),
            "ecustfd_reference": _rate(test, "ecustfd", "reference"),
            "mf3d_name_prior": _rate(test, "mf3d", "name_prior"),
            "nvreal_reject": _rate(test, "nvreal", "reject"),
        },
        "route_counts_test": {
            route: sum(row["route"] == route for row in test)
            for route in ("reject", "depth", "reference", "name_prior")
        },
    }
    del y_pol, p_mono, p_name
    dest = paths.results_action / "a5_mixed_flow" / "report.json"
    atomic_write_json(dest, clean(report))
    log(
        f"A5 test n={len(test)} reject={report['test']['reject_rate']} "
        f"vlm_wape={report['test']['vlm_route'].get('wape')} "
        f"single={best_name}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
