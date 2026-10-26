#!/usr/bin/env python3
"""E15: within-group routing opportunity on the E13/E14 public tables.

Question. After a (dataset, morphology) group has frozen one action, does the
best action still change from sample to sample, and can a rule that never sees
the label exploit that change?

The group action and every threshold are chosen on the policy split only.
Category medians on that split are leave-one-out, so a singleton category
cannot score a zero error. Calibration and test labels are held out together.
They are not used to pick tau, the group action, or a router threshold.

Decision rule:

  A router is compared with the morphology-group action on held-out WAPE.
  Beating that action is not enough. The same router must also beat the single
  action chosen for the whole dataset on the policy split, by at least 0.02
  WAPE, with a bootstrap 95% interval that excludes 0. It must also use more
  than one action on the held-out rows (modal action below 90%). Otherwise the
  paper must say group-level routing.

  A rule that only switches on the policy category count is not instance-level
  routing. It is reported separately and does not change the wording.

The oracle that sees the label is an upper bound. It is not a deployable
policy and cannot by itself justify a per-sample claim. Within one sample,
absolute error and percentage error rank actions the same way, because both
are monotone in |y-p|. WAPE and MAPE can still pick different constants once
samples are averaged.
"""
from __future__ import annotations

import argparse
from collections import defaultdict

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge

from .common import add_common_args, out_dir, save_status
from .a1_group_routing import _attach_ecustfd_semantic
from .a2_name_prior import PROMPT, _attach_categories, _query_categories, json_load
from mmcfa.data_build import build_ecustfd, build_mf3d
from mmcfa.metrics import mape, wape
from mmcfa.paths import Paths
from mmcfa.policy.group_freeze import COSTS, freeze_table
from mmcfa.runtime import atomic_write_json, log


ALPHA_RIDGE = 1.0
MIN_WAPE_DROP = 0.02
TAU = 4.0
N_BOOT = 2000
NEAR_GRAM = 1.0
NEAR_REL = 0.02
COUNT_GRID = (1, 2, 3, 5, 8)
DISAGREE_GRID = (0.05, 0.10, 0.20, 0.35, 0.50, 0.75, 1.00)
MORPHS = ("rigid", "prepared", "stacked", "liquid", "unknown")
ACTION_COST = {
    "stop": COSTS["STOP"],
    "semantic": COSTS["ASK_SEMANTIC"],
    "shrink": COSTS["ASK_SEMANTIC"],
    "llm": COSTS["ASK_SEMANTIC"],
    "view": COSTS["REQUEST_VIEW"],
    "reference": COSTS["REQUEST_REFERENCE"],
}


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
    if isinstance(obj, np.ndarray):
        return _jsonable(obj.tolist())
    return obj


def _policy_values(rows: list[dict]) -> dict[str, list[float]]:
    buckets: dict[str, list[float]] = defaultdict(list)
    for row in rows:
        if row["split"] == "policy":
            buckets[row["cat"]].append(float(row["y"]))
    return buckets


def _median(values: list[float]) -> float:
    if not values:
        return float("nan")
    return float(np.median(values))


def _drop_one(values: list[float], y: float) -> list[float]:
    kept = list(values)
    for i, value in enumerate(kept):
        if value == y:
            del kept[i]
            break
    return kept


def _stats(buckets: dict[str, list[float]], skip: tuple[str, float] | None = None) -> dict:
    use = {cat: list(values) for cat, values in buckets.items()}
    if skip is not None:
        cat, y = skip
        use[cat] = _drop_one(use.get(cat, []), y)
    med = {cat: _median(values) for cat, values in use.items() if values}
    count = {cat: len(values) for cat, values in use.items()}
    flat = [y for values in use.values() for y in values]
    return {"median": med, "count": count, "global": _median(flat)}


def _shrink(n: int, med: float, llm: float, tau: float) -> float:
    if n <= 0 or not np.isfinite(med):
        return llm if np.isfinite(llm) else float("nan")
    if not np.isfinite(llm):
        return float(med)
    return float((n * med + tau * llm) / (n + tau))


def _predictions(row: dict, stats: dict, cache: dict) -> dict[str, float]:
    cat = row["cat"]
    n = int(stats["count"].get(cat, 0))
    med = float(stats["median"].get(cat, stats["global"]))
    llm_raw = cache.get(cat, {}).get("typical_mass_g")
    llm = float(llm_raw) if llm_raw is not None and np.isfinite(llm_raw) else float("nan")
    stop = row.get("p_rgb", np.nan)
    if row["dataset"] == "mf3d":
        stop = stats["global"]
    out = {
        "stop": float(stop) if stop is not None and np.isfinite(stop) else float("nan"),
        "semantic": float(med) if n > 0 and np.isfinite(med) else float(stats["global"]),
        "shrink": _shrink(n, med if n > 0 else float("nan"), llm, TAU),
        "llm": llm if np.isfinite(llm) else float(stats["global"]),
    }
    for name, key in (("reference", "p_ref"), ("view", "p_view")):
        value = row.get(key, np.nan)
        if value is not None and np.isfinite(value):
            out[name] = float(value)
    return out


def _row_stats(row: dict, buckets: dict[str, list[float]]) -> dict:
    if row["split"] == "policy":
        return _stats(buckets, skip=(row["cat"], float(row["y"])))
    return _stats(buckets)


def _point(y, p) -> dict:
    y = np.asarray(y, dtype=float)
    p = np.asarray(p, dtype=float)
    ok = np.isfinite(y) & np.isfinite(p) & (np.abs(y) > 0)
    y, p = y[ok], p[ok]
    if len(y) == 0:
        return {"n": 0, "mape": float("nan"), "wape": float("nan"), "mae": float("nan")}
    return {
        "n": int(len(y)),
        "mape": mape(y, p),
        "wape": wape(y, p),
        "mae": float(np.mean(np.abs(y - p))),
    }


def _best_action(rows: list[dict], buckets, cache, min_n: int = 8) -> tuple[str, dict]:
    """Lowest policy WAPE. Leave-one-out predictions. Stop wins ties and small gaps."""
    preds = {row["episode_id"]: _predictions(row, _row_stats(row, buckets), cache) for row in rows}
    actions = sorted({name for item in preds.values() for name in item})
    y = np.array([row["y"] for row in rows], dtype=float)
    scores = {}
    for action in actions:
        p = np.array([preds[row["episode_id"]].get(action, np.nan) for row in rows], dtype=float)
        n = int(np.sum(np.isfinite(p) & np.isfinite(y)))
        scores[action] = {"n": n, "wape": wape(y, p) if n >= min_n else float("nan"), "point": _point(y, p)}
    stop = scores.get("stop", {}).get("wape", float("nan"))
    best, best_w = "stop", stop
    for action, rec in scores.items():
        if action == "stop" or not np.isfinite(rec["wape"]) or not np.isfinite(stop):
            continue
        if stop - rec["wape"] + 1e-12 < 0.05:
            continue
        if rec["wape"] < best_w - 1e-12:
            best, best_w = action, rec["wape"]
    return best, scores


def _freeze(rows: list[dict], buckets, cache) -> dict:
    policy = [row for row in rows if row["split"] == "policy"]
    by_group: dict[tuple[str, str], list[dict]] = defaultdict(list)
    by_dataset: dict[str, list[dict]] = defaultdict(list)
    for row in policy:
        key = (row["dataset"], row.get("morphology") or "unknown")
        by_group[key].append(row)
        by_dataset[row["dataset"]].append(row)
    dataset_choice = {}
    for ds, sub in by_dataset.items():
        action, scores = _best_action(sub, buckets, cache)
        dataset_choice[ds] = {"action": action, "n": len(sub), "scores": scores}
    groups = {}
    for key, sub in sorted(by_group.items()):
        ds, morph = key
        if len(sub) >= 8:
            action, scores = _best_action(sub, buckets, cache)
            source = "group"
        else:
            action = dataset_choice[ds]["action"]
            scores = {}
            source = "dataset_fallback"
        groups[f"{ds}|{morph}"] = {
            "dataset": ds,
            "morphology": morph,
            "n_policy": len(sub),
            "action": action,
            "source": source,
            "scores": scores,
        }
    return {"groups": groups, "dataset_fallback": dataset_choice}


def _frozen_action(row: dict, table: dict) -> str:
    key = f"{row['dataset']}|{row.get('morphology') or 'unknown'}"
    if key in table["groups"]:
        return table["groups"][key]["action"]
    return table["dataset_fallback"].get(row["dataset"], {}).get("action", "stop")


def _strict_best(preds: dict[str, float], y: float, how: str) -> str:
    def err(action: str) -> float:
        p = preds[action]
        if how == "abs":
            return abs(y - p)
        return abs(y - p) / max(abs(y), 1e-6)

    return min(preds, key=lambda action: (err(action), ACTION_COST[action], action))


def _near_best(preds: dict[str, float], y: float) -> list[str]:
    best = min(abs(y - p) for p in preds.values())
    tol = max(NEAR_GRAM, NEAR_REL * abs(y))
    return sorted(action for action, p in preds.items() if abs(y - p) <= best + tol)


def _features(row: dict, pred: dict[str, float], stats: dict) -> np.ndarray:
    stop = pred.get("stop", np.nan)
    sem = pred.get("semantic", np.nan)
    llm = pred.get("llm", np.nan)
    n_cat = float(stats["count"].get(row["cat"], 0))
    gap = abs(np.log(max(llm, 1.0)) - np.log(max(sem, 1.0))) if np.isfinite(llm) and np.isfinite(sem) else 0.0
    morph = row.get("morphology") or "unknown"
    return np.array(
        [1.0, np.log1p(max(stop, 0.0)), np.log1p(n_cat), gap] + [1.0 if morph == name else 0.0 for name in MORPHS],
        dtype=float,
    )


def _fit_error_model(policy: list[dict], buckets, cache) -> dict[str, Ridge] | None:
    by_action: dict[str, list[tuple[np.ndarray, float]]] = defaultdict(list)
    for row in policy:
        stats = _row_stats(row, buckets)
        pred = _predictions(row, stats, cache)
        feat = _features(row, pred, stats)
        for action, p in pred.items():
            if np.isfinite(p):
                by_action[action].append((feat, abs(float(row["y"]) - p)))
    models = {}
    for action, pairs in by_action.items():
        if len(pairs) < 12:
            continue
        x = np.vstack([item[0] for item in pairs])
        target = np.array([item[1] for item in pairs], dtype=float)
        model = Ridge(alpha=ALPHA_RIDGE)
        model.fit(x, target)
        models[action] = model
    return models or None


def _route_error_model(row, buckets, cache, models: dict[str, Ridge]) -> str:
    stats = _row_stats(row, buckets)
    pred = _predictions(row, stats, cache)
    feat = _features(row, pred, stats).reshape(1, -1)
    best, best_hat = None, float("inf")
    for action, p in pred.items():
        model = models.get(action)
        if model is None or not np.isfinite(p):
            continue
        hat = float(model.predict(feat)[0])
        if hat < best_hat:
            best, best_hat = action, hat
    return best or "stop"


def _choose_count_threshold(policy, buckets, cache, frozen_of) -> int | None:
    best_t, best_w = None, float("inf")
    base = _score_actions(policy, buckets, cache, lambda row, pred: frozen_of(row, pred))
    if base["n"] == 0 or not np.isfinite(base["wape"]):
        return None
    for threshold in COUNT_GRID:
        def choose(row, pred, threshold=threshold):
            n_cat = _row_stats(row, buckets)["count"].get(row["cat"], 0)
            if n_cat < threshold and np.isfinite(pred.get("shrink", np.nan)):
                return "shrink"
            return frozen_of(row, pred)

        scored = _score_actions(policy, buckets, cache, choose)
        if scored["n"] >= 8 and scored["wape"] < best_w:
            best_t, best_w = threshold, scored["wape"]
    if best_t is None or not (best_w < base["wape"] - 1e-12):
        return None
    return int(best_t)


def _choose_disagree_threshold(policy, buckets, cache) -> float | None:
    usable = [row for row in policy if np.isfinite(row.get("p_ref", np.nan)) and np.isfinite(row.get("p_rgb", np.nan))]
    if len(usable) < 12:
        return None
    best_t, best_w = None, float("inf")

    def score(threshold: float) -> float:
        y, p = [], []
        for row in usable:
            stats = _row_stats(row, buckets)
            pred = _predictions(row, stats, cache)
            ratio = abs(pred["reference"] - pred["stop"]) / max(abs(pred["stop"]), 1.0)
            action = "reference" if ratio >= threshold else "semantic"
            y.append(row["y"])
            p.append(pred[action])
        return wape(y, p)

    always = score(0.0)
    for threshold in DISAGREE_GRID:
        value = score(threshold)
        if value < best_w:
            best_t, best_w = threshold, value
    if best_t is None or not (best_w < always - 1e-12):
        return None
    return float(best_t)


def _best_by_metric(rows: list[dict], buckets, cache, metric: str) -> str:
    """Policy-split argmin. No minimum-drop gate. Leave-one-out predictions."""
    preds = {row["episode_id"]: _predictions(row, _row_stats(row, buckets), cache) for row in rows}
    actions = sorted({name for item in preds.values() for name in item})
    y = np.array([row["y"] for row in rows], dtype=float)
    best, best_value = "stop", float("inf")
    for action in actions:
        p = np.array([preds[row["episode_id"]].get(action, np.nan) for row in rows], dtype=float)
        value = _point(y, p).get(metric, float("nan"))
        if np.isfinite(value) and value < best_value:
            best, best_value = action, float(value)
    return best


def _action_list(rows, buckets, cache, choose) -> list[str]:
    chosen = []
    for row in rows:
        stats = _row_stats(row, buckets)
        pred = _predictions(row, stats, cache)
        action = choose(row, pred)
        if not np.isfinite(pred.get(action, np.nan)):
            action = "stop"
        chosen.append(action)
    return chosen


def _score_actions(rows, buckets, cache, choose) -> dict:
    y, p = [], []
    for row in rows:
        stats = _row_stats(row, buckets)
        pred = _predictions(row, stats, cache)
        action = choose(row, pred)
        value = pred.get(action, np.nan)
        if not np.isfinite(value):
            value = pred.get("stop", np.nan)
        if np.isfinite(value):
            y.append(row["y"])
            p.append(value)
    return _point(y, p)


def _bootstrap_drop(rows, buckets, cache, base_choose, alt_choose, seed: int) -> dict:
    packed = []
    for row in rows:
        stats = _row_stats(row, buckets)
        pred = _predictions(row, stats, cache)
        base = pred.get(base_choose(row, pred), np.nan)
        alt = pred.get(alt_choose(row, pred), np.nan)
        if not np.isfinite(base):
            base = pred.get("stop", np.nan)
        if not np.isfinite(alt):
            alt = pred.get("stop", np.nan)
        if np.isfinite(base) and np.isfinite(alt):
            packed.append((float(row["y"]), float(base), float(alt)))
    if len(packed) < 8:
        return {"n": len(packed)}
    y = np.array([item[0] for item in packed])
    base = np.array([item[1] for item in packed])
    alt = np.array([item[2] for item in packed])
    rng = np.random.default_rng(seed)
    drops = []
    n = len(y)
    for _ in range(N_BOOT):
        idx = rng.integers(0, n, n)
        drops.append(wape(y[idx], base[idx]) - wape(y[idx], alt[idx]))
    drops = np.asarray(drops, dtype=float)
    return {
        "n": int(n),
        "mean_wape_base_minus_challenger": float(np.mean(drops)),
        "ci95": [float(np.quantile(drops, 0.025)), float(np.quantile(drops, 0.975))],
    }


def _boot_arrays(y, base, alt, seed: int) -> dict:
    """Mean of WAPE(alt) - WAPE(base). Positive means alt is worse."""
    y = np.asarray(y, dtype=float)
    base = np.asarray(base, dtype=float)
    alt = np.asarray(alt, dtype=float)
    rng = np.random.default_rng(seed)
    n = len(y)
    gaps = np.empty(N_BOOT, dtype=float)
    for i in range(N_BOOT):
        idx = rng.integers(0, n, n)
        gaps[i] = wape(y[idx], alt[idx]) - wape(y[idx], base[idx])
    return {
        "n": int(n),
        "mean": float(np.mean(gaps)),
        "ci95": [float(np.quantile(gaps, 0.025)), float(np.quantile(gaps, 0.975))],
    }


def _gap_supports(gap: dict) -> bool:
    mean = gap.get("mean_wape_base_minus_challenger")
    ci = gap.get("ci95")
    if mean is None or not ci or gap.get("n", 0) < 20:
        return False
    return bool(mean >= MIN_WAPE_DROP and ci[0] > 0)


def decide(routers: list[dict]) -> dict:
    """Map router gaps to the wording the paper is allowed to use."""
    beat_morph = []
    instance = []
    frequency = []
    for row in routers:
        if not _gap_supports(row["heldout_gap"]):
            continue
        if row["family"] == "frequency":
            if _gap_supports(row.get("gap_vs_dataset_constant", {})):
                frequency.append(row["name"])
            continue
        beat_morph.append(row["name"])
        diverse = float(row.get("modal_fraction", 1.0)) < 0.90
        if diverse and _gap_supports(row.get("gap_vs_dataset_constant", {})):
            instance.append(row["name"])
    if instance:
        wording = "instance"
        reason = "a label-free router beat both the morphology action and the dataset constant, and it did not collapse to one action"
    elif beat_morph:
        wording = "group"
        reason = "a router beat the morphology table, but not the dataset-level constant chosen on the policy split"
    elif frequency:
        wording = "group"
        reason = "only a category-count switch beat the morphology table; that is not instance routing"
    else:
        wording = "group"
        reason = "no label-free router beat the morphology action by the pre-registered margin"
    return {
        "wording": wording,
        "reason": reason,
        "beat_morphology_freeze": beat_morph,
        "accepted_instance_routers": instance,
        "accepted_frequency_routers": frequency,
        "min_wape_drop": MIN_WAPE_DROP,
        "paper_clause": "逐样本路由" if wording == "instance" else "组级路由",
    }


def _self_check() -> None:
    weak = {"family": "instance", "name": "weak", "modal_fraction": 0.5, "heldout_gap": {"n": 100, "mean_wape_base_minus_challenger": 0.01, "ci95": [0.001, 0.02]}, "gap_vs_dataset_constant": {"n": 100, "mean_wape_base_minus_challenger": 0.04, "ci95": [0.01, 0.07]}}
    morph_only = {"family": "instance", "name": "morph", "modal_fraction": 0.4, "heldout_gap": {"n": 100, "mean_wape_base_minus_challenger": 0.04, "ci95": [0.01, 0.07]}, "gap_vs_dataset_constant": {"n": 100, "mean_wape_base_minus_challenger": -0.01, "ci95": [-0.03, 0.01]}}
    strong = {"family": "instance", "name": "strong", "modal_fraction": 0.4, "heldout_gap": {"n": 100, "mean_wape_base_minus_challenger": 0.04, "ci95": [0.01, 0.07]}, "gap_vs_dataset_constant": {"n": 100, "mean_wape_base_minus_challenger": 0.03, "ci95": [0.01, 0.05]}}
    constantish = dict(strong)
    constantish = {**strong, "name": "constantish", "modal_fraction": 0.97}
    freq = {"family": "frequency", "name": "count", "modal_fraction": 0.5, "heldout_gap": {"n": 100, "mean_wape_base_minus_challenger": 0.05, "ci95": [0.02, 0.08]}, "gap_vs_dataset_constant": {"n": 100, "mean_wape_base_minus_challenger": 0.05, "ci95": [0.02, 0.08]}}
    assert decide([weak])["wording"] == "group"
    assert decide([morph_only])["wording"] == "group"
    assert decide([morph_only])["beat_morphology_freeze"] == ["morph"]
    assert decide([strong])["wording"] == "instance"
    assert decide([constantish])["wording"] == "group"
    assert decide([freq])["wording"] == "group"
    assert decide([freq])["accepted_frequency_routers"] == ["count"]


def _group_rows(held: list[dict]) -> dict[str, list[dict]]:
    out: dict[str, list[dict]] = defaultdict(list)
    for row in held:
        out[f"{row['dataset']}|{row.get('morphology') or 'unknown'}"].append(row)
    return out


def _describe_group(name: str, rows: list[dict], buckets, cache, frozen: str) -> dict:
    records = []
    for row in rows:
        pred = _predictions(row, _row_stats(row, buckets), cache)
        finite = {action: value for action, value in pred.items() if np.isfinite(value)}
        if "stop" not in finite or frozen not in finite:
            continue
        y = float(row["y"])
        strict = _strict_best(finite, y, "abs")
        pct = _strict_best(finite, y, "pct")
        near = _near_best(finite, y)
        records.append(
            {
                "episode_id": row["episode_id"],
                "dataset": row["dataset"],
                "morphology": row.get("morphology") or "unknown",
                "cat": row["cat"],
                "split": row["split"],
                "y": y,
                "frozen": frozen,
                "strict_abs": strict,
                "strict_pct": pct,
                "frozen_in_near": frozen in near,
                "abs_pct_disagree": strict != pct,
                "pred": finite,
            }
        )
    if not records:
        return {"group": name, "n": 0}

    def take(action_of):
        return _point([rec["y"] for rec in records], [rec["pred"][action_of(rec)] for rec in records])

    actions = sorted({action for rec in records for action in rec["pred"]})
    matrix = {}
    for action in actions:
        usable = [rec for rec in records if action in rec["pred"]]
        matrix[action] = _point([rec["y"] for rec in usable], [rec["pred"][action] for rec in usable])
    counts = defaultdict(int)
    for rec in records:
        counts[rec["strict_abs"]] += 1
    modal, modal_n = max(counts.items(), key=lambda item: item[1])
    return {
        "group": name,
        "n": len(records),
        "frozen_action": frozen,
        "strict_match_frozen": float(np.mean([rec["strict_abs"] == frozen for rec in records])),
        "near_match_frozen": float(np.mean([rec["frozen_in_near"] for rec in records])),
        "modal_oracle_action": modal,
        "modal_oracle_fraction": modal_n / len(records),
        "oracle_action_counts": dict(counts),
        "abs_vs_pct_disagree": float(np.mean([rec["abs_pct_disagree"] for rec in records])),
        "frozen_point": take(lambda rec: rec["frozen"]),
        "oracle_abs_point": take(lambda rec: rec["strict_abs"]),
        "oracle_pct_point": take(lambda rec: rec["strict_pct"]),
        "matrix": matrix,
        "records": records,
    }


def _flatten_records(groups: list[dict]) -> list[dict]:
    rows = []
    for group in groups:
        for rec in group.get("records", []):
            flat = {k: rec[k] for k in ("episode_id", "dataset", "morphology", "cat", "split", "y", "frozen", "strict_abs", "strict_pct", "frozen_in_near", "abs_pct_disagree")}
            for action, value in rec["pred"].items():
                flat[f"p_{action}"] = value
            rows.append(flat)
    return rows


def main(argv=None) -> int:
    _self_check()
    ap = add_common_args(argparse.ArgumentParser())
    ap.add_argument("--skip-query", action="store_true")
    args = ap.parse_args(argv)
    paths = Paths.resolve()
    d = out_dir("a5_oracle_bound", args)
    ec = build_ecustfd(paths)
    mf = build_mf3d(paths)
    if not ec or not mf:
        save_status(d, False, error="ECUSTFD or MetaFood3D table missing")
        return 2
    _attach_ecustfd_semantic(ec, pd.read_csv(paths.file("ecustfd_csv")))
    _attach_categories(ec, mf, paths)
    rows = ec + mf
    cache_path = paths.results / "a2_name_prior" / "category_cache.json"
    cache = json_load(cache_path) if cache_path.exists() else {}
    names = sorted({row["cat"] for row in rows})
    missing = [name for name in names if not cache.get(name, {}).get("typical_mass_g")]
    if missing and not args.skip_query:
        log(f"E15 querying {len(missing)} categories")
        cache = _query_categories(names, cache)
        atomic_write_json(cache_path, _jsonable(cache))
    failed = [name for name in names if not cache.get(name, {}).get("typical_mass_g")]
    if failed:
        save_status(d, False, error=f"{len(failed)} categories have no mass prior", examples=failed[:5])
        return 2

    buckets = _policy_values(rows)
    table = _freeze(rows, buckets, cache)
    e13_table = freeze_table([row for row in rows if row["split"] == "policy"])
    held = [row for row in rows if row["split"] != "policy"]
    described = []
    for name, sub in sorted(_group_rows(held).items()):
        frozen = _frozen_action(sub[0], table)
        described.append(_describe_group(name, sub, buckets, cache, frozen))

    routers = []
    for ds in ("ecustfd", "mf3d"):
        policy = [row for row in rows if row["dataset"] == ds and row["split"] == "policy"]
        held_ds = [row for row in held if row["dataset"] == ds]
        frozen_of = lambda row, pred, table=table: _frozen_action(row, table)
        base_point = _score_actions(held_ds, buckets, cache, lambda row, pred: frozen_of(row, pred))

        models = _fit_error_model(policy, buckets, cache)
        if models is not None:
            choose = lambda row, pred, models=models: _route_error_model(row, buckets, cache, models)
            routers.append(
                {
                    "name": f"{ds}_error_model",
                    "dataset": ds,
                    "family": "instance",
                    "choose": choose,
                    "heldout": _score_actions(held_ds, buckets, cache, choose),
                    "group_heldout": base_point,
                    "heldout_gap": _bootstrap_drop(held_ds, buckets, cache, frozen_of, choose, seed=10 + len(ds)),
                }
            )

        if ds == "ecustfd":
            threshold = _choose_disagree_threshold(policy, buckets, cache)

            def disagree(row, pred, threshold=threshold):
                if threshold is None or "reference" not in pred:
                    return frozen_of(row, pred)
                ratio = abs(pred["reference"] - pred["stop"]) / max(abs(pred["stop"]), 1.0)
                return "reference" if ratio >= threshold else "semantic"

            routers.append(
                {
                    "name": "ecustfd_scale_disagreement",
                    "dataset": ds,
                    "family": "instance",
                    "threshold": threshold,
                    "choose": disagree,
                    "heldout": _score_actions(held_ds, buckets, cache, disagree),
                    "group_heldout": base_point,
                    "heldout_gap": _bootstrap_drop(held_ds, buckets, cache, frozen_of, disagree, seed=11),
                }
            )

        count_t = _choose_count_threshold(policy, buckets, cache, frozen_of)

        def by_count(row, pred, count_t=count_t):
            if count_t is None:
                return frozen_of(row, pred)
            n_cat = _row_stats(row, buckets)["count"].get(row["cat"], 0)
            if n_cat < count_t and np.isfinite(pred.get("shrink", np.nan)):
                return "shrink"
            return frozen_of(row, pred)

        routers.append(
            {
                "name": f"{ds}_category_count",
                "dataset": ds,
                "family": "frequency",
                "threshold": count_t,
                "choose": by_count,
                "heldout": _score_actions(held_ds, buckets, cache, by_count),
                "group_heldout": base_point,
                "heldout_gap": _bootstrap_drop(held_ds, buckets, cache, frozen_of, by_count, seed=12 + len(ds)),
            }
        )

    constants = {}
    for ds in ("ecustfd", "mf3d"):
        policy = [row for row in rows if row["dataset"] == ds and row["split"] == "policy"]
        held_ds = [row for row in held if row["dataset"] == ds]
        wape_action = _best_by_metric(policy, buckets, cache, "wape")
        mape_action = _best_by_metric(policy, buckets, cache, "mape")

        def _const(action):
            def choose(row, pred, action=action):
                if np.isfinite(pred.get(action, np.nan)):
                    return action
                return "stop"
            return choose

        constants[ds] = {
            "policy_wape_action": wape_action,
            "policy_mape_action": mape_action,
            "metric_winners_differ": wape_action != mape_action,
            "heldout_wape_action": _score_actions(held_ds, buckets, cache, _const(wape_action)),
            "heldout_mape_action": _score_actions(held_ds, buckets, cache, _const(mape_action)),
            "gated_dataset_action": table["dataset_fallback"][ds]["action"],
        }
    for router in routers:
        ds = router["dataset"]
        choose = router.pop("choose")
        const_action = constants[ds]["policy_wape_action"]

        def const_choose(row, pred, const_action=const_action):
            if np.isfinite(pred.get(const_action, np.nan)):
                return const_action
            return "stop"

        held_ds = [row for row in held if row["dataset"] == ds]
        actions = _action_list(held_ds, buckets, cache, choose)
        counts: dict[str, int] = defaultdict(int)
        for action in actions:
            counts[action] += 1
        router["action_counts"] = dict(counts)
        router["modal_fraction"] = (max(counts.values()) / len(actions)) if actions else 1.0
        router["dataset_constant"] = const_action
        router["dataset_constant_heldout"] = constants[ds]["heldout_wape_action"]
        router["gap_vs_dataset_constant"] = _bootstrap_drop(
            held_ds, buckets, cache, const_choose, choose, seed=30 + len(router["name"])
        )

    decision = decide(routers)
    public_groups = []
    for group in described:
        item = dict(group)
        item.pop("records", None)
        public_groups.append(item)
    report = {
        "tau": TAU,
        "near_rule": {"grams": NEAR_GRAM, "relative": NEAR_REL},
        "decision": decision,
        "e13_frozen_actions": {key: value["action"] for key, value in e13_table["groups"].items()},
        "e15_frozen_actions": {key: value["action"] for key, value in table["groups"].items()},
        "dataset_constants": constants,
        "groups": public_groups,
        "routers": routers,
        "pooled": {},
    }
    for ds in ("ecustfd", "mf3d"):
        sub = [group for group in described if group.get("group", "").startswith(ds + "|")]
        recs = [rec for group in sub for rec in group.get("records", [])]
        if not recs:
            continue
        y = np.array([rec["y"] for rec in recs], dtype=float)
        frozen_p = np.array([rec["pred"][rec["frozen"]] for rec in recs], dtype=float)
        oracle_p = np.array([rec["pred"][rec["strict_abs"]] for rec in recs], dtype=float)
        const = constants[ds]["policy_wape_action"]
        const_p = np.array([rec["pred"][const] for rec in recs], dtype=float)
        report["pooled"][ds] = {
            "n": len(recs),
            "strict_match_frozen": float(np.mean([rec["strict_abs"] == rec["frozen"] for rec in recs])),
            "near_match_frozen": float(np.mean([rec["frozen_in_near"] for rec in recs])),
            "abs_vs_pct_disagree": float(np.mean([rec["abs_pct_disagree"] for rec in recs])),
            "frozen": _point(y, frozen_p),
            "oracle_abs": _point(y, oracle_p),
            "oracle_pct": _point(y, np.array([rec["pred"][rec["strict_pct"]] for rec in recs], dtype=float)),
            "dataset_constant": const,
            "morphology_minus_constant_wape": _boot_arrays(y, const_p, frozen_p, seed=7),
            "oracle_minus_constant_wape": _boot_arrays(y, const_p, oracle_p, seed=8),
        }
    atomic_write_json(d / "report.json", _jsonable(report))
    frame = pd.DataFrame(_flatten_records(described))
    frame.to_csv(d / "per_sample.csv", index=False)
    save_status(d, True, wording=decision["wording"], paper_clause=decision["paper_clause"])
    log(f"E15 wording={decision['wording']} ({decision['paper_clause']})")
    for ds, block in report["pooled"].items():
        log(
            f"E15 {ds} n={block['n']} near_match={block['near_match_frozen']:.3f} "
            f"strict_match={block['strict_match_frozen']:.3f} "
            f"frozen_wape={block['frozen']['wape']:.3f} oracle_wape={block['oracle_abs']['wape']:.3f}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
