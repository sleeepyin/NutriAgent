#!/usr/bin/env python3
"""Fit supervised Q_φ on rgb_train weak steps and compare selectors.

Order inside the run, matching P0c: rule utility is well-defined before
``Q_φ`` is fit. State-aware BC is the saved #24 checkpoint, scored as an
ablation, not retrained. No RL. rgb_test labels are not read. Frozen n5k
and weak_steps files are hashed before and after and must not change.

    python scripts/qphi_train.py
    python scripts/qphi_train.py --limit 512
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from collections import Counter
from pathlib import Path
from typing import Any, Mapping, Sequence

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np  # noqa: E402

from vdr.controller.nutrition_loss import LOSS_DEFINITION  # noqa: E402
from vdr.controller.on_policy import load_closed_loop_policy, rgb_test_ids, rollout_dish  # noqa: E402
from vdr.controller.policy_bc import BC_ACTIONS  # noqa: E402
from vdr.controller.policy_mlp import MLP_EPOCHS, MLP_HIDDEN_SIZES, MLP_LR, MLP_SEED  # noqa: E402
from vdr.controller.q_data import FIT_JSONL, HOLDOUT_JSONL, load_q_rows  # noqa: E402
from vdr.controller.q_phi import Q_EPOCHS, Q_LR, Q_SEED, Q_UNOBSERVED_ANCHOR, QPhi  # noqa: E402
from vdr.controller.q_state import STATE_OFFSET, Z_DIM, Z_FEATURE_NAMES  # noqa: E402
from vdr.controller.selector import HeuristicUtilityPolicy, QCostPolicy  # noqa: E402
from vdr.controller.utility import ACTION_COST, UNCERTAINTY_MAJORITY  # noqa: E402
from vdr.ir.io import read_ir_jsonl  # noqa: E402
from vdr.tools.mass_estimator import DEFAULT_CHECKPOINT, assert_safe_result_dir  # noqa: E402

OUT_DIR = ROOT / "results" / "qphi"
FROZEN = (
    ROOT / "results" / "weak_steps" / "rgb_train_fit_all.jsonl",
    ROOT / "results" / "weak_steps" / "rgb_train_holdout_all.jsonl",
    ROOT / "results" / "weak_steps" / "rgb_train_fit_all.summary.json",
    ROOT / "results" / "weak_steps" / "rgb_train_holdout_all.summary.json",
    ROOT / "results" / "n5k_rgb_test_overhead_b0_b1.jsonl",
    ROOT / "results" / "n5k_rgb_test_overhead_b0_b1.json",
    ROOT / "results" / "n5k_subset_b0_b1.jsonl",
    ROOT / "results" / "n5k_subset_b0_b1.json",
)
BC_WEIGHTS = ROOT / "results" / "closed_loop" / "policy_state_mlp.npz"
BC_META = ROOT / "results" / "closed_loop" / "train_meta.json"
NOT_P0D = (
    "P0c selector comparison on logged rgb_train weak steps. "
    "Not the P0d same-budget Full-set MAE table against B5, B0, or Ours-old. "
    "No RL. rgb_test labels were not read."
)


def _sha256(path: Path) -> str | None:
    if not path.is_file():
        return None
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _frozen_hashes() -> dict[str, str | None]:
    return {str(path.relative_to(ROOT)): _sha256(path) for path in FROZEN}


def _dish_ids(path: Path) -> list[str]:
    seen: set[str] = set()
    ordered: list[str] = []
    for record in read_ir_jsonl(path):
        dish_id = record.get("dish_id")
        if isinstance(dish_id, str) and dish_id not in seen:
            seen.add(dish_id)
            ordered.append(dish_id)
    return ordered


def _load_sigma(cache_path: Path, dish_ids: Sequence[str], *, refresh: bool) -> tuple[dict[str, dict[str, Any]], str]:
    if cache_path.is_file() and not refresh:
        payload = json.loads(cache_path.read_text(encoding="utf-8"))
        if payload.get("rgb_only") is True and payload.get("depth_used") is False and payload.get("uses_rgb_test_gt") is False:
            dishes = payload.get("dishes")
            if isinstance(dishes, dict) and dishes:
                print(f"reusing {cache_path} ({len(dishes)} dishes)", flush=True)
                return dishes, str(payload.get("backbone"))
    from vdr.controller.mass_sigma import collect_dish_sigma

    print(f"RGB mass sigma for {len(dish_ids)} weak-step dishes", flush=True)
    dishes, backbone = collect_dish_sigma(dish_ids, batch_size=16)
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    cache_path.write_text(
        json.dumps(
            {
                "rgb_only": True,
                "depth_used": False,
                "uses_rgb_test_gt": False,
                "rl": False,
                "backbone": backbone,
                "checkpoint": str(DEFAULT_CHECKPOINT.relative_to(ROOT)),
                "n_dishes": len(dishes),
                "n_requested": len(set(dish_ids)),
                "note": (
                    "Dish-level ViT means and Laplace scales from estimate_mass. "
                    "Not cafe labels. Not used to choose a threshold."
                ),
                "dishes": dishes,
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    print(f"wrote {cache_path.relative_to(ROOT)} n={len(dishes)} backbone={backbone}", flush=True)
    return dishes, backbone


def _pearson(left: np.ndarray, right: np.ndarray) -> float | None:
    if left.size < 3 or float(np.std(left)) < 1e-12 or float(np.std(right)) < 1e-12:
        return None
    return float(np.corrcoef(left, right)[0, 1])


def _round(value: float | None, digits: int = 4) -> float | None:
    if value is None:
        return None
    if not np.isfinite(value):
        return None
    return round(float(value), digits)


def _hist(actions: Sequence[str]) -> dict[str, int]:
    counts = Counter(actions)
    return {name: int(counts.get(name, 0)) for name in BC_ACTIONS}


def _agree(left: Sequence[str], right: Sequence[str]) -> float:
    if len(left) != len(right) or not left:
        raise ValueError("agreement needs two equal non-empty sequences")
    hits = sum(a == b for a, b in zip(left, right))
    return hits / len(left)


def _regression(model: QPhi, data: Any) -> dict[str, Any]:
    pred = model.predict_q(data.Z)
    index = np.array([BC_ACTIONS.index(str(action)) for action in data.y], dtype=np.int64)
    logged_q = pred[np.arange(data.n_rows), index]
    measured = data.measured & np.isfinite(logged_q) & np.isfinite(data.delta_l)
    target = data.delta_l[measured]
    guess = logged_q[measured]
    zero = np.zeros_like(target)
    per_action: dict[str, Any] = {}
    y = np.asarray(data.y)
    for name in BC_ACTIONS:
        mask = measured & (y == name)
        n_mask = int(mask.sum())
        if n_mask == 0:
            per_action[name] = {"n": 0}
            continue
        per_action[name] = {
            "n": n_mask,
            "mae": _round(float(np.mean(np.abs(logged_q[mask] - data.delta_l[mask])))),
            "mean_delta_l": _round(float(np.mean(data.delta_l[mask]))),
            "mean_predicted_q": _round(float(np.mean(logged_q[mask]))),
            "pearson": _round(_pearson(logged_q[mask], data.delta_l[mask]), 4),
        }
        energy = data.energy_delta_kcal[mask]
        finite_energy = energy[np.isfinite(energy)]
        per_action[name]["mean_energy_delta_kcal"] = (
            None if finite_energy.size == 0 else _round(float(np.mean(finite_energy)))
        )
        per_action[name]["n_energy"] = int(finite_energy.size)
    return {
        "population": "tool steps whose pre and post nutri_dish both have finite energy; STOP/ABSTAIN targets are 0 and are not in this MAE",
        "n": int(measured.sum()),
        "mae": _round(float(np.mean(np.abs(guess - target)))) if target.size else None,
        "mse": _round(float(np.mean((guess - target) ** 2))) if target.size else None,
        "pearson": _round(_pearson(guess, target)) if target.size else None,
        "baseline_zero_mae": _round(float(np.mean(np.abs(zero - target)))) if target.size else None,
        "mean_delta_l": _round(float(np.mean(target))) if target.size else None,
        "mean_predicted_q": _round(float(np.mean(guess))) if target.size else None,
        "per_action": per_action,
    }


def _conditional_delta(actions: Sequence[str], data: Any) -> dict[str, Any]:
    """Mean realized ΔL where the selector picked the logged action.

    This is not a counterfactual value. Rows the selector would have sent
    down another action have no observed ΔL.
    """
    y = [str(action) for action in data.y]
    mask = np.array([picked == logged and measured for picked, logged, measured in zip(actions, y, data.measured)], dtype=bool)
    if not bool(mask.any()):
        return {"n": 0, "mean_delta_l": None}
    return {
        "n": int(mask.sum()),
        "mean_delta_l": _round(float(np.mean(data.delta_l[mask]))),
        "note": "Conditional on matching the logged action and on that action being a measured tool step.",
    }


def _mean_features(Z: np.ndarray) -> dict[str, float | None]:
    names = (
        "U_id",
        "U_mass",
        "U_nutrient",
        "U_evidence",
        "D_models",
        "sigma_mass_norm",
        "sigma_energy_norm",
        "sigma_present",
    )
    out: dict[str, float | None] = {}
    for name in names:
        column = Z[:, Z_FEATURE_NAMES.index(name)]
        out[name] = _round(float(np.mean(column)))
    return out


def _controller_stub(policy: Any, *, label: str) -> dict[str, Any]:
    """One rgb_test id, stub tools, no cafe label. Replay must match."""
    dish_id = sorted(rgb_test_ids())[0]
    calls: list[int] = []

    def mass_fn(image, candidates, component_index=None, component_hints_g=None, catalog_path=None, **kwargs):
        del image, component_index, component_hints_g, catalog_path, kwargs
        calls.append(len(list(candidates or [])))
        return {
            "portion_g": 210.0,
            "mass_g": 210.0,
            "sigma": {
                "mass_g": 12.0,
                "energy_kcal": 40.0,
                "protein_g": 2.0,
                "carb_g": 4.0,
                "fat_g": 2.0,
            },
            "nutrients": {"energy_kcal": 180.0, "protein_g": 2.0, "carb_g": 20.0, "fat_g": 1.0},
            "rgb_only": True,
            "depth_used": False,
            "portion_src": "mass_estimator",
        }

    def retrieve_fn(query, top_k=5, catalog_path=None):
        del top_k, catalog_path

        class Result:
            cands = [
                {"code": "111", "score": 0.91, "name": "Apple raw"},
                {"code": "222", "score": 0.80, "name": "Apple juice"},
            ]

        result = Result()
        result.query = query  # type: ignore[attr-defined]
        return result

    def lookup_fn(code, catalog_path=None):
        del code, catalog_path
        return {"energy_kcal": 100.0, "protein_g": 1.0, "carb_g": 10.0, "fat_g": 1.0}

    def portion_fn(code, hint_g=100.0, catalog_path=None):
        del code, hint_g, catalog_path

        class Result:
            portion_g = 100.0

        return Result()

    rolled = rollout_dish(
        dish_id,
        policy,
        image_path="overhead/rgb.png",
        perceive_fn=lambda _path: (
            [{"name": "apple", "portion_g": 180.0}],
            {"model_alias": "qwen3.8-27b", "serving": "self-check"},
        ),
        retrieve_fn=retrieve_fn,
        lookup_fn=lookup_fn,
        portion_fn=portion_fn,
        mass_fn=mass_fn,
        check_replay=True,
    )
    return {
        "selector": label,
        "dish_id": dish_id,
        "actions": [step["action_t"] for step in rolled.steps],
        "ended_reason": rolled.ended_reason,
        "n_steps": len(rolled.steps),
        "mass_calls": len(calls),
        "replay_ok": True,
        "read_cafe_label": False,
        "vlm": False,
        "note": "Stub perception and stub estimate_mass. Not an rgb_test score.",
    }


def _report(metrics: Mapping[str, Any]) -> str:
    reg = metrics["holdout_q_regression"]
    sel = metrics["holdout_selection"]
    lines = [
        "P0c U_* + supervised Q_φ hold-out summary",
        f"fit_rows={metrics['fit_rows']} holdout_rows={metrics['holdout_rows']} z_dim={metrics['z_dim']}",
        f"sigma_rows fit={metrics['fit_sigma_rows']} holdout={metrics['holdout_sigma_rows']} rgb_dishes={metrics['sigma_dishes']}",
        (
            "Q regression MAE "
            f"{reg['mae']} on n={reg['n']} measured tool steps "
            f"(zero baseline {reg['baseline_zero_mae']}, pearson {reg['pearson']})"
        ),
        (
            "RE_PORTION mean energy change "
            f"{reg['per_action']['RE_PORTION'].get('mean_energy_delta_kcal')} kcal "
            f"on n={reg['per_action']['RE_PORTION'].get('n_energy')} "
            f"(predicted Q {reg['per_action']['RE_PORTION'].get('mean_predicted_q')})"
        ),
        (
            "selection agreement "
            f"Q vs heuristic {sel['agreement_q_vs_heuristic']:.4f} "
            f"Q vs BC {sel['agreement_q_vs_bc']:.4f} "
            f"heuristic vs logged {sel['agreement_heuristic_vs_logged']:.4f} "
            f"BC vs logged {sel['agreement_bc_vs_logged']:.4f}"
        ),
        "Q actions " + " ".join(f"{name}={sel['q_histogram'][name]}" for name in BC_ACTIONS),
        "heuristic actions " + " ".join(f"{name}={sel['heuristic_histogram'][name]}" for name in BC_ACTIONS),
        "BC ablation actions " + " ".join(f"{name}={sel['bc_histogram'][name]}" for name in BC_ACTIONS),
        "logged actions " + " ".join(f"{name}={sel['logged_histogram'][name]}" for name in BC_ACTIONS),
        f"mean U hold-out {metrics['holdout_mean_u']}",
        f"stub heuristic {metrics['controller_stub']['heuristic']['actions']} ({metrics['controller_stub']['heuristic']['ended_reason']})",
        f"stub Q/Cost {metrics['controller_stub']['q_over_cost']['actions']} ({metrics['controller_stub']['q_over_cost']['ended_reason']})",
        NOT_P0D,
    ]
    return "\n".join(lines) + "\n"


def main() -> int:
    parser = argparse.ArgumentParser(description="Supervised Q_φ on logged nutrition-loss change")
    parser.add_argument("--fit", type=Path, default=FIT_JSONL)
    parser.add_argument("--holdout", type=Path, default=HOLDOUT_JSONL)
    parser.add_argument("--out-dir", type=Path, default=OUT_DIR)
    parser.add_argument("--limit", type=int, default=None, help="Limit rows in each pool. Omit for the full pools.")
    parser.add_argument("--refresh-sigma", action="store_true")
    args = parser.parse_args()
    assert_safe_result_dir(args.out_dir)
    if args.limit is not None:
        print("warning: --limit is a smoke slice, not the P0c artifact", flush=True)

    before = _frozen_hashes()
    started = time.perf_counter()
    dish_ids = _dish_ids(args.fit) + _dish_ids(args.holdout)
    sigma_path = args.out_dir / "mass_sigma_rgb_train.json"
    sigma, backbone = _load_sigma(sigma_path, dish_ids, refresh=args.refresh_sigma)
    sigma_seconds = time.perf_counter() - started

    fit = load_q_rows(args.fit, expect_pool="fit", sigma_by_dish=sigma, limit=args.limit)
    hold = load_q_rows(args.holdout, expect_pool="holdout", sigma_by_dish=sigma, limit=args.limit)
    print(
        f"rows fit={fit.n_rows} holdout={hold.n_rows} "
        f"sigma_rows fit={fit.sigma_rows} holdout={hold.sigma_rows} "
        f"supervised fit={int(fit.supervise.sum())}",
        flush=True,
    )

    model = QPhi()
    fit_started = time.perf_counter()
    model.fit(fit.Z[fit.supervise], fit.y[fit.supervise].tolist(), fit.delta_l[fit.supervise].tolist())
    fit_seconds = time.perf_counter() - fit_started
    print(f"Q fit mse={model.train_mse_:.6f} seconds={fit_seconds:.1f}", flush=True)

    heuristic = HeuristicUtilityPolicy()
    q_policy = QCostPolicy(model)
    bc = load_closed_loop_policy(BC_WEIGHTS, BC_META)
    q_actions = q_policy.predict_actions(hold.Z)
    h_actions = heuristic.predict_actions(hold.Z)
    bc_actions = bc.predict_actions(hold.Z[:, STATE_OFFSET:])
    logged = [str(action) for action in hold.y.tolist()]

    fit_reg = _regression(model, fit)
    hold_reg = _regression(model, hold)
    # Train-set mean ΔL, frozen before looking at hold-out errors as a selector.
    train_mean = float(np.mean(fit.delta_l[fit.measured])) if bool(fit.measured.any()) else 0.0
    hold_target = hold.delta_l[hold.measured]
    baseline_mean_mae = float(np.mean(np.abs(hold_target - train_mean))) if hold_target.size else None

    selection = {
        "rule": (
            "a* = argmax value/Cost over actions the tools can execute. "
            "RE_PORTION requires a code. RETRIEVE and RE_DECOMPOSE require an open component. "
            "Heuristic value is the rule utility. Q value is Q_phi. "
            "BC is argmax of the saved state MLP and does not divide by cost."
        ),
        "costs": ACTION_COST,
        "uncertainty_majority": UNCERTAINTY_MAJORITY,
        "q_histogram": _hist(q_actions),
        "heuristic_histogram": _hist(h_actions),
        "bc_histogram": _hist(bc_actions),
        "logged_histogram": _hist(logged),
        "agreement_q_vs_heuristic": _agree(q_actions, h_actions),
        "agreement_q_vs_bc": _agree(q_actions, bc_actions),
        "agreement_heuristic_vs_bc": _agree(h_actions, bc_actions),
        "agreement_q_vs_logged": _agree(q_actions, logged),
        "agreement_heuristic_vs_logged": _agree(h_actions, logged),
        "agreement_bc_vs_logged": _agree(bc_actions, logged),
        "conditional_delta_l": {
            "q_over_cost": _conditional_delta(q_actions, hold),
            "heuristic": _conditional_delta(h_actions, hold),
            "state_bc": _conditional_delta(bc_actions, hold),
        },
    }
    stub = {
        "heuristic": _controller_stub(heuristic, label="heuristic"),
        "q_over_cost": _controller_stub(q_policy, label="q_over_cost"),
    }
    if stub["heuristic"]["actions"] != ["RETRIEVE", "RE_PORTION", "STOP"]:
        raise RuntimeError(f"heuristic stub did not take mass then stop: {stub['heuristic']['actions']}")
    if stub["heuristic"]["mass_calls"] != 1:
        raise RuntimeError(f"heuristic stub mass calls {stub['heuristic']['mass_calls']}")

    metrics: dict[str, Any] = {
        "task": "P0c U_* + Q_phi",
        "rl": False,
        "uses_rgb_test_gt": False,
        "uses_train_dish_labels_for_delta_l": True,
        "holdout_used_for_selection": False,
        "kcal_proxy_in_z": False,
        "volume_tool": False,
        "legacy_regression_weights": False,
        "rgb_only_mass": True,
        "depth_used": False,
        "not_p0d": NOT_P0D,
        "loss_definition": LOSS_DEFINITION,
        "z_dim": Z_DIM,
        "state_suffix_dim": Z_DIM - STATE_OFFSET,
        "z_names": list(Z_FEATURE_NAMES),
        "hidden": list(MLP_HIDDEN_SIZES),
        "epochs": Q_EPOCHS,
        "lr": Q_LR,
        "seed": Q_SEED,
        "unobserved_anchor": Q_UNOBSERVED_ANCHOR,
        "bc_epochs_reference": MLP_EPOCHS,
        "bc_lr_reference": MLP_LR,
        "bc_seed_reference": MLP_SEED,
        "mass_backbone": backbone,
        "sigma_dishes": len(sigma),
        "fit_path": str(args.fit),
        "holdout_path": str(args.holdout),
        "fit_rows": fit.n_rows,
        "fit_dishes": fit.n_dishes,
        "holdout_rows": hold.n_rows,
        "holdout_dishes": hold.n_dishes,
        "fit_sigma_rows": fit.sigma_rows,
        "holdout_sigma_rows": hold.sigma_rows,
        "fit_limit": args.limit,
        "fit_seconds": round(fit_seconds, 2),
        "sigma_seconds": round(sigma_seconds, 2),
        "train_mse": None if model.train_mse_ is None else round(model.train_mse_, 6),
        "action_counts_fit": model.action_counts,
        "costs": ACTION_COST,
        "holdout_mean_u": _mean_features(hold.Z),
        "fit_q_regression": fit_reg,
        "holdout_q_regression": hold_reg,
        "holdout_baseline_train_mean_mae": _round(baseline_mean_mae),
        "holdout_selection": selection,
        "controller_stub": stub,
        "bc_ablation": {
            "weights": str(BC_WEIGHTS.relative_to(ROOT)),
            "note": "Saved state-aware BC (#24). Not retrained. Scored on the 79-d suffix of z. Ablation only.",
        },
    }
    # JSON cannot store the raw agreement floats alongside the rounded report
    # without help; they are plain floats already.

    args.out_dir.mkdir(parents=True, exist_ok=True)
    metrics_path = args.out_dir / "metrics_holdout.json"
    report_path = args.out_dir / "report_holdout.txt"
    weights_path = args.out_dir / "q_phi.npz"
    meta_path = args.out_dir / "train_meta.json"
    blob = model.export_numpy()
    np.savez(
        weights_path,
        mean=blob["mean"],
        std=blob["std"],
        W1=blob["W1"],
        b1=blob["b1"],
        W2=blob["W2"],
        b2=blob["b2"],
        W3=blob["W3"],
        b3=blob["b3"],
        emit=blob["emit"],
    )
    reloaded = QPhi.from_npz(weights_path)
    if reloaded.predict_actions(hold.Z[:8]) != model.predict_actions(hold.Z[:8]):
        raise RuntimeError("reloaded Q_φ disagreed with the fit model")

    after = _frozen_hashes()
    if after != before:
        changed = [key for key in before if before[key] != after.get(key)]
        raise RuntimeError(f"frozen files changed: {changed}")
    metrics["frozen_sha256"] = {key: None if value is None else value[:16] for key, value in before.items()}

    metrics_path.write_text(json.dumps(metrics, indent=2) + "\n", encoding="utf-8")
    report_path.write_text(_report(metrics), encoding="utf-8")
    meta_path.write_text(
        json.dumps(
            {
                "weights": str(weights_path.relative_to(ROOT)),
                "metrics": str(metrics_path.relative_to(ROOT)),
                "actions": list(BC_ACTIONS),
                "costs": ACTION_COST,
                "input_dim": Z_DIM,
                "emit": [bool(value > 0.5) for value in blob["emit"]],
                "action_counts": model.action_counts,
                "train_mse": metrics["train_mse"],
                "rl": False,
                "epochs": Q_EPOCHS,
                "lr": Q_LR,
                "seed": Q_SEED,
                "unobserved_anchor": Q_UNOBSERVED_ANCHOR,
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    print(_report(metrics), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
