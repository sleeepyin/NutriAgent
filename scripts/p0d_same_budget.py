#!/usr/bin/env python3
"""P0d same-budget full set: mass estimator + calibrated Q/Cost vs the controls.

The learned Q weights and the terminal-margin fence are read and not trained.
rgb_test labels are used only after the rollout, to score. Frozen n5k tables,
weak-step logs, the closed-loop on-policy trace, and the Q checkpoint are not
overwritten.

    python scripts/p0d_same_budget.py --self-check
    python scripts/p0d_same_budget.py --controls-only
    python scripts/p0d_same_budget.py --limit 1
    python scripts/p0d_same_budget.py
    python scripts/p0d_same_budget.py --score-only

``--limit`` is an official-order prefix for a smoke. It does not write the
full-set table. The rollout needs torch and torchvision for the mass
estimator. Scoring the finished trace does not.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

# Controller package init must finish before vdr.eval, or the eval package
# import cycles through the uncertainty features.
import vdr.controller.features  # noqa: E402,F401
import numpy as np  # noqa: E402

from vdr.controller.bc_data import load_bc_rows  # noqa: E402
from vdr.controller.features import build_features  # noqa: E402
from vdr.controller.policy_mlp import PolicyMLP  # noqa: E402
from vdr.controller.q_calibration import load_calibrated_policy, run_self_check as run_q_self_check  # noqa: E402
from vdr.data.n5k import load_labels, overhead_rgb, read_ids  # noqa: E402
from vdr.eval.energy_selective import (  # noqa: E402
    PRIMARY_POPULATION,
    dish_readings,
    summarize_readings,
)
from vdr.eval.fixrate import BUDGET_K, TOOL_ACTIONS  # noqa: E402
from vdr.eval.same_budget import (  # noqa: E402
    DEFINITION,
    DEFINITION_ID,
    DISPLAY_NAMES,
    METHOD_ORDER,
    TERMINAL_ACTIONS,
    SameBudgetRefused,
    abstain_write_gap,
    action_histogram,
    audit_mass_portions,
    commit_direct_vlm,
    commit_from_actions,
    committed_kcal_scores,
    comparison_section,
    executed_actions,
    format_report,
    group_steps,
    metric_block,
    nutri_float,
    run_self_check as run_metric_self_check,
)
from vdr.weak.teacher import PROTECTED_RESULT_NAMES  # noqa: E402

OUT_DIR = ROOT / "results" / "p0d"
Q_WEIGHTS = ROOT / "results" / "qphi" / "q_phi.npz"
Q_CALIBRATION = ROOT / "results" / "qphi" / "calibration.json"
OURS_OLD_JSONL = ROOT / "results" / "closed_loop" / "on_policy_rgb_test.jsonl"
DIRECT_JSONL = ROOT / "results" / "n5k_rgb_test_overhead_b0_b1.jsonl"
FIT_JSONL = ROOT / "results" / "weak_steps" / "rgb_train_fit_all.jsonl"
HOLDOUT_JSONL = ROOT / "results" / "weak_steps" / "rgb_train_holdout_all.jsonl"
FULL_JSONL = OUT_DIR / "ours_new_rgb_test.jsonl"
FULL_ROLLOUT_SUMMARY = OUT_DIR / "ours_new_rgb_test_summary.json"
SMOKE_JSONL = OUT_DIR / "ours_new_rgb_test_smoke.jsonl"
SMOKE_SUMMARY = OUT_DIR / "ours_new_rgb_test_smoke_summary.json"
FAILURES = OUT_DIR / "ours_new_failures.jsonl"
TABLE_JSON = OUT_DIR / "same_budget.json"
TABLE_DISHES = OUT_DIR / "same_budget_dishes.jsonl"
TABLE_REPORT = OUT_DIR / "report_same_budget.txt"
# Published replay of the open-loop head on the train hold-out. A sanity
# check that this process refit the same head. Not a knob, and not fit on rgb_test.
OPEN_LOOP_HOLD_ROWS = 2771
OPEN_LOOP_HOLD_HITS = 2688
OPEN_LOOP_HIT_TOLERANCE = 15
OVERHEAD_POOL_N = 507
STOP_POPULATION_N = 488
STOP_ENERGY_MAE_4DP = "96.2769"
DIRECT_ENERGY_MAE_2DP = 97.03
DIRECT_MASS_MAE_2DP = 76.46
DOCUMENTED_PREFIXES = {
    "rgb_train_fit_all.jsonl": "ed47f9239bfecb3c",
    "n5k_rgb_test_overhead_b0_b1.jsonl": "a4dfdd82e3a595f4",
    "q_phi.npz": "a59ba6e49cf45220",
    "on_policy_rgb_test.jsonl": "dc24231294d88360",
    "policy_state_mlp.npz": "f0ac0b41f99cbd72",
}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _frozen_paths() -> list[Path]:
    names = [
        FIT_JSONL,
        HOLDOUT_JSONL,
        ROOT / "results" / "weak_steps" / "rgb_train_fit_all.summary.json",
        ROOT / "results" / "weak_steps" / "rgb_train_holdout_all.summary.json",
        Q_WEIGHTS,
        Q_CALIBRATION,
        ROOT / "results" / "qphi" / "metrics_holdout.json",
        ROOT / "results" / "qphi" / "calibration_holdout.json",
        OURS_OLD_JSONL,
        ROOT / "results" / "closed_loop" / "on_policy_rgb_test_summary.json",
        ROOT / "results" / "closed_loop" / "policy_state_mlp.npz",
        ROOT / "results" / "closed_loop" / "mae_aurc_testsplit.json",
        ROOT / "results" / "closed_loop" / "mae_aurc_holdout.json",
        ROOT / "results" / "closed_loop" / "abstain_calibration.json",
        DIRECT_JSONL,
        ROOT / "results" / "n5k_rgb_test_overhead_b0_b1.json",
        ROOT / "results" / "n5k_subset_b0_b1.jsonl",
        ROOT / "results" / "n5k_subset_b0_b1.json",
        ROOT / "results" / "n5k_subset_b0_b1_n50_seed42.jsonl",
        ROOT / "results" / "n5k_subset_b0_b1_n50_seed42.json",
        ROOT / "results" / "mass_estimator" / "train_meta.json",
    ]
    return [path for path in names if path.is_file()]


def _assert_frozen(before: dict[str, str] | None = None) -> dict[str, str]:
    paths = _frozen_paths()
    digests = {path.name: _sha256(path) for path in paths}
    for name, prefix in DOCUMENTED_PREFIXES.items():
        digest = digests.get(name)
        if digest is None or not digest.startswith(prefix):
            raise SameBudgetRefused(f"{name} sha256 does not start with {prefix}")
    if before is not None:
        changed = sorted(name for name, digest in before.items() if digests.get(name) != digest)
        if changed:
            raise SameBudgetRefused("frozen files changed: " + ", ".join(changed))
    return digests


def _assert_output(path: Path) -> None:
    if path.parent.resolve() != OUT_DIR.resolve():
        raise SameBudgetRefused(f"refusing to write outside {OUT_DIR}")
    if (
        path.name in PROTECTED_RESULT_NAMES
        or path.name.startswith("n5k_")
        or "holdout" in path.name
        or path.name in {"q_phi.npz", "calibration.json", "on_policy_rgb_test.jsonl"}
    ):
        raise SameBudgetRefused(f"refusing to write {path.name}")


def _overhead_ids() -> list[str]:
    return [dish_id for dish_id in read_ids("rgb_test") if overhead_rgb(dish_id).is_file()]


def _done_ids(path: Path) -> set[str]:
    if not path.is_file():
        return set()
    done: set[str] = set()
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        dish_id = row.get("dish_id")
        if isinstance(dish_id, str):
            done.add(dish_id)
    return done


def _append(path: Path, steps: list[dict]) -> None:
    _assert_output(path)
    with path.open("a", encoding="utf-8") as handle:
        for step in steps:
            handle.write(json.dumps(step, ensure_ascii=False) + "\n")
        handle.flush()


def _terminal_kcal(steps: list[dict]) -> float | None:
    nutri = steps[-1].get("nutri_dish") if steps else None
    if not isinstance(nutri, dict):
        return None
    value = nutri.get("energy_kcal")
    if value is None:
        return None
    return float(value)


def _load_direct(path: Path, labels: dict[str, dict[str, float]]) -> dict[str, dict | None]:
    predictions: dict[str, dict | None] = {}
    mismatch = 0
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            record = json.loads(line)
            dish_id = str(record["dish_id"])
            embedded = record.get("label") or {}
            cafe = labels[dish_id]
            for key in ("energy_kcal", "mass_g"):
                if abs(float(embedded[key]) - float(cafe[key])) > 1e-3:
                    mismatch += 1
            guess = record.get("b0") or {}
            nutri = guess.get("n") if guess.get("ok") and isinstance(guess.get("n"), dict) else None
            predictions[dish_id] = nutri
    if mismatch:
        raise SameBudgetRefused(f"frozen direct-guess file disagrees with cafe labels on {mismatch} fields")
    return predictions


def _fit_open_loop() -> tuple[PolicyMLP, int]:
    """Refit the published 31-d head in memory. Does not write a checkpoint."""
    print("refitting open-loop behavior cloning on the fit pool (checkpoint not written)", flush=True)
    started = time.perf_counter()
    fit = load_bc_rows(FIT_JSONL, expect_pool="fit")
    policy = PolicyMLP()
    policy.fit(fit.X, fit.y)
    hold = load_bc_rows(HOLDOUT_JSONL, expect_pool="holdout")
    if len(hold.y) != OPEN_LOOP_HOLD_ROWS:
        raise SameBudgetRefused(f"open-loop hold-out rows {len(hold.y)} != {OPEN_LOOP_HOLD_ROWS}")
    predicted = policy.predict_actions(hold.X)
    hits = sum(pred == str(label) for pred, label in zip(predicted, hold.y.tolist()))
    elapsed = time.perf_counter() - started
    print(
        f"open-loop hold-out hits {hits}/{OPEN_LOOP_HOLD_ROWS} "
        f"(published {OPEN_LOOP_HOLD_HITS}) in {elapsed:.1f}s",
        flush=True,
    )
    if abs(hits - OPEN_LOOP_HOLD_HITS) > OPEN_LOOP_HIT_TOLERANCE:
        raise SameBudgetRefused(
            f"open-loop hold-out hits {hits} are too far from published {OPEN_LOOP_HOLD_HITS}"
        )
    return policy, hits


def _predict_open_loop(
    policy: PolicyMLP,
    steps_by_dish: dict[str, list[dict]],
) -> tuple[dict[tuple[str, int], str], dict[tuple[str, int], float]]:
    keys: list[tuple[str, int]] = []
    vectors: list[np.ndarray] = []
    for dish_id, steps in steps_by_dish.items():
        for row in steps:
            keys.append((dish_id, int(row["step_i"])))
            vectors.append(build_features(row))
    matrix = np.vstack(vectors)
    actions = policy.predict_actions(matrix)
    proba = np.asarray(policy.predict_proba(matrix), dtype=np.float64)
    column = proba[:, list(policy.actions).index("ABSTAIN")]
    if len(actions) != len(keys) or not np.isfinite(column).all():
        raise SameBudgetRefused("open-loop prediction length or probability is unusable")
    return (
        {key: action for key, action in zip(keys, actions)},
        {key: float(score) for key, score in zip(keys, column)},
    )


def _stop_block(rows: dict, *, name: str) -> dict:
    chosen = [row for row in rows.values() if row.trace_grounded_stop]
    block = metric_block(chosen)
    if block["n_scored"] != STOP_POPULATION_N:
        raise SameBudgetRefused(f"{name} STOP scored {block['n_scored']} != {STOP_POPULATION_N}")
    if f"{block['full_set_energy_mae_kcal']:.4f}" != STOP_ENERGY_MAE_4DP:
        raise SameBudgetRefused(
            f"{name} STOP energy MAE {block['full_set_energy_mae_kcal']} "
            f"does not round to {STOP_ENERGY_MAE_4DP}"
        )
    if block["n_early_decision_scored"] != 0:
        raise SameBudgetRefused(f"{name} STOP subset has {block['n_early_decision_scored']} early decisions")
    return block


def _require_pool(steps: dict[str, list[dict]], pool: list[str], *, name: str) -> None:
    if set(steps) != set(pool):
        missing = sorted(set(pool) - set(steps))
        extra = sorted(set(steps) - set(pool))
        raise SameBudgetRefused(
            f"{name} dishes do not match the overhead pool "
            f"(missing {len(missing)}, extra {len(extra)})"
        )


def _prepare_controls() -> dict:
    """Read-only controls. Labels are loaded here and are not passed to a rollout."""
    frozen = _assert_frozen()
    if len(_overhead_ids()) != OVERHEAD_POOL_N:
        raise SameBudgetRefused(f"overhead pool is {len(_overhead_ids())}, expected {OVERHEAD_POOL_N}")
    pool = _overhead_ids()
    labels = load_labels()
    missing = [dish_id for dish_id in pool if dish_id not in labels]
    if missing:
        raise SameBudgetRefused("cafe label missing for " + ", ".join(missing[:5]))
    from vdr.controller.on_policy import assert_on_policy_marker

    ours_old = group_steps(OURS_OLD_JSONL)
    assert_on_policy_marker(ours_old)
    _require_pool(ours_old, pool, name="closed-loop control")
    label_kcal = {dish_id: float(labels[dish_id]["energy_kcal"]) for dish_id in pool}
    readings = dish_readings(ours_old, executed_actions(ours_old), committed_kcal_scores(ours_old), label_kcal)
    published = summarize_readings(readings, population=PRIMARY_POPULATION)
    if published["n_scored"] != STOP_POPULATION_N:
        raise SameBudgetRefused(f"energy reader STOP n={published['n_scored']}")
    if f"{published['full_set_mae_kcal']:.4f}" != STOP_ENERGY_MAE_4DP:
        raise SameBudgetRefused(
            f"energy reader STOP MAE {published['full_set_mae_kcal']} != {STOP_ENERGY_MAE_4DP}"
        )
    direct_predictions = _load_direct(DIRECT_JSONL, labels)
    if set(direct_predictions) != set(pool):
        raise SameBudgetRefused("direct vision-language file is not the overhead pool")
    direct = commit_direct_vlm(direct_predictions, labels)
    direct_block = metric_block(list(direct.values()))
    if direct_block["n_scored"] != OVERHEAD_POOL_N:
        raise SameBudgetRefused(f"direct vision-language scored {direct_block['n_scored']}")
    if round(float(direct_block["full_set_energy_mae_kcal"]), 2) != DIRECT_ENERGY_MAE_2DP:
        raise SameBudgetRefused(f"direct energy MAE {direct_block['full_set_energy_mae_kcal']}")
    if round(float(direct_block["full_set_mass_mae_g"]), 2) != DIRECT_MASS_MAE_2DP:
        raise SameBudgetRefused(f"direct mass MAE {direct_block['full_set_mass_mae_g']}")
    policy, hits = _fit_open_loop()
    open_actions, open_scores = _predict_open_loop(policy, ours_old)
    ours_old_commit = commit_from_actions(
        ours_old,
        executed_actions(ours_old),
        committed_kcal_scores(ours_old),
        labels,
        method="ours_old",
    )
    open_commit = commit_from_actions(
        ours_old,
        open_actions,
        open_scores,
        labels,
        method="open_loop_bc",
    )
    stop_old = _stop_block(ours_old_commit, name="closed-loop control")
    stop_open = _stop_block(open_commit, name="open-loop behavior cloning")
    if abs(float(stop_old["full_set_mass_mae_g"]) - float(stop_open["full_set_mass_mae_g"])) > 1e-6:
        raise SameBudgetRefused("open-loop and closed-loop STOP mass MAE diverged")
    if abs(float(stop_old["tool_cost_scored"]["mean"]) - float(stop_open["tool_cost_scored"]["mean"])) > 1e-6:
        raise SameBudgetRefused("open-loop and closed-loop STOP tool cost diverged")
    return {
        "frozen": frozen,
        "pool": pool,
        "labels": labels,
        "ours_old_steps": ours_old,
        "ours_old": ours_old_commit,
        "open_loop_bc": open_commit,
        "direct_vlm": direct,
        "open_loop_policy": policy,
        "open_loop_holdout_hits": hits,
        "stop_closed_loop": stop_old,
        "stop_open_loop": stop_open,
        "direct_own": direct_block,
        "published_stop_energy_mae_kcal": published["full_set_mae_kcal"],
    }


def _print_controls(controls: dict) -> None:
    stop = controls["stop_closed_loop"]
    own = metric_block(list(controls["ours_old"].values()))
    print(
        "closed-loop STOP "
        f"n={stop['n_scored']} energy MAE {stop['full_set_energy_mae_kcal']:.4f} "
        f"mass MAE {stop['full_set_mass_mae_g']:.4f} "
        f"tool cost mean {stop['tool_cost_scored']['mean']:.4f}",
        flush=True,
    )
    print(
        "closed-loop own full set "
        f"scored {own['n_scored']}/{own['n_rows']} "
        f"energy MAE {own['full_set_energy_mae_kcal']:.4f} "
        f"mass MAE {own['full_set_mass_mae_g']:.4f} "
        f"tool cost mean {own['tool_cost_scored']['mean']:.4f}",
        flush=True,
    )
    direct = controls["direct_own"]
    print(
        "direct vision-language model "
        f"n={direct['n_scored']} energy MAE {direct['full_set_energy_mae_kcal']:.4f} "
        f"mass MAE {direct['full_set_mass_mae_g']:.4f} tool cost 0",
        flush=True,
    )
    print(
        f"open-loop hold-out hits {controls['open_loop_holdout_hits']}/{OPEN_LOOP_HOLD_ROWS}",
        flush=True,
    )


def _dish_rows(by_method: dict, paired: set[str]) -> list[dict]:
    rows: list[dict] = []
    for name in METHOD_ORDER:
        for dish_id in sorted(by_method[name]):
            row = by_method[name][dish_id]
            energy_error = None if row.energy_kcal is None else abs(row.energy_kcal - row.label_energy_kcal)
            mass_error = None if row.mass_g is None else abs(row.mass_g - row.label_mass_g)
            rows.append(
                {
                    "method": name,
                    "name": DISPLAY_NAMES[name],
                    "dish_id": dish_id,
                    "scored": row.scored,
                    "paired": dish_id in paired and row.scored,
                    "energy_kcal": row.energy_kcal,
                    "mass_g": row.mass_g,
                    "label_energy_kcal": row.label_energy_kcal,
                    "label_mass_g": row.label_mass_g,
                    "energy_abs_error": energy_error,
                    "mass_abs_error": mass_error,
                    "tool_count": row.tool_count,
                    "decision_action": row.decision_action,
                    "decision_step_i": row.decision_step_i,
                    "source_step_i": row.source_step_i,
                    "early_decision": row.early_decision,
                    "abstain_score": row.abstain_score,
                }
            )
    return rows


def _write_table(payload: dict, dishes: list[dict]) -> None:
    for path in (TABLE_JSON, TABLE_DISHES, TABLE_REPORT):
        _assert_output(path)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    with TABLE_DISHES.open("w", encoding="utf-8") as handle:
        for row in dishes:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    payload["artifacts"]["dishes_sha256"] = _sha256(TABLE_DISHES)
    TABLE_JSON.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    TABLE_REPORT.write_text(format_report(payload), encoding="utf-8")


def score_table() -> dict:
    """Build the full table from the finished mass-estimator trace plus the frozen controls."""
    controls = _prepare_controls()
    if not FULL_JSONL.is_file():
        raise SameBudgetRefused(f"missing {FULL_JSONL.name}; roll out before scoring")
    from vdr.controller.on_policy import assert_on_policy_marker, assert_replay_matches

    ours_new_steps = group_steps(FULL_JSONL)
    assert_on_policy_marker(ours_new_steps)
    _require_pool(ours_new_steps, controls["pool"], name="mass estimator rollout")
    policy = load_calibrated_policy(Q_WEIGHTS, Q_CALIBRATION)
    for steps in ours_new_steps.values():
        assert_replay_matches(steps, policy)
    audit = audit_mass_portions(ours_new_steps)
    labels = controls["labels"]
    ours_new = commit_from_actions(
        ours_new_steps,
        executed_actions(ours_new_steps),
        committed_kcal_scores(ours_new_steps),
        labels,
        method="ours_new",
    )
    open_on_new_actions, open_on_new_scores = _predict_open_loop(controls["open_loop_policy"], ours_new_steps)
    open_on_new = commit_from_actions(
        ours_new_steps,
        open_on_new_actions,
        open_on_new_scores,
        labels,
        method="open_loop_bc",
    )
    by_method = {
        "ours_new": ours_new,
        "open_loop_bc": controls["open_loop_bc"],
        "direct_vlm": controls["direct_vlm"],
        "ours_old": controls["ours_old"],
    }
    section = comparison_section(by_method)
    calibration = json.loads(Q_CALIBRATION.read_text(encoding="utf-8"))
    if float(policy.terminal_margin) != float(calibration["tau"]):
        raise SameBudgetRefused("loaded fence does not match calibration.json")
    q_digest = _sha256(Q_WEIGHTS)
    if not q_digest.startswith(str(calibration["weights_sha256_prefix"])):
        raise SameBudgetRefused("Q weight hash does not match the calibration record")
    dishes = _dish_rows(by_method, set(section["paired_dish_ids"]))
    payload = {
        "definition_id": DEFINITION_ID,
        "definition": DEFINITION,
        "primary": "paired_overhead",
        "method_order": list(METHOD_ORDER),
        "budget_K": int(BUDGET_K),
        "tool_actions": sorted(TOOL_ACTIONS),
        "not_tools": sorted(TERMINAL_ACTIONS),
        "pool": {
            "split": "rgb_test",
            "n_official": len(read_ids("rgb_test")),
            "n_overhead": len(controls["pool"]),
            "image": "overhead rgb.png",
            "ids_without_image_not_imputed": len(read_ids("rgb_test")) - len(controls["pool"]),
        },
        "comparison": section,
        "auxiliary": {
            "coverage_and_aurc": "Auxiliary only. Not used to choose the fence, the mass net, or a threshold.",
            "open_loop_on_ours_new_trace": {
                "note": (
                    "Auxiliary. The same open-loop policy read on the mass-estimator trace. "
                    "Not a separate vision rollout. If it does not stop early, its committed "
                    "nutrition is the mass-estimator trace's nutrition. The primary open-loop "
                    "row is the read of the closed-loop control trace."
                ),
                "own": metric_block(list(open_on_new.values())),
                "n_early_decision_scored": metric_block(list(open_on_new.values()))["n_early_decision_scored"],
            },
            "closed_loop_stop_reproduction": {
                "n": controls["stop_closed_loop"]["n_scored"],
                "energy_mae_kcal": controls["stop_closed_loop"]["full_set_energy_mae_kcal"],
                "mass_mae_g": controls["stop_closed_loop"]["full_set_mass_mae_g"],
                "tool_cost_mean": controls["stop_closed_loop"]["tool_cost_scored"]["mean"],
                "published_energy_mae_4dp": STOP_ENERGY_MAE_4DP,
                "published_n": STOP_POPULATION_N,
                "energy_reader_mae_kcal": controls["published_stop_energy_mae_kcal"],
                "open_loop_stop_energy_mae_kcal": controls["stop_open_loop"]["full_set_energy_mae_kcal"],
                "open_loop_stop_early_decisions": controls["stop_open_loop"]["n_early_decision_scored"],
                "note": (
                    "The STOP pass/corrected subset of the closed-loop trace reproduces the "
                    "published 96.2769 kcal full-set energy MAE. It is not the paired overhead table "
                    "and it is not the n=507 direct vision-language anchor."
                ),
            },
        },
        "mass_portion_audit": audit,
        "abstain_write_gap": abstain_write_gap(ours_new_steps),
        "rollout_actions": action_histogram(ours_new_steps),
        "closed_loop_actions": action_histogram(controls["ours_old_steps"]),
        "q_fence": {
            "weights": "results/qphi/q_phi.npz",
            "weights_sha256": q_digest,
            "weights_sha256_prefix": q_digest[:16],
            "calibration": "results/qphi/calibration.json",
            "calibration_sha256_prefix": _sha256(Q_CALIBRATION)[:16],
            "rule": calibration.get("rule"),
            "tau": calibration.get("tau"),
            "multiplier": calibration.get("multiplier"),
            "weights_retrained": False,
            "holdout_used_for_selection": False,
        },
        "open_loop_replay": {
            "holdout_hits": controls["open_loop_holdout_hits"],
            "holdout_rows": OPEN_LOOP_HOLD_ROWS,
            "published_hits": OPEN_LOOP_HOLD_HITS,
            "checkpoint_written": False,
            "features": "31-d build_features; observations are not an input",
        },
        "artifacts": {
            "ours_new_trajectories": "results/p0d/ours_new_rgb_test.jsonl",
            "ours_new_trajectories_sha256": _sha256(FULL_JSONL),
            "table": "results/p0d/same_budget.json",
            "dishes": "results/p0d/same_budget_dishes.jsonl",
            "report": "results/p0d/report_same_budget.txt",
            "closed_loop_trace": "results/closed_loop/on_policy_rgb_test.jsonl",
            "direct_vlm_file": "results/n5k_rgb_test_overhead_b0_b1.jsonl",
        },
        "audit_flags": {
            "rl": False,
            "q_weights_retrained": False,
            "uses_rgb_test_gt_for_thresholds": False,
            "uses_rgb_test_labels_for_scoring_only": True,
            "volume_tool": False,
            "hollow_re_portion": False,
            "not_holdout_action_agreement": True,
            "not_faithful_dietai24_or_foodagent": True,
        },
        "frozen_sha256_prefix": {name: digest[:16] for name, digest in sorted(controls["frozen"].items())},
        "status_file": "docs/STATUS.md was not modified.",
        "owner": "营养论文代码编写",
    }
    # Drop the long id list from the human report's source? Keep it in JSON.
    _write_table(payload, dishes)
    _assert_frozen(controls["frozen"])
    print(f"wrote {TABLE_REPORT.relative_to(ROOT)}", flush=True)
    return payload


def _rollout_summary(
    *,
    jsonl: Path,
    attempted: list[str],
    failures: list[dict],
    vlm: dict,
    frozen: dict[str, str],
    elapsed: float,
    complete: bool,
    full_pool: bool,
    tau: float,
) -> dict:
    histogram = {"n_dishes": 0, "n_steps": 0, "action_counts": {}, "terminal_action_counts": {}}
    if jsonl.is_file() and jsonl.stat().st_size:
        histogram = action_histogram(group_steps(jsonl))
    return {
        "protocol": "p0d_mass_calibrated_q_cost",
        "split": "rgb_test",
        "full_pool": full_pool,
        "complete": complete,
        "n_official_rgb_test": len(read_ids("rgb_test")),
        "n_with_overhead_rgb": len(_overhead_ids()),
        "n_attempted_this_process": len(attempted),
        "n_failures": len(failures),
        "failures": failures,
        "trajectories": str(jsonl.relative_to(ROOT)),
        "trajectories_sha256": _sha256(jsonl) if jsonl.is_file() else None,
        "vlm": vlm,
        "q_weights": "results/qphi/q_phi.npz",
        "q_calibration": "results/qphi/calibration.json",
        "terminal_margin": tau,
        "q_weights_retrained": False,
        "mass_tool": "estimate_mass",
        "portion_src_on_re_portion": "mass_estimator",
        **histogram,
        "uses_test_gt": False,
        "rl": False,
        "volume_tool": False,
        "labels": "This file does not contain cafe labels and does not report MAE.",
        "elapsed_seconds": round(elapsed, 2),
        "frozen_sha256_prefix": {name: digest[:16] for name, digest in sorted(frozen.items())},
        "status_file": "docs/STATUS.md was not modified.",
        "owner": "营养论文代码编写",
    }


def rollout(*, limit: int | None, resume: bool) -> Path:
    """Roll the calibrated selector. Does not read cafe labels and does not train Q."""
    if "CUDA_VISIBLE_DEVICES" not in os.environ:
        os.environ["CUDA_VISIBLE_DEVICES"] = "1"
    try:
        import torch
    except ImportError as exc:
        raise SameBudgetRefused(
            "The mass estimator needs torch and torchvision in this interpreter."
        ) from exc
    from vdr.controller.on_policy import rollout_dish
    from vdr.perception.qwen38 import Qwen38Error, server_ready
    from vdr.tools.mass_estimator import load_mass_model

    frozen = _assert_frozen()
    policy = load_calibrated_policy(Q_WEIGHTS, Q_CALIBRATION)
    calibration = json.loads(Q_CALIBRATION.read_text(encoding="utf-8"))
    tau = float(calibration["tau"])
    if float(policy.terminal_margin) != tau:
        raise SameBudgetRefused("loaded fence does not match calibration.json")
    q_before = _sha256(Q_WEIGHTS)
    if not q_before.startswith(str(calibration["weights_sha256_prefix"])):
        raise SameBudgetRefused("Q weight hash does not match the calibration record")
    load_mass_model()
    vlm = server_ready()
    ids = _overhead_ids()
    if len(ids) != OVERHEAD_POOL_N and limit is None:
        raise SameBudgetRefused(f"overhead pool is {len(ids)}, expected {OVERHEAD_POOL_N}")
    if limit is not None:
        ids = ids[:limit]
    jsonl = SMOKE_JSONL if limit is not None else FULL_JSONL
    summary_path = SMOKE_SUMMARY if limit is not None else FULL_ROLLOUT_SUMMARY
    _assert_output(jsonl)
    _assert_output(summary_path)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    if jsonl.is_file() and not resume and limit is None:
        raise SameBudgetRefused(f"{jsonl.name} already exists; pass --resume to continue")
    if limit is not None and jsonl.is_file() and not resume:
        jsonl.unlink()
    done = _done_ids(jsonl) if resume else set()
    failures: list[dict] = []
    attempted: list[str] = []
    started = time.perf_counter()
    pending = [dish_id for dish_id in ids if dish_id not in done]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(
        f"p0d rollout n={len(pending)} skipped={len(done)} "
        f"model={vlm.get('model_alias')} mass_device={device} tau={tau}",
        flush=True,
    )
    for index, dish_id in enumerate(pending, start=1):
        image = overhead_rgb(dish_id)
        t0 = time.perf_counter()
        try:
            result = rollout_dish(dish_id, policy, image_path=image)
        except (SameBudgetRefused, Qwen38Error, OSError, ValueError) as exc:
            failures.append({"dish_id": dish_id, "error": f"{type(exc).__name__}: {exc}"})
            print(f"[{index}/{len(pending)}] {dish_id} FAIL {exc}", flush=True)
            if index == 1 and not done:
                raise
            continue
        _append(jsonl, result.steps)
        attempted.append(dish_id)
        portion = sum(1 for step in result.steps if step.get("action_t") == "RE_PORTION")
        print(
            f"[{index}/{len(pending)}] {dish_id} steps={len(result.steps)} "
            f"end={result.steps[-1]['action_t']} reason={result.ended_reason} "
            f"hyps={result.n_hypotheses} re_portion={portion} kcal={_terminal_kcal(result.steps)} "
            f"{time.perf_counter() - t0:.1f}s",
            flush=True,
        )
    if failures and limit is None:
        retry_ids = [row["dish_id"] for row in failures]
        failures = []
        print(f"retrying {len(retry_ids)} failed dishes once", flush=True)
        for index, dish_id in enumerate(retry_ids, start=1):
            if dish_id in _done_ids(jsonl):
                continue
            image = overhead_rgb(dish_id)
            try:
                result = rollout_dish(dish_id, policy, image_path=image)
            except (SameBudgetRefused, Qwen38Error, OSError, ValueError) as exc:
                failures.append({"dish_id": dish_id, "error": f"{type(exc).__name__}: {exc}"})
                print(f"[retry {index}/{len(retry_ids)}] {dish_id} FAIL {exc}", flush=True)
                continue
            _append(jsonl, result.steps)
            attempted.append(dish_id)
            print(
                f"[retry {index}/{len(retry_ids)}] {dish_id} steps={len(result.steps)} "
                f"end={result.steps[-1]['action_t']}",
                flush=True,
            )
    if _sha256(Q_WEIGHTS) != q_before:
        raise SameBudgetRefused("Q weights changed during the rollout")
    frozen_after = _assert_frozen(frozen)
    payload = _rollout_summary(
        jsonl=jsonl,
        attempted=attempted,
        failures=failures,
        vlm=vlm,
        frozen=frozen_after,
        elapsed=time.perf_counter() - started,
        complete=not pending or (len(attempted) + len(failures) == len(pending) and not failures),
        full_pool=limit is None and len(ids) == OVERHEAD_POOL_N,
        tau=tau,
    )
    _assert_output(summary_path)
    summary_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    if failures and limit is None:
        _assert_output(FAILURES)
        with FAILURES.open("a", encoding="utf-8") as handle:
            for row in failures:
                handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    print(f"wrote {summary_path.relative_to(ROOT)} failures={len(failures)}", flush=True)
    if failures:
        raise SameBudgetRefused(f"{len(failures)} dishes failed; not scoring a partial pool")
    return jsonl


def main() -> int:
    parser = argparse.ArgumentParser(description="P0d same-budget full-set evaluation")
    parser.add_argument("--self-check", action="store_true")
    parser.add_argument("--controls-only", action="store_true", help="Score frozen controls. Writes nothing.")
    parser.add_argument("--limit", type=int, default=None, help="Official-order smoke prefix. Not the full set.")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--score-only", action="store_true")
    parser.add_argument("--no-score", action="store_true")
    args = parser.parse_args()
    if args.self_check:
        run_metric_self_check()
        run_q_self_check()
        print("p0d self-check ok", flush=True)
        return 0
    if args.limit is not None and args.limit < 1:
        raise SameBudgetRefused("--limit must be >= 1")
    if args.score_only and (args.limit is not None or args.controls_only):
        raise SameBudgetRefused("--score-only does not combine with --limit or --controls-only")
    if args.controls_only:
        _print_controls(_prepare_controls())
        _assert_frozen()
        print("controls checked; main table not written", flush=True)
        return 0
    if not args.score_only:
        rollout(limit=args.limit, resume=args.resume)
    if args.limit is not None:
        print("smoke prefix was not scored as the full-set table", flush=True)
        return 0
    if args.no_score:
        print("scoring skipped by --no-score", flush=True)
        return 0
    score_table()
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except SameBudgetRefused as exc:
        print(f"P0D REFUSED: {exc}", file=sys.stderr)
        raise SystemExit(1) from exc
