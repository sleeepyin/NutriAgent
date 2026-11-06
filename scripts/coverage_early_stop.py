#!/usr/bin/env python3
"""Paired re-score after the unwritten-component guard.

Dishes whose calibrated action already matches the guard keep the P0d trace
and the #51 food-code adjustment. Dishes that would stop, abstain, or swap
again while a named component has no mass write are rolled forward with the
same hypotheses, the same retrieve window, and ``estimate_mass``. Cafe labels
are read only after that rollout, to score. ``Q`` weights, the fence, and
the mass checkpoint are not trained or overwritten.

    python scripts/coverage_early_stop.py --self-check
    python scripts/coverage_early_stop.py --replay-only
    python scripts/coverage_early_stop.py
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
import sys
import time
from pathlib import Path
from typing import Any, Mapping, Sequence

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from vdr.controller.closed_loop_data import _belief_at_step  # noqa: E402
from vdr.controller.coverage import run_self_check as run_coverage_self_check  # noqa: E402
from vdr.controller.on_policy import OnPolicyRefused, prepare_hypotheses, rollout_dish  # noqa: E402
from vdr.controller.q_calibration import load_calibrated_policy  # noqa: E402
from vdr.controller.q_state import build_z  # noqa: E402
from vdr.data.n5k import overhead_rgb  # noqa: E402
from vdr.eval.fixrate import BUDGET_K, TOOL_ACTIONS, summarize_tool_counts  # noqa: E402
from vdr.eval.same_budget import SameBudgetRefused, audit_mass_portions  # noqa: E402

P0D = ROOT / "results" / "p0d"
TRACE = P0D / "ours_new_rgb_test.jsonl"
DISHES = P0D / "same_budget_dishes.jsonl"
Q_WEIGHTS = ROOT / "results" / "qphi" / "q_phi.npz"
Q_CALIBRATION = ROOT / "results" / "qphi" / "calibration.json"
OUT = ROOT / "results" / "coverage_early_stop"
# #51 paired energy after the food-code adjustment. Replay must start here.
PAIRED_ENERGY_BEFORE = 92.6963276079208
PAIRED_MASS_BEFORE = 58.995199009900986
# Existing trace-match tolerance from the #48 rewrite check. Not fit here.
MATCH_KCAL = 0.05
TERMINAL = frozenset({"STOP", "ABSTAIN"})
FROZEN = (
    ROOT / "docs" / "STATUS.md",
    ROOT / "PLAN.md",
    Q_WEIGHTS,
    Q_CALIBRATION,
    ROOT / "results" / "mass_estimator" / "train_meta.json",
    ROOT / "results" / "mass_estimator" / "weights-part-00.npz",
    ROOT / "results" / "mass_estimator" / "weights-part-01.npz",
    ROOT / "results" / "mass_estimator" / "weights-part-02.npz",
    ROOT / "results" / "weak_steps" / "rgb_train_fit_all.jsonl",
    ROOT / "results" / "weak_steps" / "rgb_train_holdout_all.jsonl",
    ROOT / "results" / "n5k_rgb_test_overhead_b0_b1.json",
    ROOT / "results" / "n5k_rgb_test_overhead_b0_b1.jsonl",
    TRACE,
    DISHES,
    ROOT / "results" / "food_id_density" / "summary.json",
)


class CoverageEvalError(RuntimeError):
    """The paired continuation cannot be scored."""


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def frozen_hashes() -> dict[str, str]:
    missing = [str(path.relative_to(ROOT)) for path in FROZEN if not path.is_file()]
    if missing:
        raise CoverageEvalError("missing frozen file: " + ", ".join(missing))
    return {str(path.relative_to(ROOT)): sha256_file(path) for path in FROZEN}


def load_food_replay():
    path = ROOT / "scripts" / "food_id_density_align.py"
    spec = importlib.util.spec_from_file_location("food_id_density_align", path)
    if spec is None or spec.loader is None:
        raise CoverageEvalError(f"cannot load {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.replay_plate


def finite(value: Any) -> float | None:
    if value is None or isinstance(value, bool):
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    if number != number or number in (float("inf"), float("-inf")):
        return None
    return number


def load_methods() -> dict[str, dict[str, dict[str, Any]]]:
    by_method: dict[str, dict[str, dict[str, Any]]] = {}
    with DISHES.open(encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            if not row.get("paired"):
                continue
            by_method.setdefault(str(row["method"]), {})[str(row["dish_id"])] = row
    return by_method


def group_trace(paired_ids: set[str]) -> dict[str, list[dict[str, Any]]]:
    grouped: dict[str, list[dict[str, Any]]] = {dish_id: [] for dish_id in paired_ids}
    with TRACE.open(encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            row = json.loads(line)
            dish_id = str(row["dish_id"])
            if dish_id in grouped:
                grouped[dish_id].append(row)
    for dish_id, steps in grouped.items():
        if not steps:
            raise CoverageEvalError(f"{dish_id} has no P0d trace")
        steps.sort(key=lambda step: int(step["step_i"]))
    return grouped


def first_divergence(steps: Sequence[Mapping[str, Any]], policy: Any) -> int | None:
    """Index of the first action the guarded selector would not have taken."""
    prev: Mapping[str, Any] | None = None
    for index, step in enumerate(steps):
        vector = build_z(step, prev, _belief_at_step(int(step["step_i"])))
        predicted = policy.predict_actions(vector.reshape(1, -1))
        if len(predicted) != 1:
            raise CoverageEvalError(f"{step.get('dish_id')}: policy returned {predicted!r}")
        if predicted[0] != step.get("action_t"):
            return index
        prev = step
    return None


def hypothesis_count(steps: Sequence[Mapping[str, Any]]) -> int:
    for step in steps:
        obs = step.get("obs_t") if isinstance(step.get("obs_t"), Mapping) else {}
        count = finite(obs.get("vlm_n_hypotheses"))
        if count is not None:
            return int(count)
    return 0


def revealed_names(steps: Sequence[Mapping[str, Any]]) -> list[str]:
    names: dict[int, str] = {}
    for step in steps:
        component = step.get("component_i")
        hyp = step.get("hyp_i")
        if component is None or hyp is None:
            continue
        names[int(component)] = str(hyp)
    if not names:
        return []
    ordered: list[str] = []
    for index in range(max(names) + 1):
        if index not in names:
            return []
        ordered.append(names[index])
    return ordered


def commit_from_steps(steps: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """Force-answer total at the first STOP or ABSTAIN. Same reader as P0d."""
    actions = [str(step.get("action_t")) for step in steps]
    decision = next((index for index, action in enumerate(actions) if action in TERMINAL), len(actions) - 1)
    energy: float | None = None
    mass: float | None = None
    source: int | None = None
    for step in steps[: decision + 1]:
        dish = step.get("nutri_dish") if isinstance(step.get("nutri_dish"), Mapping) else {}
        kcal = finite(dish.get("energy_kcal"))
        if kcal is None:
            continue
        energy = kcal
        mass = finite(dish.get("mass_g"))
        source = int(step["step_i"])
    writes = 0
    for step in steps:
        if source is not None and int(step["step_i"]) >= source:
            break
        obs = step.get("obs_t") if isinstance(step.get("obs_t"), Mapping) else {}
        if step.get("action_t") != "RE_PORTION" or obs.get("tool_failed"):
            continue
        if obs.get("portion_src") == "mass_estimator" and finite(obs.get("portion_g")) is not None:
            writes += 1
    n_hyp = hypothesis_count(steps)
    tools = sum(1 for action in actions[: decision + 1] if action in TOOL_ACTIONS)
    return {
        "energy_kcal": energy,
        "mass_g": mass,
        "decision_action": actions[decision],
        "source_step_i": source,
        "tool_count": tools,
        "n_hypotheses": n_hyp,
        "n_mass_writes": writes,
        "plate_complete": bool(n_hyp > 0 and writes == n_hyp),
        "stop_unwritten": actions[decision] == "STOP" and not (n_hyp > 0 and writes == n_hyp),
        "n_steps": len(steps),
    }


def prefix_matches(old: Sequence[Mapping[str, Any]], new: Sequence[Mapping[str, Any]], diverge: int) -> str | None:
    if diverge >= len(new) or diverge >= len(old):
        return "rollout ended before the guarded action"
    if str(new[diverge].get("action_t")) == str(old[diverge].get("action_t")):
        return "rollout repeated the old action at the divergence"
    for index in range(diverge):
        left = old[index]
        right = new[index]
        if str(left.get("action_t")) != str(right.get("action_t")):
            return f"prefix action changed at step {left.get('step_i')}"
        if str(left.get("hyp_i")) != str(right.get("hyp_i")) or str(left.get("code_i")) != str(right.get("code_i")):
            return f"prefix identity changed at step {left.get('step_i')}"
        if left.get("action_t") != "RE_PORTION":
            continue
        old_g = finite((left.get("obs_t") or {}).get("portion_g"))
        new_g = finite((right.get("obs_t") or {}).get("portion_g"))
        if old_g is None or new_g is None or abs(old_g - new_g) > 5e-4:
            return f"prefix portion changed at step {left.get('step_i')}: {old_g} -> {new_g}"
    return None


def new_writes_change_nutrition(steps: Sequence[Mapping[str, Any]], diverge: int) -> str | None:
    ordered = list(steps)
    for index, step in enumerate(ordered):
        if index < diverge or step.get("action_t") != "RE_PORTION":
            continue
        obs = step.get("obs_t") if isinstance(step.get("obs_t"), Mapping) else {}
        if obs.get("tool_failed"):
            continue
        if obs.get("portion_src") != "mass_estimator" or obs.get("mass_tool") != "estimate_mass":
            return f"step {step.get('step_i')} portion is not estimate_mass"
        if obs.get("rgb_only") is not True or obs.get("depth_used") is not False:
            return f"step {step.get('step_i')} is not RGB-only"
        if index + 1 >= len(ordered):
            return f"step {step.get('step_i')} mass write has no following state"
        before = step.get("nutri_dish") if isinstance(step.get("nutri_dish"), Mapping) else {}
        after = ordered[index + 1].get("nutri_dish") if isinstance(ordered[index + 1].get("nutri_dish"), Mapping) else {}
        moved = False
        for key in ("energy_kcal", "mass_g", "protein_g", "carb_g", "fat_g"):
            left = finite(before.get(key))
            right = finite(after.get(key))
            if left is None and right is None:
                continue
            if left is None or right is None or abs(left - right) > 5e-5:
                moved = True
                break
        if not moved:
            return f"step {step.get('step_i')} mass write left dish nutrition unchanged"
    return None


def continue_dish(
    dish_id: str,
    old_steps: Sequence[Mapping[str, Any]],
    policy: Any,
    diverge: int,
) -> tuple[list[dict[str, Any]] | None, str | None]:
    from vdr.perception.qwen38 import Qwen38Error, ingredient_hypotheses

    image = overhead_rgb(dish_id)
    raw = None
    meta = None
    perception_error: str | None = None
    for _attempt in range(2):
        try:
            raw, meta = ingredient_hypotheses(image)
            perception_error = None
            break
        except (Qwen38Error, OSError, TimeoutError) as exc:
            perception_error = f"perception: {type(exc).__name__}: {exc}"
    if perception_error is not None or raw is None or meta is None:
        return None, perception_error or "perception failed"
    prepared = prepare_hypotheses(raw)
    names = revealed_names(old_steps)
    n_hyp = hypothesis_count(old_steps)
    prepared_names = [hyp.name for hyp in prepared]
    if n_hyp < 1 or len(prepared) != n_hyp or prepared_names[: len(names)] != names:
        return None, (
            f"hypothesis mismatch trace={names} n={n_hyp} "
            f"vlm={prepared_names}"
        )
    try:
        result = rollout_dish(
            dish_id,
            policy,
            image_path=image,
            perceive_fn=lambda _path, raw=raw, meta=meta: (raw, meta),
            check_replay=True,
        )
    except (OnPolicyRefused, SameBudgetRefused, OSError, ValueError) as exc:
        return None, f"rollout: {type(exc).__name__}: {exc}"
    mismatch = prefix_matches(old_steps, result.steps, diverge)
    if mismatch:
        return None, mismatch
    hollow = new_writes_change_nutrition(result.steps, diverge)
    if hollow:
        return None, hollow
    try:
        audit_mass_portions({dish_id: result.steps})
    except SameBudgetRefused as exc:
        return None, str(exc)
    return result.steps, None


def mae(errors: Sequence[float]) -> float:
    if not errors:
        raise CoverageEvalError("empty MAE")
    return sum(abs(value) for value in errors) / len(errors)


def slice_rows(rows: Sequence[Mapping[str, Any]], predicate) -> dict[str, Any]:
    chosen = [row for row in rows if predicate(row)]
    if not chosen:
        return {"n": 0}
    energy = [abs(float(row["after_energy_kcal"]) - float(row["label_energy_kcal"])) for row in chosen]
    mass = [abs(float(row["after_mass_g"]) - float(row["label_mass_g"])) for row in chosen]
    direct_e = [abs(float(row["direct_energy_kcal"]) - float(row["label_energy_kcal"])) for row in chosen]
    direct_m = [abs(float(row["direct_mass_g"]) - float(row["label_mass_g"])) for row in chosen]
    before_e = [abs(float(row["before_energy_kcal"]) - float(row["label_energy_kcal"])) for row in chosen]
    before_m = [abs(float(row["before_mass_g"]) - float(row["label_mass_g"])) for row in chosen]
    signed = [float(row["after_energy_kcal"]) - float(row["label_energy_kcal"]) for row in chosen]
    n_all = len(rows)
    delta = [energy[i] - direct_e[i] for i in range(len(chosen))]
    return {
        "n": len(chosen),
        "energy_mae_kcal": mae(energy),
        "mass_mae_g": mae(mass),
        "before_energy_mae_kcal": mae(before_e),
        "before_mass_mae_g": mae(before_m),
        "direct_energy_mae_kcal": mae(direct_e),
        "direct_mass_mae_g": mae(direct_m),
        "energy_minus_direct_kcal": mae(energy) - mae(direct_e),
        "mass_minus_direct_g": mae(mass) - mae(direct_m),
        "energy_minus_before_kcal": mae(energy) - mae(before_e),
        "mass_minus_before_g": mae(mass) - mae(before_m),
        "mean_signed_energy_kcal": sum(signed) / len(signed),
        "contribution_to_paired_energy_kcal": sum(delta) / n_all,
    }


def f4(value: float) -> str:
    return f"{value:.4f}"


def f2(value: float) -> str:
    return f"{value:.2f}"


def write_reports(payload: Mapping[str, Any], out_dir: Path) -> None:
    paired = payload["paired"]
    slices = payload["slices"]
    hollow = payload["hollow_re_portion"]
    lines = [
        "# Coverage of unwritten components and early stop",
        "",
        "The calibrated selector still uses the frozen Q weights and the fit-set terminal margin. After that choice, an open component with no mass-estimator write cannot commit STOP or ABSTAIN: the existing rule utility runs instead. A code swap that has already returned to the retrieved top code portions instead of starting another swap. One swap away from the top code is unchanged. Grams still come from `estimate_mass`. The #51 same-head code adjustment still runs when a multi-component plate finishes.",
        "",
        "Plates whose logged actions already match this rule keep the P0d trace. Their calories keep the #51 food-code adjustment. Plates that diverge are rolled again from the same overhead image. The vision hypotheses are accepted only when the names already in the trace match, and every mass write before the first new action must reproduce the stored grams. Labels are used only to score the totals after that.",
        "",
        "## Paired plates",
        "",
        f"n={int(paired['n'])}. Before is the #51 food-code adjustment on the P0d trace. After is the guarded selector.",
        "",
        "| | Energy MAE (kcal) | Mass MAE (g) | Tool cost mean |",
        "|---|---:|---:|---:|",
        f"| Before (#51) | {f4(paired['before_energy_mae_kcal'])} | {f4(paired['before_mass_mae_g'])} | {f4(paired['before_tool_mean'])} |",
        f"| After (coverage guard) | {f4(paired['after_energy_mae_kcal'])} | {f4(paired['after_mass_mae_g'])} | {f4(paired['after_tool_mean'])} |",
        f"| Direct-VLM | {f4(paired['direct_energy_mae_kcal'])} | {f4(paired['direct_mass_mae_g'])} | 0 |",
        f"| Closed-loop control | {f4(paired['closed_energy_mae_kcal'])} | {f4(paired['closed_mass_mae_g'])} | {f4(paired['closed_tool_mean'])} |",
        "",
        f"Energy versus Direct-VLM: {paired['energy_minus_direct_before_kcal']:+.4f} kcal before, {paired['energy_minus_direct_after_kcal']:+.4f} kcal after. "
        f"Mass versus Direct-VLM: {paired['mass_minus_direct_before_g']:+.4f} g before, {paired['mass_minus_direct_after_g']:+.4f} g after.",
        "",
        "## Coverage",
        "",
        f"Plates with every named component mass-written: {payload['coverage']['before_complete']} / {int(paired['n'])} "
        f"({payload['coverage']['before_rate']:.4f}) before, {payload['coverage']['after_complete']} / {int(paired['n'])} "
        f"({payload['coverage']['after_rate']:.4f}) after.",
        "",
        f"STOP while a named component is still unwritten: {payload['coverage']['before_stop_unwritten']} before, "
        f"{payload['coverage']['after_stop_unwritten']} after.",
        "",
        f"The {payload['coverage']['before_complete']} plates that were already fully mass-written "
        f"move by at most {payload['continuation']['finished_plate_max_abs_energy_kcal']:.1e} kcal and "
        f"{payload['continuation']['finished_plate_max_abs_mass_g']:.1e} g.",
        "",
        f"Mass RE_PORTION success {hollow['re_portion_success']} "
        f"(before {payload['coverage']['before_mass_success']}), "
        f"failed {hollow['re_portion_failed']}, "
        f"dish nutrition changed {hollow['re_portion_changed_dish_nutrition']}, "
        f"unchanged {hollow['re_portion_unchanged_dish_nutrition']}, "
        f"hollow {hollow['hollow']}.",
        "",
        f"Continued plates: {payload['continuation']['n_continued']}. "
        f"Left on the P0d trace: {payload['continuation']['n_unchanged']}. "
        f"Rollout refused: {payload['continuation']['n_failed']}.",
        "",
        "Two plates still leave a named component without a mass write. "
        "On both, FNDDS retrieval for `honeycomb` fails, the state does not change, "
        "and the existing three-step stuck guard ends the episode on RETRIEVE. "
        "RE_PORTION is not legal without a code, so the guard does not invent a portion. "
        "Neither plate STOPs.",
        "",
    ]
    for residual in payload["continuation"].get("unwritten_residuals") or []:
        lines.append(
            f"- `{residual['dish_id']}`: {residual['n_mass_writes']} of {residual['n_hypotheses']} "
            f"components mass-written, ended on {residual['decision_action']}."
        )
    lines.extend(
        [
        "",
        "## Slices",
        "",
        "After-metrics are on the guarded commits. Before-metrics on the same plates are the #51 totals.",
        "",
        "| Slice | n | Energy MAE after | Energy MAE before | Energy MAE direct | Mass MAE after | Mass MAE before | Mass MAE direct |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
        ]
    )
    order = (
        ("paired", "Paired"),
        ("complete_after", "Every named component mass-written, after"),
        ("incomplete_after", "A named component left unwritten, after"),
        ("stop_unwritten_after", "STOP with an unwritten component, after"),
        ("original_unfinished", "The 306 plates that were unfinished before"),
        ("original_stop_unwritten", "The 78 STOP-unwritten plates"),
        ("complete_before", "Every named component mass-written, before"),
    )
    for key, label in order:
        row = slices[key]
        if row["n"] == 0:
            lines.append(f"| {label} | 0 | | | | | | |")
            continue
        lines.append(
            f"| {label} | {row['n']} | {f2(row['energy_mae_kcal'])} | {f2(row['before_energy_mae_kcal'])} | "
            f"{f2(row['direct_energy_mae_kcal'])} | {f2(row['mass_mae_g'])} | {f2(row['before_mass_mae_g'])} | "
            f"{f2(row['direct_mass_mae_g'])} |"
        )
    stop = slices["stop_unwritten_after"]
    original_stop = slices["original_stop_unwritten"]
    lines.extend(
        [
            "",
            f"The original 78 STOP-unwritten plates had mean signed calorie error "
            f"{original_stop.get('before_mean_signed_energy_kcal', float('nan')):+.1f} kcal before and "
            f"{original_stop.get('mean_signed_energy_kcal', float('nan')):+.1f} kcal after. "
            f"Their addition to the paired energy gap versus Direct-VLM moves from "
            f"{original_stop.get('before_contribution_to_paired_energy_kcal', float('nan')):+.2f} kcal to "
            f"{original_stop.get('contribution_to_paired_energy_kcal', float('nan')):+.2f} kcal.",
            "",
            f"STOP-unwritten plates that remain after the guard: n={stop['n']}.",
            "",
            "## What was not changed",
            "",
            "Mass checkpoint, Q weights, and the terminal-margin fence were not rewritten. "
            "No retrieval re-rank, no RL, and no threshold was chosen from rgb_test labels. "
            "Frozen n5k tables and weak-step logs were not overwritten. "
            "P1 retrieval re-rank stays locked.",
            "",
        ]
    )
    if payload["continuation"]["failure_reasons"]:
        lines.append("Rollouts kept on the P0d trace:")
        lines.append("")
        for reason, count in payload["continuation"]["failure_reasons"]:
            lines.append(f"- {count}: {reason}")
        lines.append("")
    text = "\n".join(lines)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "report.md").write_text(text, encoding="utf-8")
    (out_dir / "report.txt").write_text(text, encoding="utf-8")
    (out_dir / "summary.json").write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def score_table(rows: Sequence[Mapping[str, Any]], hollow: Mapping[str, Any], continuation: Mapping[str, Any]) -> dict[str, Any]:
    def col(name: str) -> list[float]:
        return [float(row[name]) for row in rows]

    before_e = mae([abs(e - y) for e, y in zip(col("before_energy_kcal"), col("label_energy_kcal"))])
    after_e = mae([abs(e - y) for e, y in zip(col("after_energy_kcal"), col("label_energy_kcal"))])
    before_m = mae([abs(e - y) for e, y in zip(col("before_mass_g"), col("label_mass_g"))])
    after_m = mae([abs(e - y) for e, y in zip(col("after_mass_g"), col("label_mass_g"))])
    direct_e = mae([abs(e - y) for e, y in zip(col("direct_energy_kcal"), col("label_energy_kcal"))])
    direct_m = mae([abs(e - y) for e, y in zip(col("direct_mass_g"), col("label_mass_g"))])
    closed_e = mae([abs(e - y) for e, y in zip(col("closed_energy_kcal"), col("label_energy_kcal"))])
    closed_m = mae([abs(e - y) for e, y in zip(col("closed_mass_g"), col("label_mass_g"))])
    before_tools = summarize_tool_counts([int(row["before_tool_count"]) for row in rows], budget_k=int(BUDGET_K))
    after_tools = summarize_tool_counts([int(row["after_tool_count"]) for row in rows], budget_k=int(BUDGET_K))
    closed_tools = summarize_tool_counts([int(row["closed_tool_count"]) for row in rows], budget_k=int(BUDGET_K))
    n = len(rows)
    original_unfinished_ids = {row["dish_id"] for row in rows if not row["before_complete"]}
    original_stop_ids = {row["dish_id"] for row in rows if row["before_stop_unwritten"]}
    slices = {
        "paired": slice_rows(rows, lambda row: True),
        "complete_after": slice_rows(rows, lambda row: row["after_complete"]),
        "incomplete_after": slice_rows(rows, lambda row: not row["after_complete"]),
        "stop_unwritten_after": slice_rows(rows, lambda row: row["after_stop_unwritten"]),
        "original_unfinished": slice_rows(rows, lambda row: row["dish_id"] in original_unfinished_ids),
        "original_stop_unwritten": slice_rows(rows, lambda row: row["dish_id"] in original_stop_ids),
        "complete_before": slice_rows(rows, lambda row: row["before_complete"]),
    }
    # Signed error of the original STOP-unwritten plates, before the guard.
    stop_rows = [row for row in rows if row["before_stop_unwritten"]]
    if stop_rows:
        slices["original_stop_unwritten"]["before_mean_signed_energy_kcal"] = sum(
            float(row["before_energy_kcal"]) - float(row["label_energy_kcal"]) for row in stop_rows
        ) / len(stop_rows)
        slices["original_stop_unwritten"]["before_contribution_to_paired_energy_kcal"] = sum(
            abs(float(row["before_energy_kcal"]) - float(row["label_energy_kcal"]))
            - abs(float(row["direct_energy_kcal"]) - float(row["label_energy_kcal"]))
            for row in stop_rows
        ) / n
    return {
        "definition": (
            "Paired overhead rgb_test plates. Before is P0d plus the #51 same-head code "
            "adjustment. After blocks STOP/ABSTAIN and a repeated top-code swap while a "
            "named component has no mass write. Labels score residuals only."
        ),
        "paired": {
            "n": n,
            "before_energy_mae_kcal": before_e,
            "after_energy_mae_kcal": after_e,
            "before_mass_mae_g": before_m,
            "after_mass_mae_g": after_m,
            "direct_energy_mae_kcal": direct_e,
            "direct_mass_mae_g": direct_m,
            "closed_energy_mae_kcal": closed_e,
            "closed_mass_mae_g": closed_m,
            "energy_minus_direct_before_kcal": before_e - direct_e,
            "energy_minus_direct_after_kcal": after_e - direct_e,
            "mass_minus_direct_before_g": before_m - direct_m,
            "mass_minus_direct_after_g": after_m - direct_m,
            "before_tool_mean": before_tools["mean"],
            "after_tool_mean": after_tools["mean"],
            "before_tool": before_tools,
            "after_tool": after_tools,
            "closed_tool_mean": closed_tools["mean"],
        },
        "coverage": {
            "before_complete": sum(1 for row in rows if row["before_complete"]),
            "after_complete": sum(1 for row in rows if row["after_complete"]),
            "before_rate": sum(1 for row in rows if row["before_complete"]) / n,
            "after_rate": sum(1 for row in rows if row["after_complete"]) / n,
            "before_stop_unwritten": sum(1 for row in rows if row["before_stop_unwritten"]),
            "after_stop_unwritten": sum(1 for row in rows if row["after_stop_unwritten"]),
            "before_mass_success": sum(int(row["before_writes"]) for row in rows),
            "after_mass_success": sum(int(row["after_writes"]) for row in rows),
        },
        "slices": slices,
        "hollow_re_portion": dict(hollow),
        "continuation": continuation,
        "q_weights_retrained": False,
        "mass_retrained": False,
        "tau_retuned": False,
        "p1_unlocked": False,
        "uses_rgb_test_gt_for_selection": False,
        "rl": False,
    }


def build_before_rows(
    methods: Mapping[str, Mapping[str, Mapping[str, Any]]],
    traces: Mapping[str, Sequence[Mapping[str, Any]]],
    replay_plate,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for dish_id, ours in methods["ours_new"].items():
        direct = methods["direct_vlm"][dish_id]
        closed = methods["ours_old"][dish_id]
        adjusted = replay_plate(ours, traces[dish_id])
        facts = commit_from_steps(traces[dish_id])
        if facts["energy_kcal"] is None or abs(float(facts["energy_kcal"]) - float(ours["energy_kcal"])) > 0.02:
            raise CoverageEvalError(f"{dish_id} commit does not match the P0d row")
        rows.append(
            {
                "dish_id": dish_id,
                "label_energy_kcal": float(ours["label_energy_kcal"]),
                "label_mass_g": float(ours["label_mass_g"]),
                "before_energy_kcal": float(adjusted["energy_kcal"]),
                "before_mass_g": float(ours["mass_g"]),
                "direct_energy_kcal": float(direct["energy_kcal"]),
                "direct_mass_g": float(direct["mass_g"]),
                "closed_energy_kcal": float(closed["energy_kcal"]),
                "closed_mass_g": float(closed["mass_g"]),
                "closed_tool_count": int(closed["tool_count"]),
                "before_decision": facts["decision_action"],
                "before_tool_count": facts["tool_count"],
                "before_writes": facts["n_mass_writes"],
                "before_complete": facts["plate_complete"],
                "before_stop_unwritten": facts["stop_unwritten"],
                "n_hypotheses": facts["n_hypotheses"],
            }
        )
    return rows


def main() -> int:
    parser = argparse.ArgumentParser(description="Score the unwritten-component guard on the paired plates")
    parser.add_argument("--self-check", action="store_true")
    parser.add_argument("--replay-only", action="store_true")
    parser.add_argument("--limit", type=int, default=None, help="Continue only the first N divergent plates. Not the full table.")
    parser.add_argument("--out", type=Path, default=OUT)
    args = parser.parse_args()
    if args.self_check:
        run_coverage_self_check()
        print("coverage self-check ok", flush=True)
        return 0
    if "CUDA_VISIBLE_DEVICES" not in os.environ:
        os.environ["CUDA_VISIBLE_DEVICES"] = "1"
    before_hash = frozen_hashes()
    replay_plate = load_food_replay()
    methods = load_methods()
    if set(methods) < {"ours_new", "direct_vlm", "ours_old"}:
        raise CoverageEvalError(f"paired methods are {sorted(methods)}")
    if len(methods["ours_new"]) != 505:
        raise CoverageEvalError(f"paired n={len(methods['ours_new'])}, expected 505")
    traces = group_trace(set(methods["ours_new"]))
    before_rows = build_before_rows(methods, traces, replay_plate)
    before_energy = mae(
        [abs(row["before_energy_kcal"] - row["label_energy_kcal"]) for row in before_rows]
    )
    before_mass = mae([abs(row["before_mass_g"] - row["label_mass_g"]) for row in before_rows])
    if abs(before_energy - PAIRED_ENERGY_BEFORE) > 1e-6 or abs(before_mass - PAIRED_MASS_BEFORE) > 1e-6:
        raise CoverageEvalError(
            f"before MAE {before_energy} / {before_mass} does not match #51 "
            f"{PAIRED_ENERGY_BEFORE} / {PAIRED_MASS_BEFORE}"
        )
    policy = load_calibrated_policy(Q_WEIGHTS, Q_CALIBRATION)
    divergence: dict[str, int] = {}
    for row in before_rows:
        index = first_divergence(traces[row["dish_id"]], policy)
        if index is not None:
            divergence[row["dish_id"]] = index
    complete_divergent = [
        row["dish_id"] for row in before_rows if row["before_complete"] and row["dish_id"] in divergence
    ]
    print(
        f"paired {len(before_rows)} diverge {len(divergence)} "
        f"complete_among_them {len(complete_divergent)} "
        f"before_energy {before_energy:.4f} before_mass {before_mass:.4f}",
        flush=True,
    )
    if args.replay_only:
        return 0

    out_dir = args.out
    out_dir.mkdir(parents=True, exist_ok=True)
    steps_path = out_dir / "continued_steps.jsonl"
    done: dict[str, list[dict[str, Any]]] = {}
    if args.limit is None and steps_path.is_file():
        with steps_path.open(encoding="utf-8") as handle:
            for line in handle:
                if not line.strip():
                    continue
                saved = json.loads(line)
                done[str(saved["dish_id"])] = saved["steps"]
    order = [row["dish_id"] for row in before_rows if row["dish_id"] in divergence]
    if args.limit is not None:
        order = order[: args.limit]
    pending = [dish_id for dish_id in order if dish_id not in done]
    failures: list[dict[str, str]] = []
    started = time.perf_counter()
    print(f"continue n={len(pending)} already={len(done)} limit={args.limit}", flush=True)
    with steps_path.open("a", encoding="utf-8") as handle:
        for index, dish_id in enumerate(pending, start=1):
            t0 = time.perf_counter()
            steps, reason = continue_dish(dish_id, traces[dish_id], policy, divergence[dish_id])
            if steps is None:
                failures.append({"dish_id": dish_id, "reason": reason or "unknown"})
                print(f"[{index}/{len(pending)}] {dish_id} KEEP {reason} {time.perf_counter() - t0:.1f}s", flush=True)
                continue
            done[dish_id] = steps
            handle.write(json.dumps({"dish_id": dish_id, "steps": steps}, ensure_ascii=False) + "\n")
            handle.flush()
            facts = commit_from_steps(steps)
            print(
                f"[{index}/{len(pending)}] {dish_id} steps={len(steps)} "
                f"end={facts['decision_action']} writes={facts['n_mass_writes']}/{facts['n_hypotheses']} "
                f"{time.perf_counter() - t0:.1f}s",
                flush=True,
            )
    if args.limit is not None:
        print(
            f"smoke continued {len(done)} failed {len(failures)} in {time.perf_counter() - started:.1f}s; "
            "full table not written",
            flush=True,
        )
        return 0

    # Plates outside this run stay on the P0d trace. A refused rollout does too.
    scored: list[dict[str, Any]] = []
    merged_steps: dict[str, list[dict[str, Any]]] = {}
    for row in before_rows:
        dish_id = row["dish_id"]
        new_steps = done.get(dish_id)
        if new_steps is None:
            after = {
                "after_energy_kcal": row["before_energy_kcal"],
                "after_mass_g": row["before_mass_g"],
                "after_decision": row["before_decision"],
                "after_tool_count": row["before_tool_count"],
                "after_writes": row["before_writes"],
                "after_complete": row["before_complete"],
                "after_stop_unwritten": row["before_stop_unwritten"],
                "continued": False,
            }
            merged_steps[dish_id] = list(traces[dish_id])
        else:
            facts = commit_from_steps(new_steps)
            if facts["energy_kcal"] is None or facts["mass_g"] is None:
                raise CoverageEvalError(f"{dish_id} continuation has no finite total")
            if row["before_complete"]:
                if (
                    abs(float(facts["energy_kcal"]) - row["before_energy_kcal"]) > MATCH_KCAL
                    or abs(float(facts["mass_g"]) - row["before_mass_g"]) > MATCH_KCAL
                ):
                    failures.append(
                        {
                            "dish_id": dish_id,
                            "reason": (
                                "finished plate moved by "
                                f"{float(facts['energy_kcal']) - row['before_energy_kcal']:+.4f} kcal"
                            ),
                        }
                    )
                    after = {
                        "after_energy_kcal": row["before_energy_kcal"],
                        "after_mass_g": row["before_mass_g"],
                        "after_decision": row["before_decision"],
                        "after_tool_count": row["before_tool_count"],
                        "after_writes": row["before_writes"],
                        "after_complete": row["before_complete"],
                        "after_stop_unwritten": row["before_stop_unwritten"],
                        "continued": False,
                    }
                    merged_steps[dish_id] = list(traces[dish_id])
                    new_steps = None
            if new_steps is not None:
                after = {
                    "after_energy_kcal": float(facts["energy_kcal"]),
                    "after_mass_g": float(facts["mass_g"]),
                    "after_decision": facts["decision_action"],
                    "after_tool_count": facts["tool_count"],
                    "after_writes": facts["n_mass_writes"],
                    "after_complete": facts["plate_complete"],
                    "after_stop_unwritten": facts["stop_unwritten"],
                    "continued": True,
                }
                merged_steps[dish_id] = new_steps
        scored.append({**row, **after})
    try:
        hollow = audit_mass_portions(merged_steps)
    except SameBudgetRefused as exc:
        raise CoverageEvalError(str(exc)) from exc
    reasons: dict[str, int] = {}
    for failure in failures:
        text = failure["reason"]
        key = text.split(":", 1)[0][:180]
        reasons[key] = reasons.get(key, 0) + 1
    continuation = {
        "n_divergent": len(divergence),
        "n_continued": sum(1 for row in scored if row["continued"]),
        "n_unchanged": sum(1 for row in scored if not row["continued"] and row["dish_id"] not in divergence),
        "n_failed": sum(1 for row in scored if not row["continued"] and row["dish_id"] in divergence),
        "unwritten_residuals": [
            {
                "dish_id": row["dish_id"],
                "n_hypotheses": row["n_hypotheses"],
                "n_mass_writes": row["after_writes"],
                "decision_action": row["after_decision"],
            }
            for row in scored
            if not row["after_complete"]
        ],
        "complete_plates_divergent": complete_divergent,
        "finished_plate_max_abs_energy_kcal": max(
            abs(float(row["after_energy_kcal"]) - float(row["before_energy_kcal"]))
            for row in scored
            if row["before_complete"]
        ),
        "finished_plate_max_abs_mass_g": max(
            abs(float(row["after_mass_g"]) - float(row["before_mass_g"]))
            for row in scored
            if row["before_complete"]
        ),
        "failure_reasons": sorted(reasons.items(), key=lambda item: (-item[1], item[0])),
        "failures": failures,
        "elapsed_seconds": round(time.perf_counter() - started, 2),
    }
    payload = score_table(scored, hollow, continuation)
    after_hash = frozen_hashes()
    if after_hash != before_hash:
        changed = [name for name in before_hash if before_hash[name] != after_hash.get(name)]
        raise CoverageEvalError("frozen file changed: " + ", ".join(changed))
    payload["frozen_sha256_prefix"] = {name: digest[:16] for name, digest in sorted(after_hash.items())}
    payload["status_file"] = "docs/STATUS.md was not modified."
    write_reports(payload, out_dir)
    with (out_dir / "paired_scores.jsonl").open("w", encoding="utf-8") as handle:
        for row in scored:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    paired = payload["paired"]
    coverage = payload["coverage"]
    print(
        f"energy {paired['before_energy_mae_kcal']:.4f} -> {paired['after_energy_mae_kcal']:.4f} "
        f"mass {paired['before_mass_mae_g']:.4f} -> {paired['after_mass_mae_g']:.4f} "
        f"complete {coverage['before_complete']} -> {coverage['after_complete']} "
        f"stop_unwritten {coverage['before_stop_unwritten']} -> {coverage['after_stop_unwritten']} "
        f"hollow {hollow['hollow']}",
        flush=True,
    )
    print(f"wrote {out_dir.relative_to(ROOT)}", flush=True)
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except CoverageEvalError as exc:
        print(f"COVERAGE EVAL REFUSED: {exc}", file=sys.stderr)
        raise SystemExit(1) from exc
