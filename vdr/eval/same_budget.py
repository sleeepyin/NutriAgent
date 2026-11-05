"""Same-budget full-set energy, mass, and tool cost.

The comparison is four committed answers on the overhead ``rgb_test`` dishes:

* Mass Estimator + calibrated Q/Cost selection. Its own rollout. ``RE_PORTION``
  calls ``estimate_mass``. The action is the #44 fence on the saved ``Q``
  weights. This module does not train ``Q`` and does not read the weights.
* Open-loop behavior cloning. A counterfactual read of the closed-loop
  control trace with the 31-d policy. Not a second vision rollout and not a
  mass-estimator call.
* Direct vision-language model. The frozen single-shot predictions. No
  specialist tools.
* Closed-loop control. The frozen on-policy trace (no mass specialist).

Full-set MAE is the mean absolute error on dishes with a finite committed
energy and a finite committed mass. The committed total is the latest finite
``nutri_dish`` at or before the first predicted stop or abstain, matching
``energy_selective`` force-answer. Dishes with no finite total are unscored
and are not imputed. Coverage and AURC are auxiliary. Cafe labels are an
input to scoring only; this module does not open a metadata file and does
not fit a threshold.

Tool cost counts ``RETRIEVE``, ``CORRECT_CODE``, ``RE_PORTION``, and
``RE_DECOMPOSE`` on the steps up to and including the decision. ``STOP`` and
``ABSTAIN`` are not tools. The cap is ``K = 32``. That count is not the
``Q/Cost`` denominator.
"""

from __future__ import annotations

import math
import statistics
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

from vdr.controller.policy_bc import BC_ACTIONS
from vdr.eval.abstain_calibration import score_committed_kcal
from vdr.eval.fixrate import BUDGET_K, TOOL_ACTIONS, summarize_tool_counts
from vdr.eval.selective import aurc, coverage_and_selective_mae
from vdr.eval.tool_state_audit import EPSILON_MASS_G
from vdr.ir.io import read_ir_jsonl

DEFINITION_ID = "p0d_same_budget_overhead_force_answer"
DEFINITION = (
    "Same-budget full set on official rgb_test dishes that have an overhead "
    "rgb.png. Each method commits one energy and one mass. The decision step "
    "is the first predicted STOP or ABSTAIN, or the last step if the method "
    "never predicts a terminal action. Committed nutrition is the latest finite "
    "nutri_dish at or before that step. Later totals are not credited. A dish "
    "missing energy or mass is unscored and is not imputed as zero. The paired "
    "table keeps only dishes scored by every method. Cafe labels score the "
    "error and do not choose actions, codes, grams, or thresholds. Tool cost "
    "counts RETRIEVE, CORRECT_CODE, RE_PORTION, and RE_DECOMPOSE through the "
    "decision step. STOP and ABSTAIN are not tools. The step cap is K=32. "
    "Coverage and AURC are auxiliary. This is not the hold-out action-agreement "
    "table and not a reinforcement-learning run."
)
METHOD_ORDER: tuple[str, ...] = (
    "ours_new",
    "open_loop_bc",
    "direct_vlm",
    "ours_old",
)
DISPLAY_NAMES: dict[str, str] = {
    "ours_new": "Mass Estimator + calibrated Q/Cost selection",
    "open_loop_bc": "Open-loop behavior cloning",
    "direct_vlm": "Direct vision-language model",
    "ours_old": "Closed-loop control",
}
METHOD_ROLES: dict[str, str] = {
    "ours_new": (
        "Own rollout. Local Qwen3.8-27B-VL writes component hypotheses. "
        "FNDDS retrieval and nutrition lookup write the code. RE_PORTION calls "
        "estimate_mass and is not a gram-grid rescore. The action is argmax "
        "Q/Cost with the saved terminal-margin fence. Learned Q weights are unchanged."
    ),
    "open_loop_bc": (
        "31-d open-loop policy read on the closed-loop control trace. Not a "
        "second vision rollout and not a mass-estimator call. The committed "
        "total is the force-answer nutrition at that policy's first stop or abstain."
    ),
    "direct_vlm": (
        "Frozen single-shot nutrition guess with no specialist tools. "
        "Historical energy MAE 97.03 and mass MAE 76.46 on this overhead pool "
        "are the anchor. This row is recomputed from that frozen file."
    ),
    "ours_old": (
        "Frozen on-policy closed-loop controller. No mass specialist. "
        "RE_PORTION on that trace is the earlier portion guess, not estimate_mass."
    ),
}
TERMINAL_ACTIONS = frozenset({"STOP", "ABSTAIN"})
GROUNDED_STATUS = frozenset({"pass", "corrected"})
_FORBIDDEN_KEYS = frozenset(
    {
        "train_gt_energy_kcal",
        "train_gt",
        "y_true",
        "label_kcal",
        "energy_label",
        "cafe_label",
    }
)
_NUTRI_KEYS = ("energy_kcal", "mass_g")

if set(TOOL_ACTIONS) != {"RETRIEVE", "CORRECT_CODE", "RE_PORTION", "RE_DECOMPOSE"}:
    raise RuntimeError(f"tool-cost actions drifted: {sorted(TOOL_ACTIONS)}")
if int(BUDGET_K) != 32:
    raise RuntimeError(f"budget K drifted from 32: {BUDGET_K}")
if set(TERMINAL_ACTIONS) | set(TOOL_ACTIONS) != set(BC_ACTIONS):
    raise RuntimeError("terminal and tool actions do not cover the controller set")


class SameBudgetRefused(ValueError):
    """The table would not be the same-budget full set."""


@dataclass(frozen=True)
class Commitment:
    """One dish, one method. ``None`` nutrition means unscored, not zero."""

    method: str
    dish_id: str
    energy_kcal: float | None
    mass_g: float | None
    label_energy_kcal: float
    label_mass_g: float
    decision_action: str
    decision_step_i: int
    source_step_i: int | None
    early_decision: bool
    tool_count: int
    abstain_score: float | None
    trace_terminal_action: str
    trace_grounded_stop: bool

    @property
    def scored(self) -> bool:
        return self.energy_kcal is not None and self.mass_g is not None


def _walk_keys(value: Any, found: set[str]) -> None:
    if isinstance(value, Mapping):
        for key, item in value.items():
            if str(key) in _FORBIDDEN_KEYS:
                found.add(str(key))
            _walk_keys(item, found)
    elif isinstance(value, list):
        for item in value:
            _walk_keys(item, found)


def group_steps(path: Path | str) -> dict[str, list[dict[str, Any]]]:
    """Group an IR jsonl by dish. Rows are sorted by ``step_i``.

    Label keys are refused. The file is not written.
    """
    grouped: dict[str, list[dict[str, Any]]] = {}
    for record in read_ir_jsonl(path):
        dish_id = record.get("dish_id")
        if not isinstance(dish_id, str) or not dish_id:
            raise SameBudgetRefused(f"{path}: row missing dish_id")
        action = record.get("action_t")
        if action == "VOLUME_ESTIMATE" or action not in BC_ACTIONS:
            raise SameBudgetRefused(f"{dish_id}: action {action!r} is outside the controller set")
        found: set[str] = set()
        _walk_keys(record, found)
        if found:
            raise SameBudgetRefused(f"{dish_id}: trajectory contains label keys: {sorted(found)}")
        grouped.setdefault(dish_id, []).append(record)
    if not grouped:
        raise SameBudgetRefused(f"{path}: no rows")
    ordered: dict[str, list[dict[str, Any]]] = {}
    for dish_id in sorted(grouped):
        rows = sorted(grouped[dish_id], key=lambda row: int(row["step_i"]))
        seen: list[int] = []
        for row in rows:
            step_i = row.get("step_i")
            if not isinstance(step_i, int) or isinstance(step_i, bool) or step_i < 0:
                raise SameBudgetRefused(f"{dish_id}: invalid step_i {step_i!r}")
            seen.append(step_i)
        if len(seen) != len(set(seen)):
            raise SameBudgetRefused(f"{dish_id}: duplicate step_i")
        ordered[dish_id] = rows
    return ordered


def nutri_float(nutri: Any, key: str) -> float | None:
    """Finite non-negative nutrition field, or ``None`` when the field is absent."""
    if not isinstance(nutri, Mapping):
        return None
    value = nutri.get(key)
    if value is None:
        return None
    try:
        number = float(value)
    except (TypeError, ValueError) as exc:
        raise SameBudgetRefused(f"{key} {value!r} is not a float") from exc
    if not math.isfinite(number):
        return None
    if number < -1e-3:
        raise SameBudgetRefused(f"{key} is negative ({number})")
    if number < 0.0:
        number = 0.0
    return number


def _label_pair(labels: Mapping[str, Mapping[str, float]], dish_id: str) -> tuple[float, float]:
    row = labels.get(dish_id)
    if not isinstance(row, Mapping):
        raise SameBudgetRefused(f"{dish_id}: missing cafe label")
    energy = nutri_float(row, "energy_kcal")
    mass = nutri_float(row, "mass_g")
    if energy is None or mass is None:
        raise SameBudgetRefused(f"{dish_id}: cafe label is missing energy or mass")
    return energy, mass


def _decision_index(actions: Sequence[str]) -> int:
    for index, action in enumerate(actions):
        if action in TERMINAL_ACTIONS:
            return index
    return len(actions) - 1


def commit_from_actions(
    steps_by_dish: Mapping[str, Sequence[Mapping[str, Any]]],
    actions: Mapping[tuple[str, int], str],
    scores: Mapping[tuple[str, int], float | None],
    labels: Mapping[str, Mapping[str, float]],
    *,
    method: str,
) -> dict[str, Commitment]:
    """Force-answer commitments. ``actions`` selects the decision on the logged states."""
    if method not in DISPLAY_NAMES:
        raise SameBudgetRefused(f"unknown method {method!r}")
    commitments: dict[str, Commitment] = {}
    for dish_id, steps in steps_by_dish.items():
        ordered = list(steps)
        if not ordered:
            raise SameBudgetRefused(f"{dish_id}: empty trajectory")
        chosen_actions: list[str] = []
        for row in ordered:
            key = (dish_id, int(row["step_i"]))
            action = actions.get(key)
            if action not in BC_ACTIONS or action == "VOLUME_ESTIMATE":
                raise SameBudgetRefused(f"{dish_id} step {row['step_i']}: prediction {action!r}")
            chosen_actions.append(str(action))
        decision = _decision_index(chosen_actions)
        energy: float | None = None
        mass: float | None = None
        source_step: int | None = None
        for row in ordered[: decision + 1]:
            kcal = nutri_float(row.get("nutri_dish"), "energy_kcal")
            if kcal is None:
                continue
            energy = kcal
            mass = nutri_float(row.get("nutri_dish"), "mass_g")
            source_step = int(row["step_i"])
        tool_count = sum(1 for action in chosen_actions[: decision + 1] if action in TOOL_ACTIONS)
        score_key = (dish_id, int(ordered[decision]["step_i"]))
        if score_key not in scores:
            raise SameBudgetRefused(f"{dish_id}: missing abstain score at the decision")
        score = scores[score_key]
        if score is not None:
            score = float(score)
            if not math.isfinite(score) or score < -1e-8 or score > 1.0 + 1e-6:
                raise SameBudgetRefused(f"{dish_id}: abstain score {score} is outside [0, 1]")
        last = ordered[-1]
        terminal_energy = nutri_float(last.get("nutri_dish"), "energy_kcal")
        status = last.get("s")
        grounded = (
            last.get("action_t") == "STOP"
            and status in GROUNDED_STATUS
            and terminal_energy is not None
        )
        label_energy, label_mass = _label_pair(labels, dish_id)
        commitments[dish_id] = Commitment(
            method=method,
            dish_id=dish_id,
            energy_kcal=energy,
            mass_g=mass,
            label_energy_kcal=label_energy,
            label_mass_g=label_mass,
            decision_action=chosen_actions[decision],
            decision_step_i=int(ordered[decision]["step_i"]),
            source_step_i=source_step,
            early_decision=decision < len(ordered) - 1,
            tool_count=tool_count,
            abstain_score=score,
            trace_terminal_action=str(last.get("action_t")),
            trace_grounded_stop=grounded,
        )
    return commitments


def commit_direct_vlm(
    predictions: Mapping[str, Mapping[str, Any] | None],
    labels: Mapping[str, Mapping[str, float]],
    *,
    method: str = "direct_vlm",
) -> dict[str, Commitment]:
    """One frozen guess per dish. Tool cost is 0. There is no abstain score."""
    if method != "direct_vlm":
        raise SameBudgetRefused("direct predictions are only the direct vision-language model")
    commitments: dict[str, Commitment] = {}
    for dish_id, nutri in predictions.items():
        label_energy, label_mass = _label_pair(labels, dish_id)
        energy = None if nutri is None else nutri_float(nutri, "energy_kcal")
        mass = None if nutri is None else nutri_float(nutri, "mass_g")
        commitments[dish_id] = Commitment(
            method=method,
            dish_id=dish_id,
            energy_kcal=energy,
            mass_g=mass,
            label_energy_kcal=label_energy,
            label_mass_g=label_mass,
            decision_action="STOP" if energy is not None and mass is not None else "ABSTAIN",
            decision_step_i=0,
            source_step_i=0 if energy is not None else None,
            early_decision=False,
            tool_count=0,
            abstain_score=None,
            trace_terminal_action="STOP",
            trace_grounded_stop=energy is not None and mass is not None,
        )
    return commitments


def executed_actions(
    steps_by_dish: Mapping[str, Sequence[Mapping[str, Any]]],
) -> dict[tuple[str, int], str]:
    """The action written on each IR row."""
    actions: dict[tuple[str, int], str] = {}
    for dish_id, steps in steps_by_dish.items():
        for row in steps:
            actions[(dish_id, int(row["step_i"]))] = str(row["action_t"])
    return actions


def committed_kcal_scores(
    steps_by_dish: Mapping[str, Sequence[Mapping[str, Any]]],
) -> dict[tuple[str, int], float]:
    """Locked abstain score: clip(running finite kcal / 2500, 0, 1). Not refit."""
    scores: dict[tuple[str, int], float] = {}
    for dish_id, steps in steps_by_dish.items():
        running: float | None = None
        for row in steps:
            kcal = nutri_float(row.get("nutri_dish"), "energy_kcal")
            if kcal is not None:
                running = kcal
            scores[(dish_id, int(row["step_i"]))] = score_committed_kcal(running)
    return scores


def _mean(values: Sequence[float]) -> float | None:
    if not values:
        return None
    return float(sum(values) / len(values))


def metric_block(rows: Sequence[Commitment]) -> dict[str, Any]:
    """MAE, tool cost, and auxiliary coverage / AURC for one set of dishes.

    Rows must already be the population of interest. MAE uses scored rows
    only. Tool cost is reported on scored rows and on every row in ``rows``.
    """
    ordered = sorted(rows, key=lambda row: row.dish_id)
    scored = [row for row in ordered if row.scored]
    unscored = [row.dish_id for row in ordered if not row.scored]
    energy_err = [abs(float(row.energy_kcal) - row.label_energy_kcal) for row in scored if row.energy_kcal is not None]
    mass_err = [abs(float(row.mass_g) - row.label_mass_g) for row in scored if row.mass_g is not None]
    flags = [row.decision_action == "ABSTAIN" for row in scored]
    coverage: float | None = None
    selective: float | None = None
    n_kept: int | None = None
    aurc_value: float | None = None
    if scored:
        kept = coverage_and_selective_mae(energy_err, abstain_flags=flags)
        coverage = float(kept["coverage"])
        n_kept = int(kept["n_kept"])
        selective = None if n_kept == 0 else float(kept["selective_mae"])
        if all(row.abstain_score is not None for row in scored):
            curve = aurc(
                energy_err,
                [float(row.abstain_score) for row in scored if row.abstain_score is not None],
            )
            aurc_value = float(curve["aurc"])
            energy_mae = _mean(energy_err)
            if energy_mae is not None and abs(float(curve["risk_at_full_coverage"]) - energy_mae) > 1e-6:
                raise SameBudgetRefused("AURC full-coverage risk disagrees with energy MAE")
    decisions = Counter(row.decision_action for row in scored)
    return {
        "n_rows": len(ordered),
        "n_scored": len(scored),
        "n_unscored": len(unscored),
        "unscored_dish_ids": unscored,
        "full_set_energy_mae_kcal": _mean(energy_err),
        "full_set_mass_mae_g": _mean(mass_err),
        "median_energy_abs_error_kcal": float(statistics.median(energy_err)) if energy_err else None,
        "median_mass_abs_error_g": float(statistics.median(mass_err)) if mass_err else None,
        "n_early_decision_scored": sum(1 for row in scored if row.early_decision),
        "n_abstain_scored": sum(1 for flag in flags if flag),
        "n_kept": n_kept,
        "coverage": coverage,
        "selective_energy_mae_kcal": selective,
        "aurc": aurc_value,
        "aurc_lower_is_better": True,
        "aurc_role": "auxiliary",
        "decision_actions_scored": dict(sorted(decisions.items())),
        "tool_cost_scored": summarize_tool_counts([row.tool_count for row in scored], budget_k=int(BUDGET_K)),
        "tool_cost_all_rows": summarize_tool_counts([row.tool_count for row in ordered], budget_k=int(BUDGET_K)),
    }


def scored_ids(rows: Mapping[str, Commitment]) -> set[str]:
    return {dish_id for dish_id, row in rows.items() if row.scored}


def comparison_section(
    by_method: Mapping[str, Mapping[str, Commitment]],
) -> dict[str, Any]:
    """Paired full set plus each method's own scored set."""
    missing = [name for name in METHOD_ORDER if name not in by_method]
    if missing:
        raise SameBudgetRefused("comparison is missing " + ", ".join(missing))
    paired = sorted(set.intersection(*(scored_ids(by_method[name]) for name in METHOD_ORDER)))
    if not paired:
        raise SameBudgetRefused("no dish is scored by every method")
    methods: dict[str, Any] = {}
    for name in METHOD_ORDER:
        methods[name] = {
            "name": DISPLAY_NAMES[name],
            "role": METHOD_ROLES[name],
            "own": metric_block(list(by_method[name].values())),
            "paired": metric_block([by_method[name][dish_id] for dish_id in paired]),
        }
    return {
        "paired_n": len(paired),
        "paired_dish_ids": paired,
        "methods": methods,
        "deltas_vs_closed_loop_control": _deltas(methods, "ours_old"),
        "deltas_vs_direct_vision_language_model": _deltas(methods, "direct_vlm"),
    }


def _deltas(methods: Mapping[str, Mapping[str, Any]], baseline: str) -> dict[str, dict[str, float | None]]:
    base = methods[baseline]["paired"]
    out: dict[str, dict[str, float | None]] = {}
    for name in METHOD_ORDER:
        block = methods[name]["paired"]
        out[name] = {
            "energy_mae_kcal": _subtract(block.get("full_set_energy_mae_kcal"), base.get("full_set_energy_mae_kcal")),
            "mass_mae_g": _subtract(block.get("full_set_mass_mae_g"), base.get("full_set_mass_mae_g")),
            "tool_cost_mean": _subtract(
                (block.get("tool_cost_scored") or {}).get("mean"),
                (base.get("tool_cost_scored") or {}).get("mean"),
            ),
        }
    return out


def _subtract(left: Any, right: Any) -> float | None:
    if left is None or right is None:
        return None
    return float(left) - float(right)


def abstain_write_gap(steps_by_dish: Mapping[str, Sequence[Mapping[str, Any]]]) -> dict[str, Any]:
    """Describe whether abstain terminals still carry the decision-time dish vector.

    The portion row itself stores the pre-action vector. The mass-updated vector
    is what the next decision sees. An abstain that keeps that vector lets the
    force-answer total include the mass write. A null abstain falls back to the
    previous snapshot. This is a trace description, not a second energy table.
    """
    n_abstain = 0
    n_abstain_finite = 0
    n_stop = 0
    abstain_after_mass = 0
    abstain_null_after_mass = 0
    stop_after_mass = 0
    for steps in steps_by_dish.values():
        if not steps:
            continue
        last = steps[-1]
        mass_indexes = [
            index
            for index, row in enumerate(steps)
            if row.get("action_t") == "RE_PORTION"
            and isinstance(row.get("obs_t"), Mapping)
            and row["obs_t"].get("tool_failed") is not True
            and row["obs_t"].get("portion_src") == "mass_estimator"
        ]
        after_mass = bool(mass_indexes) and mass_indexes[-1] == len(steps) - 2
        finite = nutri_float(last.get("nutri_dish"), "energy_kcal") is not None
        if last.get("action_t") == "ABSTAIN":
            n_abstain += 1
            if finite:
                n_abstain_finite += 1
            if after_mass:
                abstain_after_mass += 1
                if not finite:
                    abstain_null_after_mass += 1
        elif last.get("action_t") == "STOP":
            n_stop += 1
            if mass_indexes:
                stop_after_mass += 1
    return {
        "n_stop_terminals": n_stop,
        "n_stop_with_mass_call": stop_after_mass,
        "n_abstain_terminals": n_abstain,
        "n_abstain_finite_vector": n_abstain_finite,
        "n_abstain_immediately_after_mass": abstain_after_mass,
        "n_abstain_null_immediately_after_mass": abstain_null_after_mass,
        "note": (
            "A finite abstain vector is the dish total the selector saw, including "
            "earlier mass writes. A null abstain drops the force-answer total back "
            "to the previous written snapshot, which on a portion row is pre-mass."
        ),
    }


def action_histogram(steps_by_dish: Mapping[str, Sequence[Mapping[str, Any]]]) -> dict[str, Any]:
    actions: Counter[str] = Counter()
    terminals: Counter[str] = Counter()
    for steps in steps_by_dish.values():
        if not steps:
            continue
        for row in steps:
            actions[str(row.get("action_t"))] += 1
        terminals[str(steps[-1].get("action_t"))] += 1
    return {
        "n_dishes": len(steps_by_dish),
        "n_steps": int(sum(actions.values())),
        "action_counts": dict(sorted(actions.items())),
        "terminal_action_counts": dict(sorted(terminals.items())),
    }


def audit_mass_portions(
    steps_by_dish: Mapping[str, Sequence[Mapping[str, Any]]],
) -> dict[str, Any]:
    """Count real ``estimate_mass`` writes. A successful portion that is not that tool is refused.

    A failed call is not a hollow success. An unchanged dish total after a real
    call is reported and is not by itself a hollow tool: the specialist ran.
    """
    success = 0
    failed = 0
    changed = 0
    unchanged = 0
    no_next = 0
    hollow: list[str] = []
    for dish_id, steps in steps_by_dish.items():
        ordered = list(steps)
        for index, row in enumerate(ordered):
            if row.get("action_t") != "RE_PORTION":
                continue
            obs = row.get("obs_t")
            if not isinstance(obs, Mapping):
                hollow.append(f"{dish_id} step {row.get('step_i')}: missing obs")
                continue
            if obs.get("tool_failed") is True:
                failed += 1
                continue
            real = (
                obs.get("portion_src") == "mass_estimator"
                and obs.get("mass_tool") == "estimate_mass"
                and obs.get("rgb_only") is True
                and obs.get("depth_used") is False
                and nutri_float({"portion_g": obs.get("portion_g")}, "portion_g") is not None
            )
            if not real:
                hollow.append(
                    f"{dish_id} step {row.get('step_i')}: RE_PORTION succeeded without estimate_mass"
                )
                continue
            success += 1
            if index + 1 >= len(ordered):
                no_next += 1
                continue
            before = row.get("nutri_dish")
            after = ordered[index + 1].get("nutri_dish")
            moved = False
            for key in _NUTRI_KEYS:
                left = nutri_float(before, key)
                right = nutri_float(after, key)
                if left is None and right is None:
                    continue
                if left is None or right is None or abs(left - right) > float(EPSILON_MASS_G):
                    moved = True
                    break
            if moved:
                changed += 1
            else:
                unchanged += 1
    if hollow:
        preview = "; ".join(hollow[:5])
        raise SameBudgetRefused(f"hollow RE_PORTION ({len(hollow)}): {preview}")
    return {
        "epsilon": float(EPSILON_MASS_G),
        "re_portion_success": success,
        "re_portion_failed": failed,
        "re_portion_changed_dish_nutrition": changed,
        "re_portion_unchanged_dish_nutrition": unchanged,
        "re_portion_success_without_next_step": no_next,
        "hollow": 0,
        "note": (
            "Success means the observation is estimate_mass, RGB-only, with a finite portion. "
            "Change compares dish energy and mass with the next IR row at 5e-5. "
            "This is not the full-set MAE."
        ),
    }


def _fmt(value: Any, digits: int = 4) -> str:
    if value is None:
        return "n/a"
    return f"{float(value):.{digits}f}"


def _tool_phrase(block: Mapping[str, Any]) -> str:
    tool = block.get("tool_cost_scored") or {}
    return (
        f"mean {_fmt(tool.get('mean'))} median {_fmt(tool.get('median'))} "
        f"max {tool.get('max')} within K {tool.get('n_within_budget')}/{tool.get('n_dishes')}"
    )


def format_report(payload: Mapping[str, Any]) -> str:
    """Plain-text table. Numbers come from ``payload``."""
    section = payload["comparison"]
    methods = section["methods"]
    lines = [
        "P0d same-budget evaluation",
        f"definition: {payload['definition_id']}",
        payload["definition"],
        "",
        (
            f"Pool: official rgb_test {payload['pool']['n_official']} ids, "
            f"{payload['pool']['n_overhead']} with an overhead rgb.png. "
            "Ids without an image are not imputed."
        ),
        (
            f"Budget K={payload['budget_K']}. Tool cost counts {', '.join(payload['tool_actions'])}. "
            f"{' and '.join(payload['not_tools'])} are not tools. "
            "The Q/Cost weights are the selector's denominator, not this count."
        ),
        "Cafe labels score error only. No threshold was fit on rgb_test.",
        "Coverage and AURC are auxiliary and were not used to pick a method.",
        "This is not the hold-out action-agreement table.",
        "",
        f"Paired full set (every method has finite energy and mass), n={section['paired_n']}",
    ]
    for name in payload["method_order"]:
        block = methods[name]["paired"]
        lines.append(
            f"{methods[name]['name']}: n={block['n_scored']} "
            f"energy MAE {_fmt(block['full_set_energy_mae_kcal'])} kcal "
            f"(median {_fmt(block['median_energy_abs_error_kcal'])}) "
            f"mass MAE {_fmt(block['full_set_mass_mae_g'])} g "
            f"(median {_fmt(block['median_mass_abs_error_g'])}) "
            f"tool cost {_tool_phrase(block)}"
        )
    lines.append("")
    lines.append("Paired difference (this method minus the named control). Negative MAE is lower error.")
    for title, key in (
        ("minus closed-loop control", "deltas_vs_closed_loop_control"),
        ("minus direct vision-language model", "deltas_vs_direct_vision_language_model"),
    ):
        lines.append(title + ":")
        for name in payload["method_order"]:
            delta = section[key][name]
            lines.append(
                f"  {methods[name]['name']}: energy {_fmt(delta['energy_mae_kcal'])} kcal, "
                f"mass {_fmt(delta['mass_mae_g'])} g, tool cost {_fmt(delta['tool_cost_mean'])}"
            )
    lines.append("")
    lines.append("Own full set (dishes that method could score; unscored dishes are not imputed):")
    for name in payload["method_order"]:
        block = methods[name]["own"]
        lines.append(
            f"{methods[name]['name']}: scored {block['n_scored']}/{block['n_rows']} "
            f"energy MAE {_fmt(block['full_set_energy_mae_kcal'])} kcal "
            f"mass MAE {_fmt(block['full_set_mass_mae_g'])} g "
            f"tool cost {_tool_phrase(block)} "
            f"early decisions {block['n_early_decision_scored']}"
        )
    lines.append("")
    lines.append("Auxiliary coverage / selective energy MAE / AURC on the paired full set:")
    for name in payload["method_order"]:
        block = methods[name]["paired"]
        lines.append(
            f"{methods[name]['name']}: coverage {_fmt(block['coverage'])} "
            f"selective energy MAE {_fmt(block['selective_energy_mae_kcal'])} kcal "
            f"AURC {_fmt(block['aurc'])}"
        )
    auxiliary = payload.get("auxiliary") or {}
    on_new = auxiliary.get("open_loop_on_ours_new_trace")
    if isinstance(on_new, Mapping):
        lines.append("")
        lines.append(str(on_new.get("note") or "Open-loop read of the mass-estimator trace."))
        own = on_new.get("own") or {}
        lines.append(
            "Open-loop behavior cloning on the mass-estimator trace: "
            f"scored {own.get('n_scored')}/{own.get('n_rows')} "
            f"energy MAE {_fmt(own.get('full_set_energy_mae_kcal'))} kcal "
            f"mass MAE {_fmt(own.get('full_set_mass_mae_g'))} g "
            f"tool cost {_tool_phrase(own)} "
            f"early decisions {own.get('n_early_decision_scored')}"
        )
    reproduction = auxiliary.get("closed_loop_stop_reproduction")
    if isinstance(reproduction, Mapping):
        lines.append("")
        lines.append(
            "Reproduction, not the paired table: closed-loop control on its STOP "
            f"pass/corrected subset n={reproduction.get('n')} energy MAE "
            f"{_fmt(reproduction.get('energy_mae_kcal'))} kcal, mass MAE "
            f"{_fmt(reproduction.get('mass_mae_g'))} g, tool cost mean "
            f"{_fmt(reproduction.get('tool_cost_mean'))}. "
            "The published energy figure is 96.2769 kcal (n=488). "
            "Direct vision-language model 97.03 kcal / 76.46 g is the n=507 anchor, "
            "not this subset."
        )
    gap = payload.get("abstain_write_gap")
    if isinstance(gap, Mapping):
        lines.append("")
        lines.append(
            "Abstain terminals: "
            f"{gap.get('n_abstain_finite_vector')}/{gap.get('n_abstain_terminals')} "
            "keep a finite dish vector, which is the total the selector saw, including earlier mass writes. "
            f"Null abstains immediately after a mass write: {gap.get('n_abstain_null_immediately_after_mass')}. "
            f"Stop terminals: {gap.get('n_stop_terminals')}, "
            f"of which {gap.get('n_stop_with_mass_call')} called the mass estimator."
        )
    audit = payload.get("mass_portion_audit")
    if isinstance(audit, Mapping):
        lines.append("")
        lines.append(
            "Mass-estimator RE_PORTION audit: "
            f"success {audit.get('re_portion_success')}, "
            f"failed {audit.get('re_portion_failed')}, "
            f"dish nutrition changed {audit.get('re_portion_changed_dish_nutrition')}, "
            f"unchanged {audit.get('re_portion_unchanged_dish_nutrition')}, "
            f"hollow {audit.get('hollow')}."
        )
    fence = payload.get("q_fence") or {}
    lines.append("")
    lines.append(
        f"Q weights {fence.get('weights')} sha256 prefix {fence.get('weights_sha256_prefix')}. "
        f"Fence tau={fence.get('tau')}. Retrained={fence.get('weights_retrained')}."
    )
    lines.append(str(payload.get("status_file") or "docs/STATUS.md was not modified."))
    return "\n".join(lines) + "\n"


def run_self_check() -> None:
    """Synthetic commitments. No jsonl, no cafe file, no checkpoint, no RGB."""

    def step(step_i: int, action: str, energy: float | None, mass: float | None, status: str | None = None) -> dict[str, Any]:
        nutri = None if energy is None else {"energy_kcal": energy, "mass_g": mass}
        return {
            "dish_id": "dish_a",
            "step_i": step_i,
            "action_t": action,
            "nutri_dish": nutri,
            "s": status,
            "obs_t": {},
        }

    steps = [
        step(0, "RETRIEVE", 100.0, 40.0),
        step(1, "STOP", 180.0, 90.0, "corrected"),
    ]
    labels = {"dish_a": {"energy_kcal": 200.0, "mass_g": 100.0}, "dish_b": {"energy_kcal": 50.0, "mass_g": 20.0}}
    logged = {"dish_a": steps}
    actions = executed_actions(logged)
    scores = committed_kcal_scores(logged)
    full = commit_from_actions(logged, actions, scores, labels, method="ours_old")["dish_a"]
    if full.energy_kcal != 180.0 or full.mass_g != 90.0 or full.tool_count != 1 or full.early_decision:
        raise AssertionError(full)
    if abs(float(full.abstain_score) - (180.0 / 2500.0)) > 1e-12:
        raise AssertionError(full.abstain_score)
    early_actions = {( "dish_a", 0): "STOP", ("dish_a", 1): "RE_PORTION"}
    early_scores = {("dish_a", 0): 0.2, ("dish_a", 1): 0.2}
    early = commit_from_actions(logged, early_actions, early_scores, labels, method="open_loop_bc")["dish_a"]
    if early.energy_kcal != 100.0 or early.mass_g != 40.0 or early.tool_count != 0 or not early.early_decision:
        raise AssertionError(early)
    direct = commit_direct_vlm(
        {"dish_a": {"energy_kcal": 150.0, "mass_g": 80.0}, "dish_b": None},
        labels,
    )
    if direct["dish_a"].tool_count != 0 or direct["dish_a"].energy_kcal != 150.0:
        raise AssertionError(direct["dish_a"])
    if direct["dish_b"].scored:
        raise AssertionError("a missing direct guess must stay unscored")
    abstain_steps = {
        "dish_b": [
            {
                "dish_id": "dish_b",
                "step_i": 0,
                "action_t": "ABSTAIN",
                "nutri_dish": None,
                "s": "abstain",
                "obs_t": {},
            }
        ]
    }
    abstain = commit_from_actions(
        abstain_steps,
        {("dish_b", 0): "ABSTAIN"},
        {("dish_b", 0): 1.0},
        labels,
        method="ours_new",
    )["dish_b"]
    if abstain.scored or abstain.tool_count != 0:
        raise AssertionError(abstain)

    by_method = {
        "ours_new": {
            "dish_a": commit_from_actions(logged, actions, scores, labels, method="ours_new")["dish_a"],
            "dish_b": abstain,
        },
        "open_loop_bc": {"dish_a": early},
        "direct_vlm": direct,
        "ours_old": {"dish_a": full},
    }
    # The executed ours_new commitment above used method ours_old's twin numbers
    # via a second call. Rebuild dish_a for ours_new explicitly so the method name matches.
    by_method["ours_new"]["dish_a"] = commit_from_actions(logged, actions, scores, labels, method="ours_new")["dish_a"]
    section = comparison_section(by_method)
    if section["paired_n"] != 1 or section["paired_dish_ids"] != ["dish_a"]:
        raise AssertionError(section["paired_dish_ids"])
    paired_new = section["methods"]["ours_new"]["paired"]
    if abs(paired_new["full_set_energy_mae_kcal"] - 20.0) > 1e-9:
        raise AssertionError(paired_new)
    if abs(paired_new["full_set_mass_mae_g"] - 10.0) > 1e-9:
        raise AssertionError(paired_new)
    if abs(paired_new["aurc"] - 10.0) > 1e-9:
        raise AssertionError(paired_new["aurc"])
    if section["methods"]["direct_vlm"]["paired"]["aurc"] is not None:
        raise AssertionError("direct vision-language model has no abstain score")
    if section["methods"]["direct_vlm"]["paired"]["tool_cost_scored"]["mean"] != 0.0:
        raise AssertionError("direct vision-language model tool cost")
    if section["methods"]["ours_new"]["own"]["n_scored"] != 1 or section["methods"]["ours_new"]["own"]["n_rows"] != 2:
        raise AssertionError(section["methods"]["ours_new"]["own"])
    delta = section["deltas_vs_closed_loop_control"]["open_loop_bc"]["energy_mae_kcal"]
    if abs(float(delta) - 80.0) > 1e-9:
        raise AssertionError(delta)

    good = [
        {
            "step_i": 0,
            "action_t": "RE_PORTION",
            "nutri_dish": {"energy_kcal": 10.0, "mass_g": 10.0},
            "obs_t": {
                "portion_src": "mass_estimator",
                "mass_tool": "estimate_mass",
                "rgb_only": True,
                "depth_used": False,
                "portion_g": 80.0,
            },
        },
        {
            "step_i": 1,
            "action_t": "STOP",
            "nutri_dish": {"energy_kcal": 40.0, "mass_g": 80.0},
            "s": "pass",
            "obs_t": {},
        },
    ]
    gap = abstain_write_gap(
        {
            "dish_a": good,
            "dish_b": [
                {
                    "action_t": "RE_PORTION",
                    "obs_t": {"portion_src": "mass_estimator"},
                },
                {"action_t": "ABSTAIN", "s": "abstain"},
            ],
        }
    )
    if (
        gap["n_stop_with_mass_call"] != 1
        or gap["n_abstain_immediately_after_mass"] != 1
        or gap["n_abstain_null_immediately_after_mass"] != 1
        or gap["n_abstain_finite_vector"] != 0
    ):
        raise AssertionError(gap)
    audit = audit_mass_portions({"dish_a": good})
    if audit["re_portion_success"] != 1 or audit["re_portion_changed_dish_nutrition"] != 1 or audit["hollow"] != 0:
        raise AssertionError(audit)
    hollow = [
        {
            "step_i": 0,
            "action_t": "RE_PORTION",
            "nutri_dish": {"energy_kcal": 10.0, "mass_g": 10.0},
            "obs_t": {"portion_src": "grid_search", "portion_g": 80.0},
        },
        {
            "step_i": 1,
            "action_t": "STOP",
            "nutri_dish": {"energy_kcal": 10.0, "mass_g": 10.0},
            "s": "pass",
            "obs_t": {},
        },
    ]
    try:
        audit_mass_portions({"dish_a": hollow})
    except SameBudgetRefused:
        pass
    else:
        raise AssertionError("grid RE_PORTION must be refused as hollow")
    try:
        nutri_float({"energy_kcal": -5.0}, "energy_kcal")
    except SameBudgetRefused:
        pass
    else:
        raise AssertionError("negative energy must be refused")

    payload = {
        "definition_id": DEFINITION_ID,
        "definition": DEFINITION,
        "method_order": list(METHOD_ORDER),
        "budget_K": int(BUDGET_K),
        "tool_actions": sorted(TOOL_ACTIONS),
        "not_tools": sorted(TERMINAL_ACTIONS),
        "pool": {"n_official": 2, "n_overhead": 1},
        "comparison": section,
        "auxiliary": {},
        "mass_portion_audit": audit,
        "q_fence": {
            "weights": "results/qphi/q_phi.npz",
            "weights_sha256_prefix": "self-check",
            "tau": 0.0,
            "weights_retrained": False,
        },
        "status_file": "docs/STATUS.md was not modified.",
    }
    text = format_report(payload)
    if "Mass Estimator + calibrated Q/Cost selection" not in text or "20.0000" not in text:
        raise AssertionError(text)
    if "B5" in text or "B0" in text:
        raise AssertionError("the report leads with plain names")
