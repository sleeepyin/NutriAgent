"""P0a audit: does an executed tool change dish-level nutrition n?

Pre-state is ``nutri_dish`` on the tool row. That row is written before
``action_t`` is applied (teacher emit and on-policy ``_record`` both snapshot
the view, then run the tool). Post-state is ``nutri_dish`` on the next logged
row, which is the dish aggregate after the tool returns and before the next
action. The trajectory's final STOP total is not the post-state: later
components would confound this tool.

A step changes n when dish energy and/or mass moves by more than ε, or when a
finite value appears or disappears. ε is half of the teacher 1e-4 rounding
quantum, so a logged one-quantum difference counts and float dust does not.
It is not the FixRate 1 kcal error threshold, and it is not fit on hold-out
or on rgb_test.

Audited actions are the evidence tools: ``CORRECT_CODE``, ``RE_PORTION``,
``RE_DECOMPOSE``, and ``RETRIEVE``. ``STOP`` and ``ABSTAIN`` are not tools.
``VOLUME_ESTIMATE`` is refused. This module does not train a Mass estimator,
does not write weights, and does not read cafe or rgb_test nutrition labels.
"""

from __future__ import annotations

import math
import statistics
from collections import Counter
from dataclasses import dataclass
from typing import Any, Mapping, Sequence

from vdr.eval.fixrate import FIXRATE_EPSILON_KCAL

# Teacher ``_round_nutri`` writes four decimal places (quantum 1e-4).
NUTRI_ROUND_DECIMALS = 4
NUTRI_QUANTUM = 10.0 ** (-NUTRI_ROUND_DECIMALS)
# Half a quantum: the smallest logged nutrient step (1e-4) is a change.
EPSILON_ENERGY_KCAL = NUTRI_QUANTUM / 2.0
EPSILON_MASS_G = NUTRI_QUANTUM / 2.0
EPSILON_OTHER = NUTRI_QUANTUM / 2.0
EPSILON_LABEL = "5e-5"
# Descriptive majority for the "tool ran but n unchanged" note. Not searched.
NO_CHANGE_MAJORITY = 0.5

ENERGY_KEYS = ("energy_kcal", "energy", "kcal")
MASS_KEYS = ("mass_g", "total_mass_g", "mass", "grams")
OTHER_KEYS = ("protein_g", "carb_g", "fat_g")

AUDIT_ACTIONS = ("CORRECT_CODE", "RE_PORTION", "RE_DECOMPOSE", "RETRIEVE")
EDIT_ACTIONS = ("CORRECT_CODE", "RE_PORTION", "RE_DECOMPOSE")
TERMINAL_ACTIONS = frozenset({"STOP", "ABSTAIN"})
KNOWN_ACTIONS = frozenset(AUDIT_ACTIONS) | TERMINAL_ACTIONS

DEFINITION_ID = "p0a_tool_pre_post_dish_n"
DEFINITION = (
    "For each executed CORRECT_CODE, RE_PORTION, RE_DECOMPOSE, or RETRIEVE step, "
    "compare nutri_dish on that row with nutri_dish on the next logged row. "
    "Changed means energy_kcal and/or mass_g moved by more than epsilon, including "
    "a finite value appearing or disappearing. Both sides missing is no change. "
    "A tool that is the last row has no post-state and is no_successor, not a change. "
    "Epsilon is half the 1e-4 nutrient rounding quantum. FixRate's 1 kcal threshold "
    "is not used. Cafe labels and rgb_test nutrition are not read."
)

SENSITIVITY_EPSILONS = (0.0, float(FIXRATE_EPSILON_KCAL))


class ToolStateAuditRefused(ValueError):
    """The trajectories are not a P0a tool-state audit input."""


@dataclass(frozen=True)
class ToolStepFinding:
    """One executed evidence tool and the dish vector around it."""

    dish_id: str
    step_i: int
    action: str
    has_successor: bool
    energy_before: float | None
    energy_after: float | None
    mass_before: float | None
    mass_after: float | None
    other_changed: bool
    tool_failed: bool
    energy_key: str | None
    mass_key: str | None
    code_from: str | None
    code_to: str | None
    portion_g: float | None
    from_portion_g: float | None
    post_action: str | None


def field_changed(before: float | None, after: float | None, epsilon: float) -> bool:
    """True when a finite value appears, disappears, or moves by more than ε."""
    if epsilon < 0.0 or not math.isfinite(epsilon):
        raise ValueError(f"epsilon must be finite and >= 0, got {epsilon}")
    if before is None and after is None:
        return False
    if before is None or after is None:
        return True
    return abs(before - after) > epsilon


def _finite(value: Any) -> float | None:
    if isinstance(value, bool) or value is None:
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(number):
        return None
    return number


def _lookup(nutri: Any, keys: Sequence[str]) -> tuple[float | None, str | None]:
    if nutri is None:
        return None, None
    if not isinstance(nutri, Mapping):
        raise ToolStateAuditRefused(f"nutri_dish must be an object or null, got {type(nutri).__name__}")
    for key in keys:
        if key in nutri:
            return _finite(nutri.get(key)), key
    return None, None


def _other_changed(before: Any, after: Any, epsilon: float) -> bool:
    if before is None and after is None:
        return False
    before_map = before if isinstance(before, Mapping) else {}
    after_map = after if isinstance(after, Mapping) else {}
    if before is not None and not isinstance(before, Mapping):
        raise ToolStateAuditRefused("nutri_dish must be an object or null")
    if after is not None and not isinstance(after, Mapping):
        raise ToolStateAuditRefused("nutri_dish must be an object or null")
    for key in OTHER_KEYS:
        if field_changed(_finite(before_map.get(key)), _finite(after_map.get(key)), epsilon):
            return True
    return False


def _step_i(row: Mapping[str, Any], dish_id: str) -> int:
    step_i = row.get("step_i")
    if isinstance(step_i, bool) or not isinstance(step_i, int) or step_i < 0:
        raise ToolStateAuditRefused(f"{dish_id}: invalid step_i {step_i!r}")
    return step_i


def _ordered(steps: Sequence[Mapping[str, Any]], dish_id: str) -> list[Mapping[str, Any]]:
    if not steps:
        raise ToolStateAuditRefused(f"{dish_id}: empty trajectory")
    ordered = sorted(steps, key=lambda row: _step_i(row, dish_id))
    seen: list[int] = []
    for row in ordered:
        action = row.get("action_t")
        if action == "VOLUME_ESTIMATE":
            raise ToolStateAuditRefused(f"{dish_id}: VOLUME_ESTIMATE is not an audited action")
        if action not in KNOWN_ACTIONS:
            raise ToolStateAuditRefused(f"{dish_id}: unknown action {action!r}")
        seen.append(_step_i(row, dish_id))
        nutri = row.get("nutri_dish")
        if nutri is not None and not isinstance(nutri, Mapping):
            raise ToolStateAuditRefused(f"{dish_id}: nutri_dish must be an object or null")
    if len(seen) != len(set(seen)):
        raise ToolStateAuditRefused(f"{dish_id}: duplicate step_i")
    return ordered


def _obs_float(obs: Mapping[str, Any], key: str) -> float | None:
    if key not in obs:
        return None
    return _finite(obs.get(key))


def _audit_row(
    dish_id: str,
    step: Mapping[str, Any],
    nxt: Mapping[str, Any] | None,
) -> ToolStepFinding:
    action = str(step.get("action_t"))
    obs = step.get("obs_t")
    if obs is None:
        obs = {}
    if not isinstance(obs, Mapping):
        raise ToolStateAuditRefused(f"{dish_id}: obs_t must be an object")
    pre = step.get("nutri_dish")
    post = None if nxt is None else nxt.get("nutri_dish")
    energy_before, energy_key = _lookup(pre, ENERGY_KEYS)
    mass_before, mass_key = _lookup(pre, MASS_KEYS)
    energy_after, energy_key_after = _lookup(post, ENERGY_KEYS)
    mass_after, mass_key_after = _lookup(post, MASS_KEYS)
    code_from = obs.get("from")
    code_to = obs.get("to")
    return ToolStepFinding(
        dish_id=dish_id,
        step_i=_step_i(step, dish_id),
        action=action,
        has_successor=nxt is not None,
        energy_before=energy_before,
        energy_after=energy_after,
        mass_before=mass_before,
        mass_after=mass_after,
        other_changed=_other_changed(pre, post, EPSILON_OTHER) if nxt is not None else False,
        tool_failed=obs.get("tool_failed") is True,
        energy_key=energy_key or energy_key_after,
        mass_key=mass_key or mass_key_after,
        code_from=None if code_from is None else str(code_from),
        code_to=None if code_to is None else str(code_to),
        portion_g=_obs_float(obs, "portion_g"),
        from_portion_g=_obs_float(obs, "from_portion_g"),
        post_action=None if nxt is None else str(nxt.get("action_t")),
    )


def audit_trajectory(dish_id: str, steps: Sequence[Mapping[str, Any]]) -> list[ToolStepFinding]:
    """Audited tool steps for one dish, in ``step_i`` order."""
    ordered = _ordered(steps, dish_id)
    findings: list[ToolStepFinding] = []
    for index, step in enumerate(ordered):
        if step.get("action_t") not in AUDIT_ACTIONS:
            continue
        nxt = ordered[index + 1] if index + 1 < len(ordered) else None
        findings.append(_audit_row(dish_id, step, nxt))
    return findings


def _present(energy: float | None, mass: float | None) -> bool:
    return energy is not None or mass is not None


def _change_kind(
    finding: ToolStepFinding,
    energy_moved: bool,
    mass_moved: bool,
) -> str | None:
    if not (energy_moved or mass_moved):
        return None
    pre = _present(finding.energy_before, finding.mass_before)
    post = _present(finding.energy_after, finding.mass_after)
    if not pre and post:
        return "appear"
    if pre and not post:
        return "wipe"
    return "value_edit"


def _reason(finding: ToolStepFinding, epsilon: float) -> str:
    if not finding.has_successor:
        return "no_successor"
    if finding.tool_failed:
        return "tool_failed"
    if finding.other_changed:
        return "macro_only"
    if (
        finding.action == "RE_PORTION"
        and finding.portion_g is not None
        and finding.from_portion_g is not None
        and abs(finding.portion_g - finding.from_portion_g) <= epsilon
    ):
        return "same_portion_grams"
    if finding.code_from is not None and finding.code_from == finding.code_to:
        return "same_code"
    if (
        finding.action == "CORRECT_CODE"
        and finding.code_from is not None
        and finding.code_to is not None
        and finding.code_from != finding.code_to
    ):
        return "code_switched_same_nutri"
    if (
        finding.energy_before is None
        and finding.energy_after is None
        and finding.mass_before is None
        and finding.mass_after is None
    ):
        return "both_missing"
    return "within_epsilon"


def _outcome(finding: ToolStepFinding, epsilon: float) -> tuple[str, bool, bool, str | None]:
    if not finding.has_successor:
        return "no_successor", False, False, None
    energy_moved = field_changed(finding.energy_before, finding.energy_after, epsilon)
    mass_moved = field_changed(finding.mass_before, finding.mass_after, epsilon)
    if energy_moved or mass_moved:
        return "changed", energy_moved, mass_moved, _change_kind(finding, energy_moved, mass_moved)
    return "unchanged", False, False, None


def _rate(numer: int, denom: int) -> float | None:
    if denom <= 0:
        return None
    return round(numer / denom, 6)


def _median(values: Sequence[float]) -> float | None:
    if not values:
        return None
    return round(float(statistics.median(values)), 6)


def _empty_block(action: str | None = None) -> dict[str, Any]:
    block: dict[str, Any] = {
        "n_executed": 0,
        "n_changed": 0,
        "n_unchanged": 0,
        "n_no_successor": 0,
        "n_energy_changed": 0,
        "n_mass_changed": 0,
        "n_energy_and_mass": 0,
        "n_energy_only": 0,
        "n_mass_only": 0,
        "n_value_edit": 0,
        "n_appear": 0,
        "n_wipe": 0,
        "wipe_next_action": {},
        "n_null_flip": 0,
        "n_macro_only": 0,
        "n_dishes": 0,
        "unchanged_reasons": {},
        "tool_failed": {"n": 0, "n_changed": 0, "n_unchanged": 0, "n_no_successor": 0},
        "energy_abs_delta": {
            "n_both_finite": 0,
            "n_gt_1": 0,
            "median_both_finite": None,
            "median_when_changed": None,
        },
        "mass_abs_delta": {
            "n_both_finite": 0,
            "n_gt_1": 0,
            "median_both_finite": None,
            "median_when_changed": None,
        },
    }
    if action is not None:
        block["action"] = action
    return block


def _finish_block(
    block: dict[str, Any],
    *,
    dishes: set[str],
    reasons: Counter[str],
    energy_both: list[float],
    energy_changed: list[float],
    mass_both: list[float],
    mass_changed: list[float],
    wipe_next: Counter[str],
    epsilon_energy: float,
    epsilon_mass: float,
) -> dict[str, Any]:
    executed = int(block["n_executed"])
    parts = int(block["n_changed"]) + int(block["n_unchanged"]) + int(block["n_no_successor"])
    if parts != executed:
        raise ToolStateAuditRefused(f"outcome partition {parts} != executed {executed}")
    changed_parts = int(block["n_energy_and_mass"]) + int(block["n_energy_only"]) + int(block["n_mass_only"])
    if changed_parts != int(block["n_changed"]):
        raise ToolStateAuditRefused("energy/mass split does not match n_changed")
    kind_parts = int(block["n_value_edit"]) + int(block["n_appear"]) + int(block["n_wipe"])
    if kind_parts != int(block["n_changed"]):
        raise ToolStateAuditRefused("appear/wipe/value_edit split does not match n_changed")
    if sum(wipe_next.values()) != int(block["n_wipe"]):
        raise ToolStateAuditRefused("wipe next-action split does not match n_wipe")
    block["n_dishes"] = len(dishes)
    block["wipe_next_action"] = {key: int(wipe_next[key]) for key in sorted(wipe_next)}
    block["change_rate"] = _rate(int(block["n_changed"]), executed)
    block["no_change_rate"] = _rate(int(block["n_unchanged"]), executed)
    block["no_successor_rate"] = _rate(int(block["n_no_successor"]), executed)
    block["energy_change_rate"] = _rate(int(block["n_energy_changed"]), executed)
    block["mass_change_rate"] = _rate(int(block["n_mass_changed"]), executed)
    with_post = executed - int(block["n_no_successor"])
    block["n_with_post_state"] = with_post
    block["change_rate_given_post_state"] = _rate(int(block["n_changed"]), with_post)
    block["no_change_rate_given_post_state"] = _rate(int(block["n_unchanged"]), with_post)
    block["unchanged_reasons"] = {key: int(reasons[key]) for key in sorted(reasons)}
    block["no_change_dominates"] = bool(executed and (int(block["n_unchanged"]) / executed) > NO_CHANGE_MAJORITY)
    block["energy_abs_delta"] = {
        "n_both_finite": len(energy_both),
        "n_gt_1": sum(1 for value in energy_both if value > float(FIXRATE_EPSILON_KCAL)),
        "median_both_finite": _median(energy_both),
        "median_when_changed": _median(energy_changed),
        "epsilon": epsilon_energy,
        "gt_1_reference_kcal": float(FIXRATE_EPSILON_KCAL),
    }
    block["mass_abs_delta"] = {
        "n_both_finite": len(mass_both),
        "n_gt_1": sum(1 for value in mass_both if value > 1.0),
        "median_both_finite": _median(mass_both),
        "median_when_changed": _median(mass_changed),
        "epsilon": epsilon_mass,
        "gt_1_reference_g": 1.0,
    }
    if int(block["n_mass_changed"]) == 0 and int(block["n_energy_changed"]) > 0:
        block["mass_note"] = (
            "mass_g stayed within ε on every executed step; energy_kcal moved on the "
            "changed steps. A same-gram code swap does not move mass."
        )
    return block


def _accumulate(findings: Sequence[ToolStepFinding], epsilon: float) -> dict[str, Any]:
    """Count outcomes. ``epsilon`` is applied to both energy and mass."""
    block = _empty_block()
    dishes: set[str] = set()
    reasons: Counter[str] = Counter()
    energy_both: list[float] = []
    energy_changed: list[float] = []
    mass_both: list[float] = []
    mass_changed: list[float] = []
    wipe_next: Counter[str] = Counter()
    failed = block["tool_failed"]
    for finding in findings:
        dishes.add(finding.dish_id)
        block["n_executed"] += 1
        outcome, energy_moved, mass_moved, kind = _outcome(finding, epsilon)
        block[f"n_{outcome}"] += 1
        if finding.tool_failed:
            failed["n"] += 1
            failed[f"n_{outcome}"] += 1
        if finding.energy_before is not None and finding.energy_after is not None:
            delta = abs(finding.energy_after - finding.energy_before)
            energy_both.append(delta)
            if energy_moved:
                energy_changed.append(delta)
        if finding.mass_before is not None and finding.mass_after is not None:
            delta = abs(finding.mass_after - finding.mass_before)
            mass_both.append(delta)
            if mass_moved:
                mass_changed.append(delta)
        if outcome != "changed":
            reason = _reason(finding, epsilon)
            reasons[reason] += 1
            if reason == "macro_only":
                block["n_macro_only"] += 1
            continue
        if energy_moved:
            block["n_energy_changed"] += 1
        if mass_moved:
            block["n_mass_changed"] += 1
        if energy_moved and mass_moved:
            block["n_energy_and_mass"] += 1
        elif energy_moved:
            block["n_energy_only"] += 1
        else:
            block["n_mass_only"] += 1
        if kind is None:
            raise ToolStateAuditRefused("changed step is missing a change kind")
        block[f"n_{kind}"] += 1
        if kind == "wipe":
            wipe_next[finding.post_action or "missing"] += 1
        energy_null = (finding.energy_before is None) != (finding.energy_after is None)
        mass_null = (finding.mass_before is None) != (finding.mass_after is None)
        if energy_null or mass_null:
            block["n_null_flip"] += 1
    return _finish_block(
        block,
        dishes=dishes,
        reasons=reasons,
        energy_both=energy_both,
        energy_changed=energy_changed,
        mass_both=mass_both,
        mass_changed=mass_changed,
        wipe_next=wipe_next,
        epsilon_energy=epsilon,
        epsilon_mass=epsilon,
    )


def summarize(
    findings: Sequence[ToolStepFinding],
    *,
    n_dishes: int,
    n_rows: int,
    epsilon: float = EPSILON_ENERGY_KCAL,
) -> dict[str, Any]:
    """Per-action and rollup rates for one population."""
    by_action: dict[str, Any] = {}
    for action in AUDIT_ACTIONS:
        selected = [row for row in findings if row.action == action]
        by_action[action] = _accumulate(selected, epsilon)
        by_action[action]["action"] = action
    edit = [row for row in findings if row.action in EDIT_ACTIONS]
    overall = _accumulate(findings, epsilon)
    edit_block = _accumulate(edit, epsilon)
    keys_energy = sorted({row.energy_key for row in findings if row.energy_key})
    keys_mass = sorted({row.mass_key for row in findings if row.mass_key})
    return {
        "n_dishes": int(n_dishes),
        "n_rows": int(n_rows),
        "n_dishes_with_audited_tool": len({row.dish_id for row in findings}),
        "energy_keys_seen": keys_energy,
        "mass_keys_seen": keys_mass,
        "overall": overall,
        "edit_actions": edit_block,
        "by_action": by_action,
    }


def audit_population(steps_by_dish: Mapping[str, Sequence[Mapping[str, Any]]]) -> tuple[list[ToolStepFinding], dict[str, Any]]:
    """Audit every dish. Returns findings and the primary-ε summary."""
    if not steps_by_dish:
        raise ToolStateAuditRefused("no dishes")
    findings: list[ToolStepFinding] = []
    n_rows = 0
    n_step_gaps = 0
    for dish_id, steps in steps_by_dish.items():
        if not isinstance(dish_id, str) or not dish_id:
            raise ToolStateAuditRefused(f"invalid dish_id {dish_id!r}")
        ordered = _ordered(steps, dish_id)
        n_rows += len(ordered)
        for index in range(len(ordered) - 1):
            if _step_i(ordered[index + 1], dish_id) != _step_i(ordered[index], dish_id) + 1:
                n_step_gaps += 1
        findings.extend(audit_trajectory(dish_id, ordered))
    summary = summarize(findings, n_dishes=len(steps_by_dish), n_rows=n_rows)
    summary["n_step_gaps"] = n_step_gaps
    return findings, summary


def _sensitivity_block(findings: Sequence[ToolStepFinding], epsilon: float) -> dict[str, Any]:
    summary = summarize(findings, n_dishes=0, n_rows=0, epsilon=epsilon)
    keep = ("n_executed", "n_changed", "n_unchanged", "n_no_successor", "change_rate", "no_change_rate")

    def _slim(block: Mapping[str, Any]) -> dict[str, Any]:
        return {key: block[key] for key in keep}

    return {
        "epsilon": epsilon,
        "note": (
            "Same steps, both fields use this epsilon. Not the primary definition. "
            "0 counts any finite difference. 1 is one kcal and one gram, the FixRate "
            "energy threshold, shown only so small logged edits are visible."
        ),
        "overall": _slim(summary["overall"]),
        "edit_actions": _slim(summary["edit_actions"]),
        "by_action": {action: _slim(block) for action, block in summary["by_action"].items()},
    }


def build_payload(
    populations: Mapping[str, Mapping[str, Sequence[Mapping[str, Any]]]],
    *,
    overall_of: Sequence[str],
) -> dict[str, Any]:
    """Summarize named populations and a step-weighted rollup of ``overall_of``."""
    if not populations:
        raise ToolStateAuditRefused("no populations")
    missing = [name for name in overall_of if name not in populations]
    if missing:
        raise ToolStateAuditRefused("overall populations missing: " + ", ".join(missing))
    findings: dict[str, list[ToolStepFinding]] = {}
    summarized: dict[str, Any] = {}
    for name, grouped in populations.items():
        found, summary = audit_population(grouped)
        findings[name] = found
        summarized[name] = summary
    combined: list[ToolStepFinding] = []
    n_dishes = 0
    n_rows = 0
    for name in overall_of:
        combined.extend(findings[name])
        n_dishes += int(summarized[name]["n_dishes"])
        n_rows += int(summarized[name]["n_rows"])
    overall = summarize(combined, n_dishes=n_dishes, n_rows=n_rows)
    overall["populations"] = list(overall_of)
    overall["population_note"] = (
        "Step-weighted sum of the named full trajectories. Not one evaluation "
        "population. Dish subsets are not included, so a dish is not counted twice."
    )
    sensitivity = []
    for epsilon in SENSITIVITY_EPSILONS:
        entry: dict[str, Any] = {"epsilon": epsilon, "by_population": {}}
        for name in overall_of:
            entry["by_population"][name] = _sensitivity_block(findings[name], epsilon)
        entry["overall_step_weighted"] = _sensitivity_block(combined, epsilon)
        sensitivity.append(entry)
    payload: dict[str, Any] = {
        "definition_id": DEFINITION_ID,
        "definition": DEFINITION,
        "scope": "P0a audit gate only. Mass Estimator was not trained. No new weights. No RL.",
        "epsilon": {
            "energy_kcal": EPSILON_ENERGY_KCAL,
            "mass_g": EPSILON_MASS_G,
            "other_nutri": EPSILON_OTHER,
            "label": EPSILON_LABEL,
            "quantum": NUTRI_QUANTUM,
            "rule": "strictly greater than epsilon. A difference equal to epsilon is not a change.",
            "source": (
                "Half of the teacher _round_nutri quantum (4 decimal places, 1e-4). "
                "Not fit on hold-out or rgb_test."
            ),
            "fixrate_epsilon_kcal_not_used": float(FIXRATE_EPSILON_KCAL),
        },
        "comparison": (
            "pre = nutri_dish on the tool row (state before action_t). "
            "post = nutri_dish on the next logged row (state after the tool, before the next action)."
        ),
        "change_rule": (
            "changed if |Δenergy| > ε or |Δmass| > ε, or exactly one side of a field is finite. "
            "Both sides missing is unchanged. no_successor is a separate outcome and is not a change."
        ),
        "actions": list(AUDIT_ACTIONS),
        "edit_actions": list(EDIT_ACTIONS),
        "no_change_majority": NO_CHANGE_MAJORITY,
        "populations": summarized,
        "overall_step_weighted": overall,
        "epsilon_sensitivity": sensitivity,
    }
    payload["notes"] = interpret(payload)
    return payload


def interpret(payload: Mapping[str, Any]) -> list[str]:
    """Short factual notes. The majority cutoff is descriptive, not a searched gate."""
    notes: list[str] = []
    dominated: list[str] = []
    populations = payload["populations"]
    for name in ("logged_holdout", "on_policy_rgb_test"):
        summary = populations.get(name)
        if summary is None:
            continue
        edit = summary["edit_actions"]
        notes.append(
            f"{name} edit actions: change {edit['n_changed']}/{edit['n_executed']}, "
            f"no-change {edit['n_unchanged']}/{edit['n_executed']}, "
            f"value_edit {edit['n_value_edit']}, wipe {edit['n_wipe']}. "
            "No-change is not the majority outcome on this population."
        )
    for name, summary in populations.items():
        blocks = [("overall", summary["overall"]), ("edit_actions", summary["edit_actions"])]
        blocks.extend(summary["by_action"].items())
        for label, block in blocks:
            if int(block["n_executed"]) == 0:
                continue
            if block["no_change_dominates"]:
                dominated.append(
                    f"{name} {label}: no-change {block['no_change_rate']:.4f} "
                    f"({block['n_unchanged']}/{block['n_executed']}) is a majority"
                )
        for action, block in summary["by_action"].items():
            if int(block["n_executed"]) != 0:
                continue
            notes.append(f"{name} {action}: 0 executed steps.")
        portion = summary["by_action"]["RE_PORTION"]
        reasons = portion.get("unchanged_reasons") or {}
        same_g = int(reasons.get("same_portion_grams", 0))
        if int(portion["n_unchanged"]) > 0 and same_g > 0:
            notes.append(
                f"{name} RE_PORTION: {portion['n_unchanged']}/{portion['n_executed']} "
                "left dish energy and mass within ε; "
                f"{same_g} of those unchanged steps have portion_g and from_portion_g "
                "within ε, so the portion call did not move grams."
            )
        for action, block in summary["by_action"].items():
            wipe_next = block.get("wipe_next_action") or {}
            if int(block["n_wipe"]) <= 0:
                continue
            rendered = ", ".join(f"{key}={value}" for key, value in wipe_next.items())
            notes.append(
                f"{name} {action}: {block['n_wipe']} changed steps replace a finite dish "
                f"vector with a missing one (next action: {rendered}). "
                "Those count as written-vector changes and are not included in value_edit."
            )
    rollup = payload.get("overall_step_weighted")
    if isinstance(rollup, Mapping):
        rollup_blocks = [
            ("overall_step_weighted overall", rollup["overall"]),
            ("overall_step_weighted edit_actions", rollup["edit_actions"]),
        ]
        rollup_blocks.extend(
            (f"overall_step_weighted {action}", block) for action, block in rollup["by_action"].items()
        )
        for label, block in rollup_blocks:
            if int(block["n_executed"]) == 0:
                continue
            if block["no_change_dominates"]:
                dominated.append(
                    f"{label}: no-change {block['no_change_rate']:.4f} "
                    f"({block['n_unchanged']}/{block['n_executed']}) is a majority"
                )
    for name, summary in populations.items():
        code = summary["by_action"]["CORRECT_CODE"]
        code_reasons = code.get("unchanged_reasons") or {}
        failed = int(code_reasons.get("tool_failed", 0))
        switched = int(code_reasons.get("code_switched_same_nutri", 0))
        if int(code["n_unchanged"]) > 0:
            notes.append(
                f"{name} CORRECT_CODE: {code['n_unchanged']}/{code['n_executed']} "
                "left dish energy and mass within ε "
                f"(tool_failed {failed}, code switched but same dish energy and mass {switched}). "
                "Mass change rate is separate: a same-gram code swap does not move mass_g."
            )
    if dominated:
        notes.append(
            "Tool ran but dish energy and mass stayed within ε on a majority of executed "
            f"steps (cutoff {NO_CHANGE_MAJORITY}, descriptive, not fit on hold-out or rgb_test): "
            + "; ".join(dominated)
            + "."
        )
    else:
        notes.append(
            "No-change does not dominate any audited action or rollup "
            f"(majority cutoff {NO_CHANGE_MAJORITY} of executed steps). "
            "The cutoff is descriptive and was not fit on hold-out or rgb_test. "
            "Minority no-change steps are still counted above."
        )
    return notes


def _fmt_rate(numer: int, denom: int) -> str:
    if denom <= 0:
        return "n/a"
    return f"{numer / denom:.4f} ({numer}/{denom})"


def _fmt_block(title: str, block: Mapping[str, Any]) -> list[str]:
    executed = int(block["n_executed"])
    lines = [
        (
            f"{title}: executed={executed} changed={block['n_changed']} "
            f"unchanged={block['n_unchanged']} no_successor={block['n_no_successor']} "
            f"dishes={block['n_dishes']}"
        ),
        (
            f"  change_rate={_fmt_rate(int(block['n_changed']), executed)} "
            f"no_change_rate={_fmt_rate(int(block['n_unchanged']), executed)} "
            f"no_successor_rate={_fmt_rate(int(block['n_no_successor']), executed)}"
        ),
        (
            f"  energy_change_rate={_fmt_rate(int(block['n_energy_changed']), executed)} "
            f"mass_change_rate={_fmt_rate(int(block['n_mass_changed']), executed)} "
            f"energy_and_mass={block['n_energy_and_mass']} "
            f"energy_only={block['n_energy_only']} mass_only={block['n_mass_only']}"
        ),
        (
            f"  value_edit={block['n_value_edit']} appear={block['n_appear']} "
            f"wipe={block['n_wipe']} null_flip={block['n_null_flip']} "
            f"macro_only_unchanged={block['n_macro_only']}"
        ),
    ]
    reasons = block.get("unchanged_reasons") or {}
    if reasons:
        rendered = ", ".join(f"{key}={value}" for key, value in reasons.items())
        lines.append(f"  unchanged_or_no_successor_reasons: {rendered}")
    failed = block.get("tool_failed") or {}
    if int(failed.get("n", 0)) > 0:
        lines.append(
            "  tool_failed: "
            f"n={failed['n']} changed={failed['n_changed']} "
            f"unchanged={failed['n_unchanged']} no_successor={failed['n_no_successor']}"
        )
    energy = block.get("energy_abs_delta") or {}
    mass = block.get("mass_abs_delta") or {}
    if int(energy.get("n_both_finite", 0)) or int(mass.get("n_both_finite", 0)):
        lines.append(
            "  |Δ| both finite: "
            f"energy median={energy.get('median_both_finite')} "
            f"median_when_energy_changed={energy.get('median_when_changed')} "
            f"n_gt_1kcal={energy.get('n_gt_1')}/{energy.get('n_both_finite')}; "
            f"mass median={mass.get('median_both_finite')} "
            f"median_when_mass_changed={mass.get('median_when_changed')} "
            f"n_gt_1g={mass.get('n_gt_1')}/{mass.get('n_both_finite')}"
        )
    mass_note = block.get("mass_note")
    if mass_note:
        lines.append(f"  mass_note: {mass_note}")
    wipe_next = block.get("wipe_next_action") or {}
    if wipe_next:
        rendered = ", ".join(f"{key}={value}" for key, value in wipe_next.items())
        lines.append(f"  wipe_next_action: {rendered}")
    return lines


def format_text_report(payload: Mapping[str, Any]) -> str:
    """Plain-text report. Numbers come from ``build_payload``."""
    epsilon = payload["epsilon"]
    lines = [
        "P0a tool-state audit (dish energy and mass)",
        f"definition: {payload['definition_id']}",
        f"scope: {payload['scope']}",
        (
            f"epsilon_energy_kcal: {epsilon['label']} ({epsilon['energy_kcal']}) "
            f"epsilon_mass_g: {epsilon['label']} ({epsilon['mass_g']})"
        ),
        f"epsilon_source: {epsilon['source']}",
        f"fixrate_epsilon_kcal_not_used: {epsilon['fixrate_epsilon_kcal_not_used']}",
        f"comparison: {payload['comparison']}",
        f"change_rule: {payload['change_rule']}",
        f"edit_actions: {', '.join(payload['edit_actions'])}",
        f"also_audited: {', '.join(payload['actions'])}",
        "",
    ]
    for name, summary in payload["populations"].items():
        lines.append(
            f"[{name}] dishes={summary['n_dishes']} rows={summary['n_rows']} "
            f"dishes_with_tool={summary['n_dishes_with_audited_tool']} "
            f"step_gaps={summary.get('n_step_gaps', 0)} "
            f"energy_keys={summary['energy_keys_seen']} mass_keys={summary['mass_keys_seen']}"
        )
        lines.extend(_fmt_block("  overall", summary["overall"]))
        lines.extend(_fmt_block("  edit_actions", summary["edit_actions"]))
        for action, block in summary["by_action"].items():
            lines.extend(_fmt_block(f"  {action}", block))
        lines.append("")
    overall = payload["overall_step_weighted"]
    lines.append(
        "[overall_step_weighted] populations="
        + ",".join(overall["populations"])
        + f" dishes_sum={overall['n_dishes']} rows_sum={overall['n_rows']}"
    )
    lines.append(f"  note: {overall['population_note']}")
    lines.extend(_fmt_block("  overall", overall["overall"]))
    lines.extend(_fmt_block("  edit_actions", overall["edit_actions"]))
    for action, block in overall["by_action"].items():
        lines.extend(_fmt_block(f"  {action}", block))
    lines.append("")
    lines.append("epsilon sensitivity (not the primary definition):")
    for entry in payload["epsilon_sensitivity"]:
        for name, block in entry["by_population"].items():
            edit = block["edit_actions"]
            lines.append(
                f"  ε={entry['epsilon']} {name} edit_actions "
                f"change_rate={_fmt_rate(int(edit['n_changed']), int(edit['n_executed']))} "
                f"no_change_rate={_fmt_rate(int(edit['n_unchanged']), int(edit['n_executed']))}"
            )
    lines.append("")
    lines.append("notes:")
    for note in payload["notes"]:
        lines.append(f"- {note}")
    lines.append("")
    return "\n".join(lines)


def format_markdown_report(payload: Mapping[str, Any]) -> str:
    """Short markdown report for the same payload."""
    epsilon = payload["epsilon"]
    lines = [
        "# P0a audit: do tool steps change dish energy and mass?",
        "",
        "This is the audit gate only. Mass Estimator training was not started. No new specialist weights. No RL.",
        "",
        "Pre-state is `nutri_dish` on the tool row. Post-state is `nutri_dish` on the next logged row. "
        "A change is dish energy and/or mass moving by more than "
        f"ε = {epsilon['label']} (half of the 1e-4 nutrient rounding quantum). "
        f"That ε is not the FixRate {epsilon['fixrate_epsilon_kcal_not_used']} kcal threshold.",
        "",
        "Edit actions are `CORRECT_CODE`, `RE_PORTION`, and `RE_DECOMPOSE`. "
        "`RETRIEVE` is audited with them. Rates below use every executed step as the denominator. "
        "`no_successor` means the tool is the last row, so no post dish state was written.",
        "",
    ]
    for name, summary in payload["populations"].items():
        lines.append(f"## {name}")
        lines.append("")
        lines.append(
            f"Dishes {summary['n_dishes']}, rows {summary['n_rows']}, "
            f"dishes with an audited tool {summary['n_dishes_with_audited_tool']}."
        )
        lines.append("")
        lines.append("| slice | executed | change rate | no-change rate | energy change | mass change |")
        lines.append("| --- | ---: | ---: | ---: | ---: | ---: |")
        rows = [("edit actions", summary["edit_actions"]), ("all audited tools", summary["overall"])]
        rows.extend((action, block) for action, block in summary["by_action"].items())
        for label, block in rows:
            executed = int(block["n_executed"])
            lines.append(
                f"| {label} | {executed} | "
                f"{_fmt_rate(int(block['n_changed']), executed)} | "
                f"{_fmt_rate(int(block['n_unchanged']), executed)} | "
                f"{_fmt_rate(int(block['n_energy_changed']), executed)} | "
                f"{_fmt_rate(int(block['n_mass_changed']), executed)} |"
            )
        lines.append("")
    overall = payload["overall_step_weighted"]
    edit = overall["edit_actions"]
    lines.append("## Step-weighted rollup")
    lines.append("")
    lines.append(overall["population_note"])
    lines.append("")
    lines.append(
        "Edit actions "
        f"{_fmt_rate(int(edit['n_changed']), int(edit['n_executed']))} changed, "
        f"no-change {_fmt_rate(int(edit['n_unchanged']), int(edit['n_executed']))}. "
        "All audited tools "
        f"{_fmt_rate(int(overall['overall']['n_changed']), int(overall['overall']['n_executed']))} changed."
    )
    lines.append("")
    lines.append("## Notes")
    lines.append("")
    for note in payload["notes"]:
        lines.append(f"- {note}")
    lines.append("")
    return "\n".join(lines)


def _nutri(energy: float | None, mass: float | None, **extra: float) -> dict[str, float | None] | None:
    if energy is None and mass is None and not extra:
        return None
    row: dict[str, float | None] = {"energy_kcal": energy, "mass_g": mass}
    row.update(extra)
    return row


def _tool(
    step_i: int,
    action: str,
    energy: float | None,
    mass: float | None,
    obs: Mapping[str, Any] | None = None,
    **extra: float,
) -> dict[str, Any]:
    return {
        "step_i": step_i,
        "action_t": action,
        "nutri_dish": _nutri(energy, mass, **extra),
        "obs_t": dict(obs or {}),
        "s": None,
    }


def run_self_check() -> None:
    """Synthetic contract. No trajectories on disk and no labels."""
    if field_changed(1.0, 1.0, EPSILON_ENERGY_KCAL):
        raise AssertionError("equal values must not count as a change")
    if not field_changed(1.0, 1.0 + NUTRI_QUANTUM, EPSILON_ENERGY_KCAL):
        raise AssertionError("one rounding quantum must count as a change")
    if field_changed(1.0, 1.0 + 1e-6, EPSILON_ENERGY_KCAL):
        raise AssertionError("float dust must not count as a change")
    if not field_changed(None, 1.0, EPSILON_ENERGY_KCAL):
        raise AssertionError("appearance must count as a change")
    if field_changed(None, None, EPSILON_ENERGY_KCAL):
        raise AssertionError("missing on both sides is not a change")

    grouped: dict[str, list[dict[str, Any]]] = {
        "portion_moves": [
            _tool(0, "RE_PORTION", 200.0, 100.0, {"portion_g": 180.0, "from_portion_g": 100.0}),
            _tool(1, "STOP", 360.0, 180.0),
        ],
        "code_energy": [
            _tool(0, "CORRECT_CODE", 100.0, 100.0, {"from": "A", "to": "B"}),
            _tool(1, "STOP", 140.0, 100.0),
        ],
        "code_fail": [
            _tool(0, "CORRECT_CODE", 50.0, 80.0, {"from": "X", "to": "X", "tool_failed": True}),
            _tool(1, "STOP", 50.0, 80.0),
        ],
        "portion_same": [
            _tool(0, "RE_PORTION", 10.0, 100.0, {"portion_g": 100.0, "from_portion_g": 100.0}),
            _tool(1, "STOP", 10.0, 100.0),
        ],
        "retrieve_appear": [
            _tool(0, "RETRIEVE", None, None, {"selected_code": "1"}),
            _tool(1, "STOP", 40.0, 100.0),
        ],
        "retrieve_stuck": [
            _tool(0, "RETRIEVE", 10.0, 10.0, {"selected_code": "9"}),
        ],
        "decompose": [
            _tool(0, "RE_DECOMPOSE", 7.0, 7.0, {"tool_failed": False}),
            _tool(1, "STOP", 7.0, 7.0),
        ],
        "dust": [
            _tool(0, "CORRECT_CODE", 1.0, 1.0, {"from": "A", "to": "B"}),
            _tool(1, "STOP", 1.0 + 1e-6, 1.0),
        ],
        "one_quantum": [
            _tool(0, "RE_PORTION", 1.0, 2.0, {"portion_g": 2.0001, "from_portion_g": 2.0}),
            _tool(1, "STOP", 1.0 + NUTRI_QUANTUM, 2.0 + NUTRI_QUANTUM),
        ],
        "macro": [
            _tool(0, "CORRECT_CODE", 10.0, 10.0, {"from": "A", "to": "B"}, protein_g=1.0),
            _tool(1, "STOP", 10.0, 10.0, protein_g=2.0),
        ],
        "wipe": [
            _tool(0, "RE_PORTION", 10.0, 10.0, {"portion_g": 20.0, "from_portion_g": 10.0}),
            {
                "step_i": 1,
                "action_t": "ABSTAIN",
                "nutri_dish": {"energy_kcal": None, "mass_g": None},
                "obs_t": {},
                "s": "abstain",
            },
        ],
        "both_missing": [
            _tool(0, "RETRIEVE", None, None, {}),
            _tool(1, "RETRIEVE", None, None, {}),
        ],
    }
    # The second RETRIEVE on both_missing is the last row.
    _findings, summary = audit_population(grouped)
    del _findings
    overall = summary["overall"]
    if overall["n_executed"] != 13 or overall["n_changed"] != 5 or overall["n_unchanged"] != 6:
        raise AssertionError((overall["n_executed"], overall["n_changed"], overall["n_unchanged"], overall["n_no_successor"]))
    if overall["n_no_successor"] != 2:
        raise AssertionError(overall["n_no_successor"])
    edit = summary["edit_actions"]
    if (edit["n_executed"], edit["n_changed"], edit["n_unchanged"]) != (9, 4, 5):
        raise AssertionError((edit["n_executed"], edit["n_changed"], edit["n_unchanged"]))
    code = summary["by_action"]["CORRECT_CODE"]
    if code["n_energy_only"] != 1 or code["n_mass_changed"] != 0 or code["unchanged_reasons"].get("tool_failed") != 1:
        raise AssertionError(code)
    if code["unchanged_reasons"].get("macro_only") != 1:
        raise AssertionError(code["unchanged_reasons"])
    if code["unchanged_reasons"].get("code_switched_same_nutri") != 1:
        raise AssertionError(code["unchanged_reasons"])
    portion = summary["by_action"]["RE_PORTION"]
    if portion["n_changed"] != 3 or portion["unchanged_reasons"].get("same_portion_grams") != 1:
        raise AssertionError(portion)
    if portion["n_wipe"] != 1 or portion["n_value_edit"] != 2:
        raise AssertionError((portion["n_wipe"], portion["n_value_edit"]))
    if portion["wipe_next_action"] != {"ABSTAIN": 1}:
        raise AssertionError(portion["wipe_next_action"])
    retrieve = summary["by_action"]["RETRIEVE"]
    if retrieve["n_appear"] != 1 or retrieve["n_no_successor"] != 2 or retrieve["unchanged_reasons"].get("both_missing") != 1:
        raise AssertionError(retrieve)
    if summary["by_action"]["RE_DECOMPOSE"]["n_unchanged"] != 1:
        raise AssertionError(summary["by_action"]["RE_DECOMPOSE"])
    if not code["mass_note"]:
        raise AssertionError("same-gram code swap should explain a zero mass-change rate")

    payload = build_payload({"synthetic": grouped}, overall_of=("synthetic",))
    text = format_text_report(payload)
    markdown = format_markdown_report(payload)
    if "Mass Estimator was not trained" not in text or "Mass Estimator training was not started" not in markdown:
        raise AssertionError("report must state that Mass training did not start")
    if "5e-5" not in text or "5e-5" not in markdown:
        raise AssertionError("report must document epsilon")
    if payload["populations"]["synthetic"]["overall"]["n_changed"] != 5:
        raise AssertionError("payload overall drifted from the direct audit")
    zero = payload["epsilon_sensitivity"][0]["by_population"]["synthetic"]["edit_actions"]
    # dust (1e-6) becomes a change at ε=0, so edit changed steps go from 4 to 5.
    if int(zero["n_changed"]) != 5:
        raise AssertionError(zero)

    raised = False
    try:
        audit_population({"bad": [_tool(0, "VOLUME_ESTIMATE", None, None)]})
    except ToolStateAuditRefused:
        raised = True
    if not raised:
        raise AssertionError("VOLUME_ESTIMATE must be refused")
    raised = False
    try:
        audit_population({"bad": [_tool(0, "RETRIEVE", 1.0, 1.0), _tool(0, "STOP", 1.0, 1.0)]})
    except ToolStateAuditRefused:
        raised = True
    if not raised:
        raise AssertionError("duplicate step_i must be refused")


__all__ = [
    "AUDIT_ACTIONS",
    "DEFINITION_ID",
    "EDIT_ACTIONS",
    "EPSILON_ENERGY_KCAL",
    "EPSILON_LABEL",
    "EPSILON_MASS_G",
    "NO_CHANGE_MAJORITY",
    "ToolStateAuditRefused",
    "audit_population",
    "build_payload",
    "field_changed",
    "format_markdown_report",
    "format_text_report",
    "run_self_check",
]
