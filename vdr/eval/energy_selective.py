"""Hold-out dish energy MAE, Coverage, Selective MAE, and AURC.

Primary definition
------------------
``offline_logged_horizon_force_answer``

States are the logged weak-step trajectory. The policy does not execute
tools and no new rollout is run. The decision step is the first logged
step whose predicted action is ``STOP`` or ``ABSTAIN``. If the policy
never predicts a terminal action, the decision step is the last logged
step.

Force-answer energy is the latest finite ``nutri_dish.energy_kcal`` at or
before that decision. Later logged steps are not credited. A dish with no
finite total is unscored and is not imputed.

The error is absolute kcal against the cafe dish label supplied by the
caller. This module does not open a metadata file and does not read rgb_test.

Full-set MAE is the mean error on the scored dishes, including dishes whose
decision action is ``ABSTAIN``. Coverage and Selective MAE use
``coverage_and_selective_mae`` with that abstain flag (keep the complement).
AURC uses ``aurc``. ``abstain_score`` is caller-supplied and must lie in
``[0, 1]``. Higher score is lower confidence. The #28 table passed softmax
``P(ABSTAIN)``. A later run may pass a calibrated score from
``vdr.eval.abstain_calibration``; this module does not choose which.
No threshold is fit here. A fixed secondary cut at 0.5 is reported beside
the flag rule and is not the primary coverage.

Primary population: logged terminal action ``STOP`` with status ``pass`` or
``corrected`` and a finite terminal ``nutri_dish`` energy. Policy abstentions
on those dishes stay in the full-set MAE.

Sensitivity population: every dish with a finite force-answer energy,
including a logged ``ABSTAIN`` trajectory whose earlier steps wrote a
partial total.

``vdr.eval.selective`` is not modified. No model is fit here.
"""

from __future__ import annotations

import math
import statistics
from dataclasses import dataclass
from typing import Any, Mapping, Sequence

import numpy as np

from vdr.controller.policy_bc import BC_ACTIONS
from vdr.eval.metrics import mae as pair_mae
from vdr.eval.metrics import mape as pair_mape
from vdr.eval.selective import aurc, coverage_and_selective_mae, risk_coverage_curve

DEFINITION_ID = "offline_logged_horizon_force_answer"
DEFINITION = (
    "Offline logged-horizon force-answer. States stay the logged weak-step "
    "trajectory; the policy does not execute tools. The decision step is the "
    "first predicted STOP or ABSTAIN, or the last logged step if the policy "
    "never predicts a terminal action. Force-answer energy is the latest finite "
    "nutri_dish.energy_kcal at or before that step. Later logged calories are "
    "not credited. Dishes with no finite total are unscored and not imputed. "
    "Error is absolute kcal versus the cafe dish label passed in by the caller. "
    "Full-set MAE averages that error on scored dishes and keeps policy "
    "ABSTAIN decisions in the average. Coverage and Selective MAE keep the "
    "dishes whose decision action is not ABSTAIN (vdr.eval.selective, flag "
    "rule). AURC uses the caller-supplied abstain_score at the decision step "
    "(higher means less confident). The score may be softmax P(ABSTAIN) or a "
    "calibrated score recorded by the caller. Ties keep sorted dish_id order. "
    "The primary population is a logged STOP with status pass or corrected "
    "and a finite terminal energy. A sensitivity population also scores "
    "logged ABSTAIN trajectories when an earlier step has a finite total. "
    "A secondary score cut at 0.5 is reported and was not fit. "
    "This is not an on-policy rollout and not an rgb_test VLM evaluation."
)
PRIMARY_POPULATION = "logged_grounded_stop"
SENSITIVITY_POPULATION = "any_finite_horizon_energy"
PRIMARY_POPULATION_NOTE = (
    "Logged terminal action is STOP, top-level s is pass or corrected, and "
    "terminal nutri_dish.energy_kcal is finite. This is the weak-step "
    "train-energy audit set. Policy ABSTAIN decisions remain in the full-set MAE."
)
SENSITIVITY_POPULATION_NOTE = (
    "Every dish with a finite force-answer energy, including logged ABSTAIN "
    "trajectories whose earlier steps wrote a partial nutri_dish total. "
    "A partial total is not the teacher's emitted dish answer."
)
# Conventional probability cut. Not selected on hold-out or rgb_test.
SECONDARY_ABSTAIN_THRESHOLD = 0.5
CURVE_FRACTIONS = (0.1, 0.25, 0.5, 0.75, 1.0)
GROUNDED_STATUS = frozenset({"pass", "corrected"})
TERMINAL_ACTIONS = frozenset({"STOP", "ABSTAIN"})
KNOWN_ACTIONS = frozenset(BC_ACTIONS)

if "VOLUME_ESTIMATE" in KNOWN_ACTIONS or "ABSTAIN" not in KNOWN_ACTIONS or "STOP" not in KNOWN_ACTIONS:
    raise RuntimeError(f"energy eval action set is not the BC head: {tuple(BC_ACTIONS)}")


class EnergyEvalRefused(ValueError):
    """The trajectories or predictions are not an energy-table input."""


@dataclass(frozen=True)
class DishReading:
    """One dish under one policy. Logged fields do not depend on the policy."""

    dish_id: str
    label_kcal: float
    energy_kcal: float | None
    error_kcal: float | None
    abstain_flag: bool
    abstain_score: float
    decision_action: str
    decision_step_i: int
    decision_is_last: bool
    logged_terminal_action: str
    logged_terminal_status: str | None
    logged_terminal_energy_kcal: float | None
    grounded_stop: bool


def _finite_energy(nutri: Any) -> float | None:
    if not isinstance(nutri, Mapping):
        return None
    value = nutri.get("energy_kcal")
    if value is None:
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(number):
        return None
    if number < -1e-3:
        raise EnergyEvalRefused(f"nutri_dish.energy_kcal is negative ({number})")
    return number


def _ordered_steps(steps: Sequence[Mapping[str, Any]], dish_id: str) -> list[Mapping[str, Any]]:
    ordered = sorted(steps, key=lambda row: int(row["step_i"]))
    seen: list[int] = []
    for row in ordered:
        action = row.get("action_t")
        if action == "VOLUME_ESTIMATE":
            raise EnergyEvalRefused(f"{dish_id}: VOLUME_ESTIMATE is not an energy-table action")
        if action not in KNOWN_ACTIONS:
            raise EnergyEvalRefused(f"{dish_id}: unknown logged action {action!r}")
        step_i = row.get("step_i")
        if not isinstance(step_i, int) or isinstance(step_i, bool) or step_i < 0:
            raise EnergyEvalRefused(f"{dish_id}: invalid step_i {step_i!r}")
        seen.append(step_i)
    if len(seen) != len(set(seen)):
        raise EnergyEvalRefused(f"{dish_id}: duplicate step_i")
    return ordered


def _require_score(score: Any, dish_id: str, step_i: int) -> float:
    try:
        number = float(score)
    except (TypeError, ValueError) as exc:
        raise EnergyEvalRefused(f"{dish_id} step {step_i}: abstain_score {score!r}") from exc
    if not math.isfinite(number) or number < -1e-8 or number > 1.0 + 1e-6:
        raise EnergyEvalRefused(f"{dish_id} step {step_i}: abstain_score {number} is outside [0, 1]")
    return number


def _require_action(action: Any, dish_id: str, step_i: int) -> str:
    if not isinstance(action, str) or action not in KNOWN_ACTIONS:
        raise EnergyEvalRefused(f"{dish_id} step {step_i}: prediction {action!r} is not a BC action")
    if action == "VOLUME_ESTIMATE":
        raise EnergyEvalRefused(f"{dish_id} step {step_i}: VOLUME_ESTIMATE prediction")
    return action


def _label_kcal(labels: Mapping[str, float], dish_id: str) -> float:
    if dish_id not in labels:
        raise EnergyEvalRefused(f"{dish_id}: missing cafe energy label")
    try:
        number = float(labels[dish_id])
    except (TypeError, ValueError) as exc:
        raise EnergyEvalRefused(f"{dish_id}: label {labels[dish_id]!r} is not a float") from exc
    if not math.isfinite(number):
        raise EnergyEvalRefused(f"{dish_id}: label is not finite")
    return number


def dish_readings(
    steps_by_dish: Mapping[str, Sequence[Mapping[str, Any]]],
    predictions: Mapping[tuple[str, int], str],
    abstain_scores: Mapping[tuple[str, int], float],
    label_kcal: Mapping[str, float],
) -> list[DishReading]:
    """Build one reading per dish. Dish ids are sorted. Does not fit a model."""
    if not steps_by_dish:
        raise EnergyEvalRefused("no dishes")
    readings: list[DishReading] = []
    for dish_id in sorted(steps_by_dish):
        if not isinstance(dish_id, str) or not dish_id:
            raise EnergyEvalRefused(f"invalid dish id {dish_id!r}")
        ordered = _ordered_steps(steps_by_dish[dish_id], dish_id)
        if not ordered:
            raise EnergyEvalRefused(f"{dish_id}: empty trajectory")
        actions: list[str] = []
        scores: list[float] = []
        for row in ordered:
            step_i = int(row["step_i"])
            key = (dish_id, step_i)
            if key not in predictions:
                raise EnergyEvalRefused(f"{dish_id} step {step_i}: missing prediction")
            if key not in abstain_scores:
                raise EnergyEvalRefused(f"{dish_id} step {step_i}: missing abstain_score")
            actions.append(_require_action(predictions[key], dish_id, step_i))
            scores.append(_require_score(abstain_scores[key], dish_id, step_i))
        decision = len(ordered) - 1
        for index, action in enumerate(actions):
            if action in TERMINAL_ACTIONS:
                decision = index
                break
        energy: float | None = None
        for row in ordered[: decision + 1]:
            value = _finite_energy(row.get("nutri_dish"))
            if value is not None:
                energy = value
        last = ordered[-1]
        terminal_energy = _finite_energy(last.get("nutri_dish"))
        status = last.get("s")
        status_text = status if isinstance(status, str) else None
        grounded = (
            last.get("action_t") == "STOP"
            and status_text in GROUNDED_STATUS
            and terminal_energy is not None
        )
        label = _label_kcal(label_kcal, dish_id)
        error = None if energy is None else abs(energy - label)
        readings.append(
            DishReading(
                dish_id=dish_id,
                label_kcal=label,
                energy_kcal=energy,
                error_kcal=error,
                abstain_flag=actions[decision] == "ABSTAIN",
                abstain_score=scores[decision],
                decision_action=actions[decision],
                decision_step_i=int(ordered[decision]["step_i"]),
                decision_is_last=decision == len(ordered) - 1,
                logged_terminal_action=str(last.get("action_t")),
                logged_terminal_status=status_text,
                logged_terminal_energy_kcal=terminal_energy,
                grounded_stop=grounded,
            )
        )
    return readings


def _average_ranks(values: Sequence[float]) -> np.ndarray:
    order = sorted(range(len(values)), key=lambda index: float(values[index]))
    ranks = np.empty(len(values), dtype=np.float64)
    start = 0
    while start < len(order):
        end = start
        while end + 1 < len(order) and float(values[order[end + 1]]) == float(values[order[start]]):
            end += 1
        average = (start + end) / 2.0
        for pos in range(start, end + 1):
            ranks[order[pos]] = average
        start = end + 1
    return ranks


def _spearman(xs: Sequence[float], ys: Sequence[float]) -> float | None:
    """Spearman correlation. Positive means larger abstain_score lines up with larger error."""
    if len(xs) < 3:
        return None
    left = _average_ranks(xs)
    right = _average_ranks(ys)
    if float(left.std()) == 0.0 or float(right.std()) == 0.0:
        return None
    return float(np.corrcoef(left, right)[0, 1])


def _pearson(xs: Sequence[float], ys: Sequence[float]) -> float | None:
    if len(xs) < 3:
        return None
    left = np.asarray(xs, dtype=np.float64)
    right = np.asarray(ys, dtype=np.float64)
    if float(left.std()) == 0.0 or float(right.std()) == 0.0:
        return None
    return float(np.corrcoef(left, right)[0, 1])


def _curve_sample(curve: Sequence[Mapping[str, float]], n: int) -> list[dict[str, float]]:
    if n < 1 or not curve:
        return []
    samples: list[dict[str, float]] = []
    for fraction in CURVE_FRACTIONS:
        k = min(n, max(1, int(round(fraction * n))))
        point = curve[k - 1]
        samples.append(
            {
                "target_coverage": float(fraction),
                "k": float(point["k"]),
                "coverage": float(point["coverage"]),
                "risk": float(point["risk"]),
            }
        )
    return samples


def _in_population(reading: DishReading, population: str) -> bool:
    if population == PRIMARY_POPULATION:
        return reading.grounded_stop
    if population == SENSITIVITY_POPULATION:
        return True
    raise EnergyEvalRefused(f"unknown population {population!r}")


def summarize_readings(readings: Sequence[DishReading], *, population: str) -> dict[str, Any]:
    """Full-set MAE plus selective.py Coverage / Selective MAE / AURC.

    ``readings`` must already be in sorted dish_id order. That order is the
    AURC tie break when abstain scores are equal.
    """
    chosen = [reading for reading in readings if _in_population(reading, population)]
    scored = [reading for reading in chosen if reading.energy_kcal is not None and reading.error_kcal is not None]
    unscored = [reading for reading in chosen if reading.energy_kcal is None]
    errors = [float(reading.error_kcal) for reading in scored if reading.error_kcal is not None]
    flags = [bool(reading.abstain_flag) for reading in scored]
    scores = [float(reading.abstain_score) for reading in scored]
    labels = [float(reading.label_kcal) for reading in scored]
    energies = [float(reading.energy_kcal) for reading in scored if reading.energy_kcal is not None]
    n_kept = sum(1 for flag in flags if not flag)
    kept_errors = [err for err, flag in zip(errors, flags) if not flag]

    if errors:
        flag_metrics = coverage_and_selective_mae(errors, abstain_flags=flags)
        threshold_metrics = coverage_and_selective_mae(
            errors,
            abstain_scores=scores,
            threshold=SECONDARY_ABSTAIN_THRESHOLD,
        )
        aurc_metrics = aurc(errors, scores)
        curve = risk_coverage_curve(errors, scores)
        via_mae = pair_mae(labels, energies)
        if abs(via_mae - aurc_metrics["risk_at_full_coverage"]) > 1e-8:
            raise EnergyEvalRefused("full-set MAE disagrees with AURC risk at full coverage")
        if abs(via_mae - (sum(errors) / len(errors))) > 1e-8:
            raise EnergyEvalRefused("full-set MAE disagrees with the mean absolute error")
        if abs(flag_metrics["selective_mae"] - (sum(kept_errors) / len(kept_errors) if kept_errors else 0.0)) > 1e-8:
            raise EnergyEvalRefused("selective MAE disagrees with the keep-set mean")
        full_mae: float | None = float(via_mae)
        mape = float(pair_mape(labels, energies))
        median = float(statistics.median(errors))
        signed = float(sum(energy - label for energy, label in zip(energies, labels)) / len(energies))
        spearman = _spearman(scores, errors)
        pearson = _pearson(energies, labels)
        aurc_value: float | None = float(aurc_metrics["aurc"])
        risk_full: float | None = float(aurc_metrics["risk_at_full_coverage"])
        risk_min: float | None = float(aurc_metrics["risk_at_min_coverage"])
        sample = _curve_sample(curve, len(errors))
    else:
        flag_metrics = {"coverage": 0.0, "selective_mae": 0.0, "n": 0.0, "n_kept": 0.0}
        threshold_metrics = {"coverage": 0.0, "selective_mae": 0.0, "n": 0.0, "n_kept": 0.0}
        full_mae = None
        mape = None
        median = None
        signed = None
        spearman = None
        pearson = None
        aurc_value = None
        risk_full = None
        risk_min = None
        sample = []

    equal_terminal = [
        reading.energy_kcal == reading.logged_terminal_energy_kcal
        for reading in scored
        if reading.logged_terminal_energy_kcal is not None
    ]
    return {
        "population": population,
        "n_population": len(chosen),
        "n_scored": len(scored),
        "n_unscored": len(unscored),
        "unscored_dish_ids": [reading.dish_id for reading in unscored],
        "n_policy_abstain": sum(1 for reading in scored if reading.abstain_flag),
        "n_early_decision": sum(1 for reading in chosen if not reading.decision_is_last),
        "n_energy_equals_logged_terminal": sum(1 for flag in equal_terminal if flag),
        "n_energy_differs_from_logged_terminal": sum(1 for flag in equal_terminal if not flag),
        "decision_actions": _counts(reading.decision_action for reading in chosen),
        "full_set_mae_kcal": full_mae,
        "full_set_definition": (
            "Mean |force-answer energy − cafe label| over scored dishes. "
            "Policy ABSTAIN rows stay in this mean when they have a finite energy."
        ),
        "coverage": float(flag_metrics["coverage"]),
        "selective_mae_kcal": None if n_kept == 0 else float(flag_metrics["selective_mae"]),
        "n_kept": n_kept,
        "coverage_of_population": (n_kept / len(chosen)) if chosen else None,
        "aurc": aurc_value,
        "aurc_lower_is_better": True,
        "risk_at_full_coverage": risk_full,
        "risk_at_min_coverage": risk_min,
        "spearman_abstain_score_vs_abs_error": spearman,
        "spearman_note": (
            "Positive: a higher abstain_score lines up with larger absolute error, "
            "which is the ranking AURC rewards. Negative: the score is anti-aligned. "
            "The score is whatever the caller passed (softmax P(ABSTAIN) or a calibrated score)."
        ),
        "secondary_score_threshold": {
            "threshold": SECONDARY_ABSTAIN_THRESHOLD,
            "rule": "keep where abstain_score < 0.5",
            "note": (
                "Fixed conventional cut. Not selected on hold-out or rgb_test. "
                "Primary coverage uses the ABSTAIN action flag, not this cut."
            ),
            "coverage": float(threshold_metrics["coverage"]) if errors else None,
            "selective_mae_kcal": (
                None
                if not errors or int(threshold_metrics["n_kept"]) == 0
                else float(threshold_metrics["selective_mae"])
            ),
            "n_kept": int(threshold_metrics["n_kept"]) if errors else 0,
        },
        "secondary_literature": {
            "mape_percent": mape,
            "median_abs_error_kcal": median,
            "mean_signed_error_kcal": signed,
            "pearson_r_energy_vs_label": pearson,
            "note": "PLAN §7 also lists MAPE and r. They are not the headline.",
        },
        "curve_sample": sample,
        "selective_api": "vdr.eval.selective.coverage_and_selective_mae + aurc",
    }


def _counts(actions: Any) -> dict[str, int]:
    counts: dict[str, int] = {}
    for action in actions:
        counts[str(action)] = counts.get(str(action), 0) + 1
    return dict(sorted(counts.items()))


def _logged_signature(reading: DishReading) -> tuple[Any, ...]:
    return (
        reading.dish_id,
        reading.label_kcal,
        reading.logged_terminal_action,
        reading.logged_terminal_status,
        reading.logged_terminal_energy_kcal,
        reading.grounded_stop,
    )


def _aligned(readings_by_method: Mapping[str, Sequence[DishReading]]) -> dict[str, list[DishReading]]:
    if not readings_by_method:
        raise EnergyEvalRefused("no methods")
    aligned: dict[str, list[DishReading]] = {}
    signatures: list[tuple[Any, ...]] | None = None
    for name in sorted(readings_by_method):
        rows = list(readings_by_method[name])
        if [row.dish_id for row in rows] != sorted(row.dish_id for row in rows):
            raise EnergyEvalRefused(f"{name}: readings are not sorted by dish_id")
        sig = [_logged_signature(row) for row in rows]
        if signatures is None:
            signatures = sig
        elif sig != signatures:
            raise EnergyEvalRefused(f"{name}: logged trajectory fields differ from the other methods")
        aligned[name] = rows
    return aligned


def logged_terminal_audit(readings: Sequence[DishReading]) -> dict[str, Any]:
    """Teacher terminal energy on grounded STOP dishes. Independent of the policy."""
    grounded = [reading for reading in readings if reading.grounded_stop]
    raw: list[float] = []
    rounded: list[float] = []
    for reading in grounded:
        if reading.logged_terminal_energy_kcal is None:
            raise EnergyEvalRefused(f"{reading.dish_id}: grounded STOP has no terminal energy")
        err = abs(reading.logged_terminal_energy_kcal - reading.label_kcal)
        raw.append(err)
        rounded.append(round(err, 4))
    if not raw:
        return {
            "n": 0,
            "mae_kcal": None,
            "mae_rounded_like_weak_step_summary": None,
            "rounding": (
                "Each dish absolute error is rounded to 4 decimals, then the mean "
                "is rounded to 4 decimals. Same rounding as the weak-step summary audit."
            ),
            "note": (
                "Logged teacher STOP energy versus the cafe label. Not a policy row. "
                "Logged ABSTAIN dishes are outside this audit."
            ),
        }
    summary_mae = round(sum(rounded) / len(rounded), 4)
    return {
        "n": len(grounded),
        "mae_kcal": sum(raw) / len(raw),
        "mae_rounded_like_weak_step_summary": summary_mae,
        "rounding": (
            "Each dish absolute error is rounded to 4 decimals, then the mean "
            "is rounded to 4 decimals. Same rounding as the weak-step summary audit."
        ),
        "note": (
            "Logged teacher STOP energy versus the cafe label. Not a policy row. "
            "Logged ABSTAIN dishes are outside this audit."
        ),
    }


def _energy_identity(rows: Sequence[DishReading], *, population: str) -> tuple[tuple[str, float | None], ...]:
    return tuple(
        (reading.dish_id, reading.energy_kcal)
        for reading in rows
        if _in_population(reading, population)
    )


def _identical_groups(
    aligned: Mapping[str, Sequence[DishReading]],
    *,
    population: str,
) -> list[list[str]]:
    groups: list[list[str]] = []
    keys_for: dict[str, tuple[tuple[str, float | None], ...]] = {
        name: _energy_identity(rows, population=population) for name, rows in aligned.items()
    }
    for name in sorted(keys_for):
        placed = False
        for group in groups:
            if keys_for[group[0]] == keys_for[name]:
                group.append(name)
                placed = True
                break
        if not placed:
            groups.append([name])
    for group in groups:
        group.sort()
    groups.sort(key=lambda group: group[0])
    return groups


def _paired(
    aligned: Mapping[str, Sequence[DishReading]],
    *,
    population: str,
) -> dict[str, Any]:
    """Dishes in the population that every method can score. Same dish order."""
    names = sorted(aligned)
    by_id = {name: {row.dish_id: row for row in rows} for name, rows in aligned.items()}
    first = aligned[names[0]]
    common: list[str] = []
    for reading in first:
        if not _in_population(reading, population):
            continue
        dish_id = reading.dish_id
        if all(
            _in_population(by_id[name][dish_id], population) and by_id[name][dish_id].energy_kcal is not None
            for name in names
        ):
            common.append(dish_id)
    blocks = {
        name: summarize_readings(
            [by_id[name][dish_id] for dish_id in common],
            population=population,
        )
        for name in names
    }
    return {
        "population": population,
        "n_common_scored": len(common),
        "note": (
            "Intersection of dishes in this population with a finite force-answer "
            "energy under every method. Use it when one method leaves a dish unscored."
        ),
        "methods": blocks,
    }


def evaluate_energy_table(
    readings_by_method: Mapping[str, Sequence[DishReading]],
) -> dict[str, Any]:
    """Score each method. Does not read a file and does not fit a threshold."""
    aligned = _aligned(readings_by_method)
    reference_name = sorted(aligned)[0]
    audit = logged_terminal_audit(aligned[reference_name])
    methods: dict[str, Any] = {}
    for name, rows in aligned.items():
        methods[name] = {
            "primary_logged_grounded_stop": summarize_readings(rows, population=PRIMARY_POPULATION),
            "sensitivity_any_finite_horizon_energy": summarize_readings(
                rows,
                population=SENSITIVITY_POPULATION,
            ),
        }
    return {
        "metric": "Full-set energy MAE + Coverage + Selective MAE + AURC",
        "definition_id": DEFINITION_ID,
        "definition": DEFINITION,
        "populations": {
            PRIMARY_POPULATION: PRIMARY_POPULATION_NOTE,
            SENSITIVITY_POPULATION: SENSITIVITY_POPULATION_NOTE,
        },
        "headline": "primary_logged_grounded_stop",
        "methods": methods,
        "identical_force_answer_energy": {
            PRIMARY_POPULATION: _identical_groups(aligned, population=PRIMARY_POPULATION),
            SENSITIVITY_POPULATION: _identical_groups(aligned, population=SENSITIVITY_POPULATION),
        },
        "paired_common_scored": {
            PRIMARY_POPULATION: _paired(aligned, population=PRIMARY_POPULATION),
            SENSITIVITY_POPULATION: _paired(aligned, population=SENSITIVITY_POPULATION),
        },
        "teacher_terminal_audit": audit,
        "audit": {
            "uses_test_gt_for_thresholds": False,
            "threshold_fit": False,
            "secondary_threshold": SECONDARY_ABSTAIN_THRESHOLD,
            "online_rollout": False,
            "vlm": False,
            "volume_tool": False,
            "legacy_regression_weights": False,
            "rl": False,
            "model_fit_in_this_module": False,
        },
    }


def _fmt(value: float | None, digits: int = 4) -> str:
    if value is None:
        return "n/a"
    return f"{value:.{digits}f}"


def _method_line(name: str, block: Mapping[str, Any]) -> str:
    return (
        f"{name}: n_scored={block['n_scored']}/{block['n_population']} "
        f"full_MAE={_fmt(block['full_set_mae_kcal'])} "
        f"coverage={_fmt(block['coverage'])} ({block['n_kept']}/{block['n_scored']}) "
        f"selective_MAE={_fmt(block['selective_mae_kcal'])} "
        f"AURC={_fmt(block['aurc'])} "
        f"spearman={_fmt(block['spearman_abstain_score_vs_abs_error'])} "
        f"early={block['n_early_decision']}"
    )


def _identical_line(groups: Sequence[Sequence[str]]) -> str:
    parts = [" = ".join(group) for group in groups if len(group) > 1]
    if not parts:
        return "force-answer kcal: each method has its own energy vector"
    return "force-answer kcal identical: " + "; ".join(parts)


def format_energy_report(payload: Mapping[str, Any]) -> str:
    """Plain-text table. Numbers come from ``evaluate_energy_table``."""
    methods: Mapping[str, Any] = payload["methods"]
    order = list(payload.get("method_order") or sorted(methods))
    lines = [
        "Full-set energy MAE / Coverage / Selective MAE / AURC",
        f"definition: {payload['definition_id']}",
        f"headline population: {payload['headline']}",
        payload["populations"][PRIMARY_POPULATION],
        "",
        str(
            payload.get("primary_title")
            or "Primary — logged STOP pass/corrected (force-answer keeps policy ABSTAIN):"
        ),
    ]
    for name in order:
        lines.append(_method_line(name, methods[name]["primary_logged_grounded_stop"]))
    lines.append(_identical_line(payload["identical_force_answer_energy"][PRIMARY_POPULATION]))
    for name in order:
        block = methods[name]["primary_logged_grounded_stop"]
        if block["n_unscored"]:
            lines.append(f"  {name} unscored: {', '.join(block['unscored_dish_ids'])}")
    paired = payload["paired_common_scored"][PRIMARY_POPULATION]
    lines.append(
        f"paired common scored n={paired['n_common_scored']} "
        "(dishes every method can score):"
    )
    for name in order:
        if name in paired["methods"]:
            lines.append("  " + _method_line(name, paired["methods"][name]))
    lines.extend(
        [
            "",
            "Sensitivity — any finite horizon energy, including logged-abstain partial totals:",
            payload["populations"][SENSITIVITY_POPULATION],
        ]
    )
    for name in order:
        lines.append(_method_line(name, methods[name]["sensitivity_any_finite_horizon_energy"]))
    lines.append(_identical_line(payload["identical_force_answer_energy"][SENSITIVITY_POPULATION]))
    audit = payload["teacher_terminal_audit"]
    lines.extend(
        [
            "",
            (
                f"{payload.get('terminal_audit_title') or 'teacher terminal audit (not a policy row)'}: "
                f"n={audit['n']} MAE={_fmt(audit['mae_kcal'])} "
                f"summary_round={_fmt(audit['mae_rounded_like_weak_step_summary'])}"
            ),
            "",
            (
                f"Secondary score cut abstain_score < {SECONDARY_ABSTAIN_THRESHOLD} "
                "is in the json under secondary_score_threshold. "
                "Primary coverage is the ABSTAIN action flag."
            ),
            "AURC is kcal (same units as MAE), trapezoid from vdr.eval.selective, lower better.",
            "Spearman > 0 means a higher abstain_score lines up with larger absolute error.",
            "",
            str(payload.get("definition") or DEFINITION),
        ]
    )
    sources = payload.get("abstain_score_sources")
    if isinstance(sources, Mapping):
        lines.extend(["", "Abstain score source:"])
        for key, text in sources.items():
            lines.append(f"  {key}: {text}")
    reference = payload.get("softmax_p_abstain_reference")
    if isinstance(reference, Mapping):
        lines.extend(
            [
                "",
                "Softmax P(ABSTAIN) reference (not the headline; same actions, #28 score):",
            ]
        )
        for name in order:
            block = reference.get(name)
            if isinstance(block, Mapping):
                lines.append("  " + _method_line(name, block))
    scope = payload.get("scope")
    if isinstance(scope, Mapping):
        lines.extend(["", "Scope:"])
        for key in (
            "split",
            "labels",
            "rgb_test",
            "on_policy",
            "offline_holdout",
            "vlm",
            "b0_b1",
            "fixrate_and_accuracy",
            "b5",
            "closed_loop_weights",
            "abstain_score",
            "mask",
            "fixrate",
        ):
            if key in scope:
                lines.append(f"  {key}: {scope[key]}")
    sanity = payload.get("loader_sanity")
    if isinstance(sanity, Mapping):
        lines.extend(["", "Loader sanity (action agreement is not method proof):"])
        for key, value in sanity.items():
            lines.append(f"  {key}: {value}")
    return "\n".join(lines) + "\n"


def _check(condition: bool, message: str) -> None:
    if not condition:
        raise AssertionError(message)


def _stop_step(step_i: int, energy: float | None, *, status: str = "corrected") -> dict[str, Any]:
    return {
        "step_i": step_i,
        "action_t": "STOP",
        "s": status,
        "nutri_dish": None if energy is None else {"energy_kcal": energy},
    }


def _tool_step(step_i: int, action: str, energy: float | None) -> dict[str, Any]:
    return {
        "step_i": step_i,
        "action_t": action,
        "s": None,
        "nutri_dish": None if energy is None else {"energy_kcal": energy},
    }


def run_self_check() -> None:
    """Synthetic contract. Does not read weak steps, weights, or labels on disk."""
    _check(SECONDARY_ABSTAIN_THRESHOLD == 0.5, "secondary cut stays at 0.5")
    _check("VOLUME_ESTIMATE" not in KNOWN_ACTIONS, "volume tool is outside the head")

    good = aurc([1.0, 100.0], [0.1, 0.9])["aurc"]
    bad = aurc([1.0, 100.0], [0.9, 0.1])["aurc"]
    _check(good < bad, "aligned abstain scores must produce a lower AURC")

    early_steps = [
        _tool_step(0, "RETRIEVE", 40.0),
        _tool_step(1, "RE_PORTION", 70.0),
        _stop_step(2, 100.0),
    ]
    follow_pred = {( "early", 0): "RETRIEVE", ("early", 1): "RE_PORTION", ("early", 2): "STOP"}
    follow_score = {("early", 0): 0.2, ("early", 1): 0.2, ("early", 2): 0.05}
    early_pred = {("early", 0): "STOP", ("early", 1): "STOP", ("early", 2): "STOP"}
    early_score = {("early", 0): 0.1, ("early", 1): 0.1, ("early", 2): 0.1}
    labels = {"early": 90.0}
    follow = dish_readings({"early": early_steps}, follow_pred, follow_score, labels)[0]
    early = dish_readings({"early": early_steps}, early_pred, early_score, labels)[0]
    _check(follow.decision_is_last and follow.energy_kcal == 100.0, "follow-through uses the terminal total")
    _check(follow.error_kcal == 10.0, "follow-through error is |100-90|")
    _check(early.decision_step_i == 0 and early.energy_kcal == 40.0, "early STOP does not read later calories")
    _check(early.error_kcal == 50.0, "early STOP error is |40-90|, not the terminal error")
    _check(follow.grounded_stop and early.grounded_stop, "logged STOP stays grounded for both policies")

    flag_steps = {
        "a_stop": [_stop_step(0, 50.0)],
        "b_abstain": [_stop_step(0, 5.0)],
    }
    flag_pred = {("a_stop", 0): "STOP", ("b_abstain", 0): "ABSTAIN"}
    flag_score = {("a_stop", 0): 0.9, ("b_abstain", 0): 0.1}
    flag_labels = {"a_stop": 40.0, "b_abstain": 105.0}
    flag_rows = dish_readings(flag_steps, flag_pred, flag_score, flag_labels)
    flag_table = evaluate_energy_table({"flag_case": flag_rows})
    primary = flag_table["methods"]["flag_case"]["primary_logged_grounded_stop"]
    _check(primary["n_scored"] == 2 and primary["full_set_mae_kcal"] == 55.0, "force-answer keeps the abstention")
    _check(primary["n_kept"] == 1 and primary["selective_mae_kcal"] == 10.0, "flag keep-set is the STOP dish")
    _check(abs(primary["coverage"] - 0.5) < 1e-12, "flag coverage is 1/2")
    secondary = primary["secondary_score_threshold"]
    _check(secondary["n_kept"] == 1 and secondary["selective_mae_kcal"] == 100.0, "0.5 cut is not the flag rule")
    _check(primary["aurc"] is not None and primary["risk_at_full_coverage"] == 55.0, "AURC full-coverage risk is the MAE")

    partial_steps = {
        "logged_abstain": [
            _tool_step(0, "RETRIEVE", 15.0),
            {
                "step_i": 1,
                "action_t": "ABSTAIN",
                "s": "abstain",
                "nutri_dish": {"energy_kcal": None},
            },
        ]
    }
    partial_pred = {("logged_abstain", 0): "RETRIEVE", ("logged_abstain", 1): "STOP"}
    partial_score = {("logged_abstain", 0): 0.0, ("logged_abstain", 1): 0.2}
    partial = dish_readings(partial_steps, partial_pred, partial_score, {"logged_abstain": 80.0})[0]
    _check(not partial.grounded_stop and partial.energy_kcal == 15.0, "abstain log can still carry a partial total")
    partial_table = evaluate_energy_table({"partial": [partial]})
    _check(
        partial_table["methods"]["partial"]["primary_logged_grounded_stop"]["n_population"] == 0,
        "logged abstain is outside the primary population",
    )
    sensitivity = partial_table["methods"]["partial"]["sensitivity_any_finite_horizon_energy"]
    _check(sensitivity["n_scored"] == 1 and sensitivity["full_set_mae_kcal"] == 65.0, "partial enters the sensitivity MAE")

    empty_steps = {
        "none": [
            {
                "step_i": 0,
                "action_t": "ABSTAIN",
                "s": "abstain",
                "nutri_dish": {"energy_kcal": None},
            }
        ]
    }
    empty = dish_readings(
        empty_steps,
        {("none", 0): "ABSTAIN"},
        {("none", 0): 1.0},
        {"none": 10.0},
    )
    empty_table = evaluate_energy_table({"empty": empty})
    _check(
        empty_table["methods"]["empty"]["sensitivity_any_finite_horizon_energy"]["n_unscored"] == 1,
        "null energy is unscored",
    )
    _check(
        empty_table["methods"]["empty"]["sensitivity_any_finite_horizon_energy"]["full_set_mae_kcal"] is None,
        "unscored dishes are not imputed as zero error",
    )

    audit_rows = dish_readings({"early": early_steps}, follow_pred, follow_score, labels)
    audit = logged_terminal_audit(audit_rows)
    _check(audit["n"] == 1 and audit["mae_rounded_like_weak_step_summary"] == 10.0, "terminal audit ignores the policy")

    paired = evaluate_energy_table(
        {
            "follow": dish_readings({"early": early_steps}, follow_pred, follow_score, labels),
            "early": dish_readings({"early": early_steps}, early_pred, early_score, labels),
        }
    )
    _check(
        paired["methods"]["follow"]["primary_logged_grounded_stop"]["full_set_mae_kcal"] == 10.0,
        "paired follow MAE",
    )
    _check(
        paired["methods"]["early"]["primary_logged_grounded_stop"]["full_set_mae_kcal"] == 50.0,
        "paired early MAE",
    )
    groups = paired["identical_force_answer_energy"][PRIMARY_POPULATION]
    _check(all(len(group) == 1 for group in groups), "different horizon energies stay in different groups")
    text = format_energy_report(paired)
    _check("full_MAE=10.0000" in text and "full_MAE=50.0000" in text, "report carries both MAEs")
    _check("AURC=" in text, "report names AURC")

    raised = False
    try:
        dish_readings(
            {"bad": [{"step_i": 0, "action_t": "VOLUME_ESTIMATE", "s": None, "nutri_dish": None}]},
            {("bad", 0): "STOP"},
            {("bad", 0): 0.0},
            {"bad": 1.0},
        )
    except EnergyEvalRefused:
        raised = True
    _check(raised, "VOLUME_ESTIMATE is refused")

    raised = False
    try:
        dish_readings(
            {"bad": [_stop_step(0, 1.0), _stop_step(0, 2.0)]},
            {("bad", 0): "STOP"},
            {("bad", 0): 0.0},
            {"bad": 1.0},
        )
    except EnergyEvalRefused:
        raised = True
    _check(raised, "duplicate step_i is refused")


__all__ = [
    "DEFINITION_ID",
    "DishReading",
    "EnergyEvalRefused",
    "PRIMARY_POPULATION",
    "SECONDARY_ABSTAIN_THRESHOLD",
    "SENSITIVITY_POPULATION",
    "dish_readings",
    "evaluate_energy_table",
    "format_energy_report",
    "logged_terminal_audit",
    "run_self_check",
]
