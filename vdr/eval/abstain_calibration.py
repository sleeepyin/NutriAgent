"""Abstain score for selective energy risk. Not a new action head.

The closed-loop MLP still emits the action. This module only replaces the
score that AURC sorts by. Higher score means less confident.

Why this score
--------------
On the fit pool, absolute kcal error grows with the committed
``nutri_dish.energy_kcal``. Softmax ``P(ABSTAIN)`` does not rank that error:
its smallest values include large residuals, and those rows dominate AURC.
A fit-internal slice (numeric dish id modulo 5 equals 0) compares a fixed
candidate list. The locked rule is

    abstain_score = clip(committed_kcal / ENERGY_CAP_KCAL, 0, 1)

``ENERGY_CAP_KCAL`` is the existing 2500 ceiling in
``vdr.controller.state_features``. It is not a threshold searched on the
official hold-out or on rgb_test. Missing energy scores 1 (least confident).
Dishes with no finite total stay unscored in the energy table; the 1 is only
a placeholder on earlier steps.

The official rgb_train hold-out and rgb_test labels are not inputs to
``select_rule``. Action argmax is untouched, so force-answer horizons stay
on the logged trajectory the saved policy already produces.
"""

from __future__ import annotations

import math
from typing import Any, Mapping, Sequence

from vdr.controller.state_features import ENERGY_CAP_KCAL
from vdr.eval.selective import aurc

SCORE_RULE_ID = "committed_kcal_over_energy_cap"
# Fixed before the official hold-out energy table. Not chosen by sweeping moduli.
FIT_VAL_MODULUS = 5
CANDIDATE_RULES = (
    "committed_kcal_over_energy_cap",
    "softmax_p_abstain",
    "obs_nutri_energy_norm",
    "stop_gap_energy",
)
_MIN_SPLIT = 5


class AbstainCalibrationError(ValueError):
    """The score or the fit-internal selection is not usable."""


def score_committed_kcal(energy_kcal: float | None) -> float:
    """Map a committed total to an abstain score in ``[0, 1]``."""
    if energy_kcal is None:
        return 1.0
    try:
        number = float(energy_kcal)
    except (TypeError, ValueError) as exc:
        raise AbstainCalibrationError(f"committed kcal {energy_kcal!r} is not a float") from exc
    if not math.isfinite(number) or number < -1e-3:
        raise AbstainCalibrationError(f"committed kcal {number} is not a non-negative finite total")
    if number < 0.0:
        number = 0.0
    capped = min(number, float(ENERGY_CAP_KCAL))
    return capped / float(ENERGY_CAP_KCAL)


def _finite_kcal(nutri: Any) -> float | None:
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
        raise AbstainCalibrationError(f"nutri_dish.energy_kcal is negative ({number})")
    return number


def logged_committed_kcal_scores(
    steps_by_dish: Mapping[str, Sequence[Mapping[str, Any]]],
) -> dict[tuple[str, int], float]:
    """Score every logged step by the latest finite total at or before that step.

    The value at the decision step is ``score_committed_kcal`` of the
    force-answer energy. Later logged totals are not visible at that step.
    """
    scores: dict[tuple[str, int], float] = {}
    for dish_id, steps in steps_by_dish.items():
        ordered = sorted(steps, key=lambda row: int(row["step_i"]))
        seen: set[int] = set()
        running: float | None = None
        for row in ordered:
            step_i = int(row["step_i"])
            if step_i in seen:
                raise AbstainCalibrationError(f"{dish_id}: duplicate step_i {step_i}")
            seen.add(step_i)
            value = _finite_kcal(row.get("nutri_dish"))
            if value is not None:
                running = value
            scores[(dish_id, step_i)] = score_committed_kcal(running)
    return scores


def dish_in_fit_val(dish_id: str) -> bool:
    """True for the fit-internal validation slice. Not the official hold-out."""
    digits = "".join(ch for ch in dish_id if ch.isdigit())
    if not digits:
        raise AbstainCalibrationError(f"dish id {dish_id!r} has no digits for the fit-internal split")
    return int(digits) % FIT_VAL_MODULUS == 0


def _clip01(value: float) -> float:
    number = float(value)
    if not math.isfinite(number):
        raise AbstainCalibrationError("candidate score is not finite")
    if number < 0.0:
        return 0.0
    if number > 1.0:
        return 1.0
    return number


def candidate_scores(row: Mapping[str, Any]) -> dict[str, float]:
    """Fixed candidates. The set is not extended from hold-out results."""
    energy = row.get("energy_kcal")
    enorm = score_committed_kcal(None if energy is None else float(energy))
    p_stop = _clip01(float(row["p_stop"]))
    return {
        "committed_kcal_over_energy_cap": enorm,
        "softmax_p_abstain": _clip01(float(row["p_abstain"])),
        "obs_nutri_energy_norm": _clip01(float(row["obs_energy_norm"])),
        "stop_gap_energy": _clip01(1.0 - (1.0 - enorm) * p_stop),
    }


def select_rule(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """Pick the candidate with the lowest fit-internal validation AURC.

    Ties break by candidate name. Does not read the official hold-out.
    """
    if len(rows) < _MIN_SPLIT * 2:
        raise AbstainCalibrationError(f"need at least {_MIN_SPLIT * 2} fit rows, got {len(rows)}")
    val_flags = [dish_in_fit_val(str(row["dish_id"])) for row in rows]
    n_val = sum(val_flags)
    n_train = len(rows) - n_val
    if n_val < _MIN_SPLIT or n_train < _MIN_SPLIT:
        raise AbstainCalibrationError(f"fit-internal split is too small (train={n_train}, val={n_val})")

    per_rule: dict[str, dict[str, float]] = {}
    for rule in CANDIDATE_RULES:
        val_errors: list[float] = []
        val_scores: list[float] = []
        for row, is_val in zip(rows, val_flags):
            if not is_val:
                continue
            val_errors.append(float(row["error_kcal"]))
            val_scores.append(candidate_scores(row)[rule])
        per_rule[rule] = {
            "n": float(n_val),
            "aurc": float(aurc(val_errors, val_scores)["aurc"]),
            "mae": float(sum(val_errors) / len(val_errors)),
        }
    winner = min(per_rule, key=lambda name: (per_rule[name]["aurc"], name))
    return {
        "rule_id": SCORE_RULE_ID,
        "winner": winner,
        "winner_matches_locked_rule": winner == SCORE_RULE_ID,
        "energy_cap_kcal": float(ENERGY_CAP_KCAL),
        "formula": (
            "clip(committed nutri_dish.energy_kcal / ENERGY_CAP_KCAL, 0, 1); "
            "missing energy -> 1"
        ),
        "fit_internal_val": {
            "modulus": FIT_VAL_MODULUS,
            "val_when": "int(digits of dish_id) % 5 == 0",
            "n_train": n_train,
            "n_val": n_val,
            "val_mae": per_rule[winner]["mae"],
            "selection_metric": "validation AURC, lower better; tie break is the candidate name",
            "candidates": per_rule,
        },
        "uses_official_holdout": False,
        "uses_test_gt": False,
        "threshold_fit_on_holdout": False,
        "rl": False,
    }


def run_self_check() -> None:
    """Synthetic contract. Does not read weak steps or labels on disk."""
    if abs(float(ENERGY_CAP_KCAL) - 2500.0) > 1e-9:
        raise AbstainCalibrationError("ENERGY_CAP_KCAL drifted from the state-feature ceiling 2500")
    if score_committed_kcal(0.0) != 0.0:
        raise AbstainCalibrationError("zero kcal must be the most confident score")
    if score_committed_kcal(1250.0) != 0.5:
        raise AbstainCalibrationError("half the cap must score 0.5")
    if score_committed_kcal(2500.0) != 1.0 or score_committed_kcal(9000.0) != 1.0:
        raise AbstainCalibrationError("the cap must saturate the score at 1")
    if score_committed_kcal(None) != 1.0:
        raise AbstainCalibrationError("missing energy must be least confident")
    if not score_committed_kcal(100.0) < score_committed_kcal(800.0):
        raise AbstainCalibrationError("the score must increase with committed kcal")

    steps = {
        "dish_1": [
            {"step_i": 0, "nutri_dish": {"energy_kcal": 100.0}},
            {"step_i": 1, "nutri_dish": None},
            {"step_i": 2, "nutri_dish": {"energy_kcal": 400.0}},
        ]
    }
    scored = logged_committed_kcal_scores(steps)
    if scored[("dish_1", 0)] != score_committed_kcal(100.0):
        raise AbstainCalibrationError("step 0 must see 100 kcal")
    if scored[("dish_1", 1)] != score_committed_kcal(100.0):
        raise AbstainCalibrationError("a null total must keep the previous committed kcal")
    if scored[("dish_1", 2)] != score_committed_kcal(400.0):
        raise AbstainCalibrationError("step 2 must not keep the earlier total")

    # Validation ids are multiples of 5. Errors equal the committed kcal, so
    # that score is perfectly ordered. Softmax P(ABSTAIN) is reversed.
    explicit = [
        ("dish_30", 1600.0),
        ("dish_11", 80.0),
        ("dish_10", 50.0),
        ("dish_16", 640.0),
        ("dish_20", 400.0),
        ("dish_12", 160.0),
        ("dish_25", 800.0),
        ("dish_13", 320.0),
        ("dish_15", 200.0),
        ("dish_17", 1280.0),
    ]
    built = []
    for dish_id, energy in explicit:
        built.append(
            {
                "dish_id": dish_id,
                "energy_kcal": energy,
                "error_kcal": energy,
                "p_abstain": 1.0 - score_committed_kcal(energy),
                "p_stop": 0.5,
                "obs_energy_norm": 0.5,
            }
        )
    chosen = select_rule(built)
    if chosen["winner"] != SCORE_RULE_ID:
        raise AbstainCalibrationError(f"synthetic selection picked {chosen['winner']}")
    val = chosen["fit_internal_val"]["candidates"]
    if not val[SCORE_RULE_ID]["aurc"] < val["softmax_p_abstain"]["aurc"]:
        raise AbstainCalibrationError("aligned kcal score should beat reversed P(ABSTAIN)")
    if dish_in_fit_val("dish_10") is not True or dish_in_fit_val("dish_11") is not False:
        raise AbstainCalibrationError("fit-internal slice disagrees with modulo 5")
