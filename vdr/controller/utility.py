"""Rule utility and the shared ``argmax value / cost`` rule.

Costs are fixed positive weights. They were not searched on hold-out or on
rgb_test.

- ``CORRECT_CODE`` 1 — swap a candidate already in hand
- ``RE_PORTION`` 1 — commit the Mass Estimator grams
- ``STOP`` 1 and ``ABSTAIN`` 1 — decisions, not catalog or vision calls
- ``RETRIEVE`` 2 — another FNDDS query
- ``RE_DECOMPOSE`` 3 — another perception pass; the live path does not take one

``UNCERTAINTY_MAJORITY = 0.5`` is the point where a ``[0, 1]`` uncertainty
exceeds its complement (``U > 1 - U``). It is not a threshold fit on data.

The heuristic is the first controller. It does not imitate the weak-step
teacher and it does not read labels. ``Q_φ`` uses the same costs and the
same tie break, with learned ``ΔL`` in place of these rules.
"""

from __future__ import annotations

import math
from typing import Mapping

import numpy as np

from vdr.controller.policy_bc import BC_ACTIONS
from vdr.controller.q_state import Z_DIM, Z_FEATURE_NAMES

# Fixed. Not fit on hold-out or rgb_test.
ACTION_COST: dict[str, float] = {
    "CORRECT_CODE": 1.0,
    "RE_PORTION": 1.0,
    "STOP": 1.0,
    "ABSTAIN": 1.0,
    "RETRIEVE": 2.0,
    "RE_DECOMPOSE": 3.0,
}
# Equal scores keep the earlier action. STOP beats a tied tool.
TIE_ORDER: tuple[str, ...] = (
    "STOP",
    "ABSTAIN",
    "CORRECT_CODE",
    "RE_PORTION",
    "RETRIEVE",
    "RE_DECOMPOSE",
)
UNCERTAINTY_MAJORITY = 0.5

_INDEX = {name: index for index, name in enumerate(Z_FEATURE_NAMES)}

if set(ACTION_COST) != set(BC_ACTIONS) or set(TIE_ORDER) != set(BC_ACTIONS):
    raise RuntimeError("action costs drifted from the BC action set")
if any(cost <= 0.0 for cost in ACTION_COST.values()):
    raise RuntimeError("every action cost must be positive")
if "VOLUME_ESTIMATE" in ACTION_COST:
    raise RuntimeError("volume cost is not part of the controller")


def values_over_cost(values: Mapping[str, float]) -> dict[str, float]:
    """``value / cost``. Non-finite values other than ``-inf`` are refused."""
    scores: dict[str, float] = {}
    for action in BC_ACTIONS:
        if action not in values:
            raise KeyError(f"missing value for {action}")
        value = float(values[action])
        if math.isnan(value) or (math.isinf(value) and value > 0.0):
            raise ValueError(f"{action} value is not a usable score: {value}")
        if value == float("-inf"):
            scores[action] = float("-inf")
        else:
            scores[action] = value / ACTION_COST[action]
    return scores


def legal_actions(row: np.ndarray) -> dict[str, bool]:
    """Tools the closed-loop rollout can actually run from this ``z``.

    ``RE_PORTION`` without a code and ``RETRIEVE`` / ``RE_DECOMPOSE`` without
    an open component return a failed observation and do not change ``n``.
    They stay out of the argmax. ``STOP`` and ``ABSTAIN`` are always available.
    These are the tool preconditions, not a threshold fit on hold-out.
    """
    vector = np.asarray(row, dtype=np.float64)
    if vector.shape != (Z_DIM,):
        raise ValueError(f"legal_actions expected shape {(Z_DIM,)}, got {vector.shape}")
    open_ = _component(vector, "gate_component_open") >= 0.5
    code = _component(vector, "gate_code_present") >= 0.5
    return {
        "RETRIEVE": open_,
        "CORRECT_CODE": code,
        "RE_PORTION": code,
        "RE_DECOMPOSE": open_,
        "ABSTAIN": True,
        "STOP": True,
    }


def argmax_value_over_cost(
    values: Mapping[str, float],
    legal: Mapping[str, bool] | None = None,
) -> str:
    """``a* = argmax_a value(a) / Cost(a)``, with the fixed tie order.

    ``legal`` drops actions the tools cannot execute. A missing map keeps
    every action. Dropped actions score as ``-inf``.
    """
    scored = dict(values)
    if legal is not None:
        for action in BC_ACTIONS:
            if action not in legal:
                raise KeyError(f"legal map is missing {action}")
            if not legal[action]:
                scored[action] = float("-inf")
    scores = values_over_cost(scored)
    best_action: str | None = None
    best_score: float | None = None
    for action in TIE_ORDER:
        score = scores[action]
        if best_score is None or score > best_score + 1e-12:
            best_score = score
            best_action = action
    if best_action is None:
        raise RuntimeError("argmax_value_over_cost had no actions")
    return best_action


def _component(row: np.ndarray, name: str) -> float:
    return float(row[_INDEX[name]])


def heuristic_utility(row: np.ndarray) -> dict[str, float]:
    """Per-action utility in ``[0, 1]`` from one ``z`` row. No labels."""
    vector = np.asarray(row, dtype=np.float64)
    if vector.shape != (Z_DIM,):
        raise ValueError(f"heuristic_utility expected shape {(Z_DIM,)}, got {vector.shape}")
    if not np.isfinite(vector).all():
        raise ValueError("heuristic_utility received a non-finite z")

    u_id = _component(vector, "U_id")
    u_mass = _component(vector, "U_mass")
    u_nut = _component(vector, "U_nutrient")
    u_evi = _component(vector, "U_evidence")
    open_ = _component(vector, "gate_component_open") >= 0.5
    code = _component(vector, "gate_code_present") >= 0.5
    top1 = _component(vector, "gate_code_top1") >= 0.5
    in_cands = _component(vector, "gate_code_in_cands") >= 0.5
    applied = _component(vector, "gate_specialist_applied") >= 0.5
    dish = _component(vector, "gate_dish_energy") >= 0.5
    force_abstain = _component(vector, "safety_force_abstain") >= 0.5
    force_stop = _component(vector, "safety_force_stop") >= 0.5

    utility = {action: 0.0 for action in BC_ACTIONS}
    if force_abstain:
        utility["ABSTAIN"] = 1.0
        return utility
    if force_stop:
        utility["STOP"] = 1.0
        return utility

    needs_fix = open_ and code and in_cands and not top1
    if open_:
        utility["RETRIEVE"] = u_id
    if needs_fix:
        # An alternate candidate is in hand. Do not portion that code yet.
        utility["CORRECT_CODE"] = 1.0
    elif open_ and code and not applied and u_id < UNCERTAINTY_MAJORITY:
        # Identity is not the majority problem, and Mass has not been committed.
        utility["RE_PORTION"] = 1.0
    elif open_ and code:
        utility["RE_PORTION"] = u_mass

    if open_ and code and applied and u_nut >= UNCERTAINTY_MAJORITY:
        if needs_fix:
            utility["CORRECT_CODE"] = max(utility["CORRECT_CODE"], u_nut)
        else:
            utility["RETRIEVE"] = max(utility["RETRIEVE"], u_nut)

    if open_ and not code:
        utility["RE_DECOMPOSE"] = u_evi

    if not open_ and not dish:
        utility["ABSTAIN"] = 1.0
    elif not open_ and u_evi >= UNCERTAINTY_MAJORITY:
        utility["ABSTAIN"] = u_evi

    if open_:
        utility["STOP"] = 0.0
    else:
        utility["STOP"] = max(0.0, 1.0 - max(u_id, u_mass, u_nut, u_evi))
    return utility


def heuristic_action(row: np.ndarray) -> str:
    """``argmax utility / cost`` for one ``z`` row."""
    return argmax_value_over_cost(heuristic_utility(row), legal_actions(row))


def run_self_check() -> None:
    """Costs, tie break, and the four specialist rules. No labels."""
    from vdr.controller.q_state import build_z

    def _row(**overrides: object) -> np.ndarray:
        state = {
            "hyp_i": "apple",
            "code_i": None,
            "cands_i": [],
            "portion_g_i": None,
            "portion_src_i": None,
            "verify_codes": [],
            "evidence_edges": [],
            "nutri_i": None,
            "nutri_dish": None,
            "step_i": 0,
        }
        state.update(overrides)
        return build_z(state, None, {"tools_used": 0})

    if heuristic_action(_row()) != "RETRIEVE":
        raise AssertionError("an open component with no code should retrieve")

    portion_state = _row(
        code_i="111",
        cands_i=[{"code": "111", "score": 0.95}, {"code": "222", "score": 0.4}],
        portion_g_i=100.0,
        portion_src_i="grid_search",
        nutri_i={"energy_kcal": 100.0, "protein_g": 1.0, "carb_g": 10.0, "fat_g": 1.0, "mass_g": 100.0},
        evidence_edges=[{"type": "n<-code×portion", "src": "n", "dst": "c"}],
    )
    if heuristic_action(portion_state) != "RE_PORTION":
        raise AssertionError(heuristic_action(portion_state))

    correct_state = _row(
        code_i="222",
        cands_i=[{"code": "111", "score": 0.95}, {"code": "222", "score": 0.7}],
        portion_g_i=100.0,
        portion_src_i="grid_search",
    )
    if heuristic_action(correct_state) != "CORRECT_CODE":
        raise AssertionError(heuristic_action(correct_state))

    done = build_z(
        {
            "hyp_i": None,
            "code_i": None,
            "cands_i": [],
            "portion_g_i": None,
            "verify_codes": [],
            "evidence_edges": [{"type": "n<-aggregate", "src": "dish", "dst": "n"}],
            "nutri_dish": {"energy_kcal": 180.0, "mass_g": 200.0, "protein_g": 2.0, "carb_g": 20.0, "fat_g": 4.0},
            "nutri_i": None,
            "step_i": 4,
        }
    )
    if heuristic_action(done) != "STOP":
        raise AssertionError(heuristic_action(done))

    empty = build_z(
        {
            "hyp_i": None,
            "code_i": None,
            "cands_i": [],
            "portion_g_i": None,
            "verify_codes": [],
            "evidence_edges": [],
            "nutri_dish": None,
            "nutri_i": None,
            "step_i": 0,
        }
    )
    if heuristic_action(empty) != "ABSTAIN":
        raise AssertionError(heuristic_action(empty))

    forced = build_z(
        {
            "hyp_i": "apple",
            "code_i": "111",
            "cands_i": [{"code": "111", "score": 0.99}],
            "portion_g_i": 50.0,
            "verify_codes": [],
            "evidence_edges": [],
            "step_i": 1,
        },
        None,
        {"tools_used": 1, "safety_force_abstain": True},
    )
    if heuristic_action(forced) != "ABSTAIN":
        raise AssertionError("safety_force_abstain")

    tied = {action: 0.0 for action in BC_ACTIONS}
    tied["RETRIEVE"] = 0.4
    tied["CORRECT_CODE"] = 0.4
    if argmax_value_over_cost(tied) != "CORRECT_CODE":
        raise AssertionError("cheaper code fix should beat retrieve at the same utility")
    negative = {action: -0.2 for action in BC_ACTIONS}
    negative["STOP"] = 0.0
    if argmax_value_over_cost(negative) != "STOP":
        raise AssertionError("non-positive tool values should lose to STOP at 0")
    prefer_portion = {action: 0.0 for action in BC_ACTIONS}
    prefer_portion["RE_PORTION"] = 0.5
    prefer_portion["STOP"] = 0.3
    if argmax_value_over_cost(prefer_portion) != "RE_PORTION":
        raise AssertionError("higher Q/Cost should select RE_PORTION")
    prefer_stop = {action: 0.0 for action in BC_ACTIONS}
    prefer_stop["RE_PORTION"] = 0.2
    prefer_stop["STOP"] = 0.3
    if argmax_value_over_cost(prefer_stop) != "STOP":
        raise AssertionError("STOP should win when its value/cost is higher")
    blocked = {action: 0.0 for action in BC_ACTIONS}
    blocked["RE_PORTION"] = 1.0
    blocked["STOP"] = 0.0
    no_code = _row()
    if argmax_value_over_cost(blocked, legal_actions(no_code)) == "RE_PORTION":
        raise AssertionError("RE_PORTION is not executable without a code")
