"""Decision calibration for ``argmax Q/Cost``. Not a second fit, and not RL.

On the fit weak steps, ``Q(STOP)`` and ``Q(ABSTAIN)`` sit near 0. Specialist
scores are often slightly negative, so a terminal action wins by a margin of
about 0.001. That pile is far below the fit residual of ``Q`` (about 0.03).
Inside the same pile the mass action is not the best specialist score, so
subtracting a constant from stop still does not call the mass estimator: the
noisy specialist ranking retrieves or swaps codes and can loop.

The fence is ``median(excess) + 3 * MAD(excess)`` on fit rows where the
uncalibrated rule already picked a terminal action and a specialist was legal.
``excess`` is the best terminal ``Q/Cost`` minus the best legal specialist
``Q/Cost``. ``MAD`` is the median absolute deviation from that median. The
multiplier is the usual robust fence. It is not a threshold scored on the
official hold-out or on rgb_test labels.

At decision time, a terminal win whose excess is below the fence is unresolved.
The existing rule utility selects instead. A specialist that already wins
``Q/Cost``, and a terminal win above the fence, stay on ``Q/Cost``.
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any, Mapping

import numpy as np

from vdr.controller.policy_bc import BC_ACTIONS
from vdr.controller.q_state import Z_DIM
from vdr.controller.utility import argmax_value_over_cost, heuristic_action, legal_actions, values_over_cost

RULE_ID = "terminal_margin_fence"
# Conventional robust fence. Not swept on hold-out or on rgb_test.
MARGIN_MAD_MULTIPLIER = 3.0
SPECIALIST_ACTIONS: tuple[str, ...] = (
    "RETRIEVE",
    "CORRECT_CODE",
    "RE_PORTION",
    "RE_DECOMPOSE",
)
TERMINAL_ACTIONS: tuple[str, ...] = ("STOP", "ABSTAIN")

if set(SPECIALIST_ACTIONS).union(TERMINAL_ACTIONS) != set(BC_ACTIONS):
    raise RuntimeError("calibration actions drifted from the controller set")


class QCalibrationError(ValueError):
    """The margin fence cannot be computed or applied."""


def _payload(row_q: np.ndarray) -> dict[str, float]:
    if row_q.shape != (len(BC_ACTIONS),):
        raise QCalibrationError(f"Q row width {row_q.shape} != {len(BC_ACTIONS)}")
    return {action: float(row_q[index]) for index, action in enumerate(BC_ACTIONS)}


def _apply_legal(values: Mapping[str, float], legal: Mapping[str, bool]) -> dict[str, float]:
    scored = {action: float(values[action]) for action in BC_ACTIONS}
    for action in BC_ACTIONS:
        if not legal[action]:
            scored[action] = float("-inf")
    return values_over_cost(scored)


def terminal_excess(scores: Mapping[str, float]) -> float | None:
    """Terminal ``Q/Cost`` minus the best legal specialist, or ``None`` if none is legal."""
    tool_scores = [float(scores[action]) for action in SPECIALIST_ACTIONS if math.isfinite(float(scores[action]))]
    if not tool_scores:
        return None
    terminal_scores = [float(scores[action]) for action in TERMINAL_ACTIONS]
    if not all(math.isfinite(score) for score in terminal_scores):
        raise QCalibrationError("a terminal Q/Cost is not finite")
    return max(terminal_scores) - max(tool_scores)


def calibrated_decision(
    values: Mapping[str, float],
    row_z: np.ndarray,
    terminal_margin: float,
) -> tuple[str, str]:
    """Return ``(action, source)``.

    ``source`` is ``q_over_cost`` or ``rule_utility``. The rule utility is
    used only when a terminal action beats every legal specialist by less
    than ``terminal_margin``.
    """
    margin = float(terminal_margin)
    if not math.isfinite(margin) or margin < 0.0:
        raise QCalibrationError(f"terminal margin must be finite and non-negative, got {terminal_margin}")
    vector = np.asarray(row_z, dtype=np.float64)
    if vector.shape != (Z_DIM,):
        raise QCalibrationError(f"z shape {vector.shape} != {(Z_DIM,)}")
    legal = legal_actions(vector)
    scores = _apply_legal(values, legal)
    q_action = argmax_value_over_cost(values, legal)
    if q_action not in TERMINAL_ACTIONS:
        return q_action, "q_over_cost"
    excess = terminal_excess(scores)
    if excess is None or excess >= margin:
        return q_action, "q_over_cost"
    return heuristic_action(vector), "rule_utility"


def calibrated_action(values: Mapping[str, float], row_z: np.ndarray, terminal_margin: float) -> str:
    """Action only. Same rule as ``calibrated_decision``."""
    return calibrated_decision(values, row_z, terminal_margin)[0]


def terminal_win_excesses(Z: np.ndarray, Q: np.ndarray) -> np.ndarray:
    """Excess on rows whose uncalibrated ``Q/Cost`` action is terminal and a specialist is legal.

    Rows with no legal specialist are omitted. Those decisions are stop versus
    abstain, and the fence does not apply to them.
    """
    matrix = np.asarray(Z, dtype=np.float64)
    scores = np.asarray(Q, dtype=np.float64)
    if matrix.ndim != 2 or matrix.shape[1] != Z_DIM:
        raise QCalibrationError(f"Z shape {matrix.shape} is not (n, {Z_DIM})")
    if scores.shape != (matrix.shape[0], len(BC_ACTIONS)):
        raise QCalibrationError(f"Q shape {scores.shape} does not match Z and the action set")
    if not np.isfinite(matrix).all():
        raise QCalibrationError("Z contains a non-finite feature")
    found: list[float] = []
    for row_z, row_q in zip(matrix, scores):
        payload = _payload(row_q)
        legal = legal_actions(row_z)
        chosen = argmax_value_over_cost(payload, legal)
        if chosen not in TERMINAL_ACTIONS:
            continue
        excess = terminal_excess(_apply_legal(payload, legal))
        if excess is None:
            continue
        if not math.isfinite(excess):
            raise QCalibrationError("terminal excess is not finite")
        found.append(excess)
    return np.asarray(found, dtype=np.float64)


def robust_fence(excesses: np.ndarray) -> dict[str, float | int]:
    """``median + 3 * MAD`` for a one-dimensional sample of terminal excesses."""
    sample = np.asarray(excesses, dtype=np.float64).reshape(-1)
    if sample.size < 3:
        raise QCalibrationError(f"need at least 3 terminal excesses, got {sample.size}")
    if not np.isfinite(sample).all():
        raise QCalibrationError("terminal excesses contain a non-finite value")
    median = float(np.median(sample))
    mad = float(np.median(np.abs(sample - median)))
    if mad < 0.0 or not math.isfinite(mad) or not math.isfinite(median):
        raise QCalibrationError("median or MAD is not usable")
    tau = median + MARGIN_MAD_MULTIPLIER * mad
    return {
        "n": int(sample.size),
        "median": median,
        "mad": mad,
        "multiplier": MARGIN_MAD_MULTIPLIER,
        "tau": float(tau),
        "mean": float(np.mean(sample)),
        "fraction_below_tau": float(np.mean(sample < tau)),
    }


def load_calibrated_policy(weights_path: str | Path, calibration_path: str | Path) -> Any:
    """Reload ``Q`` and the saved fence. Does not train and does not read labels."""
    from vdr.controller.q_phi import QPhi
    from vdr.controller.selector import QCostPolicy

    path = Path(calibration_path)
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("rule") != RULE_ID:
        raise QCalibrationError(f"{path.name} rule is {payload.get('rule')!r}, expected {RULE_ID}")
    if payload.get("holdout_used_for_selection") is not False:
        raise QCalibrationError("calibration artifact was selected on the hold-out")
    if payload.get("uses_rgb_test_gt") is not False or payload.get("rl") is not False:
        raise QCalibrationError("calibration artifact records test labels or RL")
    try:
        tau = float(payload["tau"])
    except (KeyError, TypeError, ValueError) as exc:
        raise QCalibrationError("calibration artifact is missing tau") from exc
    if not math.isfinite(tau) or tau < 0.0:
        raise QCalibrationError(f"stored terminal margin is not usable: {tau}")
    if float(payload.get("multiplier", -1)) != MARGIN_MAD_MULTIPLIER:
        raise QCalibrationError("stored multiplier does not match the fence")
    model = QPhi.from_npz(weights_path)
    return QCostPolicy(model, terminal_margin=tau)


def run_self_check() -> None:
    """The fence calls mass on the unresolved stub and keeps a confident stop.

    The Q values are the shape measured on the real stub: every specialist is
    slightly negative, stop is slightly positive, and the mass score is the
    worst specialist. No jsonl and no cafe label.
    """
    from vdr.controller.on_policy import rgb_test_ids, rollout_dish
    from vdr.controller.q_state import build_z
    from vdr.controller.selector import QCostPolicy

    sample = np.array([0.001, 0.0012, 0.0013, 0.0014, 0.05], dtype=np.float64)
    fence = robust_fence(sample)
    if abs(float(fence["median"]) - 0.0013) > 1e-12 or abs(float(fence["mad"]) - 0.0001) > 1e-12:
        raise AssertionError(fence)
    if abs(float(fence["tau"]) - (0.0013 + 3.0 * 0.0001)) > 1e-12:
        raise AssertionError(fence)

    open_row = build_z(
        {
            "hyp_i": "apple",
            "code_i": None,
            "cands_i": [],
            "portion_g_i": None,
            "verify_codes": [],
            "evidence_edges": [],
            "nutri_i": None,
            "nutri_dish": None,
            "step_i": 0,
        },
        None,
        {"tools_used": 0},
    )
    # Stop leads by 0.10 over retrieval after the cost of 2. That is above the fence.
    confident = {action: 0.0 for action in BC_ACTIONS}
    confident["STOP"] = 0.30
    confident["RETRIEVE"] = 0.40
    action, source = calibrated_decision(confident, open_row, 0.05)
    if action != "STOP" or source != "q_over_cost":
        raise AssertionError((action, source))
    # The same state with the measured near-zero gap defers to retrieval.
    noisy = {action: -0.002 for action in BC_ACTIONS}
    noisy["RE_DECOMPOSE"] = float("-inf")
    noisy["ABSTAIN"] = -1e-5
    noisy["STOP"] = 0.0002
    noisy["RETRIEVE"] = -0.00214
    action, source = calibrated_decision(noisy, open_row, 0.002)
    if action != "RETRIEVE" or source != "rule_utility":
        raise AssertionError((action, source))
    if calibrated_action(noisy, open_row, 0.0) != "STOP":
        raise AssertionError("a zero margin must keep the uncalibrated stop")

    class _NearZeroQ:
        actions = BC_ACTIONS

        def predict_q(self, Z: np.ndarray) -> np.ndarray:
            matrix = np.asarray(Z, dtype=np.float64)
            if matrix.ndim == 1:
                matrix = matrix.reshape(1, -1)
            out = np.zeros((matrix.shape[0], len(BC_ACTIONS)), dtype=np.float64)
            order = {name: index for index, name in enumerate(BC_ACTIONS)}
            out[:, order["RETRIEVE"]] = -0.00179
            out[:, order["RE_PORTION"]] = -0.0031
            out[:, order["RE_DECOMPOSE"]] = float("-inf")
            out[:, order["ABSTAIN"]] = -1e-5
            out[:, order["STOP"]] = 0.00044
            out[:, order["CORRECT_CODE"]] = -0.00137
            return out

    def mass_fn(image, candidates, component_index=None, component_hints_g=None, catalog_path=None, **kwargs):
        del image, candidates, component_index, component_hints_g, catalog_path, kwargs
        mass_fn.calls += 1  # type: ignore[attr-defined]
        return {
            "portion_g": 210.0,
            "mass_g": 210.0,
            "sigma": {"mass_g": 12.0, "energy_kcal": 40.0},
            "nutrients": {"energy_kcal": 180.0, "protein_g": 2.0, "carb_g": 20.0, "fat_g": 1.0},
            "rgb_only": True,
            "depth_used": False,
            "portion_src": "mass_estimator",
        }

    mass_fn.calls = 0  # type: ignore[attr-defined]

    class _Cands:
        cands = [
            {"code": "111", "score": 0.91, "name": "Apple raw"},
            {"code": "222", "score": 0.80, "name": "Apple juice"},
        ]

    def retrieve_fn(query, top_k=5, catalog_path=None):
        del top_k, catalog_path
        found = _Cands()
        found.query = query  # type: ignore[attr-defined]
        return found

    def lookup_fn(code, catalog_path=None):
        del code, catalog_path
        return {"energy_kcal": 100.0, "protein_g": 1.0, "carb_g": 10.0, "fat_g": 1.0}

    class _Portion:
        portion_g = 100.0

    dish_id = sorted(rgb_test_ids())[0]
    raw = QCostPolicy(_NearZeroQ())
    raw_roll = rollout_dish(
        dish_id,
        raw,
        image_path="overhead/rgb.png",
        perceive_fn=lambda _path: ([{"name": "apple", "portion_g": 180.0}], {"model_alias": "self-check"}),
        retrieve_fn=retrieve_fn,
        lookup_fn=lookup_fn,
        portion_fn=lambda code, hint_g=100.0, catalog_path=None: _Portion(),
        mass_fn=mass_fn,
        check_replay=True,
    )
    if [step["action_t"] for step in raw_roll.steps] != ["STOP"] or mass_fn.calls != 0:
        raise AssertionError(([step["action_t"] for step in raw_roll.steps], mass_fn.calls))

    mass_fn.calls = 0  # type: ignore[attr-defined]
    calibrated = QCostPolicy(_NearZeroQ(), terminal_margin=0.002)
    rolled = rollout_dish(
        dish_id,
        calibrated,
        image_path="overhead/rgb.png",
        perceive_fn=lambda _path: ([{"name": "apple", "portion_g": 180.0}], {"model_alias": "self-check"}),
        retrieve_fn=retrieve_fn,
        lookup_fn=lookup_fn,
        portion_fn=lambda code, hint_g=100.0, catalog_path=None: _Portion(),
        mass_fn=mass_fn,
        check_replay=True,
    )
    actions = [step["action_t"] for step in rolled.steps]
    if actions != ["RETRIEVE", "RE_PORTION", "STOP"]:
        raise AssertionError(actions)
    if mass_fn.calls != 1:
        raise AssertionError(mass_fn.calls)
    if rolled.steps[1]["obs_t"].get("portion_src") != "mass_estimator":
        raise AssertionError(rolled.steps[1]["obs_t"])
