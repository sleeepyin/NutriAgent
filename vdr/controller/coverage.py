"""Keep a named component from ending the episode before a mass write.

The calibrated selector still uses the frozen ``Q`` weights and the fit-set
terminal-margin fence. This guard runs after that choice. It does not refit
``Q``, it does not change ``τ``, and it does not read a cafe label.

Two cases are rewritten, and only while the open component has no
``portion_src=mass_estimator`` write:

- ``STOP`` / ``ABSTAIN`` defer to the existing rule utility, which already
  scores a terminal action at 0 while a component is open.
- A ``CORRECT_CODE`` that comes back to the retrieved top code after another
  ``CORRECT_CODE`` would start a second swap cycle. The mass write is taken
  instead. One swap away from the top code is left alone.

A closed component, and a component that already has a mass write, keep the
fence decision, including a confident stop.
"""

from __future__ import annotations

from typing import Mapping

import numpy as np

from vdr.controller.policy_bc import BC_ACTIONS
from vdr.controller.q_calibration import calibrated_decision
from vdr.controller.q_state import Z_DIM, Z_FEATURE_NAMES
from vdr.controller.utility import heuristic_action, legal_actions

COVERAGE_SOURCE = "coverage_unwritten"
# Prefer a real portion once identity has returned to the retrieved code.
_WRITE_ORDER: tuple[str, ...] = ("RE_PORTION", "RETRIEVE", "CORRECT_CODE", "RE_DECOMPOSE")

_INDEX = {name: index for index, name in enumerate(Z_FEATURE_NAMES)}


def _gate(row: np.ndarray, name: str) -> bool:
    return float(row[_INDEX[name]]) >= 0.5


def component_needs_mass_write(row_z: np.ndarray) -> bool:
    """True when the open component has not been committed by ``estimate_mass``."""
    vector = np.asarray(row_z, dtype=np.float64)
    if vector.shape != (Z_DIM,):
        raise ValueError(f"coverage guard expected shape {(Z_DIM,)}, got {vector.shape}")
    return _gate(vector, "gate_component_open") and not _gate(vector, "gate_specialist_applied")


def _returned_to_top(row: np.ndarray) -> bool:
    """The previous action swapped a code and the current code is the top candidate again."""
    return _gate(row, "obs_prev_action_CORRECT_CODE") and _gate(row, "gate_code_top1")


def _fallback_write(legal: Mapping[str, bool]) -> str | None:
    for action in _WRITE_ORDER:
        if legal[action]:
            return action
    return None


def apply_coverage_guard(
    action: str,
    source: str,
    row_z: np.ndarray,
) -> tuple[str, str]:
    """Return ``(action, source)`` after the unwritten-component guard.

    ``source`` stays the fence source when the guard does not rewrite.
    A rewrite is ``coverage_unwritten``.
    """
    if action not in BC_ACTIONS:
        raise ValueError(f"coverage guard received {action!r}")
    vector = np.asarray(row_z, dtype=np.float64)
    if not component_needs_mass_write(vector):
        return action, source
    legal = legal_actions(vector)
    if action == "CORRECT_CODE" and _returned_to_top(vector) and legal["RE_PORTION"]:
        return "RE_PORTION", COVERAGE_SOURCE
    if action not in {"STOP", "ABSTAIN"}:
        return action, source
    rule = heuristic_action(vector)
    if rule == "CORRECT_CODE" and _returned_to_top(vector) and legal["RE_PORTION"]:
        return "RE_PORTION", COVERAGE_SOURCE
    if rule not in {"STOP", "ABSTAIN"}:
        return rule, COVERAGE_SOURCE
    fallback = _fallback_write(legal)
    if fallback is None:
        return action, source
    return fallback, COVERAGE_SOURCE


def guarded_action(
    values: Mapping[str, float],
    row_z: np.ndarray,
    terminal_margin: float,
) -> str:
    """Fence first, then the unwritten-component guard. ``Q`` weights are not an input."""
    action, source = calibrated_decision(values, row_z, terminal_margin)
    return apply_coverage_guard(action, source, row_z)[0]


def run_self_check() -> None:
    """Confident stop on an open component retrieves, then portions. No labels."""
    from vdr.controller.on_policy import rgb_test_ids, rollout_dish
    from vdr.controller.q_state import build_z
    from vdr.controller.selector import QCostPolicy

    def _row(**overrides: object) -> dict[str, object]:
        state: dict[str, object] = {
            "hyp_i": "apple",
            "code_i": "111",
            "cands_i": [
                {"code": "111", "score": 0.91, "name": "Apple raw"},
                {"code": "222", "score": 0.80, "name": "Apple juice"},
            ],
            "portion_g_i": 100.0,
            "portion_src_i": "grid_search",
            "verify_codes": [],
            "evidence_edges": [{"type": "n<-code×portion", "src": "n0", "dst": "c0"}],
            "nutri_i": {"energy_kcal": 100.0, "protein_g": 1.0, "carb_g": 10.0, "fat_g": 1.0, "mass_g": 100.0},
            "nutri_dish": {"energy_kcal": 100.0, "protein_g": 1.0, "carb_g": 10.0, "fat_g": 1.0, "mass_g": 100.0},
            "step_i": 2,
        }
        state.update(overrides)
        return state

    top = build_z(
        _row(),
        {"action_t": "CORRECT_CODE", "obs_t": {"from": "222", "to": "111"}, "step_i": 1},
        {"tools_used": 2},
    )
    if apply_coverage_guard("CORRECT_CODE", "q_over_cost", top) != ("RE_PORTION", COVERAGE_SOURCE):
        raise AssertionError("a swap that has returned to the top code should portion")
    fresh = build_z(_row(), {"action_t": "RETRIEVE", "obs_t": {}, "step_i": 1}, {"tools_used": 1})
    if apply_coverage_guard("CORRECT_CODE", "q_over_cost", fresh)[0] != "CORRECT_CODE":
        raise AssertionError("the first correction off a retrieve must stay available")
    alternate = build_z(
        _row(code_i="222"),
        {"action_t": "CORRECT_CODE", "obs_t": {"from": "111", "to": "222"}, "step_i": 1},
        {"tools_used": 2},
    )
    if apply_coverage_guard("CORRECT_CODE", "q_over_cost", alternate)[0] != "CORRECT_CODE":
        raise AssertionError("the swap back toward the top code must stay available")

    closed = build_z(
        {
            "hyp_i": None,
            "code_i": None,
            "cands_i": [],
            "portion_g_i": None,
            "verify_codes": [],
            "evidence_edges": [{"type": "n<-aggregate", "src": "dish", "dst": "n_dish"}],
            "nutri_i": None,
            "nutri_dish": {"energy_kcal": 180.0, "protein_g": 2.0, "carb_g": 20.0, "fat_g": 1.0, "mass_g": 210.0},
            "step_i": 3,
        },
        {"action_t": "RE_PORTION", "obs_t": {"portion_src": "mass_estimator"}, "step_i": 2},
        {"tools_used": 3},
    )
    confident = {action: 0.0 for action in BC_ACTIONS}
    confident["STOP"] = 0.30
    confident["RETRIEVE"] = 0.40
    if guarded_action(confident, closed, 0.05) != "STOP":
        raise AssertionError("a finished component must still be allowed to stop")
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
    if not component_needs_mass_write(open_row):
        raise AssertionError("an open component with no portion is unwritten")
    if guarded_action(confident, open_row, 0.05) != "RETRIEVE":
        raise AssertionError("a confident stop on an unwritten component should retrieve")

    class _ConfidentStop:
        actions = BC_ACTIONS

        def predict_q(self, Z: np.ndarray) -> np.ndarray:
            matrix = np.asarray(Z, dtype=np.float64)
            if matrix.ndim == 1:
                matrix = matrix.reshape(1, -1)
            out = np.full((matrix.shape[0], len(BC_ACTIONS)), -0.05, dtype=np.float64)
            order = {name: index for index, name in enumerate(BC_ACTIONS)}
            out[:, order["STOP"]] = 0.30
            out[:, order["RETRIEVE"]] = 0.40
            out[:, order["ABSTAIN"]] = -0.20
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
    raw = QCostPolicy(_ConfidentStop())
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
        raise AssertionError("uncalibrated Q must still be free to stop before the guard")

    mass_fn.calls = 0  # type: ignore[attr-defined]
    guarded = QCostPolicy(_ConfidentStop(), terminal_margin=0.05)
    rolled = rollout_dish(
        dish_id,
        guarded,
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
    portion = rolled.steps[1]["obs_t"]
    if portion.get("portion_src") != "mass_estimator" or portion.get("mass_tool") != "estimate_mass":
        raise AssertionError(portion)
    before = rolled.steps[1]["nutri_dish"]["energy_kcal"]
    after = rolled.steps[2]["nutri_dish"]["energy_kcal"]
    if before == after:
        raise AssertionError("the mass write did not change the dish total")
