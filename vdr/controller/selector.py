"""Policies the closed-loop rollout can call.

``feature_kind="state"`` is the saved state-aware BC ablation: 79-d, no
``U_*``. ``feature_kind="z"`` is the error state. Heuristic utility and
``Q_φ`` both select with ``argmax value / Cost``. A positive
``terminal_margin`` is the Q calibration fence: an unresolved stop or
abstain defers to the rule utility. ``None`` keeps the uncalibrated argmax.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from vdr.controller.policy_bc import BC_ACTIONS
from vdr.controller.q_calibration import calibrated_action
from vdr.controller.q_phi import QPhi
from vdr.controller.q_state import Z_DIM, build_z
from vdr.controller.state_features import build_state_features
from vdr.controller.utility import heuristic_action, legal_actions, argmax_value_over_cost


def decision_features(
    policy: Any,
    h_t: Any,
    o_prev: Any,
    b_t: Any,
) -> np.ndarray:
    """Feature vector the rollout and the replay check both use.

    A z-policy reads σ only from the previous observation (and from whatever
    ``build_z`` already resolves there). Replay of the written IR therefore
    sees the same vector the rollout scored.
    """
    kind = getattr(policy, "feature_kind", "state")
    if kind == "state":
        return build_state_features(h_t, o_prev, b_t)
    if kind == "z":
        return build_z(h_t, o_prev, b_t)
    raise ValueError(f"unknown feature_kind {kind!r}")


class HeuristicUtilityPolicy:
    """Rule utility over the six controller actions. ``feature_kind`` is ``z``."""

    feature_kind = "z"

    def predict_actions(self, Z: Any) -> list[str]:
        matrix = np.asarray(Z, dtype=np.float64)
        if matrix.ndim == 1:
            matrix = matrix.reshape(1, -1)
        if matrix.ndim != 2 or matrix.shape[1] != Z_DIM:
            raise ValueError(f"Z must have shape (n, {Z_DIM}), got {matrix.shape}")
        return [heuristic_action(row) for row in matrix]


class QCostPolicy:
    """``a* = argmax Q_φ(z, a) / Cost(a)``. ``feature_kind`` is ``z``.

    ``terminal_margin`` is the fit-set fence from ``q_calibration``. When it
    is set, a stop or abstain that beats the best legal specialist by less
    than that fence uses the rule utility. The learned ``Q`` weights are
    unchanged.
    """

    feature_kind = "z"

    def __init__(self, model: QPhi, *, terminal_margin: float | None = None) -> None:
        if model.actions != BC_ACTIONS:
            raise ValueError("QCostPolicy requires the BC action order")
        if terminal_margin is not None:
            margin = float(terminal_margin)
            if not np.isfinite(margin) or margin < 0.0:
                raise ValueError(f"terminal_margin must be finite and non-negative, got {terminal_margin}")
            self.terminal_margin: float | None = margin
        else:
            self.terminal_margin = None
        self.model = model

    def predict_actions(self, Z: Any) -> list[str]:
        matrix = np.asarray(Z, dtype=np.float64)
        if matrix.ndim == 1:
            matrix = matrix.reshape(1, -1)
        if matrix.ndim != 2 or matrix.shape[1] != Z_DIM:
            raise ValueError(f"Z must have shape (n, {Z_DIM}), got {matrix.shape}")
        values = self.model.predict_q(matrix)
        chosen: list[str] = []
        for row_z, row_q in zip(matrix, values):
            payload = {action: float(row_q[index]) for index, action in enumerate(self.model.actions)}
            if self.terminal_margin is None:
                chosen.append(argmax_value_over_cost(payload, legal_actions(row_z)))
            else:
                chosen.append(calibrated_action(payload, row_z, self.terminal_margin))
        return chosen

    def predict_q(self, Z: Any) -> np.ndarray:
        return self.model.predict_q(Z)
