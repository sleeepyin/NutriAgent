"""Learnable controller: open-loop B5, closed-loop state BC, supervised Q_φ.

Open-loop ``PolicyBC`` / ``PolicyMLP`` call ``build_features`` only.
Closed-loop ``PolicyStateMLP`` uses ``build_state_features`` when
``CLOSED_LOOP_READY`` is true. ``QPhi`` reads ``build_z`` and is supervised
regression of nutrition-loss change, not RL.
"""

from vdr.controller.dagger import (
    DAGGER_MAX_STEPS,
    DaggerRoundResult,
    fit_pool_dishes,
    run_dagger_round,
    stack_aggregate,
)
from vdr.controller.features import FEATURE_DIM, FEATURE_NAMES, build_features
from vdr.controller.policy_bc import BC_ACTIONS, PolicyBC
from vdr.controller.policy_mlp import PolicyMLP
from vdr.controller.q_phi import QPhi
from vdr.controller.q_state import (
    Z_DIM,
    Z_FEATURE_NAMES,
    build_z,
)
from vdr.controller.state_features import (
    CLOSED_LOOP_READY,
    STATE_FEATURE_DIM,
    STATE_FEATURE_NAMES,
    build_state_features,
)
from vdr.controller.utility import ACTION_COST, argmax_value_over_cost, heuristic_action

__all__ = [
    "BC_ACTIONS",
    "CLOSED_LOOP_READY",
    "DAGGER_MAX_STEPS",
    "DaggerRoundResult",
    "FEATURE_DIM",
    "FEATURE_NAMES",
    "PolicyBC",
    "PolicyMLP",
    "STATE_FEATURE_DIM",
    "STATE_FEATURE_NAMES",
    "Z_DIM",
    "Z_FEATURE_NAMES",
    "ACTION_COST",
    "QPhi",
    "argmax_value_over_cost",
    "build_features",
    "build_state_features",
    "build_z",
    "fit_pool_dishes",
    "heuristic_action",
    "run_dagger_round",
    "stack_aggregate",
]
