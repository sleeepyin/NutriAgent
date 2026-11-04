"""Learnable controller: open-loop B5, closed-loop state BC, DAgger round.

Open-loop ``PolicyBC`` / ``PolicyMLP`` call ``build_features`` only.
Closed-loop ``PolicyStateMLP`` uses ``build_state_features`` when
``CLOSED_LOOP_READY`` is true.
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
from vdr.controller.state_features import (
    CLOSED_LOOP_READY,
    STATE_FEATURE_DIM,
    STATE_FEATURE_NAMES,
    build_state_features,
)

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
    "build_features",
    "build_state_features",
    "fit_pool_dishes",
    "run_dagger_round",
    "stack_aggregate",
]
