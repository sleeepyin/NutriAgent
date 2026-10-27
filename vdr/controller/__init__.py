"""Learnable controller: pre-action features, BC policies, and one DAgger round."""

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

__all__ = [
    "BC_ACTIONS",
    "DAGGER_MAX_STEPS",
    "DaggerRoundResult",
    "FEATURE_DIM",
    "FEATURE_NAMES",
    "PolicyBC",
    "PolicyMLP",
    "build_features",
    "fit_pool_dishes",
    "run_dagger_round",
    "stack_aggregate",
]
