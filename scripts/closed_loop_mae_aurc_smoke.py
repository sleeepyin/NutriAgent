#!/usr/bin/env python3
"""Synthetic energy-table contract plus a tiny predict_proba check.

Does not read weak steps and does not write results/.
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np  # noqa: E402

from vdr.controller.features import FEATURE_DIM  # noqa: E402
from vdr.controller.policy_bc import BC_ACTIONS  # noqa: E402
from vdr.controller.policy_mlp import PolicyMLP  # noqa: E402
from vdr.controller.policy_state_mlp import PolicyStateMLP  # noqa: E402
from vdr.controller.state_features import STATE_FEATURE_DIM  # noqa: E402
from vdr.eval.energy_selective import (  # noqa: E402
    SECONDARY_ABSTAIN_THRESHOLD,
    run_self_check,
)


def _proba_matches_actions(policy: PolicyMLP | PolicyStateMLP, matrix: np.ndarray) -> None:
    actions = policy.predict_actions(matrix)
    proba = policy.predict_proba(matrix)
    if proba.shape != (matrix.shape[0], len(policy.actions)):
        raise AssertionError(f"proba shape {proba.shape}")
    if not np.allclose(proba.sum(axis=1), 1.0):
        raise AssertionError("proba rows do not sum to 1")
    via = [policy.actions[int(index)] for index in np.argmax(proba, axis=1)]
    if via != list(actions):
        raise AssertionError("proba argmax disagrees with predict_actions")
    masked = [name for name, weight in (policy._class_weight or {}).items() if weight == 0.0]
    for name in masked:
        column = proba[:, policy.actions.index(name)]
        if float(np.max(column)) > 1e-8:
            raise AssertionError(f"{name} should have probability 0")


def main() -> int:
    run_self_check()
    rng = np.random.default_rng(0)
    labels = np.array(["RETRIEVE"] * 40 + ["ABSTAIN"] * 8, dtype=object)
    b5 = PolicyMLP()
    b5.fit(rng.normal(size=(48, FEATURE_DIM)), labels)
    _proba_matches_actions(b5, rng.normal(size=(12, FEATURE_DIM)))
    closed = PolicyStateMLP()
    closed.fit(rng.normal(size=(48, STATE_FEATURE_DIM)), labels)
    _proba_matches_actions(closed, rng.normal(size=(12, STATE_FEATURE_DIM)))
    if "VOLUME_ESTIMATE" in BC_ACTIONS:
        raise AssertionError("volume tool entered the BC head")
    print(
        "closed_loop_mae_aurc_smoke ok "
        f"secondary_threshold={SECONDARY_ABSTAIN_THRESHOLD} "
        f"B5={FEATURE_DIM} state={STATE_FEATURE_DIM}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
