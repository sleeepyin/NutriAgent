#!/usr/bin/env python3
"""Contract smoke: closed-loop loader + B5 dim unchanged."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np

from vdr.controller.bc_data import load_bc_rows
from vdr.controller.closed_loop_data import (
    HOLDOUT_JSONL,
    FIT_JSONL,
    load_closed_loop_rows,
    observation_block_norm,
)
from vdr.controller.features import FEATURE_DIM, build_features
from vdr.controller.policy_mlp import PolicyMLP
from vdr.controller.state_features import OBS_SLICE, STATE_FEATURE_DIM


def main() -> int:
    ok = True
    ok &= FEATURE_DIM == 31
    ok &= STATE_FEATURE_DIM == 79
    ok &= PolicyMLP().n_actions == 6

    if not FIT_JSONL.is_file():
        print("SKIP: fit jsonl missing")
        return 0

    fit_bc = load_bc_rows(FIT_JSONL, expect_pool="fit", limit=64)
    ok &= fit_bc.feature_dim == 31

    fit_cl = load_closed_loop_rows(FIT_JSONL, expect_pool="fit", limit=64)
    ok &= fit_cl.feature_dim == STATE_FEATURE_DIM

    norms = observation_block_norm(fit_cl.X)
    steps = np.array(fit_cl.step_indices)
    ok &= np.all(norms[steps == 0] >= 0.0)
    later = norms[steps > 0]
    ok &= later.size > 0 and np.any(later > 0.0)

    row = {
        "hyp_i": "rice",
        "code_i": None,
        "cands_i": [],
        "portion_g_i": None,
        "verify_codes": [],
        "step_i": 0,
    }
    ok &= build_features(row).shape == (31,)

    print(
        f"closed_loop_smoke ok={ok} B5={FEATURE_DIM} state={STATE_FEATURE_DIM} "
        f"obs_slice={OBS_SLICE} step_gt0_obs_norm_mean={later.mean():.4f}"
    )
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
