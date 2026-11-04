"""Load weak-step JSONL into closed-loop BC rows (h_t, o_{t-1}, b_t, a_t).

Each dish trajectory is grouped and ordered by ``step_i``. Step 0 uses
``o_prev=None``. Later steps pass the previous IR row (contains ``obs_t``).
``build_state_features`` stacks the vector; train GT and rgb_test are refused.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from vdr.controller.bc_data import BCDataRefused, assert_bc_jsonl_path
from vdr.controller.policy_bc import BC_ACTIONS
from vdr.controller.state_features import (
    OBS_SLICE,
    STATE_FEATURE_DIM,
    STATE_FEATURE_NAMES,
    build_state_features,
)
from vdr.data.n5k import test_split_ids
from vdr.ir.io import read_ir_jsonl
from vdr.weak.teacher import pool_of_dish

FIT_JSONL = Path("results/weak_steps/rgb_train_fit_all.jsonl")
HOLDOUT_JSONL = Path("results/weak_steps/rgb_train_holdout_all.jsonl")


@dataclass(frozen=True)
class ClosedLoopDataset:
    """In-memory state features and action labels."""

    X: np.ndarray
    y: np.ndarray
    dish_ids: tuple[str, ...]
    step_indices: tuple[int, ...]
    path: str
    pool: str | None

    @property
    def n_rows(self) -> int:
        return int(self.X.shape[0])

    @property
    def n_dishes(self) -> int:
        return len(set(self.dish_ids))

    @property
    def feature_dim(self) -> int:
        return int(self.X.shape[1]) if self.X.ndim == 2 else 0


def _belief_at_step(step_i: int) -> dict[str, Any]:
    return {
        "tools_used": max(int(step_i), 0),
        "step_i": int(step_i),
        "safety_force_abstain": False,
        "safety_force_stop": False,
    }


def load_closed_loop_rows(
    path: Path | str,
    *,
    expect_pool: str | None = None,
    limit: int | None = None,
    mask_obs: str = "none",
    shuffle_seed: int | None = None,
) -> ClosedLoopDataset:
    """Read weak steps and stack ``build_state_features`` with ``action_t`` labels."""
    if expect_pool is not None and expect_pool not in {"fit", "holdout"}:
        raise ValueError(f"expect_pool must be fit or holdout, got {expect_pool!r}")
    if limit is not None and limit < 1:
        raise ValueError("limit must be >= 1")
    if mask_obs not in ("none", "zero", "shuffle"):
        raise ValueError(mask_obs)
    if mask_obs == "shuffle" and shuffle_seed is None:
        raise ValueError("shuffle requires shuffle_seed")

    src = Path(path)
    assert_bc_jsonl_path(src)
    if not src.is_file():
        raise FileNotFoundError(src)

    forbidden = test_split_ids()
    allowed = set(BC_ACTIONS)
    by_dish: dict[str, list[dict[str, Any]]] = {}

    for record in read_ir_jsonl(src):
        dish_id = record.get("dish_id")
        if not isinstance(dish_id, str) or not dish_id:
            raise BCDataRefused(f"{src.name}: row missing dish_id")
        if dish_id in forbidden:
            raise BCDataRefused(f"{src.name}: refusing test dish {dish_id}")
        slot = pool_of_dish(dish_id)
        if expect_pool is not None and slot != expect_pool:
            raise BCDataRefused(
                f"{src.name}: {dish_id} is in the {slot} pool, not {expect_pool}"
            )
        action = record.get("action_t")
        if action == "VOLUME_ESTIMATE":
            raise BCDataRefused(f"{src.name}: VOLUME_ESTIMATE is not a BC label")
        if not isinstance(action, str) or action not in allowed:
            raise BCDataRefused(f"{src.name}: action {action!r} is outside the BC set")
        step = record.get("step_i")
        if not isinstance(step, int) or step < 0:
            raise BCDataRefused(f"{src.name}: invalid step_i on {dish_id}")
        by_dish.setdefault(dish_id, []).append(record)

    if not by_dish:
        raise BCDataRefused(f"{src.name}: no weak-step rows")

    features: list[np.ndarray] = []
    labels: list[str] = []
    dish_ids: list[str] = []
    step_indices: list[int] = []

    for dish_id in sorted(by_dish):
        steps = sorted(by_dish[dish_id], key=lambda row: int(row["step_i"]))
        prev: dict[str, Any] | None = None
        for row in steps:
            step_i = int(row["step_i"])
            b_t = _belief_at_step(step_i)
            vec = build_state_features(
                row,
                prev,
                b_t,
                mask_obs=mask_obs,  # type: ignore[arg-type]
                shuffle_seed=shuffle_seed,
            )
            features.append(vec)
            labels.append(str(row["action_t"]))
            dish_ids.append(dish_id)
            step_indices.append(step_i)
            prev = row
            if limit is not None and len(labels) >= limit:
                break
        if limit is not None and len(labels) >= limit:
            break

    X = np.vstack(features)
    if X.shape[1] != STATE_FEATURE_DIM or X.shape[1] != len(STATE_FEATURE_NAMES):
        raise RuntimeError(f"feature width {X.shape[1]} != {STATE_FEATURE_DIM}")
    return ClosedLoopDataset(
        X=X,
        y=np.asarray(labels, dtype=object),
        dish_ids=tuple(dish_ids),
        step_indices=tuple(step_indices),
        path=str(src),
        pool=expect_pool,
    )


def observation_block_norm(X: np.ndarray) -> np.ndarray:
    """L2 norm of the observation slice for each row."""
    block = X[:, OBS_SLICE]
    return np.linalg.norm(block, axis=1)
