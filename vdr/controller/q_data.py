"""Logged rows for supervised ``Q_φ``.

Fit and hold-out weak steps only. rgb_test ids are refused. The cafe label
is the train-split dish total used as the reference for ``ΔL``; it is not
written into ``z``.

σ timing matches the live controller. The step that logs ``RE_PORTION`` does
not see a scale. Later steps do: from the previous observation when that
observation carries ``estimate_mass`` σ, otherwise from a dish-level RGB
forward of the P0b checkpoint. Historical teacher portions did not store σ,
so the forward is what makes ``U_mass`` / ``U_nutrient`` see the specialist
on those later steps.

The supervised target is the clipped energy-error change. Tool steps whose
pre or post dish energy is missing are kept for selection tallies and are
not regression targets.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

import numpy as np

from vdr.controller.bc_data import BCDataRefused, assert_bc_jsonl_path
from vdr.controller.mass_sigma import estimate_from_sigma_record
from vdr.controller.nutrition_loss import energy_abs_error, nutrition_loss
from vdr.controller.policy_bc import BC_ACTIONS
from vdr.controller.q_state import Z_DIM, build_z
from vdr.controller.state_features import STATE_FEATURE_DIM
from vdr.controller.uncertainty import UNCERTAINTY_NAMES
from vdr.data.n5k import load_labels, test_split_ids
from vdr.ir.io import read_ir_jsonl
from vdr.weak.teacher import pool_of_dish

FIT_JSONL = Path("results/weak_steps/rgb_train_fit_all.jsonl")
HOLDOUT_JSONL = Path("results/weak_steps/rgb_train_holdout_all.jsonl")
TOOL_ACTIONS = frozenset({"RETRIEVE", "CORRECT_CODE", "RE_PORTION", "RE_DECOMPOSE"})
TERMINAL_ACTIONS = frozenset({"STOP", "ABSTAIN"})
_SIGMA_PRESENT = UNCERTAINTY_NAMES.index("sigma_present")


class QDataRefused(ValueError):
    """The trajectories are not a legal supervised-Q sample."""


@dataclass(frozen=True)
class QDataset:
    """Error states, logged actions, and nutrition-loss changes."""

    Z: np.ndarray
    y: np.ndarray
    delta_l: np.ndarray
    measured: np.ndarray
    supervise: np.ndarray
    energy_delta_kcal: np.ndarray
    dish_ids: tuple[str, ...]
    step_indices: tuple[int, ...]
    sigma_rows: int
    path: str
    pool: str

    @property
    def n_rows(self) -> int:
        return int(self.Z.shape[0])

    @property
    def n_dishes(self) -> int:
        return len(set(self.dish_ids))


def _belief(step_i: int) -> dict[str, Any]:
    return {
        "tools_used": max(int(step_i), 0),
        "step_i": int(step_i),
        "safety_force_abstain": False,
        "safety_force_stop": False,
    }


def _labels_without_test() -> dict[str, dict[str, float]]:
    banned = test_split_ids()
    labels = {dish_id: row for dish_id, row in load_labels().items() if dish_id not in banned}
    leaked = banned.intersection(labels)
    if leaked:
        raise QDataRefused("test ids remained in the Q label table")
    return labels


def load_q_rows(
    path: Path | str,
    *,
    expect_pool: str,
    sigma_by_dish: Mapping[str, Mapping[str, Any]] | None = None,
    limit: int | None = None,
) -> QDataset:
    """Stack ``z``, the logged action, and ``ΔL`` for one weak-step pool."""
    if expect_pool not in {"fit", "holdout"}:
        raise ValueError(f"expect_pool must be fit or holdout, got {expect_pool!r}")
    if limit is not None and limit < 1:
        raise ValueError("limit must be >= 1")
    src = Path(path)
    assert_bc_jsonl_path(src)
    if not src.is_file():
        raise FileNotFoundError(src)

    banned = test_split_ids()
    labels = _labels_without_test()
    allowed = set(BC_ACTIONS)
    by_dish: dict[str, list[dict[str, Any]]] = {}
    for record in read_ir_jsonl(src):
        dish_id = record.get("dish_id")
        if not isinstance(dish_id, str) or not dish_id:
            raise QDataRefused(f"{src.name}: row missing dish_id")
        if dish_id in banned:
            raise QDataRefused(f"{src.name}: refusing test dish {dish_id}")
        slot = pool_of_dish(dish_id)
        if slot != expect_pool:
            raise QDataRefused(f"{src.name}: {dish_id} is in the {slot} pool, not {expect_pool}")
        action = record.get("action_t")
        if action == "VOLUME_ESTIMATE":
            raise QDataRefused(f"{src.name}: VOLUME_ESTIMATE is not a Q action")
        if not isinstance(action, str) or action not in allowed:
            raise QDataRefused(f"{src.name}: action {action!r} is outside the controller set")
        step = record.get("step_i")
        if isinstance(step, bool) or not isinstance(step, int) or step < 0:
            raise QDataRefused(f"{src.name}: invalid step_i on {dish_id}")
        by_dish.setdefault(dish_id, []).append(record)
    if not by_dish:
        raise BCDataRefused(f"{src.name}: no weak-step rows")

    features: list[np.ndarray] = []
    labels_y: list[str] = []
    deltas: list[float] = []
    measured: list[bool] = []
    supervise: list[bool] = []
    energy_deltas: list[float] = []
    dish_ids: list[str] = []
    step_indices: list[int] = []
    sigma_rows = 0

    for dish_id in sorted(by_dish):
        if dish_id not in labels:
            raise QDataRefused(f"{dish_id} has no train-split cafe label")
        label = labels[dish_id]
        steps = sorted(by_dish[dish_id], key=lambda row: int(row["step_i"]))
        prev: dict[str, Any] | None = None
        seen_portion = False
        cached = None if sigma_by_dish is None else sigma_by_dish.get(dish_id)
        for index, row in enumerate(steps):
            injected = None
            if seen_portion and cached is not None:
                injected = estimate_from_sigma_record(cached)
            vector = build_z(row, prev, _belief(int(row["step_i"])), mass_estimate=injected)
            action = str(row["action_t"])
            nxt = steps[index + 1] if index + 1 < len(steps) else None
            if action in TERMINAL_ACTIONS:
                delta = 0.0
                is_measured = False
                do_supervise = True
                energy_delta = float("nan")
            elif action in TOOL_ACTIONS and nxt is not None:
                before = nutrition_loss(row.get("nutri_dish"), label)
                after = nutrition_loss(nxt.get("nutri_dish"), label)
                err_before = energy_abs_error(row.get("nutri_dish"), label)
                err_after = energy_abs_error(nxt.get("nutri_dish"), label)
                if before is None or after is None:
                    # Energy appeared or disappeared. That is not a kcal edit
                    # comparable to a portion correction, so it is not a target.
                    delta = 0.0
                    is_measured = False
                    do_supervise = False
                    energy_delta = float("nan")
                else:
                    delta = before - after
                    is_measured = True
                    do_supervise = True
                    energy_delta = float(err_before - err_after) if err_before is not None and err_after is not None else float("nan")
            else:
                prev = row
                if action == "RE_PORTION":
                    seen_portion = True
                continue
            if do_supervise and (not np.isfinite(delta) or delta < -1.0000001 or delta > 1.0000001):
                raise QDataRefused(f"{dish_id} step {row['step_i']}: delta L {delta} is outside [-1, 1]")
            if float(vector[_SIGMA_PRESENT]) >= 0.5:
                sigma_rows += 1
            features.append(vector)
            labels_y.append(action)
            deltas.append(float(delta))
            measured.append(is_measured)
            supervise.append(do_supervise)
            energy_deltas.append(float(energy_delta))
            dish_ids.append(dish_id)
            step_indices.append(int(row["step_i"]))
            prev = row
            if action == "RE_PORTION":
                seen_portion = True
            if limit is not None and len(labels_y) >= limit:
                break
        if limit is not None and len(labels_y) >= limit:
            break

    if not features:
        raise QDataRefused(f"{src.name}: no Q rows")
    Z = np.vstack(features)
    if Z.shape[1] != Z_DIM or Z.shape[1] < STATE_FEATURE_DIM:
        raise RuntimeError(f"z width {Z.shape[1]} != {Z_DIM}")
    return QDataset(
        Z=Z,
        y=np.asarray(labels_y, dtype=object),
        delta_l=np.asarray(deltas, dtype=np.float64),
        measured=np.asarray(measured, dtype=bool),
        supervise=np.asarray(supervise, dtype=bool),
        energy_delta_kcal=np.asarray(energy_deltas, dtype=np.float64),
        dish_ids=tuple(dish_ids),
        step_indices=tuple(step_indices),
        sigma_rows=sigma_rows,
        path=str(src),
        pool=expect_pool,
    )
