"""Load weak-step JSONL into behavior-cloning rows (h, a).

Training rows come from the fit pool. Action accuracy is computed on the
hold-out pool only. ``rgb_test`` paths and the frozen B0/B1 result names
are refused before the file is read. ``obs_t`` is not copied into ``X``.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from vdr.controller.features import FEATURE_DIM, FEATURE_NAMES, build_features
from vdr.controller.policy_bc import BC_ACTIONS
from vdr.data.n5k import test_split_ids
from vdr.ir.io import read_ir_jsonl
from vdr.weak.teacher import PROTECTED_RESULT_NAMES, pool_of_dish

FIT_JSONL = Path("results/weak_steps/rgb_train_fit_all.jsonl")
HOLDOUT_JSONL = Path("results/weak_steps/rgb_train_holdout_all.jsonl")


class BCDataRefused(ValueError):
    """The path or a row is not a train-pool weak step."""


@dataclass(frozen=True)
class BCDataset:
    """In-memory (h, a) table. ``y`` is the action string for each row."""

    X: np.ndarray
    y: np.ndarray
    dish_ids: tuple[str, ...]
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


def assert_bc_jsonl_path(path: Path) -> None:
    """Reject test-split names and frozen eval artifacts before opening them."""
    name = path.name
    lowered = name.lower()
    if "rgb_test" in lowered or "depth_test" in lowered or "test_gt" in lowered:
        raise BCDataRefused(
            f"refusing {name}: BC rows are train weak steps only, not a test split"
        )
    if name in PROTECTED_RESULT_NAMES or "n5k_subset_b0_b1" in name or "n5k_rgb_test" in name:
        raise BCDataRefused(f"refusing {name}: that file is a frozen eval artifact")
    if name.endswith(".summary.json") or not name.endswith(".jsonl"):
        raise BCDataRefused(
            f"refusing {name}: BC loader reads weak-step jsonl, not a summary or GT audit"
        )


def load_bc_rows(
    path: Path | str,
    *,
    expect_pool: str | None = None,
    limit: int | None = None,
) -> BCDataset:
    """Read IR steps and stack ``build_features`` with the ``action_t`` label.

    ``expect_pool`` is ``fit`` or ``holdout``. Every dish must land in that
    pool under ``pool_of_dish``. A test dish id aborts the load.
    """
    if expect_pool is not None and expect_pool not in {"fit", "holdout"}:
        raise ValueError(f"expect_pool must be fit or holdout, got {expect_pool!r}")
    if limit is not None and limit < 1:
        raise ValueError("limit must be >= 1")

    src = Path(path)
    assert_bc_jsonl_path(src)
    if not src.is_file():
        raise FileNotFoundError(src)

    forbidden = test_split_ids()
    features: list[np.ndarray] = []
    labels: list[str] = []
    dish_ids: list[str] = []
    allowed = set(BC_ACTIONS)

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
        features.append(build_features(record))
        labels.append(action)
        dish_ids.append(dish_id)
        if limit is not None and len(labels) >= limit:
            break

    if not labels:
        raise BCDataRefused(f"{src.name}: no weak-step rows")

    X = np.vstack(features)
    if X.shape[1] != FEATURE_DIM or X.shape[1] != len(FEATURE_NAMES):
        raise RuntimeError(f"feature width {X.shape[1]} != {FEATURE_DIM}")
    return BCDataset(
        X=X,
        y=np.asarray(labels, dtype=object),
        dish_ids=tuple(dish_ids),
        path=str(src),
        pool=expect_pool,
    )


def action_counts(y: Any) -> dict[str, int]:
    """Count action labels. Order follows ``BC_ACTIONS`` then any extras."""
    labels = [str(item) for item in np.asarray(y, dtype=object).tolist()]
    counts = {name: 0 for name in BC_ACTIONS}
    extra: dict[str, int] = {}
    for label in labels:
        if label in counts:
            counts[label] += 1
        else:
            extra[label] = extra.get(label, 0) + 1
    counts.update(extra)
    return counts
