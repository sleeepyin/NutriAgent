"""Route-state features. Never include unselected counterfactual outcomes."""
from __future__ import annotations

import numpy as np

from ..schemas import UncertaintyFactors


FEATURE_NAMES = [
    "pred_w",
    "log_pred_w",
    "ens_disagree",
    "u_scale",
    "u_geometry",
    "u_density",
    "u_domain",
    "u_quality",
    "ood",
    "has_ref",
    "has_view",
    "has_depth",
    "has_sem",
    "p_rigid",
    "p_prepared",
    "remaining_budget",
    "step",
]


def pack_features(
    pred_w: float,
    ens_disagree: float,
    u: UncertaintyFactors,
    ood: float,
    mask: dict[str, bool],
    p_rigid: float,
    p_prepared: float,
    remaining_budget: float,
    step: int,
) -> np.ndarray:
    pred_w = float(max(pred_w, 1e-3))
    return np.array(
        [
            pred_w,
            float(np.log(pred_w)),
            float(ens_disagree),
            u.u_scale,
            u.u_geometry,
            u.u_density,
            u.u_domain,
            u.u_quality,
            float(ood),
            1.0 if mask.get("REQUEST_REFERENCE") else 0.0,
            1.0 if mask.get("REQUEST_VIEW") else 0.0,
            1.0 if mask.get("REQUEST_DEPTH") else 0.0,
            1.0 if mask.get("ASK_SEMANTIC") else 0.0,
            float(p_rigid),
            float(p_prepared),
            float(remaining_budget),
            float(step),
        ],
        dtype=float,
    )
