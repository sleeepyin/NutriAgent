"""Error-aware state ``z`` for the non-RL controller.

``z = [U_*, D_*, σ, gates | closed-loop state]``.

The suffix is ``build_state_features`` unchanged: pre-action 31-d, previous
observation, and budget. State-aware BC keeps training on that suffix only.
``Q_φ`` reads the whole vector. Masking ``U_*`` zeros the uncertainty and
disagreement prefix and leaves the gates and the closed-loop suffix.
"""

from __future__ import annotations

from typing import Any, Mapping

import numpy as np

from vdr.controller.state_features import (
    STATE_FEATURE_DIM,
    STATE_FEATURE_NAMES,
    MaskMode,
    build_state_features,
)
from vdr.controller.uncertainty import (
    MASKED_UNCERTAINTY,
    UNCERTAINTY_NAMES,
    uncertainty_parts,
)

Z_FEATURE_NAMES: tuple[str, ...] = UNCERTAINTY_NAMES + STATE_FEATURE_NAMES
Z_DIM = len(Z_FEATURE_NAMES)
STATE_OFFSET = len(UNCERTAINTY_NAMES)

if STATE_OFFSET != len(UNCERTAINTY_NAMES):
    raise RuntimeError("state offset drifted from the uncertainty block")
if Z_DIM != STATE_OFFSET + STATE_FEATURE_DIM:
    raise RuntimeError(f"z width {Z_DIM} != {STATE_OFFSET} + {STATE_FEATURE_DIM}")


def build_z(
    h_t: Mapping[str, Any],
    o_prev: Mapping[str, Any] | None = None,
    b_t: Mapping[str, Any] | None = None,
    *,
    mass_estimate: Mapping[str, Any] | None = None,
    mask_u: bool = False,
    mask_obs: MaskMode = "none",
    shuffle_seed: int | None = None,
) -> np.ndarray:
    """Fixed-width float64 error state. Does not read train or test labels.

    ``mass_estimate`` is used only when the previous observation has no RGB
    specialist payload. ``mask_u`` zeros ``U_*``, ``D_*``, and the σ squashes.
    """
    if mask_obs not in ("none", "zero", "shuffle"):
        raise ValueError(mask_obs)
    # uncertainty_parts prefers a σ already stored on o_prev over mass_estimate.
    parts = uncertainty_parts(h_t, o_prev, mass_estimate=mass_estimate)
    head = np.array([parts[name] for name in UNCERTAINTY_NAMES], dtype=np.float64)
    if mask_u:
        head[:MASKED_UNCERTAINTY] = 0.0
    state = build_state_features(
        h_t,
        o_prev,
        b_t,
        mask_obs=mask_obs,
        shuffle_seed=shuffle_seed,
    )
    vector = np.concatenate([head, state])
    if vector.shape != (Z_DIM,):
        raise RuntimeError(f"z width {vector.shape} != {Z_DIM}")
    if not np.isfinite(vector).all():
        raise RuntimeError("build_z produced a non-finite value")
    return vector


def run_self_check() -> None:
    """The suffix matches the closed-loop state. Labels do not enter z."""
    from vdr.controller.uncertainty import run_self_check as _uncertainty_check

    _uncertainty_check()
    row = {
        "hyp_i": "apple",
        "code_i": "111",
        "cands_i": [{"code": "111", "score": 0.9}, {"code": "222", "score": 0.4}],
        "portion_g_i": 100.0,
        "portion_src_i": "grid_search",
        "verify_codes": [],
        "evidence_edges": [{"type": "n<-code×portion", "src": "n", "dst": "c"}],
        "nutri_i": {"energy_kcal": 100.0, "protein_g": 1.0, "carb_g": 10.0, "fat_g": 1.0, "mass_g": 100.0},
        "nutri_dish": {"energy_kcal": 100.0, "protein_g": 1.0, "carb_g": 10.0, "fat_g": 1.0, "mass_g": 100.0},
        "step_i": 1,
        "train_gt_energy_kcal": 4000.0,
    }
    belief = {"tools_used": 1, "safety_force_abstain": False, "safety_force_stop": False}
    z = build_z(row, None, belief)
    state = build_state_features(row, None, belief)
    if z.shape != (Z_DIM,) or not np.allclose(z[STATE_OFFSET:], state):
        raise AssertionError("z suffix is not the closed-loop state")
    if any("committed_kcal" in name or name == "abstain_score" for name in Z_FEATURE_NAMES):
        raise AssertionError("kcal proxy is still in z")
    masked = build_z(row, None, belief, mask_u=True)
    if np.count_nonzero(masked[:MASKED_UNCERTAINTY]) != 0:
        raise AssertionError("mask_u left an uncertainty coordinate")
    if not np.allclose(masked[MASKED_UNCERTAINTY:STATE_OFFSET], z[MASKED_UNCERTAINTY:STATE_OFFSET]):
        raise AssertionError("mask_u changed the gates")
    if not np.allclose(masked[STATE_OFFSET:], state):
        raise AssertionError("mask_u changed the closed-loop suffix")
    zeroed = build_z(row, {"obs_t": {"cands": [{"code": "111", "score": 0.5}]}}, belief, mask_obs="zero")
    if np.allclose(zeroed[STATE_OFFSET:], z[STATE_OFFSET:]):
        raise AssertionError("mask_obs did not touch the observation suffix")
    labeled = dict(row)
    labeled["train_gt_energy_kcal"] = 1.0
    if not np.allclose(build_z(labeled, None, belief), z):
        raise AssertionError("a label key changed z")
