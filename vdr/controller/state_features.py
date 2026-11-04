"""Closed-loop state features. Sketch only. Not trained. Not ready.

Main interface (PLAN.md): ``π(a_t | h_t, o_{t-1}, b_t)`` via
``build_state_features``. The vector is

    [ pre_action_31d | obs_prev_block | budget_block ]

``build_features`` remains the open-loop 31-d ablation **B5**. ``PolicyBC``,
``PolicyMLP``, and DAgger keep calling that function. This module has no
``fit`` and does not load weights, open a volume tool, or read test GT.

``CLOSED_LOOP_READY`` stays false until a later change actually trains a
policy on these features. Returning a vector here is the interface sketch,
not a claim that closed-loop control works.

Fixed scales (not fit on hold-out or rgb_test):

- ``OBS_SCORE_GAP_TAU = 0.05``
- ``ENERGY_CAP_KCAL = 2500`` (same ceiling as ``rule_verify.ENERGY_MAX``)
- ``UNRESOLVED_CAP = 8``, ``KEPT_CAP = 16``, ``TOOL_CAP = 32``
- The legacy step trio stays inside the 31-d prefix. The budget block does
  not repeat it. ``safety_volume_disabled`` and ``safety_legacy_weights_off``
  are hard 1s.

``h_t["obs_t"]`` and ``train_gt_*`` are not encoded. Only the ``o_prev``
argument fills the observation block. Pass either the previous ``obs_t``
dict, or the previous IR row (a mapping that contains ``obs_t``). On a
previous IR row, ``action_t``, ``verify_codes``, ``s``, and
``abstain_reason`` may be read. Nutrients come from the observation payload
first (``nutrients`` / ``nutri`` / ``n`` / ``nutri_i`` inside ``obs_t``).
If that payload has none, a previous IR row may supply ``nutri_dish`` then
``nutri_i`` (tool-reported public totals, not a train or test label).
"""

from __future__ import annotations

import math
from typing import Any, Literal, Mapping

import numpy as np

from vdr.controller.features import (
    CAND_SLOTS,
    FEATURE_DIM,
    FEATURE_NAMES,
    PORTION_SCALE_G,
    VERIFY_CODE_VOCAB,
    _candidates,
    _finite,
    _text,
    _verify_flags,
    build_features,
)

MaskMode = Literal["none", "zero", "shuffle"]

# Interface sketch. Flip only when a closed-loop policy is actually trained.
CLOSED_LOOP_READY = False

OBS_SCORE_GAP_TAU = 0.05
ENERGY_CAP_KCAL = 2500.0
UNRESOLVED_CAP = 8.0
KEPT_CAP = 16.0
TOOL_CAP = 32.0

# actions.json enum, volume tool excluded. Not the confusion-matrix column order.
PREV_ACTIONS: tuple[str, ...] = (
    "RETRIEVE",
    "RE_PORTION",
    "RE_DECOMPOSE",
    "ABSTAIN",
    "STOP",
    "CORRECT_CODE",
)
TERMINAL_STATES: tuple[str, ...] = ("pass", "corrected", "abstain", "none")

_NUTRI_CONTAINERS = ("nutrients", "nutri", "n", "nutri_i")
_FORBIDDEN_KEYS = frozenset(
    {
        "train_gt_energy_kcal",
        "train_gt",
        "gold",
        "y_true",
        "label_kcal",
        "test_gt",
    }
)


def _obs_feature_names() -> tuple[str, ...]:
    names = [
        "obs_n_cands_norm",
        "obs_top1_score",
        "obs_top2_score",
        "obs_score_gap",
        "obs_score_mean",
        "obs_selected_is_top1",
        "obs_selected_in_cands",
        "obs_selected_rank_norm",
        "obs_score_entropy_norm",
        "obs_top1_uncertainty",
        "obs_ambiguous",
    ]
    names.extend(f"obs_verify_{name}" for name in VERIFY_CODE_VOCAB)
    names.extend(
        [
            "obs_verify_other",
            "obs_any_verify",
            "obs_error_code_present",
            "obs_tool_failed",
            "obs_nutri_energy_norm",
            "obs_nutri_mass_norm",
            "obs_nutri_present",
            "obs_delta_portion_norm",
        ]
    )
    names.extend(f"obs_prev_action_{name}" for name in PREV_ACTIONS)
    names.extend(f"obs_terminal_s_{name}" for name in TERMINAL_STATES)
    names.extend(
        [
            "obs_abstain_signal",
            "obs_n_unresolved_norm",
            "obs_n_kept_norm",
            "obs_code_changed",
            "obs_missing",
        ]
    )
    return tuple(names)


OBS_FEATURE_NAMES = _obs_feature_names()
BUDGET_FEATURE_NAMES: tuple[str, ...] = (
    "budget_tools_used_norm",
    "safety_force_abstain",
    "safety_force_stop",
    "safety_volume_disabled",
    "safety_legacy_weights_off",
)
STATE_FEATURE_NAMES = FEATURE_NAMES + OBS_FEATURE_NAMES + BUDGET_FEATURE_NAMES
OBS_DIM = len(OBS_FEATURE_NAMES)
BUDGET_DIM = len(BUDGET_FEATURE_NAMES)
STATE_FEATURE_DIM = len(STATE_FEATURE_NAMES)

PRE_SLICE = slice(0, FEATURE_DIM)
OBS_SLICE = slice(FEATURE_DIM, FEATURE_DIM + OBS_DIM)
BUDGET_SLICE = slice(FEATURE_DIM + OBS_DIM, STATE_FEATURE_DIM)


def _entropy_norm(scores: list[float]) -> float:
    if len(scores) < 2:
        return 0.0
    clipped = [max(score, 0.0) for score in scores]
    total = sum(clipped)
    if total <= 0.0:
        return 0.0
    entropy = 0.0
    for score in clipped:
        if score <= 0.0:
            continue
        prob = score / total
        entropy -= prob * math.log(prob)
    return entropy / math.log(len(clipped))


def _error_text(obs: Mapping[str, Any]) -> str | None:
    for key in ("error", "error_code"):
        value = obs.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()
    return None


def _split_prev(
    o_prev: Mapping[str, Any] | None,
) -> tuple[Mapping[str, Any] | None, Mapping[str, Any] | None]:
    """Return ``(obs_payload, previous_ir_row_or_none)``."""
    if o_prev is None:
        return None, None
    nested = o_prev.get("obs_t")
    if isinstance(nested, Mapping):
        return nested, o_prev
    return o_prev, None


def _prev_action(
    obs: Mapping[str, Any] | None,
    row: Mapping[str, Any] | None,
    b_t: Mapping[str, Any] | None,
) -> str | None:
    if b_t is not None:
        named = _text(b_t.get("prev_action"))
        if named is not None:
            return named
    if row is not None:
        named = _text(row.get("action_t"))
        if named is not None:
            return named
    if obs is not None:
        return _text(obs.get("action_t") or obs.get("prev_action"))
    return None


def _verify_source(
    obs: Mapping[str, Any] | None,
    row: Mapping[str, Any] | None,
) -> Any:
    if row is not None and isinstance(row.get("verify_codes"), (list, tuple)):
        return row.get("verify_codes")
    if obs is not None and isinstance(obs.get("verify_codes"), (list, tuple)):
        return obs.get("verify_codes")
    return None


def _terminal_and_reason(
    obs: Mapping[str, Any] | None,
    row: Mapping[str, Any] | None,
) -> tuple[str | None, str | None]:
    status = None
    reason = None
    if obs is not None:
        status = _text(obs.get("s"))
        if isinstance(obs.get("abstain_reason"), str) and obs.get("abstain_reason").strip():
            reason = obs.get("abstain_reason").strip()
    if row is not None:
        if status is None:
            status = _text(row.get("s"))
        if reason is None and isinstance(row.get("abstain_reason"), str):
            text = row.get("abstain_reason").strip()
            if text:
                reason = text
    return status, reason


def _has_nutri_keys(source: Mapping[str, Any]) -> bool:
    return any(key in source and source.get(key) is not None for key in (
        "energy_kcal",
        "protein_g",
        "carb_g",
        "fat_g",
        "mass_g",
    ))


def _copy_nutri(source: Mapping[str, Any]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for key in ("energy_kcal", "protein_g", "carb_g", "fat_g", "mass_g"):
        if key in _FORBIDDEN_KEYS or key not in source or source.get(key) is None:
            continue
        out[key] = source[key]
    return out


def _nutri_payload(obs: Mapping[str, Any], row: Mapping[str, Any] | None) -> dict[str, Any]:
    for key in _NUTRI_CONTAINERS:
        value = obs.get(key)
        if isinstance(value, Mapping) and _has_nutri_keys(value):
            return _copy_nutri(value)
    if row is not None:
        for key in ("nutri_dish", "nutri_i"):
            value = row.get(key)
            if isinstance(value, Mapping) and _has_nutri_keys(value):
                return _copy_nutri(value)
    return _copy_nutri(obs)


def _obs_block(
    o_prev: Mapping[str, Any] | None,
    b_t: Mapping[str, Any] | None,
) -> np.ndarray:
    values = {name: 0.0 for name in OBS_FEATURE_NAMES}
    if o_prev is None:
        values["obs_missing"] = 1.0
        return np.array([values[name] for name in OBS_FEATURE_NAMES], dtype=np.float64)

    obs, row = _split_prev(o_prev)
    assert obs is not None
    codes, scores = _candidates(obs.get("cands"))
    n_scores = len(scores)
    values["obs_n_cands_norm"] = min(n_scores, CAND_SLOTS) / float(CAND_SLOTS)
    top1 = scores[0] if n_scores else 0.0
    top2 = scores[1] if n_scores >= 2 else 0.0
    gap = (top1 - top2) if n_scores >= 2 else 0.0
    values["obs_top1_score"] = top1
    values["obs_top2_score"] = top2
    values["obs_score_gap"] = gap
    values["obs_score_mean"] = (sum(scores) / n_scores) if n_scores else 0.0
    values["obs_score_entropy_norm"] = _entropy_norm(scores)
    values["obs_top1_uncertainty"] = max(0.0, 1.0 - top1) if n_scores else 0.0
    values["obs_ambiguous"] = 1.0 if n_scores >= 2 and gap < OBS_SCORE_GAP_TAU else 0.0

    selected = _text(obs.get("selected_code"))
    if selected is None:
        selected = _text(obs.get("to"))
    in_cands = selected is not None and selected in codes
    values["obs_selected_is_top1"] = 1.0 if in_cands and selected == codes[0] else 0.0
    values["obs_selected_in_cands"] = 1.0 if in_cands else 0.0
    if in_cands and selected is not None:
        values["obs_selected_rank_norm"] = codes.index(selected) / float(CAND_SLOTS - 1)
    elif selected is not None:
        values["obs_selected_rank_norm"] = 1.0

    one_hot, other, _count_norm, any_verify = _verify_flags(_verify_source(obs, row))
    for name, flag in zip(VERIFY_CODE_VOCAB, one_hot):
        values[f"obs_verify_{name}"] = flag
    values["obs_verify_other"] = other
    values["obs_any_verify"] = any_verify

    err = _error_text(obs)
    values["obs_error_code_present"] = 1.0 if err is not None else 0.0
    prev_action = _prev_action(obs, row, b_t)
    failed = False
    if obs.get("tool_failed") is True or obs.get("failed") is True:
        failed = True
    if err is not None:
        low = err.lower()
        if any(token in low for token in ("keyerror", "not found", "miss", "failed", "empty")):
            failed = True
    if prev_action == "RETRIEVE" and n_scores == 0:
        failed = True
    values["obs_tool_failed"] = 1.0 if failed else 0.0

    nutri = _nutri_payload(obs, row)
    energy = _finite(nutri.get("energy_kcal"))
    mass = _finite(nutri.get("mass_g"))
    if any(key in nutri for key in ("energy_kcal", "protein_g", "carb_g", "fat_g", "mass_g")):
        values["obs_nutri_present"] = 1.0
    if energy is not None:
        values["obs_nutri_energy_norm"] = min(max(energy, 0.0), ENERGY_CAP_KCAL) / ENERGY_CAP_KCAL
    if mass is not None:
        values["obs_nutri_mass_norm"] = min(max(mass, 0.0), PORTION_SCALE_G) / PORTION_SCALE_G
    portion = _finite(obs.get("portion_g"))
    portion_from = _finite(obs.get("from_portion_g"))
    if portion is not None and portion_from is not None:
        values["obs_delta_portion_norm"] = (
            min(abs(portion - portion_from), PORTION_SCALE_G) / PORTION_SCALE_G
        )

    if prev_action in PREV_ACTIONS:
        values[f"obs_prev_action_{prev_action}"] = 1.0

    status, reason = _terminal_and_reason(obs, row)
    if status not in ("pass", "corrected", "abstain"):
        status = "none"
    values[f"obs_terminal_s_{status}"] = 1.0
    abstain = status == "abstain" or reason is not None or prev_action == "ABSTAIN"
    values["obs_abstain_signal"] = 1.0 if abstain else 0.0

    n_unresolved = _finite(obs.get("n_unresolved"))
    n_kept = _finite(obs.get("n_kept"))
    if n_unresolved is not None:
        values["obs_n_unresolved_norm"] = min(max(n_unresolved, 0.0), UNRESOLVED_CAP) / UNRESOLVED_CAP
    if n_kept is not None:
        values["obs_n_kept_norm"] = min(max(n_kept, 0.0), KEPT_CAP) / KEPT_CAP

    src_raw = obs.get("from")
    dst_raw = obs.get("to")
    if isinstance(src_raw, str) and isinstance(dst_raw, str) and src_raw.strip() and dst_raw.strip():
        values["obs_code_changed"] = 1.0 if src_raw.strip() != dst_raw.strip() else 0.0

    return np.array([values[name] for name in OBS_FEATURE_NAMES], dtype=np.float64)


def _budget_block(b_t: Mapping[str, Any] | None) -> np.ndarray:
    tools = 0.0
    if b_t is not None:
        raw = b_t.get("tools_used", b_t.get("n_tools"))
        parsed = _finite(raw)
        if parsed is not None:
            tools = max(parsed, 0.0)
    force_abstain = 1.0 if b_t is not None and bool(b_t.get("safety_force_abstain")) else 0.0
    force_stop = 1.0 if b_t is not None and bool(b_t.get("safety_force_stop")) else 0.0
    values = {
        "budget_tools_used_norm": min(tools, TOOL_CAP) / TOOL_CAP,
        "safety_force_abstain": force_abstain,
        "safety_force_stop": force_stop,
        # Hard bans. Callers cannot turn these off in the sketch.
        "safety_volume_disabled": 1.0,
        "safety_legacy_weights_off": 1.0,
    }
    return np.array([values[name] for name in BUDGET_FEATURE_NAMES], dtype=np.float64)


def mask_observation_block(
    vec: np.ndarray,
    *,
    mode: MaskMode,
    obs_slice: slice = OBS_SLICE,
    rng: np.random.Generator | None = None,
) -> np.ndarray:
    """Copy ``vec``. Pre-action and budget blocks stay put."""
    out = np.array(vec, dtype=np.float64, copy=True)
    if mode == "none":
        return out
    if mode == "zero":
        out[obs_slice] = 0.0
        return out
    if mode == "shuffle":
        if rng is None:
            raise ValueError("shuffle requires an RNG")
        block = out[obs_slice].copy()
        rng.shuffle(block)
        out[obs_slice] = block
        return out
    raise ValueError(mode)


def build_state_features(
    h_t: Mapping[str, Any],
    o_prev: Mapping[str, Any] | None,
    b_t: Mapping[str, Any] | None = None,
    *,
    mask_obs: MaskMode = "none",
    shuffle_seed: int | None = None,
) -> np.ndarray:
    """Fixed-width float64 sketch for a closed-loop policy. Not trained.

    ``mask_obs="zero"`` nulls the observation block (H2). ``"shuffle"``
    permutes that block with ``shuffle_seed``. ``"none"`` is the unmasked
    sketch. This function does not fit a threshold and does not read
    ``train_gt_*``.
    """
    if not isinstance(h_t, Mapping):
        raise TypeError(f"build_state_features expects a mapping, got {type(h_t).__name__}")
    if mask_obs not in ("none", "zero", "shuffle"):
        raise ValueError(mask_obs)
    if mask_obs == "shuffle" and shuffle_seed is None:
        raise ValueError("shuffle requires shuffle_seed")
    if o_prev is not None and not isinstance(o_prev, Mapping):
        raise TypeError(f"o_prev must be a mapping or None, got {type(o_prev).__name__}")
    if b_t is not None and not isinstance(b_t, Mapping):
        raise TypeError(f"b_t must be a mapping or None, got {type(b_t).__name__}")

    pre = build_features(h_t)
    obs = _obs_block(o_prev, b_t)
    budget = _budget_block(b_t)
    vector = np.concatenate([pre, obs, budget])
    rng = np.random.default_rng(shuffle_seed) if mask_obs == "shuffle" else None
    vector = mask_observation_block(vector, mode=mask_obs, obs_slice=OBS_SLICE, rng=rng)
    if vector.shape != (STATE_FEATURE_DIM,):
        raise RuntimeError(f"state feature width {vector.shape} != {STATE_FEATURE_DIM}")
    if not np.isfinite(vector).all():
        raise RuntimeError("build_state_features produced a non-finite value")
    return vector


def build_features_for_ablation(
    state: Mapping[str, Any],
    *,
    mode: Literal["B5_open_loop", "main_closed_loop"] = "B5_open_loop",
    o_prev: Mapping[str, Any] | None = None,
    b_t: Mapping[str, Any] | None = None,
    mask_obs: MaskMode = "none",
    shuffle_seed: int | None = None,
) -> np.ndarray:
    """Dispatch. Default is open-loop B5 (``build_features`` only).

    ``main_closed_loop`` returns the untrained sketch. It does not train.
    """
    if mode == "B5_open_loop":
        return build_features(state)
    if mode == "main_closed_loop":
        return build_state_features(
            state,
            o_prev,
            b_t,
            mask_obs=mask_obs,
            shuffle_seed=shuffle_seed,
        )
    raise ValueError(mode)
