"""Pre-action state features h for behavior cloning.

The vector is built only from the fields the teacher exposes before the
action: ``hyp_i``, ``code_i``, ``cands_i``, ``portion_g_i``, ``verify_codes``,
and ``step_i``. The label ``action_t`` is not an input. ``obs_t`` is the tool
return; on ``CORRECT_CODE`` and ``RE_PORTION`` it carries the teacher-chosen
code or grams, so it is privileged and is not read.

Layout (see ``FEATURE_NAMES``):

- perception summary of ``hyp_i`` (present, clipped character length, token count)
- ``code_i`` relative to the written Top-k list
- Top-k score gap and score summary
- ``portion_g_i`` present and scaled grams
- ``verify_codes`` one-hot over the rule_verify vocabulary, plus an other bit
- error proxy with no ground truth: how many rules fired, and ``1 - top1``
- step budget from ``step_i`` against a fixed cap

Scales (``STEP_BUDGET``, ``PORTION_SCALE_G``, text caps, Top-k width) are
fixed constants. They are not fit on the train hold-out and they are not
fit on rgb_test.
"""

from __future__ import annotations

import math
from typing import Any, Mapping

import numpy as np

# Fields the policy is allowed to see. Anything else on the IR row is ignored.
PRE_ACTION_FIELDS = (
    "hyp_i",
    "code_i",
    "cands_i",
    "portion_g_i",
    "verify_codes",
    "step_i",
)

# Present on IR rows or audit dicts. Listed so callers can see what is dropped.
IGNORED_STATE_FIELDS = (
    "obs_t",
    "action_t",
    "nutri_i",
    "nutri_dish",
    "evidence_edges",
    "portion_src_i",
    "s",
    "abstain_reason",
    "component_i",
    "dish_id",
    "train_gt_energy_kcal",
)

# Stable order matching vdr.tools.rule_verify. Unknown codes use verify_other.
VERIFY_CODE_VOCAB = (
    "ENERGY_OUTLIER",
    "PROTEIN_OUTLIER",
    "CARB_OUTLIER",
    "FAT_OUTLIER",
    "ENERGY_ATWATER_MISMATCH",
    "UNGROUNDED",
    "PORTION_MISSING",
    "PORTION_EXTREME",
    "CODE_CONF_GAP",
)

# Width of the candidate list the teacher writes (top_k=5). Not a threshold.
CAND_SLOTS = 5
# Fixed cap for the step-budget features. Not tuned on hold-out or rgb_test.
STEP_BUDGET = 32
# Same gram ceiling as rule_verify.PORTION_MAX_G. A scale, not a new cutoff.
PORTION_SCALE_G = 1500.0
HYP_CHAR_CAP = 64.0
HYP_TOKEN_CAP = 8.0

_VERIFY_INDEX = {name: i for i, name in enumerate(VERIFY_CODE_VOCAB)}


def _feature_names() -> tuple[str, ...]:
    names = [
        "hyp_present",
        "hyp_char_len_norm",
        "hyp_token_len_norm",
        "code_present",
        "code_is_top1",
        "code_in_cands",
        "code_rank_norm",
        "cands_present",
        "n_cands_norm",
        "top1_score",
        "top2_score",
        "score_gap",
        "score_mean",
        "portion_present",
        "portion_g_norm",
    ]
    names.extend(f"verify_{name}" for name in VERIFY_CODE_VOCAB)
    names.extend(
        [
            "verify_other",
            "verify_count_norm",
            "any_verify",
            "top1_uncertainty",
            "step_frac",
            "step_remaining_frac",
            "step_over_budget",
        ]
    )
    return tuple(names)


FEATURE_NAMES = _feature_names()
FEATURE_DIM = len(FEATURE_NAMES)


def _text(value: Any) -> str | None:
    if not isinstance(value, str):
        return None
    text = value.strip()
    return text or None


def _finite(value: Any) -> float | None:
    if isinstance(value, bool) or value is None:
        return None
    if isinstance(value, (int, float)):
        number = float(value)
        if math.isfinite(number):
            return number
    return None


def _candidates(value: Any) -> tuple[list[str], list[float]]:
    if not isinstance(value, (list, tuple)):
        return [], []
    codes: list[str] = []
    scores: list[float] = []
    for cand in value:
        if not isinstance(cand, Mapping):
            continue
        code = _text(cand.get("code"))
        if code is None:
            continue
        score = _finite(cand.get("score"))
        codes.append(code)
        scores.append(0.0 if score is None else score)
    return codes, scores


def _verify_flags(value: Any) -> tuple[list[float], float, float, float]:
    """One-hot, other bit, unique-count norm, any-trigger."""
    one_hot = [0.0] * len(VERIFY_CODE_VOCAB)
    if not isinstance(value, (list, tuple)):
        return one_hot, 0.0, 0.0, 0.0
    seen: set[str] = set()
    other = 0.0
    for item in value:
        if not isinstance(item, str) or not item:
            continue
        seen.add(item)
        index = _VERIFY_INDEX.get(item)
        if index is None:
            other = 1.0
        else:
            one_hot[index] = 1.0
    if not seen:
        return one_hot, 0.0, 0.0, 0.0
    known = sum(one_hot)
    count_norm = min(known + other, float(len(VERIFY_CODE_VOCAB))) / float(len(VERIFY_CODE_VOCAB))
    return one_hot, other, count_norm, 1.0


def build_features(state: Mapping[str, Any], **kwargs: Any) -> np.ndarray:
    """Return h as a 1-d float64 vector. ``kwargs`` are ignored.

    Only ``PRE_ACTION_FIELDS`` are read. Passing ``obs_t`` or a train-GT
    key on ``state`` does not change the vector.
    """
    del kwargs  # callers cannot inject obs_t or labels through kwargs
    if not isinstance(state, Mapping):
        raise TypeError(f"build_features expects a mapping, got {type(state).__name__}")

    hyp = _text(state.get("hyp_i"))
    if hyp is None:
        hyp_present = 0.0
        hyp_char = 0.0
        hyp_token = 0.0
    else:
        hyp_present = 1.0
        hyp_char = min(len(hyp), HYP_CHAR_CAP) / HYP_CHAR_CAP
        n_tokens = len([tok for tok in hyp.split() if tok])
        hyp_token = min(n_tokens, HYP_TOKEN_CAP) / HYP_TOKEN_CAP

    code = _text(state.get("code_i"))
    codes, scores = _candidates(state.get("cands_i"))
    code_present = 1.0 if code is not None else 0.0
    in_cands = code is not None and code in codes
    code_is_top1 = 1.0 if in_cands and code == codes[0] else 0.0
    code_in_cands = 1.0 if in_cands else 0.0
    if in_cands and code is not None:
        rank = codes.index(code)
        code_rank_norm = rank / float(CAND_SLOTS - 1)
    elif code is not None:
        code_rank_norm = 1.0
    else:
        code_rank_norm = 0.0

    n_scores = len(scores)
    cands_present = 1.0 if n_scores else 0.0
    n_cands_norm = min(n_scores, CAND_SLOTS) / float(CAND_SLOTS)
    top1 = scores[0] if n_scores else 0.0
    top2 = scores[1] if n_scores >= 2 else 0.0
    score_gap = (top1 - top2) if n_scores >= 2 else 0.0
    score_mean = (sum(scores) / n_scores) if n_scores else 0.0
    top1_uncertainty = max(0.0, 1.0 - top1) if n_scores else 0.0

    portion = _finite(state.get("portion_g_i"))
    if portion is None:
        portion_present = 0.0
        portion_norm = 0.0
    else:
        portion_present = 1.0
        portion_norm = min(max(portion, 0.0), PORTION_SCALE_G) / PORTION_SCALE_G

    one_hot, verify_other, verify_count_norm, any_verify = _verify_flags(state.get("verify_codes"))

    step = _finite(state.get("step_i"))
    step_value = 0.0 if step is None else max(step, 0.0)
    step_frac = min(step_value, STEP_BUDGET) / float(STEP_BUDGET)
    step_remaining = max(STEP_BUDGET - step_value, 0.0) / float(STEP_BUDGET)
    step_over = 1.0 if step_value >= STEP_BUDGET else 0.0

    values = {
        "hyp_present": hyp_present,
        "hyp_char_len_norm": hyp_char,
        "hyp_token_len_norm": hyp_token,
        "code_present": code_present,
        "code_is_top1": code_is_top1,
        "code_in_cands": code_in_cands,
        "code_rank_norm": code_rank_norm,
        "cands_present": cands_present,
        "n_cands_norm": n_cands_norm,
        "top1_score": top1,
        "top2_score": top2,
        "score_gap": score_gap,
        "score_mean": score_mean,
        "portion_present": portion_present,
        "portion_g_norm": portion_norm,
        "verify_other": verify_other,
        "verify_count_norm": verify_count_norm,
        "any_verify": any_verify,
        "top1_uncertainty": top1_uncertainty,
        "step_frac": step_frac,
        "step_remaining_frac": step_remaining,
        "step_over_budget": step_over,
    }
    for name, flag in zip(VERIFY_CODE_VOCAB, one_hot):
        values[f"verify_{name}"] = flag

    vector = np.array([values[name] for name in FEATURE_NAMES], dtype=np.float64)
    if vector.shape != (FEATURE_DIM,):
        raise RuntimeError(f"feature width {vector.shape} != {FEATURE_DIM}")
    if not np.isfinite(vector).all():
        raise RuntimeError("build_features produced a non-finite value")
    return vector
