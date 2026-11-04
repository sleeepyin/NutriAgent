# State-Aware Feature Design / 状态感知特征设计

> Landed design note for NutriAgent / VDR (P0). Aligned with [`PLAN.md`](../PLAN.md) after `#17` / `7ad3cfb`.
> Method center = **Evidence-Aware Closed-Loop Tool Control**.
> Main policy: \(\pi(a_t \mid h_t, o_{t-1}, b_t)\) — **must** read the prior tool observation.
> Open-loop 31-d BC (no \(o_{t-1}\)) = **ablation B5 only**, not the main method.
> The encoder in `vdr/controller/state_features.py` is **trainable** (`CLOSED_LOOP_READY = True` after `scripts/closed_loop_bc_train.py` + `results/closed_loop/metrics_holdout.json`). Open-loop 31-d remains ablation **B5**.

---

## 1. Motivation / 动机

Repo `vdr/controller/features.py::build_features` deliberately builds a **31-d pre-action** vector from
`hyp_i, code_i, cands_i, portion_g_i, verify_codes, step_i` and **excludes** `obs_t`.

That design was correct for an open-loop BC ablation (STATUS: MLP hold-out Acc 0.9700 but
**ABSTAIN 0/24** — ABSTAIN and STOP collide on pre-action fields; the distinguishing signal sits
in `obs_t` / `abstain_reason`). It is **not** the main closed-loop policy feature.

This document proposes `build_state_features(h_t, o_prev, b_t)` for the main method and maps the
existing open-loop path to ablation **B5**.

**Scope of this file:** interface design + field lists + masking API + the landed sketch layout.  
**Out of scope:** training a closed-loop policy, claiming the sketch is ready, changing frozen eval artifacts.

---

## 2. Policy interface / 策略接口

### 2.1 Main (closed-loop)

\[
\pi(a_t \mid h_t, o_{t-1}, b_t)
\]

| Symbol | Meaning |
|--------|---------|
| \(h_t\) | Pre-action IR snapshot at step \(t\) (current public state) |
| \(o_{t-1}\) | Tool observation returned by action \(a_{t-1}\) (previous step's `obs_t`) |
| \(b_t\) | Budget / safety control signals |
| \(a_t\) | Discrete action ∈ {RETRIEVE, CORRECT_CODE, RE_PORTION, RE_DECOMPOSE, ABSTAIN, STOP} |

No `VOLUME_ESTIMATE` in the main head (Tanabe; actions.json default OFF).

### 2.2 Ablation B5 (open-loop)

\[
\pi_{\mathrm{B5}}(a_t \mid h_t) \equiv \texttt{Policy\*(build\_features}(h_t)\texttt{)}
\]

Existing 31-d path. Kept for paper table A1 / open-loop BC comparison only.

---

## 3. Proposed API / 提议 API

```python
from typing import Any, Literal, Mapping, Sequence
import numpy as np

MaskMode = Literal["none", "zero", "shuffle"]

def build_state_features(
    h_t: Mapping[str, Any],
    o_prev: Mapping[str, Any] | None,
    b_t: Mapping[str, Any] | None = None,
    *,
    mask_obs: MaskMode = "none",
    shuffle_seed: int | None = None,
) -> np.ndarray:
    """Fixed-width float64 vector for closed-loop policy.

    Concatenation (stable order; document names in STATE_FEATURE_NAMES):
      [ pre_action_31d | obs_prev_block | budget_block ]

    - h_t: current IR pre-action fields (same contract as build_features).
    - o_prev: previous step's obs_t. None on the first step of a dish → obs block = zeros
      plus a ``obs_missing=1`` flag.
    - b_t: remaining budget / safety flags. None → defaults (step from h_t only).
    - mask_obs: H2 ablation — ``zero`` nulls the obs block; ``shuffle`` permutes obs
      across a batch externally OR shuffles scalar slots with ``shuffle_seed``.
      ``none`` = main method path.

    Does NOT read privileged teacher-chosen targets as *labels*. Encoding
    observation *structure* (candidate list returned by RETRIEVE, portion
    returned by RE_PORTION, abstain payload, error codes) is allowed; copying
    train-GT kcal into the vector is forbidden.
    """
    ...


def build_features(state: Mapping[str, Any], **kwargs: Any) -> np.ndarray:
    """EXISTING open-loop 31-d builder. Ablation B5-only path.

    Remains the sole feature function for PolicyBC / PolicyMLP BC smoke until
    closed-loop lands. Must keep ignoring obs_t / nutri / evidence / labels.
    """
    ...  # unchanged from vdr/controller/features.py
```

The sketch in `vdr/controller/state_features.py` implements this signature. `build_features` in `vdr/controller/features.py` is unchanged in width and in the fields it reads. Constants `STATE_FEATURE_NAMES`, `STATE_FEATURE_DIM`, `OBS_FEATURE_NAMES`, and `BUDGET_FEATURE_NAMES` are defined next to it. `CLOSED_LOOP_READY` is false.

Suggested constants (landed next to `FEATURE_NAMES`):

```python
STATE_FEATURE_NAMES: tuple[str, ...]  # pre + obs + budget names
STATE_FEATURE_DIM: int                 # FEATURE_DIM + OBS_DIM + BUDGET_DIM
OBS_FEATURE_NAMES: tuple[str, ...]
BUDGET_FEATURE_NAMES: tuple[str, ...]
```

Compatibility shim (landed as `build_features_for_ablation`; default mode is still B5):

```python
def build_features_for_ablation(
    state: Mapping[str, Any],
    *,
    mode: Literal["B5_open_loop", "main_closed_loop"] = "B5_open_loop",
    o_prev: Mapping[str, Any] | None = None,
    b_t: Mapping[str, Any] | None = None,
) -> np.ndarray:
    if mode == "B5_open_loop":
        return build_features(state)
    return build_state_features(state, o_prev, b_t, mask_obs="none")
```

---

## 4. What stays from the current 31-d pre-action block / 保留的 31 维

From `FEATURE_NAMES` / `PRE_ACTION_FIELDS` (repo `features.py`):

| Block | Fields |
|-------|--------|
| Perception | `hyp_present`, `hyp_char_len_norm`, `hyp_token_len_norm` |
| Code vs Top-k | `code_present`, `code_is_top1`, `code_in_cands`, `code_rank_norm` |
| Candidate list | `cands_present`, `n_cands_norm`, `top1_score`, `top2_score`, `score_gap`, `score_mean` |
| Portion | `portion_present`, `portion_g_norm` |
| Verify one-hot | `verify_{ENERGY_OUTLIER,…,CODE_CONF_GAP}`, `verify_other` |
| Error proxies | `verify_count_norm`, `any_verify`, `top1_uncertainty` |
| Step (legacy) | `step_frac`, `step_remaining_frac`, `step_over_budget` |

**Scales stay fixed** (STEP_BUDGET=32, PORTION_SCALE_G=1500, CAND_SLOTS=5, HYP caps).  
Do not refit on hold-out or rgb_test.

When `build_state_features` is used, the legacy step trio may stay for continuity **or** move entirely into \(b_t\); either way document the chosen layout in `STATE_FEATURE_NAMES` and keep B5 using the original 31-d layout unchanged.

---

## 5. Encoding \(o_{t-1}\) / 观测块字段候选

Source: previous IR row's `obs_t` (and, when needed for flags only, previous `action_t`).  
Teacher `_obs(...)` keys observed in `vdr/weak/teacher.py` include:
`label_quality`, `hypothesis_source`, `teacher_version`, `privileged`, `volume_tool`,
`query`, `cands`, `selected_code`, `from`/`to` (CORRECT_CODE), `portion_g`,
`from_portion_g`, `portion_src`, `portion_note`, `s`, `n_kept`, `n_unresolved`.

### 5.1 Candidate list stats (from `o_prev["cands"]` or equivalent)

| Feature (proposed name) | Encoding |
|-------------------------|----------|
| `obs_n_cands_norm` | min(n, 5)/5 |
| `obs_top1_score`, `obs_top2_score`, `obs_score_gap`, `obs_score_mean` | same scales as pre-action |
| `obs_selected_is_top1` | 1 if `selected_code == cands[0].code` |
| `obs_selected_in_cands` | membership |
| `obs_selected_rank_norm` | rank / 4, or 1.0 if missing |

### 5.2 Ambiguity

| Feature | Encoding |
|---------|----------|
| `obs_score_entropy_norm` | normalized Shannon entropy over Top-k scores (0 if empty) |
| `obs_top1_uncertainty` | max(0, 1 − top1) |
| `obs_ambiguous` | 1 if score_gap < τ_gap (fixed a priori, e.g. 0.05; **not** fit on test) |

### 5.3 Verify / error codes after the tool return

Prefer verify codes recomputed on the **post-action** public state if available in the
trajectory; else encode structured error flags inside `obs_t`:

| Feature | Encoding |
|---------|----------|
| `obs_verify_*` / `obs_verify_other` | same VERIFY_CODE_VOCAB one-hot (+ other) |
| `obs_any_verify` | any trigger |
| `obs_error_code_present` | 1 if obs carries an explicit `error` / `error_code` string |
| `obs_tool_failed` | 1 if empty cands after RETRIEVE, KeyError-style miss, etc. |

### 5.4 Nutrient aggregates (structure only — no train GT)

| Feature | Encoding |
|---------|----------|
| `obs_nutri_energy_norm` | clip(energy_kcal, 0, E_CAP)/E_CAP from obs payload if present |
| `obs_nutri_mass_norm` | clip(mass_g, 0, PORTION_SCALE_G)/PORTION_SCALE_G |
| `obs_nutri_present` | 1 if any nutri keys present |
| `obs_delta_portion_norm` | |portion_g − from_portion_g| / PORTION_SCALE_G when both exist |

**Forbidden:** `train_gt_energy_kcal`, any test GT, teacher oracle mass as a *label channel*
beyond what the tool legitimately returned in `obs_t`.

### 5.5 Last action result / terminal hints

| Feature | Encoding |
|---------|----------|
| `obs_prev_action_*` | one-hot over the 6 actions (previous `action_t`) |
| `obs_terminal_s_*` | one-hot over {pass, corrected, abstain, none} from `obs_t["s"]` |
| `obs_abstain_signal` | 1 if obs indicates abstain / abstain_reason non-empty |
| `obs_n_unresolved_norm` | min(n_unresolved, U_CAP)/U_CAP |
| `obs_n_kept_norm` | min(n_kept, K_CAP)/K_CAP |
| `obs_code_changed` | 1 if CORRECT_CODE-style `from` ≠ `to` |
| `obs_missing` | 1 if `o_prev is None` (first step) |

This block is exactly what separates ABSTAIN from STOP in STATUS (empty pre-action
fields collide; `obs_t` + `abstain_reason` differ).

---

## 6. Budget block \(b_t\) / 预算与安全

| Feature | Encoding |
|---------|----------|
| `budget_step_frac` | step_i / STEP_BUDGET |
| `budget_remaining_frac` | max(STEP_BUDGET − step_i, 0) / STEP_BUDGET |
| `budget_over` | 1 if step_i ≥ STEP_BUDGET |
| `budget_tools_used_norm` | cumulative tool calls / TOOL_CAP (fixed) |
| `safety_force_abstain` | hard guardrail flag from rule layer |
| `safety_force_stop` | hard guardrail flag |
| `safety_volume_disabled` | always 1 while VOLUME_ESTIMATE stays OFF |
| `safety_legacy_weights_off` | always 1 (hard ban on old regression weights) |

If legacy step features remain inside the 31-d block, avoid duplicating them in \(b_t\)
without renaming; prefer a single source of truth in `STATE_FEATURE_NAMES`.

---

## 7. Explicit mapping: open-loop → B5 only / 映射声明

| Path | Function | Reads \(o_{t-1}\)? | Role in paper |
|------|----------|-------------------|----------------|
| Main Ours-BC / closed-loop | `build_state_features(h_t, o_prev, b_t)` | **Yes** | Main method |
| Ablation B5 / open-loop BC | `build_features(h_t)` (31-d) | **No** | Ablation only |
| H2 mask-zero | `build_state_features(..., mask_obs="zero")` | structurally present, zeroed | Mechanism ablation |
| H2 mask-shuffle | `build_state_features(..., mask_obs="shuffle")` or batch shuffle of obs rows | scrambled | Mechanism ablation |

**Do not** call the 31-d open-loop Acc (0.9700 MLP / 0.9206 linear / 0.9592 DAgger r1)
the main-method story. Overall Acc is not the headline; report per-class PR/F1
(especially ABSTAIN), Provenance Validity, Coverage / Selective MAE / AURC,
FixRate/WorsenRate (ε later).

---

## 8. Masking ablation API for H2 / H2 掩码消融

Hypothesis H2 (PLAN): removing step-level / evidence-related signal hurts.

Operationalization for **observation channel**:

```python
def mask_observation_block(
    vec: np.ndarray,
    *,
    mode: MaskMode,
    obs_slice: slice,
    rng: np.random.Generator | None = None,
) -> np.ndarray:
    """In-place-safe copy. Pre-action and budget blocks untouched."""
    out = vec.copy()
    if mode == "none":
        return out
    if mode == "zero":
        out[obs_slice] = 0.0
        return out
    if mode == "shuffle":
        if rng is None:
            raise ValueError("shuffle requires an RNG")
        block = out[obs_slice].copy()
        rng.shuffle(block)  # within-vector slot shuffle for unit tests;
        # batch-level row shuffle of the obs block is preferred in eval scripts.
        out[obs_slice] = block
        return out
    raise ValueError(mode)
```

Eval script contract (future, not in this P0):

1. Build full state features with `mask_obs="none"`.
2. Report main metrics.
3. Re-run with `mask_obs="zero"` and with batch-shuffled \(o_{t-1}\) rows.
4. Expect ABSTAIN / evidence-sensitive metrics to degrade under masking if H2 holds.

---

## 9. Evidence chain reminder / 证据链

Image Region → Component → FNDDS Code → Portion → Nutrient.

Feature design must not break provenance: nutrients in the obs block are *reported*
tool outputs to be validated by `provenance_validity` (see `scripts/`), not substitutes
for the FNDDS catalog path.

FNDDS release pinned: **2019–2020** (5624 codes).

---

## 10. Landed sketch layout / 已落地的草图

`vdr/controller/state_features.py` returns a fixed-width vector. It does not train.

| Choice | What landed |
|--------|-------------|
| Prefix | Unchanged 31-d `build_features(h_t)`. Step trio stays here and is **not** copied into \(b_t\). |
| Observation | Fields in §5, in `OBS_FEATURE_NAMES`. `h_t["obs_t"]` is ignored. `train_gt_*` is ignored. |
| First step | `o_prev is None` → observation block zeros and `obs_missing=1`. |
| Previous IR row | If `o_prev` contains `obs_t`, that payload is the observation. Row `action_t`, `verify_codes`, `s`, and `abstain_reason` may be read. Row `nutri_i` is not. |
| Nutrients | Observation payload first (`nutrients` / `nutri` / `n` / `nutri_i`). If that payload has none and `o_prev` is a previous IR row, `nutri_dish` then `nutri_i` (tool-reported public totals). `train_gt_*` is never read. |
| Budget | `budget_tools_used_norm`, `safety_force_abstain`, `safety_force_stop`, `safety_volume_disabled=1`, `safety_legacy_weights_off=1`. The last two are hard 1s. |
| Caps (a priori) | `OBS_SCORE_GAP_TAU=0.05`, `ENERGY_CAP_KCAL=2500`, `UNRESOLVED_CAP=8`, `KEPT_CAP=16`, `TOOL_CAP=32`. Not fit on hold-out or rgb_test. |
| Previous-action one-hot | `actions.json` enum order, volume excluded. Not the confusion-matrix column order. |
| Ready flag | `CLOSED_LOOP_READY = False`. `PolicyBC` / `PolicyMLP` / DAgger still call `build_features` only. |

`build_features_for_ablation(..., mode="B5_open_loop")` is `build_features`. `mode="main_closed_loop"` returns the sketch and still does not train.

---

## 11. Non-claims / 非声明

- This note does **not** train a model.
- `build_state_features` exists as a sketch. Closed-loop control is **not** ready. Do not quote it as a trained Ours policy.
- No RL. No volume tool. No legacy regressor weights. No test-GT supervision.
- Open-loop 31-d accuracy (MLP **0.9700**, linear **0.9206**, DAgger r1 **0.9592**) stays an ablation reference. Hold-out **ABSTAIN 0/24** stays a red flag.
- Frozen `results/n5k_*` and `results/weak_steps/*.jsonl` are not rewritten by this landing.

---

## 12. Paths

| Artifact | Path |
|----------|------|
| This note | `docs/STATE_FEATURE_DESIGN.md` |
| Open-loop B5 | `vdr/controller/features.py` `build_features` |
| Closed-loop sketch + mask | `vdr/controller/state_features.py` |
| Action confusion | `scripts/action_confusion_matrix.py`, `docs/ACTION_CONFUSION.md` |
| Provenance | `scripts/provenance_validity.py` |
| Selective metrics | `scripts/selective_metrics.py` |
| Mock catalog | `tests/fixtures/fndds_mock_foods.json` |
| Smoke (no GPU) | `scripts/p0_eval_smoke.py` |
| Index | `docs/VDR_P0.md` |
