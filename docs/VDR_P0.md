# VDR P0 — Evidence-Aware Closed-Loop (landed scripts)

Offline scaffolding for **Verified Dietary Reasoning**. Truth sources: [`PLAN.md`](../PLAN.md) and [`docs/STATUS.md`](STATUS.md) after `#17` / `7ad3cfb`.

**No closed-loop training in this landing.** No DAgger/BC re-run. No RL. No volume tool. No legacy regressor weights. No test-GT supervision. Frozen `results/n5k_*` and `results/weak_steps/*.jsonl` stay as they are.

- Method center: Evidence-Aware Closed-Loop Tool Control
- Main policy (not trained here): \(\pi(a_t \mid h_t, o_{t-1}, b_t)\) — must read the tool observation
- Open-loop 31-d BC (`build_features`, no \(o_{t-1}\)) = **ablation B5 only**
- Overall Acc is not the story. Report per-class PR/F1 (especially ABSTAIN), Provenance Validity, Coverage / Selective MAE / AURC
- Hold-out **ABSTAIN 0/24** is a red flag. The synthetic demo locks ABSTAIN F1 = 0 and Acc ≈ 0.9700

FNDDS release pinned: **2019–2020** (5624 codes) when a real catalog is loaded. Actions: RETRIEVE, CORRECT_CODE, RE_PORTION, RE_DECOMPOSE, ABSTAIN, STOP (no VOLUME_ESTIMATE).

---

## Layout

| Path | Role |
|------|------|
| [`docs/STATE_FEATURE_DESIGN.md`](STATE_FEATURE_DESIGN.md) | Closed-loop feature interface. Sketch is untrained |
| [`docs/ACTION_CONFUSION.md`](ACTION_CONFUSION.md) | Confusion-matrix notes |
| `vdr/controller/features.py` | Open-loop B5 `build_features` (31-d) |
| `vdr/controller/state_features.py` | `build_state_features` sketch. `CLOSED_LOOP_READY` is false |
| `vdr/eval/action_metrics.py` | Per-class PR/F1 + confusion matrix |
| `vdr/eval/provenance.py` | `n == code × portion/100 × per100g`, release pin |
| `vdr/eval/selective.py` | Coverage, Selective MAE, AURC |
| `scripts/action_confusion_matrix.py` | CLI, including `--demo` |
| `scripts/provenance_validity.py` | CLI. Mock offline; real catalog must be 2019–2020 |
| `scripts/selective_metrics.py` | CLI |
| `tests/fixtures/fndds_mock_foods.json` | 3-food mock. Not the 5624-code release |
| `scripts/p0_eval_smoke.py` | CPU smoke. No GPU |

---

## 1) State features

Read [`STATE_FEATURE_DESIGN.md`](STATE_FEATURE_DESIGN.md).

```python
build_state_features(h_t, o_prev, b_t, *, mask_obs="none"|"zero"|"shuffle") -> np.ndarray
build_features(state)  # existing 31-d → B5-only path
```

`PolicyBC`, `PolicyMLP`, and DAgger still call `build_features` only.

---

## 2) Action confusion matrix

```bash
python3 scripts/action_confusion_matrix.py --demo
```

Expect **ABSTAIN F1 = 0** and overall accuracy **0.9700** for the documented BC MLP hold-out synthetic case (recalls: RETRIEVE 1273/1285, RE_PORTION 917/949, STOP 356/356, CORRECT_CODE 142/157, ABSTAIN 0/24, RE_DECOMPOSE 0/0).

Offline JSONL lines look like `{"action_t":"…","pred_action":"…"}`.

---

## 3) Provenance validity

```bash
python3 scripts/provenance_validity.py --demo
```

Uses `tests/fixtures/fndds_mock_foods.json`. Checks `n == code × portion/100 × per100g` within `atol` (default 1e-2). The demo expects 2 passes and 2 failures.

Real catalog (list of foods, not the mock):

```bash
python3 scripts/provenance_validity.py \
  --catalog data/fndds/2019-2020/foods.json \
  --jsonl claims.jsonl
```

A real catalog whose version, release, or path does not identify **2019–2020** is rejected. The toy catalog under `data/fndds/toy/` is not that release. The mock is marked `kind=mock` and is not claimed to be the 5624-code file.

---

## 4) Selective metrics

```bash
python3 scripts/selective_metrics.py --demo
```

Coverage \(C = |keep| / N\). Selective MAE is the mean error on the keep set. AURC sorts by confidence = ascending `abstain_score`, sets \(c_k = k/N\), \(r_k =\) mean error of the top-\(k\) confident rows, and integrates risk versus coverage with the trapezoid, including \(0.5 \cdot r_1 \cdot c_1\) from the origin. Lower AURC is better. Formula text lives in `vdr/eval/selective.py`.

---

## Smoke

```bash
python3 scripts/p0_eval_smoke.py
```

CPU only. It does not fit a policy and does not write under `results/`.
