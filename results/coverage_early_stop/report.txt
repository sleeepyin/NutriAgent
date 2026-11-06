# Coverage of unwritten components and early stop

The calibrated selector still uses the frozen Q weights and the fit-set terminal margin. After that choice, an open component with no mass-estimator write cannot commit STOP or ABSTAIN: the existing rule utility runs instead. A code swap that has already returned to the retrieved top code portions instead of starting another swap. One swap away from the top code is unchanged. Grams still come from `estimate_mass`. The #51 same-head code adjustment still runs when a multi-component plate finishes.

Plates whose logged actions already match this rule keep the P0d trace. Their calories keep the #51 food-code adjustment. Plates that diverge are rolled again from the same overhead image. The vision hypotheses are accepted only when the names already in the trace match, and every mass write before the first new action must reproduce the stored grams. Labels are used only to score the totals after that.

## Paired plates

n=505. Before is the #51 food-code adjustment on the P0d trace. After is the guarded selector.

| | Energy MAE (kcal) | Mass MAE (g) | Tool cost mean |
|---|---:|---:|---:|
| Before (#51) | 92.6963 | 58.9952 | 4.6970 |
| After (coverage guard) | 64.2792 | 32.9908 | 6.0257 |
| Direct-VLM | 97.1312 | 76.3465 | 0 |
| Closed-loop control | 100.4110 | 62.4099 | 6.5782 |

Energy versus Direct-VLM: -4.4348 kcal before, -32.8520 kcal after. Mass versus Direct-VLM: -17.3513 g before, -43.3557 g after.

## Coverage

Plates with every named component mass-written: 199 / 505 (0.3941) before, 503 / 505 (0.9960) after.

STOP while a named component is still unwritten: 78 before, 0 after.

The 199 plates that were already fully mass-written move by at most 3.4e-05 kcal and 0.0e+00 g.

Mass RE_PORTION success 1479 (before 827), failed 0, dish nutrition changed 1479, unchanged 0, hollow 0.

Continued plates: 308. Left on the P0d trace: 197. Rollout refused: 0.

Two plates still leave a named component without a mass write. On both, FNDDS retrieval for `honeycomb` fails, the state does not change, and the existing three-step stuck guard ends the episode on RETRIEVE. RE_PORTION is not legal without a code, so the guard does not invent a portion. Neither plate STOPs.

- `dish_1565020895`: 5 of 7 components mass-written, ended on RETRIEVE.
- `dish_1565020926`: 7 of 8 components mass-written, ended on RETRIEVE.

## Slices

After-metrics are on the guarded commits. Before-metrics on the same plates are the #51 totals.

| Slice | n | Energy MAE after | Energy MAE before | Energy MAE direct | Mass MAE after | Mass MAE before | Mass MAE direct |
|---|---:|---:|---:|---:|---:|---:|---:|
| Paired | 505 | 64.28 | 92.70 | 97.13 | 32.99 | 59.00 | 76.35 |
| Every named component mass-written, after | 503 | 63.89 | 92.42 | 97.22 | 32.58 | 58.39 | 75.65 |
| A named component left unwritten, after | 2 | 162.76 | 162.36 | 75.18 | 137.28 | 211.23 | 252.50 |
| STOP with an unwritten component, after | 0 | | | | | | |
| The 306 plates that were unfinished before | 306 | 63.68 | 110.58 | 101.10 | 39.85 | 82.77 | 85.93 |
| The 78 STOP-unwritten plates | 78 | 60.93 | 144.11 | 104.85 | 44.56 | 103.18 | 100.29 |
| Every named component mass-written, before | 199 | 65.20 | 65.20 | 91.02 | 22.44 | 22.44 | 61.61 |

The original 78 STOP-unwritten plates had mean signed calorie error -87.6 kcal before and +8.4 kcal after. Their addition to the paired energy gap versus Direct-VLM moves from +6.06 kcal to -6.78 kcal.

STOP-unwritten plates that remain after the guard: n=0.

## What was not changed

Mass checkpoint, Q weights, and the terminal-margin fence were not rewritten. No retrieval re-rank, no RL, and no threshold was chosen from rgb_test labels. Frozen n5k tables and weak-step logs were not overwritten. P1 retrieval re-rank stays locked.
