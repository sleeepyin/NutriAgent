# External held-out evaluation

The paired 505 rgb_test dishes remain a development / diagnostic set. This table is the cafe2 held-out named in `results/heldout_plan/`. #53 does not prove the method. #51 is a density / kcal heuristic (automatic Exact on the development food-code audit was 0). On the separate one-step audit, Q ≈ rule ≪ oracle. Nothing here retrained Mass or Q, refit τ, or opened P1. Held-out labels were not used to choose a method, stop early, move τ, or edit a code or a portion. Cafe2 energy targets are ingredient-calorie sums because the dish-level energy column is 0 on every cafe2 row. Mass targets are the dish-level gram totals.

External cafe2 held-out. The main method is the frozen calibrated Q/Cost selector with the coverage guard and the same-head density swap already inside the rollout. Direct-VLM is one Qwen2.5-VL nutrition guess. Mass-only is one frozen estimate_mass RGB forward. Cafe2 dish-level energy is 0 on every row, so the energy target is the sum of ingredient calories. Mass is the dish-level gram total, which matches the ingredient grams. Labels are read only here, to score. No threshold, code, or portion was chosen from these labels.

Planned dishes: 228. Trajectories: 228. Direct-VLM finite: 228. Mass-only finite: 228. Paired (all three finite): 228.

## Main table

MAE is the mean absolute error on the paired dishes. Coverage is the fraction of those dishes whose every named component has a mass-estimator write. Tool calls count RETRIEVE, CORRECT_CODE, RE_PORTION, and RE_DECOMPOSE up to the committing step. Direct-VLM and Mass-only do not take those controller actions. Real latency, token, and dollar cost are gate ④ and are not in this table.

| Method | n | Energy MAE (kcal) | Mass MAE (g) | Coverage | Steps mean | Tool calls mean |
|---|---:|---:|---:|---:|---:|---:|
| Coverage-Constrained + density/same-head | 228 | 116.8494 | 50.9483 | 0.9737 | 6.2193 | 5.2456 |
| Direct-VLM | 228 | 105.6157 | 73.7456 |  | 1 | 0 |
| Mass-only (frozen Mass) | 228 | 75.3870 | 51.7107 |  | 1 | 0 |

Energy MAE minus Direct-VLM: +11.2337 kcal. Mass MAE minus Direct-VLM: -22.7973 g. Energy MAE minus Mass-only: +41.4624 kcal. Mass MAE minus Mass-only: -0.7625 g.

## Coverage and budget

Paired dishes with every named component mass-written: 222 / 228 (0.9737). STOP while a named component is still unwritten: 0. Mass-estimator writes: 575.

6 dishes still leave a named component without a mass write. They ended on RETRIEVE, not on STOP. The rollout does not invent a portion when retrieval has not produced a code.

- `dish_1571946063`: 3 of 5 components mass-written, ended on RETRIEVE.
- `dish_1572464284`: 3 of 4 components mass-written, ended on RETRIEVE.
- `dish_1574269685`: 4 of 5 components mass-written, ended on RETRIEVE.
- `dish_1575664962`: 3 of 6 components mass-written, ended on RETRIEVE.
- `dish_1575923893`: 1 of 2 components mass-written, ended on RETRIEVE.
- `dish_1576100046`: 1 of 2 components mass-written, ended on RETRIEVE.

Steps on the paired dishes: mean 6.2193, median 5.0, min 3, max 15.

Tool calls on the paired dishes: mean 5.2456, median 4.0, min 2, max 14. Within K=32: 228 / 228.

Same figures on every trajectory, including dishes a baseline did not score: coverage 222 / 228, steps mean 6.2193, tool mean 5.2456.

Dishes where the frozen same-head density swap applied: 106. That swap is the #51 kcal heuristic. It is not an identity correction.

## What this run did not do

Mass, Q, and τ were not retrained. P1 stayed closed. There is no RL. Gate ④ real tool cost, gate ⑤ faithful B1/B2, and gate ⑥ provenance were not started. The development 505 energy MAE 64.2792 and mass MAE 32.9908 are not rows of this table.
