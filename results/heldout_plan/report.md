# External held-out plan

This file is the split only. It is not an evaluation and it does not select a method.

## How the paired 505 is identified

A development dish is a dish_id in results/p0d/same_budget_dishes.jsonl with method=='ours_new' and paired==true. That set equals the paired ids of direct_vlm, ours_old, and open_loop_bc, and equals results/coverage_early_stop/paired_scores.jsonl.

Count: **505**. Every id is an official rgb_test id and a cafe1 id. None are in rgb_train. `results/coverage_early_stop/summary.json` records paired n=505.

## How fit is identified

Fit is an official rgb_train dish_id with pool_of_dish=='fit': int(sha256(dish_id).hexdigest()[:8], 16) % 10 != 0. This is vdr.weak.teacher.pool_of_dish. The same ids are the dish set of results/weak_steps/rgb_train_fit_all.jsonl. Mass training used the overhead-RGB subset of this set. Q and tau used the weak steps of this set. The rgb_train internal hold-out (% 10 == 0) is not fit and is also excluded from this external list.

Count: **3679** dishes, of which **2498** have an overhead RGB image. The rgb_train internal hold-out is **380** dishes and is not part of fit.

## Held-out

Held-out is every dish_id in data/nutrition5k/metadata/dish_metadata_cafe2.csv whose overhead rgb.png exists. The list is sorted by dish_id. Cafe2 is the source. The first CSV field is the only field read. No nutrition number, error, or component count is a filter. Ids are kept whole; decision steps are not split. Missing images are listed and are not a label drop.

Cafe2 has **238** dishes. **228** have `realsense_overhead/<dish_id>/rgb.png` and are the held-out list. **10** cafe2 dishes have no overhead image and are listed in `cafe2_missing_overhead.txt`. They were not dropped because of a label.

Each held-out id appears once. The whole dish is on one side of the split.

## Disjointness

| Set | Intersection with held-out |
|---|---:|
| fit | 0 |
| dev505 | 0 |
| rgb_train | 0 |
| rgb_train_internal_holdout | 0 |
| rgb_test | 0 |
| cafe1 | 0 |

`fit ∩ held-out dish_id = ∅` and `dev505 ∩ held-out dish_id = ∅`.

Held-out is cafe2 only. The paired 505 and rgb_train are cafe1.

Held-out membership is fixed before evaluation. It is not a method choice, an early stop, a tau choice, or a food-code or portion edit.

## Scoring labels

Membership above does not read nutrition numbers. Scoring does, and only after predictions exist.

sum of ingredient energy_kcal on the same cafe2 row. The dish-level energy column is 0 on every cafe2 row, so that column is not used.

dish-level mass_g. It equals the sum of ingredient grams on every cafe2 row.

Checked on all 238 cafe2 rows: dish-level energy column is zero on 238, and the largest dish-mass gap versus ingredient grams is 0.000019 g. Ingredient-energy sum is zero on 0 rows; those rows stay in the list. The paired 505 keeps cafe1 dish-level labels.

The #53 development numbers are not a row of a later held-out table. #53 does not prove the method. #51 is a density / kcal heuristic (automatic Exact on the development audit was 0). On the separate one-step audit, Q ≈ rule ≪ oracle. Those statements are framing; this script did not recompute them.
