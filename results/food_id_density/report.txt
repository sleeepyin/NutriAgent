# Food-code density alignment on finished multi-component plates

The mass estimator's grams stay. On a plate whose every named component already has a mass write, the committed calorie is still the bound food code's FNDDS kcal per 100 g times those grams. This pass may replace that code with another code from the same retrieve window when three things are true: the new code's FNDDS head does not add a food noun, it does not introduce a dish form or a prepared form the query did not ask for, and the plate's food-table energy moves at least 15% closer to the mass estimator's own calorie total. The 15% figure is the existing density-gain cutoff. It was not chosen on this test set.

Actions and grams are the ones already stored for the paired comparison. The selector is not asked again, and the mass network is not trained again. Labels are used only to score the calorie and the gram error after the code choice.

## Paired plates

n=505. Mass mean absolute error is unchanged because the grams are unchanged.

| | Energy MAE (kcal) | Mass MAE (g) |
|---|---:|---:|
| Mass estimator + calibrated selection, before | 98.5521 | 58.9952 |
| Same plates after the food-code swap | 92.6963 | 58.9952 |
| Direct vision-language model | 97.1312 | 76.3465 |
| Closed-loop control | 100.4110 | 62.4099 |

Energy MAE changes by -5.8558 kcal against the previous food-table commit and by -4.4348 kcal against the direct vision-language model. Mass MAE stays at 58.9952 g.

## Finished multi-component plates

n=50. These are the plates where every named component received a mass write and the plate named more than one component. The swap ran on 37 of them and changed 68 component codes. Absolute calorie error fell on 30 plates and rose on 7.

| | Energy MAE (kcal) | Mass MAE (g) |
|---|---:|---:|
| Food-table commit, before | 190.79 | 44.30 |
| After the same-head swap | 131.65 | 44.30 |
| Direct vision-language model | 180.78 | 136.10 |
| Stored calorie head (not a commit) | 77.75 | 44.30 |

On this slice the food-table energy MAE was +10.01 kcal versus the direct model before the swap and -49.13 kcal after it. The stored calorie head is listed only as a reading of the forward pass the swap uses as its target. The scored commit is still the food-table product.

## Portion checks

Successful mass portions in the saved trajectories: 827. Hollow portions (a success that is not the mass estimator): 0. Portions whose dish energy and mass were unchanged at the next step: 0. The swap does not add a portion step and does not put the old grams back, so a portion that already changed the dish total still changes it.

## Examples

Absolute calorie error on the finished multi-component slice. A negative change is a smaller error.

- `dish_1563984167`: label 292 kcal, mass-estimator head 270 kcal, food table 595 → 336 kcal, absolute error 303 → 44 (-259).
  - cottage cheese: Cottage cheese, farmer's → Cheese, cottage, dry curd (364 g)
- `dish_1566316757`: label 448 kcal, mass-estimator head 487 kcal, food table 776 → 701 kcal, absolute error 328 → 253 (-75).
  - diced potatoes: Potato, NFS → Potato, baked, NFS (208 g)
  - gravy: Gravy, NFS → Gravy, poultry (125 g)
- `dish_1560368019`: label 437 kcal, mass-estimator head 628 kcal, food table 496 → 593 kcal, absolute error 59 → 156 (+97).
  - spinach: Spinach, raw → Spinach, fresh, cooked, fat added, NS as to fat type (94 g)
  - asparagus: Asparagus, raw → Asparagus, fresh, cooked, fat added, NS as to fat type (63 g)
  - stuffed eggplant: Eggplant, raw → Eggplant, cooked, fat added (71 g)
  - pizza: Pizza, cheese, stuffed crust → Pizza, extra cheese, thin crust (118 g)

## What this does not do

The mass network is not trained again. The Q weights and the terminal-margin fence are not changed and are not refit. Retrieval is not reranked and no new candidate list is fetched. Coverage and early stopping are untouched. `docs/STATUS.md` is untouched. Labels were not used to pick a code, a cutoff, or a gram weight.

The food noun is the FNDDS text before the first comma. A preparation written after that comma, such as `Apple, dried` against `Apple, raw`, can still be selected when it pulls the plate total toward the calorie head. Seven of the fifty finished multi-component plates have a larger calorie error after the swap. The head itself remains closer than the swapped table (77.75 kcal versus 131.65 kcal on that slice).

A later full pass would let the selector see the swapped calorie total before it stops. This table holds the recorded actions fixed, which is the comparison that isolates the food-code change.

```bash
python scripts/food_id_density_align.py
```
