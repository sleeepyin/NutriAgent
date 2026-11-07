# Food-id semantic audit of the #51 same-head swaps

Automatic Exact / Near / Wrong labels are suggestions only. They do not close the audit on the 37/50 code-swapped multi-component dishes. A person has to confirm every review row.

This folder is a read of the frozen #51 artifacts. The mass estimator, the calibrated selector, and the #51 / #53 / #57 evaluators are not run. Grams and actions stay the ones already stored. Labels are used only to score identity and macros after the code choice that #51 already recorded.

## What was compared

Finished multi-component plates in the frozen summary: **50**. Plates where the same-head swap changed a code: **37**. Component codes changed: **68**. Absolute calorie error fell on **30** of the 50 and rose on **7**.

The frozen `results/food_id_density/summary.json` stores those dish totals and three example swaps. The rest of the component codes are recovered by `align_multicomponent_codes` on the frozen P0d trace (`results/p0d/ours_new_rgb_test.jsonl` plus the paired dish file). Recovery is accepted only when the applied flag, the reason, the changed-code count, the dish energy, and the three published examples match the frozen summary.

Ground-truth food codes are Nutrition5K `ingr_*` ids on the cafe ingredient rows. FNDDS codes and `ingr_*` ids are different systems, so Exact is not string equality of the two codes. It is the noun rule below, applied to the swapped FNDDS name and the matched cafe ingredient name.

## Taxonomy that is not in the repo

FNDDS `foods.json` was checked (5624 foods). Alias rows that are anything other than the lowercased description plus the head clause: **0**. Record keys: aliases, code, fdc_id, name, per_100g, portions. There is no food-group, WWEIA, or taxonomy field. Nutrition5K `ingredients_metadata.csv` columns: ingr, id, cal/g, fat(g), carb(g), protein(g). Those columns are id, name, and per-gram macros. No family and no synonym column.

The fallback is the token rule in `scripts/food_id_semantic_audit.py`. It uses the repository lemma function from FNDDS retrieval (`_lemma` / `_TOKEN_RE`). The token lists were not fit on these 50 plates. A hit on a variety word is Near, never Exact. Commodity synonyms are not rewritten into Near.

### Exact

Food-noun sets are equal, the code does not negate a noun, and preparation beyond raw / fresh / NFS / NS / plain, cut, and style are the same as the cafe name.

### Near

Any of the following, and not Exact:

- The food nouns match, and preparation, cut, or style differs (raw apple versus dried apple, chicken versus canned chicken meat, cottage cheese versus dry-curd cottage cheese).
- One food-noun set contains the other (potato versus sweet potato, pizza versus cheese pizza, broccoli versus chinese broccoli).
- The food nouns overlap and every unshared noun is in the variety/color list (white rice versus brown rice).

### Wrong

No cafe ingredient shares a food noun with the vision query; the code negates a cafe noun; the code is a meat analog and the cafe ingredient is not; or the leftover nouns are not a subset and not variety qualifiers (pike versus salmon, beef versus steak, sweet potato versus yam).

The matched cafe row is the ingredient whose food nouns overlap the vision query the most (Jaccard, then grams). Ties are flagged ambiguous and the heavier row is kept only as a placeholder. Ingredients named `deprecated` are skipped. The swapped FNDDS name is not used to choose the cafe row.

### Token lists

- Default prep, ignored when it is the only prep: fresh, nfs, ns, plain, raw.
- Density-changing prep: baked, barbecued, bbq, blanched, boiled, breaded, broiled, canned, cooked, creamed, dehydrated, drained, dried, fried, frozen, grilled, mashed, microwaved, pickled, poached, refried, roasted, salted, saute, sauteed, scrambled, smoked, steamed, stewed, toasted, unroasted, unsalted.
- Cut: back, bone, boneless, breast, brisket, chopped, chuck, chunk, diced, drumstick, drumsticks, filet, fillet, ground, leg, legs, loin, meat, neck, only, oxtail, oxtails, peel, piece, rib, ribs, round, shank, shredded, skin, skinless, slice, sliced, strip, strips, thigh, wing, wings.
- Style: crust, curd, dry, extra, farmer, lowfat, nonfat, reduced, skim, stuffed, thick, thin.
- Variety/color qualifiers: baby, bell, black, brown, cherry, chile, chili, chinese, dark, garbanzo, glutinous, golden, green, hot, kidney, light, navy, new, pinto, purple, red, roma, spanish, sweet, vidalia, white, whole, wild, yellow, yolk.
- Faux-meat tokens: imitation, meatless, vegan, vegetarian.

## Identity of the swapped codes

Denominator is the **68** swapped component codes, not the 50 plates. The label is the post-swap code against the matched cafe ingredient. Pre-swap labels use the same cafe row.

| | n | Exact | Near | Wrong |
|---|---:|---:|---:|---:|
| Post-swap code | 68 | 0 (0.00%) | 50 (73.53%) | 18 (26.47%) |
| Pre-swap code, same cafe row | 68 | 27 (39.71%) | 23 (33.82%) | 18 (26.47%) |

Across the 68 codes the auto label improved on **1**, stayed the same on **39**, and worsened on **28**.

| Pre \ Post | Exact | Near | Wrong |
|---|---:|---:|---:|
| Exact | 0 | 27 | 0 |
| Near | 0 | 22 | 1 |
| Wrong | 0 | 1 | 17 |

Codes whose auto label moved up the order Exact > Near > Wrong. The move is still a suggestion:

- `dish_1560801829` query `pizza`: Pizza, cheese, stuffed crust → Pizza, no cheese, thick crust against cafe pepperoni pizza (Wrong → Near). shared {pizza}; extra code {}; extra cafe {pepperoni}

On the 37 swapped plates the worst auto label is Exact on **0**, Near with no Wrong code on **22**, and Wrong on at least one code on **15**.

## Macro error

Protein, carbohydrate, fat, and energy are the food-table sums: each component's frozen FNDDS per-100 g value times the stored grams. The ground truth is the cafe dish total. Grams are the mass estimator's grams, and the vision list is not the cafe ingredient list, so this MAE mixes identity, preparation, missing foods, and portion error. The pre-swap to post-swap change is only the code swap.

Energy on the 50-plate slice is the #51 food-table energy. Recomputed MAE is **190.7886 → 131.6452** kcal. The frozen summary reports **190.7887 → 131.6452**.

| Slice | n | Protein MAE before | after | Carb MAE before | after | Fat MAE before | after | Energy MAE before | after |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Finished multi-component | 50 | 10.5747 | 10.0661 | 13.0033 | 16.1715 | 17.5631 | 11.9737 | 190.7886 | 131.6452 |
| Code-swapped plates | 37 | 9.9250 | 9.2377 | 12.2415 | 16.5228 | 17.3938 | 9.8406 | 179.2208 | 99.2972 |
| Finished plates with no code change | 13 | 12.4240 | 12.4240 | 15.1716 | 15.1716 | 18.0448 | 18.0448 | 223.7124 | 223.7124 |

Units are grams for protein, carbohydrate, and fat, and kcal for energy. MAE after minus MAE before: swapped plates protein **-0.6873**, carb **+4.2813**, fat **-7.5531**, energy **-79.9236**.

## kcal getting better ≠ food identity becoming correct

Of the **30** swapped plates whose absolute energy error fell, **0** have every swapped code auto-labeled Exact, and **30** do not. Among those 30, **12** have at least one Wrong code and **18** are Near with no Wrong code.

Of the **50** swapped codes that sit on an energy-improved plate, **50** are Near or Wrong. Post-swap counts on those codes: Exact **0**, Near **36**, Wrong **14**.

Energy error rose on **7** swapped plates. Of those, **0** are all Exact and **7** are not.

The disagreement count used in this audit is the plate count **30 / 30**: calorie error fell, and the auto identity of the swapped codes is not all Exact. That count is a suggestion until the review pack is confirmed by a person.

On the 37 swapped plates, energy MAE changed by **-79.9236 kcal**, protein by **-0.6873 g**, carbohydrate by **+4.2813 g**, and fat by **-7.5531 g**. **27** codes moved from auto-Exact to auto-Near, and **0** swapped codes are auto-Exact after the swap.

## Human review

Review rows: **68** component rows covering **37** dishes. Every one of those rows has `needs_human_review=yes` and `auto_label_is_suggestion_only=yes`. The other **13** finished multi-component plates did not change a code and are not in the review pack.

A person has to confirm:

1. Automatic Exact / Near / Wrong labels are suggestions only. They do not close the audit on the 37/50 code-swapped multi-component dishes. A person has to confirm every review row.
2. Confirm which cafe ingredient, if any, the vision query was naming. The match uses food-noun overlap with the query. It has no synonym table, so steak against beef, yam against sweet potato, and salmon against fish stay unmatched or Wrong until a person links them.
3. Confirm Exact: the food nouns match, and the code does not add a preparation, cut, or style the cafe name lacks. Raw, fresh, NFS, NS, and plain are treated as unspecified.
4. Confirm Near: same food nouns with a different preparation, cut, or style; one name's food nouns contained in the other; or a variety/color difference such as sweet versus plain potato, white versus brown rice, or chinese broccoli. Near is not an identity success. A qualifier can name a different commodity.
5. Confirm Wrong: the query shares no food noun with any cafe ingredient, the code negates a cafe noun such as no cheese, the code is a meat analog the cafe ingredient is not, or the leftover nouns are not a subset or a variety qualifier.
6. A smaller calorie error is not evidence that the swapped code is the right food. Rows with energy_improved_but_identity_not_all_exact are the disagreement set and have to be read one by one.
7. Ambiguous ties, and matches under the existing 8 g / 8 kcal ingredient floor, need a person to pick the referent. Deprecated cafe rows are ignored.

Tables: `human_review_components.csv` (one row per swapped code), `human_review_dishes.csv` (one row per swapped dish), and `human_review.md`.

## What this audit does not do

It does not retrain the mass estimator, Q, or τ. It does not rerank retrieval, run reinforcement learning, or retune rgb_test thresholds. It does not rerun #51, #53, or #57. It does not edit `docs/STATUS.md`, so gates ③–⑦ and P1 stay locked. It does not overwrite `results/food_id_density/`, `results/coverage_early_stop/`, or `results/qphi_counterfactual/`.

```bash
python scripts/food_id_semantic_audit.py --self-check
python scripts/food_id_semantic_audit.py
```
