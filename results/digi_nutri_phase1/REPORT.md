# DiGI-Nutri Phase 1

Verdict: `q0_fail_stop_wait_triste`

Q0 failed. The privileged teachers are not clearly better than B1 on energy and the three macros. Stop. Do not keep distilling. Wait for Triste.

This file is the single official holdout read on the Nutrition5k depth test split. Student numbers use one overhead RGB image. Teacher numbers use that teacher's privileged inputs and are only a Q0 headroom check. Mass is not a pass condition. STATUS.md was not edited. Phase 2 was not started. The pull request base is digi-nutri, not main.

## Rules applied

A teacher is clearly better than B1 on a nutrient when the plate-grouped 95% CI of (teacher - B1) lies entirely below 0 for both per-dish MAE and per-dish PMAE. Q0 passes only when the geometry teacher and the names-only ingredient teacher both clear that bar on energy, fat, carb, and protein. Mass is recorded and is not required.

Stably better means the plate-grouped 95% CI of (method - baseline) is strictly below 0 on both per-dish MAE and per-dish PMAE. Resamples draw physical plates, using dish_id unless a reshoot file groups ids. A CI that touches or crosses 0 is not a pass. Mass is not a pass condition.

Q0 pass: False. Q1 opened: False. Q1 pass: None. Q2 opened: False. Q2 pass: None.

Q1 and Q2 were not opened. Any descriptive contrast below is not a license to keep distilling or to change the loss.

## Holdout

Dishes: 507. Plate groups: 507. PMAE is `100 * |pred-y| / max(|y|, eps)`.

### MAE

| nutrient | geometry teacher (privileged) | ingredient teacher (privileged, names only) | B1 RGB | B2a RGB | B2b RGB | B3 RGB |
|---|---|---|---|---|---|---|
| energy (headline) | 63.3402 | 60.1223 | 62.0600 | 62.3177 | 62.8174 | 63.7823 |
| mass (recorded) | 45.9163 | 39.5519 | 43.2574 | 41.9541 | 42.6078 | 45.0299 |
| fat (headline) | 4.4551 | 4.2150 | 4.4115 | 4.1666 | 4.4450 | 4.1591 |
| carb (headline) | 6.2738 | 5.8796 | 6.2011 | 5.8764 | 6.2920 | 5.9944 |
| protein (headline) | 5.7563 | 5.8715 | 5.9805 | 5.8001 | 5.7203 | 5.7491 |

### PMAE percent

| nutrient | geometry teacher (privileged) | ingredient teacher (privileged, names only) | B1 RGB | B2a RGB | B2b RGB | B3 RGB |
|---|---|---|---|---|---|---|
| energy (headline) | 32.1369 | 29.0393 | 31.9892 | 30.2067 | 31.1155 | 31.1952 |
| mass (recorded) | 30.0469 | 31.1967 | 30.6880 | 29.0352 | 31.3380 | 32.4156 |
| fat (headline) | 56.4372 | 42.9622 | 52.0969 | 50.3787 | 50.2677 | 51.5059 |
| carb (headline) | 44.5188 | 39.1282 | 46.0278 | 40.7373 | 40.7822 | 37.7862 |
| protein (headline) | 54.4011 | 47.1457 | 50.1300 | 50.2981 | 49.9601 | 50.6460 |

## Plate-grouped 95% intervals of (method − baseline)

### geometry_teacher_minus_b1

| nutrient | role | MAE difference | PMAE difference | stable |
|---|---|---|---|---|
| energy (kcal) | headline | 1.2801 [-1.2849, 3.9860] | 0.1477 [-2.0579, 2.4204] | False |
| mass (g) | recorded | 2.6589 [0.4422, 4.8664] | -0.6411 [-3.2618, 1.5861] | False |
| fat (g) | headline | 0.0437 [-0.2006, 0.2972] | 4.3403 [0.4318, 8.4662] | False |
| carb (g) | headline | 0.0727 [-0.2248, 0.3816] | -1.5091 [-6.2729, 3.2279] | False |
| protein (g) | headline | -0.2242 [-0.5937, 0.1323] | 4.2711 [-0.4104, 9.7534] | False |

### ingredient_teacher_minus_b1

| nutrient | role | MAE difference | PMAE difference | stable |
|---|---|---|---|---|
| energy (kcal) | headline | -1.9377 [-4.6149, 0.8129] | -2.9499 [-5.8865, 0.0394] | False |
| mass (g) | recorded | -3.7055 [-6.1853, -1.3113] | 0.5088 [-2.5762, 4.0835] | False |
| fat (g) | headline | -0.1965 [-0.4165, 0.0357] | -9.1347 [-14.5068, -3.9196] | False |
| carb (g) | headline | -0.3214 [-0.6383, 0.0024] | -6.8997 [-11.6948, -2.8481] | False |
| protein (g) | headline | -0.1089 [-0.4900, 0.2525] | -2.9843 [-6.3403, 0.3853] | False |

### b3_minus_b1

| nutrient | role | MAE difference | PMAE difference | stable |
|---|---|---|---|---|
| energy (kcal) | headline | 1.7223 [-0.4748, 3.8955] | -0.7941 [-3.1831, 1.8421] | False |
| mass (g) | recorded | 1.7724 [-0.0502, 3.5504] | 1.7276 [-0.1015, 4.0133] | False |
| fat (g) | headline | -0.2524 [-0.4730, -0.0350] | -0.5910 [-4.3322, 2.7143] | False |
| carb (g) | headline | -0.2067 [-0.4647, 0.0652] | -8.2416 [-14.9650, -2.7329] | False |
| protein (g) | headline | -0.2314 [-0.5461, 0.0723] | 0.5160 [-3.2626, 4.7257] | False |

### b3_minus_b2a

| nutrient | role | MAE difference | PMAE difference | stable |
|---|---|---|---|---|
| energy (kcal) | headline | 1.4646 [-0.7326, 3.6522] | 0.9885 [-0.8584, 3.0837] | False |
| mass (g) | recorded | 3.0758 [1.0213, 5.1266] | 3.3804 [1.5172, 5.2842] | False |
| fat (g) | headline | -0.0075 [-0.2195, 0.2037] | 1.1272 [-2.3034, 4.2599] | False |
| carb (g) | headline | 0.1180 [-0.1684, 0.3997] | -2.9511 [-6.8359, 0.5536] | False |
| protein (g) | headline | -0.0510 [-0.3564, 0.2524] | 0.3479 [-3.5014, 4.3814] | False |

### b3_minus_b2b

| nutrient | role | MAE difference | PMAE difference | stable |
|---|---|---|---|---|
| energy (kcal) | headline | 0.9649 [-1.2960, 3.2028] | 0.0797 [-1.8442, 2.4415] | False |
| mass (g) | recorded | 2.4221 [0.7790, 4.0419] | 1.0776 [-0.3158, 2.4166] | False |
| fat (g) | headline | -0.2859 [-0.5128, -0.0645] | 1.2383 [-1.3266, 4.2193] | False |
| carb (g) | headline | -0.2976 [-0.5696, -0.0331] | -2.9960 [-6.9276, 0.2470] | False |
| protein (g) | headline | 0.0288 [-0.2735, 0.3226] | 0.6859 [-3.2363, 5.5181] | False |

## Frozen training setup

Backbone `resnet101`, fusion `add`, image 256, batch 32, teacher epochs 30, student epochs 30, AdamW lr 0.0001, weight decay 0.0001, cosine eta_min 1e-06.

Warmup epochs 5, distillation ramp epochs 5. Loss weights geo 1.0, semantic 1.0, aux 0.2. Depth fill `median`. Text model `openai/clip-vit-base-patch32`. Gram weighting: False.

geometry teacher probe fit batch 32

## Last training epoch (train fit, not the holdout)

- b1: task/loss 0.17800479144745315 MAE [36.963952502913465, 38.40031579377085, 1.679571356388998, 2.7671421780998324, 2.291555986771483]
- b2a: task/loss 0.16991853103658472 MAE [34.66157841734377, 34.27302635486522, 1.514573036975071, 2.7024288496074247, 2.1012749858931]
- b2b: task/loss 0.20587267142841587 MAE [36.32975841624718, 37.91062103914193, 1.714779815729234, 2.903380853158456, 2.3239048277024525]
- b3: task/loss 0.20761965108073616 MAE [35.77471139972939, 38.67103053628316, 1.605238343116245, 2.8233537362291923, 2.1848904558429703]
- geometry_teacher: task/loss 0.1786633939005451 MAE [40.01798946267033, 39.91402578804055, 1.5489097576515347, 2.6905140942564545, 2.0789872618282543]
- ingredient_teacher: task/loss 0.1519979046753722 MAE [29.774063625626574, 29.21678384352873, 1.543016058513888, 2.617941568273518, 2.1493488268827994]

## Git

HEAD `4ed130274273357f9df5ab02f08b0b9aacadeb8a`. Branch `cursor/digi-nutri-phase1-abd6`. main `2e62d3c8a2c2a3b7d0a8d5e3ca72c19d949ed08e`. digi-nutri `09b9df0f13f9538d04ec5897342b5aabfa479985`.

The metrics JSON next to this file is the holdout record. A second evaluation must refuse to overwrite it.
