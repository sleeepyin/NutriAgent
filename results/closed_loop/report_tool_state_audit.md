# P0a audit: do tool steps change dish energy and mass?

This is the audit gate only. Mass Estimator training was not started. No new specialist weights. No RL.

Pre-state is `nutri_dish` on the tool row. Post-state is `nutri_dish` on the next logged row. A change is dish energy and/or mass moving by more than ε = 5e-5 (half of the 1e-4 nutrient rounding quantum). That ε is not the FixRate 1.0 kcal threshold.

Edit actions are `CORRECT_CODE`, `RE_PORTION`, and `RE_DECOMPOSE`. `RETRIEVE` is audited with them. Rates below use every executed step as the denominator. `no_successor` means the tool is the last row, so no post dish state was written.

## logged_holdout

Dishes 380, rows 2771, dishes with an audited tool 380.

| slice | executed | change rate | no-change rate | energy change | mass change |
| --- | ---: | ---: | ---: | ---: | ---: |
| edit actions | 1106 | 1.0000 (1106/1106) | 0.0000 (0/1106) | 1.0000 (1106/1106) | 0.8580 (949/1106) |
| all audited tools | 2391 | 0.9770 (2336/2391) | 0.0230 (55/2391) | 0.9770 (2336/2391) | 0.9113 (2179/2391) |
| CORRECT_CODE | 157 | 1.0000 (157/157) | 0.0000 (0/157) | 1.0000 (157/157) | 0.0000 (0/157) |
| RE_PORTION | 949 | 1.0000 (949/949) | 0.0000 (0/949) | 1.0000 (949/949) | 1.0000 (949/949) |
| RE_DECOMPOSE | 0 | n/a | n/a | n/a | n/a |
| RETRIEVE | 1285 | 0.9572 (1230/1285) | 0.0428 (55/1285) | 0.9572 (1230/1285) | 0.9572 (1230/1285) |

## on_policy_rgb_test

Dishes 507, rows 3822, dishes with an audited tool 506.

| slice | executed | change rate | no-change rate | energy change | mass change |
| --- | ---: | ---: | ---: | ---: | ---: |
| edit actions | 1795 | 0.9175 (1647/1795) | 0.0825 (148/1795) | 0.9175 (1647/1795) | 0.7755 (1392/1795) |
| all audited tools | 3325 | 0.9417 (3131/3325) | 0.0553 (184/3325) | 0.9417 (3131/3325) | 0.8626 (2868/3325) |
| CORRECT_CODE | 326 | 0.7822 (255/326) | 0.2178 (71/326) | 0.7822 (255/326) | 0.0000 (0/326) |
| RE_PORTION | 1469 | 0.9476 (1392/1469) | 0.0524 (77/1469) | 0.9476 (1392/1469) | 0.9476 (1392/1469) |
| RE_DECOMPOSE | 0 | n/a | n/a | n/a | n/a |
| RETRIEVE | 1530 | 0.9699 (1484/1530) | 0.0235 (36/1530) | 0.9699 (1484/1530) | 0.9647 (1476/1530) |

## logged_holdout_stop_pass_corrected

Dishes 356, rows 2606, dishes with an audited tool 356.

| slice | executed | change rate | no-change rate | energy change | mass change |
| --- | ---: | ---: | ---: | ---: | ---: |
| edit actions | 1061 | 1.0000 (1061/1061) | 0.0000 (0/1061) | 1.0000 (1061/1061) | 0.8549 (907/1061) |
| all audited tools | 2250 | 0.9836 (2213/2250) | 0.0164 (37/2250) | 0.9836 (2213/2250) | 0.9151 (2059/2250) |
| CORRECT_CODE | 154 | 1.0000 (154/154) | 0.0000 (0/154) | 1.0000 (154/154) | 0.0000 (0/154) |
| RE_PORTION | 907 | 1.0000 (907/907) | 0.0000 (0/907) | 1.0000 (907/907) | 1.0000 (907/907) |
| RE_DECOMPOSE | 0 | n/a | n/a | n/a | n/a |
| RETRIEVE | 1189 | 0.9689 (1152/1189) | 0.0311 (37/1189) | 0.9689 (1152/1189) | 0.9689 (1152/1189) |

## on_policy_stop_pass_corrected

Dishes 488, rows 3641, dishes with an audited tool 488.

| slice | executed | change rate | no-change rate | energy change | mass change |
| --- | ---: | ---: | ---: | ---: | ---: |
| edit actions | 1722 | 0.9164 (1578/1722) | 0.0836 (144/1722) | 0.9164 (1578/1722) | 0.7741 (1333/1722) |
| all audited tools | 3153 | 0.9496 (2994/3153) | 0.0504 (159/3153) | 0.9496 (2994/3153) | 0.8693 (2741/3153) |
| CORRECT_CODE | 314 | 0.7803 (245/314) | 0.2197 (69/314) | 0.7803 (245/314) | 0.0000 (0/314) |
| RE_PORTION | 1408 | 0.9467 (1333/1408) | 0.0533 (75/1408) | 0.9467 (1333/1408) | 0.9467 (1333/1408) |
| RE_DECOMPOSE | 0 | n/a | n/a | n/a | n/a |
| RETRIEVE | 1431 | 0.9895 (1416/1431) | 0.0105 (15/1431) | 0.9895 (1416/1431) | 0.9839 (1408/1431) |

## Step-weighted rollup

Step-weighted sum of the named full trajectories. Not one evaluation population. Dish subsets are not included, so a dish is not counted twice.

Edit actions 0.9490 (2753/2901) changed, no-change 0.0510 (148/2901). All audited tools 0.9564 (5467/5716) changed.

## Notes

- logged_holdout edit actions: change 1106/1106, no-change 0/1106, value_edit 1098, wipe 8. No-change is not the majority outcome on this population.
- on_policy_rgb_test edit actions: change 1647/1795, no-change 148/1795, value_edit 1639, wipe 8. No-change is not the majority outcome on this population.
- logged_holdout RE_DECOMPOSE: 0 executed steps.
- logged_holdout RE_PORTION: 8 changed steps replace a finite dish vector with a missing one (next action: ABSTAIN=8). Those count as written-vector changes and are not included in value_edit.
- logged_holdout RETRIEVE: 19 changed steps replace a finite dish vector with a missing one (next action: ABSTAIN=12, RETRIEVE=7). Those count as written-vector changes and are not included in value_edit.
- on_policy_rgb_test RE_DECOMPOSE: 0 executed steps.
- on_policy_rgb_test RE_PORTION: 77/1469 left dish energy and mass within ε; 77 of those unchanged steps have portion_g and from_portion_g within ε, so the portion call did not move grams.
- on_policy_rgb_test RE_PORTION: 8 changed steps replace a finite dish vector with a missing one (next action: ABSTAIN=8). Those count as written-vector changes and are not included in value_edit.
- logged_holdout_stop_pass_corrected RE_DECOMPOSE: 0 executed steps.
- logged_holdout_stop_pass_corrected RETRIEVE: 6 changed steps replace a finite dish vector with a missing one (next action: RETRIEVE=6). Those count as written-vector changes and are not included in value_edit.
- on_policy_stop_pass_corrected RE_DECOMPOSE: 0 executed steps.
- on_policy_stop_pass_corrected RE_PORTION: 75/1408 left dish energy and mass within ε; 75 of those unchanged steps have portion_g and from_portion_g within ε, so the portion call did not move grams.
- on_policy_rgb_test CORRECT_CODE: 71/326 left dish energy and mass within ε (tool_failed 58, code switched but same dish energy and mass 11). Mass change rate is separate: a same-gram code swap does not move mass_g.
- on_policy_stop_pass_corrected CORRECT_CODE: 69/314 left dish energy and mass within ε (tool_failed 56, code switched but same dish energy and mass 11). Mass change rate is separate: a same-gram code swap does not move mass_g.
- No-change does not dominate any audited action or rollup (majority cutoff 0.5 of executed steps). The cutoff is descriptive and was not fit on hold-out or rgb_test. Minority no-change steps are still counted above.
