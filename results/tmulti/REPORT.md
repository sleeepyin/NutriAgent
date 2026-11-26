# Unified privileged teacher T_multi

One training run of T_multi under the frozen #91 recipe.
The comparator is the frozen #91 same-setting B1. B1 was not retrained.
Geometric augmentation, head dropout, the split learning rate, and the
explicit height readout are the pre-registered regularization and volume
features. This record does not treat T_multi as a new method.
Distillation was not run.

T_multi parameters: 57895814.

## Train metrics

Eval mode, no augmentation, kept official train plates.
These numbers are not a Q0 gate. A train advantage is not a pass.

| Model | Energy | Fat | Carb | Protein | Mass |
|---|---:|---:|---:|---:|---:|
| frozen #91 B1 | 19.3102 | 1.1887 | 1.9318 | 1.7235 | 13.0775 |
| t_multi | 11.5764 | 0.7134 | 1.3462 | 0.9449 | 9.2824 |

## Official holdout

Split `official_depth_test`, n=507, groups=507.
Eval mode, no augmentation. Plate nutrition is predicted mass times predicted density.
The B1 column is the frozen #91 holdout prediction, not a new run.

### MAE

| Model | Energy | Fat | Carb | Protein | Mass |
|---|---:|---:|---:|---:|---:|
| frozen #91 B1 | 47.8534 | 3.4686 | 4.6763 | 4.6013 | 28.2641 |
| t_multi | 44.4616 | 3.1782 | 4.1075 | 3.7287 | 26.1155 |

### Dataset PMAE

| Model | Energy | Fat | Carb | Protein | Mass |
|---|---:|---:|---:|---:|---:|
| frozen #91 B1 | 18.7317 | 27.0334 | 23.7573 | 26.3255 | 14.2382 |
| t_multi | 17.4041 | 24.7702 | 20.8678 | 21.3327 | 13.1558 |

### Per-plate PMAE

| Model | Energy | Fat | Carb | Protein | Mass |
|---|---:|---:|---:|---:|---:|
| frozen #91 B1 | 27.9168 | 81.2066 | 59.8861 | 45.9589 | 20.7816 |
| t_multi | 24.3715 | 53.9998 | 41.1625 | 35.9357 | 20.1849 |

### Headline intervals

Each row is the 95% plate-grouped CI of (T_multi - frozen #91 B1).
An interval is below zero only when its upper end is strictly below 0.
Mass is recorded after this table and is not one of the 12.

| Nutrient | Metric | Lo | Hi | Below 0 |
|---|---|---:|---:|---|
| energy | mae | -7.040422 | 1.047149 | no |
| energy | dataset_pmae | -2.786671 | 0.412676 | no |
| energy | plate_pmae | -6.015752 | -0.735401 | yes |
| fat | mae | -0.507258 | -0.037517 | yes |
| fat | dataset_pmae | -4.112708 | -0.294465 | yes |
| fat | plate_pmae | -42.230042 | -14.658709 | yes |
| carb | mae | -0.834301 | -0.324627 | yes |
| carb | dataset_pmae | -4.276023 | -1.662786 | yes |
| carb | plate_pmae | -29.057732 | -10.763595 | yes |
| protein | mae | -1.150231 | -0.595901 | yes |
| protein | dataset_pmae | -6.582163 | -3.412140 | yes |
| protein | plate_pmae | -13.690965 | -6.507668 | yes |

Intervals below zero: 10 / 12.

### Mass intervals (not a pass condition)

| Metric | Lo | Hi | Below 0 |
|---|---:|---:|---|
| mae | -3.205579 | -1.162751 | yes |
| dataset_pmae | -1.595168 | -0.596937 | yes |
| plate_pmae | -1.811601 | 0.739747 | no |

Verdict: `q0_fail_stop_wait_triste`.

The run stops on this verdict. Distillation stays locked.
