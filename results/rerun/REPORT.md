# Same-setting dual-teacher rerun

One training run of B1, T_geo, and T_ing under the frozen recipe.
The comparator is the B1 trained in this run.
Geometric augmentation, head dropout, the split learning rate, and the
explicit height aggregates are the pre-registered regularization and
volume readout. This record does not treat them as a new method.
Distillation was not run.

Parameter counts: `{"b1": 43550533, "t_geo": 56568454, "t_ing": 44877893}`.

## Train metrics

Eval mode, no augmentation, kept official train plates.
These numbers are not a Q0 gate.

| Model | Energy | Fat | Carb | Protein | Mass |
|---|---:|---:|---:|---:|---:|
| b1 | 19.3102 | 1.1887 | 1.9318 | 1.7235 | 13.0775 |
| t_geo | 21.1034 | 1.2642 | 2.0080 | 1.6974 | 16.0026 |
| t_ing | 13.4242 | 0.8853 | 1.5985 | 1.1189 | 11.3559 |

## Official holdout

Split `official_depth_test`, n=507, groups=507.
Eval mode, no augmentation. Plate nutrition is each model's own predicted mass
times its own predicted density.

### MAE

| Model | Energy | Fat | Carb | Protein | Mass |
|---|---:|---:|---:|---:|---:|
| b1 | 47.8534 | 3.4686 | 4.6763 | 4.6013 | 28.2641 |
| t_geo | 45.2309 | 3.3607 | 4.9656 | 4.4307 | 28.9883 |
| t_ing | 43.2897 | 3.1002 | 4.0493 | 3.7102 | 26.6398 |

### Dataset PMAE

| Model | Energy | Fat | Carb | Protein | Mass |
|---|---:|---:|---:|---:|---:|
| b1 | 18.7317 | 27.0334 | 23.7573 | 26.3255 | 14.2382 |
| t_geo | 17.7052 | 26.1930 | 25.2269 | 25.3495 | 14.6030 |
| t_ing | 16.9453 | 24.1624 | 20.5719 | 21.2269 | 13.4200 |

### Per-plate PMAE

| Model | Energy | Fat | Carb | Protein | Mass |
|---|---:|---:|---:|---:|---:|
| b1 | 27.9168 | 81.2066 | 59.8861 | 45.9589 | 20.7816 |
| t_geo | 27.9609 | 72.8555 | 54.7820 | 45.9922 | 22.5341 |
| t_ing | 21.7172 | 39.0605 | 34.3317 | 33.2214 | 21.1932 |

### Headline intervals

Each row is the 95% plate-grouped CI of (teacher - this run's B1).
An interval is below zero only when its upper end is strictly below 0.
Mass intervals are omitted here and are not part of the 24.

| Teacher | Nutrient | Metric | Lo | Hi | Below 0 |
|---|---|---|---:|---:|---|
| t_geo | energy | mae | -5.033238 | -0.167188 | yes |
| t_geo | energy | dataset_pmae | -1.958143 | -0.068366 | yes |
| t_geo | energy | plate_pmae | -1.777581 | 1.804410 | no |
| t_geo | fat | mae | -0.267956 | 0.058448 | no |
| t_geo | fat | dataset_pmae | -2.076363 | 0.452091 | no |
| t_geo | fat | plate_pmae | -17.619163 | -0.507216 | yes |
| t_geo | carb | mae | 0.074238 | 0.513745 | no |
| t_geo | carb | dataset_pmae | 0.378879 | 2.591945 | no |
| t_geo | carb | plate_pmae | -11.763622 | 0.305618 | no |
| t_geo | protein | mae | -0.438725 | 0.103143 | no |
| t_geo | protein | dataset_pmae | -2.502843 | 0.602571 | no |
| t_geo | protein | plate_pmae | -2.954904 | 2.726252 | no |
| t_ing | energy | mae | -9.058217 | 0.552375 | no |
| t_ing | energy | dataset_pmae | -3.586800 | 0.201525 | no |
| t_ing | energy | plate_pmae | -9.204133 | -2.931855 | yes |
| t_ing | fat | mae | -0.690619 | -0.026614 | yes |
| t_ing | fat | dataset_pmae | -5.544373 | -0.190833 | yes |
| t_ing | fat | plate_pmae | -62.429266 | -25.937747 | yes |
| t_ing | carb | mae | -1.003319 | -0.271318 | yes |
| t_ing | carb | dataset_pmae | -5.188049 | -1.361803 | yes |
| t_ing | carb | plate_pmae | -40.639720 | -15.021597 | yes |
| t_ing | protein | mae | -1.316757 | -0.453238 | yes |
| t_ing | protein | dataset_pmae | -7.566322 | -2.639534 | yes |
| t_ing | protein | plate_pmae | -17.770575 | -7.759047 | yes |

Intervals below zero: 13 / 24.

Verdict: `q0_fail_stop_wait_triste`.

The run stops on this verdict. Distillation stays locked.
