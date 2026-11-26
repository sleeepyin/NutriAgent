# DiGI-Nutri v3 step 1

One new ConvNeXt-Small RGB baseline, then one residual-Atwater T_multi.
The comparator is that new B1. Mass times density stays the plate estimate.
Energy density is 9 fat + 4 carb + 4 protein plus a learned residual.
The residual penalty is the frozen 0.1 SmoothL1 term. Labels are not forced onto Atwater.
Train metrics are not a gate. Student training was not started.
docs/STATUS.md and README.md were not edited.

B1 parameters: 49849701.
T_multi parameters: 62254886.

## Train metrics

Eval mode, no augmentation, kept official train plates.
A train advantage is not a Q0 pass.

### MAE

| Model | Energy | Fat | Carb | Protein | Mass |
|---|---:|---:|---:|---:|---:|
| B1 | 19.3118 | 1.6105 | 2.3362 | 2.1536 | 18.2261 |
| T_multi | 10.6164 | 0.7443 | 1.4073 | 1.0254 | 9.9920 |

## Official holdout

Split `official_depth_test`, n=507, groups=507.
Eval mode, no augmentation. This file is the single holdout read.
Plate nutrition is predicted mass times predicted density.
The B1 column is the ConvNeXt-Small baseline trained in this round.

### MAE

| Model | Energy | Fat | Carb | Protein | Mass |
|---|---:|---:|---:|---:|---:|
| B1 | 48.8898 | 3.7182 | 4.8823 | 5.3086 | 30.5763 |
| T_multi | 41.6484 | 2.9578 | 4.0021 | 3.9731 | 25.6730 |

### Dataset PMAE

| Model | Energy | Fat | Carb | Protein | Mass |
|---|---:|---:|---:|---:|---:|
| B1 | 19.1374 | 28.9788 | 24.8038 | 30.3719 | 15.4030 |
| T_multi | 16.3029 | 23.0527 | 20.3319 | 22.7310 | 12.9329 |

### Per-plate PMAE

| Model | Energy | Fat | Carb | Protein | Mass |
|---|---:|---:|---:|---:|---:|
| B1 | 29.1023 | 98.7861 | 75.0347 | 56.0794 | 19.6647 |
| T_multi | 19.5712 | 41.3644 | 36.1526 | 40.5315 | 20.0658 |

### Relative gate

Each row is the 95% plate-grouped CI of (T_multi - this round's B1).
An interval is below zero only when its upper end is strictly below 0.
Mass is recorded after this table and is not one of the 12.

| Nutrient | Metric | Lo | Hi | Below 0 |
|---|---|---:|---:|---|
| energy | mae | -10.934191 | -3.299728 | yes |
| energy | dataset_pmae | -4.312229 | -1.293840 | yes |
| energy | plate_pmae | -12.444201 | -6.700239 | yes |
| fat | mae | -1.001192 | -0.516236 | yes |
| fat | dataset_pmae | -7.909969 | -3.949823 | yes |
| fat | plate_pmae | -76.344484 | -41.563790 | yes |
| carb | mae | -1.157635 | -0.619282 | yes |
| carb | dataset_pmae | -5.889927 | -3.102774 | yes |
| carb | plate_pmae | -65.137161 | -19.348231 | yes |
| protein | mae | -1.727879 | -0.970652 | yes |
| protein | dataset_pmae | -9.724646 | -5.569769 | yes |
| protein | plate_pmae | -20.770096 | -10.878865 | yes |

Intervals below zero: 12 / 12.
Relative gate: pass.

### Mass intervals (not a pass condition)

| Metric | Lo | Hi | Below 0 |
|---|---:|---:|---|
| mae | -6.651166 | -3.273486 | yes |
| dataset_pmae | -3.257144 | -1.704293 | yes |
| plate_pmae | -1.921469 | 4.242870 | no |

### Absolute gate

Dataset PMAE only. Per-plate PMAE is not an absolute gate.

T_multi energy dataset PMAE: 16.3029 (limit 13.5).
Mean of energy, mass, fat, carb, and protein: 19.0703 (limit 15.5).
Absolute gate: fail.

Verdict: `q0_fail_stop_wait_triste`.

The run stops on this verdict. Student training stays locked.

