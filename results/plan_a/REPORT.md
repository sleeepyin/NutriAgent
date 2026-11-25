# Plan A geometry teacher

This run retrains only the geometry teacher. Separate encoders and a zero-initialized
residual are a BatchNorm repair. They are not a new method, and predicted mass times
predicted density is the same product as the frozen run.

## D_table audit

The count was computed with the border median that includes missing-depth zeros,
before any height-code change. Height is `clip(D_table - D_raw, 0, 150 mm) / 150`,
then one 3×3 closing.

Kept train plates: 2754 of 2758.
`D_table == 0`: 65 / 2754 = 0.023602033406.
All-zero height maps: 65 / 2754.
Those two sets are the same plates.

Every kept plate has at least one zero on the 1-pixel border (2754 / 2754). The median fraction of border zeros is 0.295170. The median table depth on the full kept set is 403.1000 mm.

65 of 2754 kept training plates have D_table exactly 0, and the height map is all zeros on exactly those plates. A blank height map on 65 plates is not a rare leftover, so missing-depth zeros are excluded from the border median before Plan A training. Desktop subtraction, the 0-15 cm clip, and one 3x3 closing stay unchanged. This judgment is frozen with the audit. It is not a percentage cutoff and it is not revised after train or holdout error.

Training uses `apply_d_table_zero_exclusion` = true.

## Train gate

Eval mode, no augmentation, on the same kept training plates. A tie with frozen B1 fails. Mass is recorded and is not a pass condition.

| | energy | fat | carb | protein | mass |
|---|---:|---:|---:|---:|---:|
| frozen B1 | 11.600400 | 0.526000 | 1.034300 | 0.646900 | 6.876100 |
| T_geo Plan A | 7.059961 | 0.296463 | 0.689384 | 0.376770 | 3.501899 |

Train gate pass: True.
Verdict: `train_gate_pass`.

Batch size 32. Measured parameters 56562054.

## Holdout

The train gate passed, so the official 507-plate holdout was opened once. There are 507 plate groups. Distillation was not started.

Holdout MAE:

| | energy | fat | carb | protein | mass |
|---|---:|---:|---:|---:|---:|
| frozen B1 | 48.1623 | 3.6721 | 5.0107 | 4.7746 | 27.9739 |
| T_geo Plan A | 47.5375 | 3.6065 | 4.8233 | 4.7237 | 27.7704 |
| frozen T_ing | 47.4253 | 3.3340 | 4.2720 | 3.8153 | 29.4733 |

Verdict: `q0_fail_stop_wait_triste`.

Ten headline intervals do not lie entirely below 0. A bound above 0 is a stop. Mass is not in this list.

- T_geo energy MAE [-1.745151, 0.555765], dataset PMAE [-0.691526, 0.218502]
- T_geo fat MAE [-0.153833, 0.021462], dataset PMAE [-1.183327, 0.171459]
- T_geo protein MAE [-0.175112, 0.067864], dataset PMAE [-1.001901, 0.395980]
- T_ing energy MAE [-5.452755, 4.532127], dataset PMAE [-2.162930, 1.736250]
- T_ing fat MAE [-0.711062, 0.050513], dataset PMAE [-5.688505, 0.369609]
