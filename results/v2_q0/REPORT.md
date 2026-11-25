# DiGI-Nutri v2 Q0

Status commit `faad495acb679eb04eb54b710fd9add4f3144f4a`. Backbone `resnet101`. Image 384, epochs 60, AdamW lr 0.0001, weight decay 0.01, cosine eta_min 0.0, batch 32.

Parameter counts: {"b1": 43550533, "t_geo": 45651781, "t_ing": 44877893}.

Train exclusions:
- `dish_1558109714`: rgb_missing, depth_missing
- `dish_1564159636`: depth_zero
- `dish_1557862384`: rgb_missing, depth_missing
- `dish_1558109511`: rgb_missing, depth_missing

Kept train 2754 of 2758. Official test 507 dishes in 507 plate groups. Reshoot rule: exact nutrition within 0.001 and timestamp gap at most 120 seconds, inside the evaluated split.
Cross-split pairs under that rule: 4. The official split was not rewritten.

USDA name collisions with different per-gram vectors are not resolved by picking a row. Ambiguous names: deprecated. A dish uses the unique ingredient id.

## Train MAE

| model | energy | fat | carb | protein | mass |
|---|---:|---:|---:|---:|---:|
| b1 | 11.6004 | 0.5260 | 1.0343 | 0.6469 | 6.8761 |
| t_geo | 76.0853 | 6.8015 | 12.9356 | 10.8645 | 40.4084 |
| t_ing | 6.2960 | 0.3339 | 0.8170 | 0.4493 | 4.7687 |

Train gate pass: False.

Train-gate failures:
- t_geo energy: 76.0853 vs B1 11.6004
- t_geo fat: 6.8015 vs B1 0.5260
- t_geo carb: 12.9356 vs B1 1.0343
- t_geo protein: 10.8645 vs B1 0.6469

Official holdout was not evaluated. Code at training time was `4d0491a4c055d3b09263f19c311967f40d64a16c`.

On 64 kept training dishes, energy MAE was 10.04 for B1 in eval mode and 60.45 for `T_geo` in eval mode. The same `T_geo` checkpoint in train mode, where batch-norm uses the current batch, had energy MAE 28.71 and a prediction mean of 242.7, against 192.1 in eval mode. The shared backbone runs the height map through the same batch-norm layers, so those running statistics are not RGB-only. The loss, split, and preprocessing were left as frozen. This slice is not a holdout result.

Distillation was not started.
