# 估计层 E1–E5

这些脚本是现有实现。R0 要求它们在同一套严格 hold-out 上重跑（菜品不相交，测试半再对半，重复 10 次，分位数用顺序统计量）。统一入口还没写。

| 脚本 | 对应 |
|---|---|
| `train_weight_4ch.py` | E1 的四通道点估计 |
| `conformal_methods.py` | E2 的 Split / Normalized / Mondrian / AEC，以及附录里的 MTC、MLNS |
| `aec_s.py` | E2 / E3 的 AEC-S、AEC-S2m |
| `cqr_baseline.py`、`train_density_cqr.py` | E2 要的 CQR |
| `train_nutrients.py`、`nutrient_intervals.py`、`build_nutrient_labels.py` | E5 |

三通道训练脚本原先不在本仓库。E1 重跑时和四通道脚本放在一起。

重跑前的数值在 `results/estimation/previous/`。营养素的现有结果在 `results/estimation/e5/`。
