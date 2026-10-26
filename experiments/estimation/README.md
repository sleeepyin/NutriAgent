# 估计层

C1、C2 已写在 `results/estimation/r0/`。C3、C4 还没跑，输出目录是 `results/v4/`。定义见 `docs/论文方案_v4.0.md`，与 v3.0 的贡献一相同。

| 脚本 | 用途 |
|---|---|
| `run_r0.py` | 复现 C1、C2，并写出营养素点估计 `e5_point.json` |
| `aec_s.py` | AEC-S 的搜索和收缩。`run_r0.py` 从这里取函数 |
| `conformal_methods.py` | 固定宽度、按预测值缩放、分段 |
| `train_weight_4ch.py`、`train_density_cqr.py` | 四通道重量和营养素 CQR |
| `train_nutrients.py`、`nutrient_intervals.py`、`build_nutrient_labels.py` | 营养素网络。旧的五候选、七候选区间结果不再写入 `results/` |

三通道重量在 `weights/nvreal/deep_weight_*_3ch.pth`。旧区间的 pickle 在 `backup/v3_excluded/results/estimation/e5/`。
