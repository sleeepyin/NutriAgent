# v3.0 不使用的文件

2026-10-23（北京时间）从工作区移入。当前方案是 `docs/论文方案_v4.0.md`。v3.0 已移到 `backup/论文方案_v3.0.md`。这里的脚本不要接回 `experiments/` 或 `src/mmcfa`。

| 内容 | 原因 |
|---|---|
| `experiments/action/`、`results/action/`、`src/mmcfa/policy/`、`tests/test_a5_oracle.py`、`docs/notes/` | 名称先验、自动参照、域外拒答、分组路由、混合流。方案第 6 节不列入实验 |
| `experiments/vlm/`、`cache/vlm/`、`results/vlm/` | 旧提示词。一次调用同时要了食物名、克数、域外标记和参照物。v4.0 用 `cache/vlm_v4/`，不读这里 |
| `experiments/diagnosis/d6_capacity/`、`results/diagnosis/d6_*`、`weights/dinov2/`、`third_party/dinov2/` | 换骨干和看图直接报重量。不是 v3.0 的测量 |
| `experiments/diagnosis/d3_mf3d_sam3d/`、`weights/sam3d/`、`third_party/sam3d/`、`data/metafood3d_3d_mesh/` | 三维重建。体积真值已经在 D1 的标签表里 |
| `experiments/appendix/`、`results/appendix/`、`experiments/figures/`、`figures/` | 加权共形和旧图。v3.0 的四张图还没画 |
| `results/estimation/e5/`、`results/estimation/previous/`、`results/estimation/r0/e4_per_bin.json` | 五候选、七候选营养素区间，以及不进正文的逐段覆盖 |
| `weights/nutrition5k/` 里不带 `_final` 的重量副本、`density_*_v4.pth`、`nutrient_conformal_calibrator.pkl` | 副本和旧校准器。工作区只留 `_final` 和 CQR |
| `weights/nvreal/joint_*.pth`、`nvonly_resnet18_3ch.pth` | 联合训练和未完成的 NV-only。域偏移用冻住的三通道集成 |
| `data/nutritionverse-real.zip` | 与 `data/nvreal/raw/` 同一套 892 个文件 |
| `docs/paper.md`、`docs/paper_cn.md` | 指向 v2.0 的正文占位 |
| `logs/` | 当时的运行日志 |
| `experiments/run_v2_remaining.*`、`experiments/diagnosis/run_remaining.sh` | 上面这些实验的启动脚本 |

方案第 4 节引用的三份对照仍在原处：`results/diagnosis/d4_ecustfd/summary.json`、`experiments/diagnosis/d1_mf3d_oracle/phase1_results.csv`、`results/diagnosis/d5_nvreal/summary.json`。折外密度缓存和五折权重也留在工作区。
