# D5 NV-Real 域偏移

v4.0 不把域偏移列为实验。这里的汇总留在 `results/diagnosis/d5_nvreal/summary.json`，不写入 `docs/实验结果汇总_v4.0.md` 的 W、N、X 表。

外部验证的无 VLM 重量仍用 `weights/nvreal/deep_weight_*_3ch.pth`。联合训练和 NV-only 的检查点在 `backup/v3_excluded/weights/nvreal/`，v4.0 不加载它们。
