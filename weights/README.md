# 权重

v4.0 还要加载的检查点。公开骨干和已经不用的检查点在 `backup/v3_excluded/weights/`。

`weights/nutrition5k/`

```
deep_weight_resnet18_final.pth
deep_weight_efficientnet_final.pth
density_resnet18_final.pth
density_efficientnet_final.pth
density_resnet18_cqr.pth
density_efficientnet_cqr.pth
weight_cqr_resnet18.pth
```

四通道 `deep_weight_*_final.pth` 是 C1 的中心，也是 C3，以及 N5K 密度臂（深度保留）的重量。`density_*_final.pth` 是每克营养素。`density_*_cqr.pth` 只给 C3 的分位数回归对照。`weight_cqr_resnet18.pth` 是 C1 里那一行 CQR。

`weights/nvreal/`：`deep_weight_resnet18_3ch.pth`、`deep_weight_efficientnet_3ch.pth`。这是在 Nutrition5K 上训的三通道集成。N5K 的尺度臂和 NV-Real 外部验证用它，不把深度填进第四通道。联合训练和 NV-only 的检查点已备份。

`weights/metafood3d/`：`mf3d_density_resnet18_fold{0-4}.pth`。MF3D 各重量臂用对应折的折外密度，不重训。
