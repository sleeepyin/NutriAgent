# 权重

目录是空的。训练脚本会把新权写到对应子目录；旧校验和在 `backup/sha256sum.txt`。

`weights/nutrition5k/`

```
deep_weight_resnet18_final.pth
deep_weight_efficientnet_final.pth
deep_weight_resnet18.pth
deep_weight_efficientnet.pth
density_resnet18_final.pth
density_efficientnet_final.pth
density_resnet18.pth
density_efficientnet.pth
density_resnet18_v4.pth
density_efficientnet_v4.pth
density_resnet18_cqr.pth
density_efficientnet_cqr.pth
nutrient_conformal_calibrator.pkl
```

`weights/nvreal/`：三通道重量权重 `deep_weight_resnet18_3ch.pth`、`deep_weight_efficientnet_3ch.pth`，以及联合训练和 NV-only 权重。

`weights/metafood3d/`：D2 的 `mf3d_density_resnet18_fold{0-4}.pth`。

`weights/dinov2/dinov2_vits14_pretrain.pth`：D6 的骨干。

`weights/sam3d/checkpoints/`：SAM 3D 权重大约 13 GB。代码在 `third_party/sam3d/`。
