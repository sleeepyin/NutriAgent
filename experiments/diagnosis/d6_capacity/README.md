# D6 容量对照

`dinov2_ablation.py` 是已经做完的 DINOv2-small 对照。骨干权重放到 `weights/dinov2/dinov2_vits14_pretrain.pth`，代码在 `third_party/dinov2/`。现有预测在 `results/diagnosis/d6_dinov2/`。

VLM 直接看图估重还没写。它只读 `cache/vlm/` 里的 `weight_guess_g`，不另问一遍模型。
