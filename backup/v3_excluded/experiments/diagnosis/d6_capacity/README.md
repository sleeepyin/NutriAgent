# D6 容量对照

`dinov2_ablation.py` 会训练 DINOv2-small 加深度分支的回归头，并保存到 `weights/dinov2/d6_dinov2_small_head.pth`。公开骨干是 `weights/dinov2/dinov2_vits14_pretrain.pth`。

这个回归头目前不存在。2026-09-12 的预测在 `backup/pre_r0_results/diagnosis/d6_dinov2/`，那次测试集是 652 张，不能当作这次结果。

VLM 直接看图估重还没写。它只读 `cache/vlm/` 里的 `weight_guess_g`。
