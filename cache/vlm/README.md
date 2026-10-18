# VLM 缓存

D6、A2b、A3、A4、A5 共用一次看图推理。推理脚本还没写。每张留出集图像一份 JSON，文件名用数据集和样本 id。

提示词里不放重量标签。四个留出集是：Nutrition5K 测试半、ECUSTFD 88、MetaFood3D 376（每物体一个视角）、NV-Real 菜品划分测试 177。

```json
{"food_name": "...", "form": "rigid|prepared|liquid|stacked",
 "references": [{"object": "coin", "bbox": [], "size_mm": 25.0, "conf": 0.9}],
 "weight_guess_g": 120, "scene": "...", "out_of_domain": false}
```
