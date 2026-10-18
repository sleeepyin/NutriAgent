# Nutrition5K

原始数据和裁剪缓存都还没有。估计层重跑之前放到这里。

```
raw/metadata/dish_metadata_cafe1.csv
raw/metadata/dish_metadata_cafe2.csv
raw/yolo_crops/dish_weights_full.csv
raw/imagery/                         官方 RGB-D，目录随发行包
cache/deep_crops_cache.npz           2608 训练 / 652 测试，4 通道 224²
cache/nutrient_labels.npz
cache/nutrient_density.npz
cache/aec_v2_pred_cache.npz
cache/nutrient_conformal_cache.npz
cache/cqr_pred_cache.npz
cache/features_cache_v3.pkl
```

`experiments/estimation/build_nutrient_labels.py` 读 `raw/yolo_crops` 和 `raw/metadata`，把标签写进 `cache/`。
