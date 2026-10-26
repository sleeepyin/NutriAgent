# Nutrition5K

2791/698 的四通道裁剪在 `cache/deep_crops_cache.npz`。

```
raw/metadata/
raw/yolo_crops/dish_weights_full.csv
cache/deep_crops_cache.npz
cache/nutrient_labels.npz
cache/nutrient_density.npz
cache/aec_v2_pred_cache.npz
cache/nutrient_conformal_cache.npz
cache/cqr_pred_cache.npz
```

`experiments/estimation/build_nutrient_labels.py` 读 `raw/yolo_crops` 和 `raw/metadata`，把标签写进 `cache/`。
