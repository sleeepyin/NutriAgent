# NutritionVerse-Real

`raw/` 里是手动标注图像、COCO 标注和 `nutritionverse_dish_metadata3.csv`。压缩包在 `backup/v3_excluded/data/nutritionverse-real.zip`，和 `raw/` 是同一套 892 个文件。

还缺这些派生文件，重做 D5 时放到对应位置：

```
cache/nvreal_dish_cache.npz
derived/phase2b_per_image.csv
derived/phase2b_results.json
derived/phase3_conformal_results.json
derived/phase4_nutrient_results.json
```

跨域实验脚本原先不在这个仓库里。位置留在 `experiments/diagnosis/d5_nvreal_shift/`。继续下载用 `python experiments/download_nvreal.py`。
