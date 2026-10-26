# MM-CFA

单张食物图像的重量和营养素用区间来报。方案在 `docs/论文方案_v4.0.md`，已有数字和未跑实验的占位在 `docs/实验结果汇总_v4.0.md`。

两个贡献：

1. AEC-S，只做共形预测。重量、营养素各一个对比和一个消融，共 4 个。定义与 v3.0 相同。C1、C2 已有数字。
2. 视觉语言模型补尺度和密度。ECUSTFD、MetaFood3D、Nutrition5K 三个有条件的数据集上都做，样本用满。问法和融合系数冻住后，NutritionVerse-Real 做没有参照物、体积和深度时的外部验证。域偏移和数据集合并不做。这 21 对还没跑。

v4.0 不使用的脚本、结果、权重和三维网格在 `backup/v3_excluded/`。v3.0 的方案和汇总在 `backup/论文方案_v3.0.md`、`backup/实验结果汇总_v3.0.md`。再上一版在 `backup/论文方案_v2.0_精简版.md`、`backup/实验结果汇总.md`。

## 目录

```
experiments/estimation/     贡献一。C1、C2 在 results/estimation/r0/
experiments/diagnosis/      尺度和密度的已有对照，以及 W、N 实验要读的标签和预测
src/mmcfa/                  路径、指标、共形分位数、数据表、本机 Qwen
data/                       四个数据集。Nutrition5K 缓存是 2791/698
weights/                    v4.0 还要加载的检查点
cache/vlm_v4/               尺度和密度提示词的看图缓存
results/v4/                 C3、C4 和 W、N、X 的输出目录
backup/v3_excluded/         名称先验、拒答、路由、换骨干、三维重建、旧营养素区间
```

路径都从仓库根目录解析，入口是 `src/mmcfa/paths.py`。

## 还要用的脚本

在仓库根目录执行。

```bash
python experiments/diagnosis/d1_mf3d_oracle/oracle.py
python experiments/diagnosis/d4_ecustfd_scale/reference_scale.py
```

D1 只读标签表。D2 的折外密度已经在 `results/diagnosis/d2_density/density_pred_cache.npy`，MF3D 的密度臂直接用这份缓存和 `weights/metafood3d/`。D4 的汇总在 `results/diagnosis/`。D5 的跨域汇总留在原地，v4.0 不把它列为实验。

## 还没完成的部分

完成标准写在 `docs/论文方案_v4.0.md`。C3、C4 和 W、N、X 写到 `results/v4/`。看图写入 `cache/vlm_v4/`。旧提示词的 JSON 在 `backup/v3_excluded/cache/vlm/`，不读入 v4.0。

## 数据和权重

重量和营养素中心在 `weights/nutrition5k/`。外部验证的三通道集成在 `weights/nvreal/deep_weight_*_3ch.pth`。MetaFood3D 的五折密度在 `weights/metafood3d/`。

C1、C2 来自 `results/estimation/r0/e2_intervals.json` 和 `e3_ablation.json`。同一次运行的点估计在 `e1_point.json`、`e5_point.json`。

MetaFood3D 的 Blender 渲染留在 `data/metafood3d/`，密度臂和尺度臂都看这些图。三维网格在 `backup/v3_excluded/data/metafood3d_3d_mesh/`。NV-Real 的压缩包在 `backup/v3_excluded/data/nutritionverse-real.zip`，解压后的图在 `data/nvreal/raw/`。外部验证使用全部有标签的图像，不在三个数据集的融合系数写死之前打分。

## 环境

```bash
conda create -n nutri python=3.10 -y
conda activate nutri
pip install -r requirements.txt
```
