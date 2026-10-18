# MM-CFA

单张食物图像估重不准，是因为缺信息，不是因为模型不够强。仓库按论文方案 v2.0 分成三层：估计（E1–E5）、诊断（D1–D6）、行动（A1–A5）。方案正文在 `docs/论文方案_v2.0_精简版.md`。

当前稿在 `docs/paper_cn.md` 与 `docs/paper.md`。实验笔记在 `docs/notes/`。

## 目录

```
experiments/estimation/     E1–E5 已有脚本。统一重跑（R0）还没写
experiments/diagnosis/      D1–D4、D6 的 DINOv2 已有脚本
experiments/action/         A1、A2a、A5 的 oracle 上界已有脚本
experiments/appendix/       加权共形（附录）
experiments/figures/        诊断图
src/mmcfa/                  行动层共用的路径、指标、共形分位数和数据表
third_party/dinov2/         DINOv2 代码
data/                       四个数据集的位置。除 NV-Real 原图外都还空着
weights/                    权重位置，目前是空的
cache/vlm/                  一次看图推理的 JSON 缓存，D6 / A2b / A3 / A4 / A5 共用
results/                    已有结果，以及待补实验的空位
figures/                    已有诊断图
backup/                     不再进入正文的旧方案和旧代码
```

路径都从仓库根目录解析，入口是 `src/mmcfa/paths.py`。

## 已经能用的脚本

在仓库根目录执行。

```bash
python experiments/diagnosis/d1_mf3d_oracle/oracle.py
python -m experiments.action.a1_group_routing
python -m experiments.action.a2_name_prior --skip-query
python -m experiments.action.a5_oracle_bound --skip-query
python experiments/figures/make_figures.py
```

A2a 和 A5 的 `--skip-query` 读取 `results/action/a2_name_prior/category_cache.json`。重新问模型时去掉这个参数，本机服务密钥用环境变量 `QWEN_API_KEY_FILE` 指定。

估计层脚本在权重和 `data/nutrition5k/cache/deep_crops_cache.npz` 放回之前跑不起来。D2 需要 MetaFood3D 渲染图，D3 需要 SAM 3D 代码和权重，D4 重跑需要 ECUSTFD 原图和 `density.xls`。D1 和上面三条行动层命令只读仓库里的 CSV。

## 留着但还没写的部分

| 位置 | 以后要放什么 |
|---|---|
| `experiments/estimation/` | R0：E1–E5 在严格 hold-out 上一次重跑。3 通道训练脚本原先不在这个仓库里 |
| `experiments/diagnosis/d5_nvreal_shift/` | NV-Real 零样本、冻结共形、联合训练。脚本原先不在这个仓库里 |
| `experiments/diagnosis/d6_capacity/` | VLM 直接看图估重。DINOv2 脚本已经在 |
| `experiments/action/a3_auto_reference/` | A3，只做 ECUSTFD |
| `experiments/action/a4_ood_reject/` | A4 |
| `experiments/action/a5_mixed_flow/` | A5 端到端。oracle 上界是 `experiments/action/a5_oracle_bound.py` |
| `cache/vlm/` | 四个留出集各问一次的 JSON |
| `experiments/diagnosis/d4_ecustfd_scale/` | 从现有 CSV 补 WAPE 的脚本还没写。对照表已经在 |

## 数据和权重

Nutrition5K 的裁剪缓存、四通道和三通道权重、营养素权重都不在本机。MetaFood3D 渲染图、ECUSTFD 原图、SAM 3D 权重也不在。NV-Real 原图和压缩包在 `data/nvreal/`，缺的是菜品缓存和逐图预测。各目录里的 README 写了期望的文件名。

`results/estimation/previous/` 是重跑前的估计层输出，不能当作 v2.0 的表 1。`results/estimation/e5/` 是营养素区间的现有结果。

## 环境

```bash
conda create -n nutri python=3.10 -y
conda activate nutri
pip install -r requirements.txt
```

SAM 3D 用单独的环境，安装脚本是 `experiments/diagnosis/d3_mf3d_sam3d/setup_sam3d.sh`。代码放到 `third_party/sam3d/`，或用 `SAM3D_CODE` 指向别处。
