# MM-CFA 可复现包

**从不确定性到行动：共形估计、信息诊断与主动获取**

正文在 `docs/paper_cn.md` 与 `docs/paper.md`。主结果使用诚实协议：AEC-S 覆盖率 90.4%、宽度 74 g；严格 hold-out 四通道 MAPE 22.0%。不再把 MAPE 与 WAPE 都称作 PMAE。开发口径 9.44% 不进入摘要。

本包复现三层结果：域内估计、四数据集诊断、公开数据上的行动层可行性评测。权重、裁剪缓存和原始大数据不入库。

## 1. 布局

目录结构与实验服务器一致——**解压到 `/home` 后，脚本内所有绝对路径直接生效**：

```
mm-cfa-repro/
├── Nutri_0811/                  # 主管线 (Nutrition5K)
│   ├── *.py                     # canonical 脚本 (Appendix C 清单) + 论文图表生成
│   ├── models/                  # 最终权重 + 数据缓存 + 冻结校准器 (~850MB)
│   ├── results/                 # 重量/共形结果 (pkl + 图)
│   ├── results_deep/dinov2_ablation/  # DINOv2-small 消融结果 (任务4)
│   ├── results_nutrient/        # 营养素结果
│   ├── results_weighted/
│   ├── dinov2_hub/              # DINOv2 官方 hub 代码 (免下载)
│   ├── figures/                 # 论文图 1-5 (300dpi)
│   └── paper.md / paper_cn.md   # 论文全文 (双语)
├── nvreal/                      # 跨域实验 (NutritionVerse-Real, §5.8)
│   ├── *.py                     # Phase 0-4 脚本 + 3ch 权重 + 联合训练脚本
│   ├── nvreal_dish_cache.npz    # 880 菜品裁剪缓存 (联合训练免下载)
│   ├── deep_weight_*_{joint,nvonly}.pth  # 多域联合训练权重
│   ├── phase2b/3/4_*.csv|json   # 跨域结果
│   └── results_audit/           # 作废首版结果 + 修复后训练日志 (诚实性记录)
├── metafood3d/                  # MetaFood3D + SAM 3D (§6.2-6.3)
├── ecustfd/                     # ECUSTFD 参照物实验 (§6.4)
├── docs/                        # paper.md, paper_cn.md, 实验日志, 环境 freeze
├── requirements.txt
└── run_all.sh
```

## 2. 环境

```bash
conda create -n nutri python=3.10 -y
conda activate nutri
pip install -r requirements.txt   # CPU 推理可装 cpu 版 torch
```

GPU 训练在单卡 RTX 3090 (24GB) 上验证；CPU 可跑全部推理与评测（NV-Real Phase 2b-4 即 CPU 完成的）。

## 3. 原始数据（不在包内，按需下载）

| 数据集 | 体积 | 获取 | 何时需要 |
|---|---|---|---|
| Nutrition5K 裁剪缓存 | — | **已包含** (`models/deep_crops_cache.npz`, 2608+652×4ch×224²) | 不需要 |
| NutritionVerse-Real | 1.11GB | `python nvreal/nvreal_dl.py`（Kaggle 公开集） | 仅 §5.8 零样本部分 |
| NV-Real 菜品缓存 | — | **已包含** (`nvreal/nvreal_dish_cache.npz`, 880 菜品, 180/45 菜品划分) | 联合训练不需要下载 |
| ECUSTFD | ~390MB | `git clone https://gh-proxy.com/https://github.com/.../ECUSTFD` | 仅 §6.4 |
| MetaFood3D | 69GB | Kaggle `metafood3d` | 仅 §6.2-6.3 |
| SAM 3D 权重 | 13GB | `metafood3d/setup_sam3d.sh`（需 HF token） | 仅 §6.3 |

**关键设计**：主管线（§5 全部数字）只依赖 `deep_crops_cache.npz`——N5K 原始数据、YOLO 分割、裁剪对齐已全部预计算进缓存，无需重新下载 30GB 原图。

## 4. 复现顺序与预期数字

`bash run_all.sh` 一键跑完（约 40min，GPU 训练另计）。分步：

### A. 主管线（零下载，立即可跑）

| 步骤 | 命令 (cwd=Nutri_0811) | 预期结果 |
|---|---|---|
| 1. 生产推理+区间验证 | `python test_100_samples_final.py` | 打印 WAPE（MAE/均值），不是 MAPE |
| 2. 重量共形实验 | `python conformal_deep_experiment_final.py` | 全局权重只在校准集上拟合；分位数为顺序统计量 |
| 3. AEC-S 诚实协议 | `python aec_s_v2_experiment.py` | 诚实 AEC-S 90.4%/74g；AEC-S2m 90.4%/76g/Winkler 171 |
| 4. 营养素共形 (AEC-S2m-N) | `python nutrient_conformal_final.py` | 覆盖 89.7–90.6%，nWinkler 2.42 |
| 5. CQR 基线（可选） | `python nutrient_cqr_experiment.py` | 最好 2.55（被 2.42 支配） |
| 6. 加权共形基线（可选） | `python nutrient_weighted_conformal_experiment.py` | 被 AEC-S2m-5W 支配（表 9） |
| 7. 论文图表 | `python make_paper_figures.py` | `figures/` 图 1–5（数字与正文对齐） |

（可选）从头重训：`python train_deep_v2_final.py`（GPU, ~1h, hold-out 22.0%）、`python train_nutrient_v4_final.py`（密度）、`train_density_cqr.py`（CQR）。

### B. 跨域 §5.8（需先下载 NV-Real）

```bash
python nvreal/nvreal_dl.py                     # 1.11GB → /home/nvreal/nvreal_data/
python nvreal/nvreal_phase0_prep.py            # GT 配对 (多实例求和语义)
python nvreal/train_deep_v2_3ch.py             # GPU ~40min → 域内 3ch 31.4%
python nvreal/nvreal_phase2b_zeroshot.py       # → PMAE 103.7%, ρ=0.16
python nvreal/nvreal_phase3_conformal.py       # → 覆盖 85.9%→32.8%
python nvreal/nvreal_phase4_nutrient.py        # → 营养覆盖 91.1%→44-56%
```

**多域联合训练（实验 22，零下载可复现）**：包内已含 `nvreal_dish_cache.npz`（880 菜品裁剪）与全部权重，跳过上面下载/训练步骤直接评测：

```bash
python nvreal/train_joint_multidomain.py       # 检测已有权重则直接评测
# → NV-only: NV 25.6% / N5K 124.7% (对称坍缩)
# → Joint:   N5K 31.5% (零源域代价) + NV 23.9% (vs 零样本 77.4%, 菜品划分)
```

（如需重建缓存：`python nvreal/nvreal_build_joint_cache.py`，需先下载原始数据。）

**DINOv2-small 现代 backbone 消融（任务4，零下载可复现）**：包内已含 `models/dinov2_vits14_pretrain.pth`（85MB）与 `dinov2_hub/` 官方代码，脚本自动回退到包内路径：

```bash
cd Nutri_0811
python train_dinov2_ablation.py        # GPU ~27min → 标定 PMAE 31.3% vs 4ch 基线 22.0%
# 判读：现代强主干也打不破天花板 → 容量非瓶颈，信息论天花板成立（论文 §6.5 证据 iii）
```

### C. 外部数据集 §6.2–6.4（需下载对应数据集）

```bash
python metafood3d/mf3d_phase1_oracle.py        # 密度离散 21.8% (Gate 1)
python metafood3d/mf3d_phase2_learned_density.py  # 30.3% vs 26.2% (Gate 2, 证伪)
python metafood3d/mf3d_oracle6_sam3d.py        # SAM 3D 零样本 (需 sam3d env)
python ecustfd/ecustfd_reference_experiment.py # 硬币定标 7.9–12.9%
```

## 5. 诚实性记录（audit）

- `nvreal/results_audit/`：**首版 Phase 2 结果作废**（任务错位：整盘模型 × 单食材裁剪 = 694% 无效）+ 3ch 首训爆炸日志（augment clamp bug）。保留供审计。
- NV-Real GT 语义：metadata 同类多实例列为重复列，正确语义是**求和**（251 菜 `sum==total_food_weight` 校验和验证）。`nvreal_phase0_prep.py` 已修复。
- v2 4ch 训练脚本存在 augment clamp bug（ImageNet 归一化数据被 clamp(0,1)）；4ch 模型靠未受损的深度通道存活，22.0% 是"带伤"成绩。3ch 重训脚本 `train_deep_v2_3ch.py` 已修复（monkey-patch）。4ch **未**重训以保持与论文基线可比。
- 完整失败链（21 项）：`docs/历史方案_final.md`。

## 6. 校验和

`sha256sum.txt` 覆盖全部权重与缓存。任何文件不匹配请勿使用。

## 7. 硬件参考

- GPU 训练：RTX 3090 24GB（4ch/3ch 训练各 ~11GB 显存）
- CPU 评测：16 线程，NV-Real 880 图全程推理 ~10min
- 磁盘：包本体 ~1.5GB；全量数据（含 MetaFood3D+SAM3D）另需 ~85GB
