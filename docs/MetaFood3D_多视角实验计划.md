# MetaFood3D 多视角实验计划（2026-08-21）

> 数据集下载中 → `/home/metafood3d/`（RGB-D 视频 3.06GB + Blender 渲染 19GB + 标签 CSV 已就位）
> 本文档是下载完成后的直接执行手册。所有结论遵循项目惯例：诚实协议、失败也记录。

---

## 0. 背景与动机

**现状（已证实的天花板）**：
- 单目 RGB-D 重量预测 R²=0.39 —— 61% 重量方差不可恢复（尺度歧义，无物理参照）
- 18 项优化全部证伪，≥100g 子集 PMAE 18.4% 是单目范式的物理上限
- 营养素区间 nWinkler 2.42 受限于密度模型 PMAE 33-37%

**MetaFood3D 提供的新信号**（Nutrition5K 没有的）：
| 信号 | 说明 | 攻击哪个瓶颈 |
|------|------|--------------|
| **GT Volume (cm³)** | 637 物体的精确体积 | 重量 = 体积 × 密度，绕过尺度歧义 |
| **多视角 RGB-D 视频** | 每物体环绕拍摄 | 三角化恢复绝对尺度 |
| **Blender 渲染 + 相机参数** | 完美深度 + 已知位姿 | 几何监督的合成训练数据 |
| **基准点标记 (fiducial markers)** | 拍摄时可见 | 真实视频的位姿估计 |
| **物理密度标签** | weight/volume 可直接算出 | 全新的密度监督信号 |

**范式转变**：`预测重量（回归）` → `测量体积（几何） × 查密度（先验）`

---

## 1. 数据资产清单

```
/home/metafood3d/
├── metafood3d_labels.csv              # 637 物体标签（已就位，需清洗）
├── metafood3d_rgbd_videos.zip         # 3.06GB 真实 RGB-D 视频帧
├── metafood3d_blender_renders.zip     # 19GB Blender 多视角渲染 RGB+Depth
├── kaggle_parallel_dl.py              # 下载器（断点续传）
├── monitor.py                         # 实时监控面板
└── download.log
```

标签列：`Object_name, Food_Type, FNDDS Food Code, Weight(g), Energy(Kcal), Protein(g), Fat(g), Carbs(g), Volume`

**已知数据质量问题（Phase 0 处理）**：
- Volume 有负值（密度出现 -0.38 g/cm³）→ 坏扫描件，需剔除
- 密度离群到 7.45 g/cm³ → 检查是否标签错位
- 重量范围 1–2091g，中位数 93g（偏轻，与 Nutrition5K 分布不同——跨数据集比较要小心）

---

## 2. 总体思路：两条攻击路线

### 路线 P2（优先）：sim2real 几何预训练 —— 直接攻 Nutrition5K 的 18.4%

```
Blender 渲染 (完美深度+相机参数, ~数千张/物体)
    → 训练单视角"体积预测头"（深度→点云→体积，可微 or 两阶段）
    → 637 真实物体微调
    → 迁移到 Nutrition5K：替换/增强 v2 重量模型
```
- **优点**：部署端不变（仍是单张 RGB-D），直接攻击 R²=0.39
- **核心假设**：合成几何监督能教会模型"从深度感知体积"，而这正是单目模型缺的物理锚点
- 19GB 渲染集就是为这条路下载的

### 路线 P1：推理时多视角融合 —— 体积 × 密度范式验证

```
多帧 RGB-D + 基准点位姿 → TSDF 融合 → watertight mesh → 体积
    → 体积 × 密度(类别查表/FNDDS) → 重量
```
- **优点**：精度上限最高（CVPR'24 挑战赛冠军 volume MAPE 10.97%）
- **代价**：改变产品交互（用户需环绕拍摄）
- **价值**：验证"范式"本身——单目 vs 多视角的信息差到底值多少 PMAE，这是论文 Limitations 的直接回应

---

## 3. 阶段计划

### Phase 0：数据工程（下载完成后立即，~半天）

| 任务 | 说明 |
|------|------|
| 解压验证 | unzip 两个包，核对目录结构：`RGBD_videos/<类别>/<物体>/original/frame_*.jpg`（确认有无 depth 流/flip 流） |
| 标签清洗 | 剔除 Volume≤0 与密度∉[0.05, 3.0] 的行；输出 `metafood3d_labels_clean.csv`；报告剔除数量 |
| 对齐检查 | 视频目录 ↔ 标签行 一一对应；统计类别分布（131 类，平均 ~5 物体/类） |
| 产出脚本 | `mf3d_phase0_validate.py`（沙箱可跑） |

**判定**：可用物体 ≥ 550 则继续；否则记录坏数据比例并调整样本量预期。

### Phase 1：Oracle 上界实验（~半天，CPU，先算清楚值不值得做）

**在写任何 3D 管线之前，用 GT 体积做误差预算分解**（对数空间：log W = log V + log ρ，误差可加）：

| 实验 | 设置 | 回答的问题 |
|------|------|-----------|
| Oracle-1 | GT Volume × GT density | sanity（应≈0%） |
| **Oracle-2** | **GT Volume × LOO 类别均值密度** | **体积完美时，密度查表误差是多少？→ 体积×密度范式的上界** |
| Oracle-3 | GT Volume × 全局均值密度 | 无类别信息的下界 |
| Oracle-4 | v2 重量模型 zero-shot 到 MetaFood3D 帧 | 跨域差距有多大（预期很差，佐证域差距） |
| Oracle-5 | 灵敏度扫描：GT Volume×(1±ε) × LOO 密度，ε∈{5,10,15,20}% | **体积估计需要多准才能打赢单目？** |
| **Oracle-6 (SAM 3D)** | SAM 3D Objects zero-shot：单帧 RGB+mask → Gaussian splat → 体积 | **商业级单图 3D 基础模型的尺度先验，能否从单图恢复食物体积？**（详见 §3.5） |

### Oracle-6 最终结果 — **已判定（2026-08-24，路线全面闭合）**

**实测结果（`mf3d_oracle6_sam3d.py`，130 物体 = 30 大类 × 4 实例分层抽样 + pilot 10，250s/物体，显存峰值 19.6GB 稳定零 OOM）**

| 分析层 | 结果 | 结论 |
|--------|------|------|
| 绝对尺度 | metric MAPE **100.0%**，中位 ratio 5.6e-5 | 输出在规范化潜在空间，连单位一致性都没有 ❌ |
| 全局校准后 | MAPE **87.3%** | 一个统一系数救不回来（类别异质偏置）❌ |
| 每类校准后 | 体积 MAPE 中位 **33.3%** | 有真实实例级信号但不够 ✗ |
| 类间 r（类均值） | **+0.92** | "杏仁小碗大"的类别级知识真实存在 |
| 类内 r（28 类） | 中位 **+0.88**，范围 [-0.98, +0.99] | 刚体类好（Avocado/Broccoli/Salmon >0.95），**堆叠类崩盘**（Almond(bowl) -0.10 / Blueberries -0.41 / Bacon -0.98） |

**核心发现：**
1. **SAM 3D 尺度先验是"三十把各自歪的尺子"**——类间尺度知识来自 3.14M mesh 训练分布的语义记忆，但每类的系统偏置不同，全局校准无效
2. **蓝莓悖论的 3D 投影**：孔隙率同时摧毁质量方程两个乘数——散装蓝莓体积追踪失败（几何侧）+ 密度随堆积状态变 33%（语义侧），与 Gate 1/2 的密度侧发现从两个独立方向闭环
3. **最好情况 ≈ 40% 重量误差**（33.3% 体积 + 21.8% 密度离散），2.2 倍差于要突破的 18.4%

**判定：**
- ❌ SAM 3D 零样本体积路线证伪——绝对尺度不可用，相对尺度类别异质
- ✅ 但作为论文证据价值最高：2025 最强单图 3D 基础模型也无法从单图逃脱信息论上限 → **天花板是 semantic 不是 geometric 的论证完整闭环**
- 已写入 paper.md 6.3 节（SAM 3D Objects: The Strongest Single-Image Baseline Also Fails at Scale）

**产出脚本**：`mf3d_phase1_oracle.py`

### Go/No-Go Gate 1 — **已判定（2026-08-22）**

**实测结果（`mf3d_phase1_oracle.py` + `mf3d_phase1b_hetero.py`，n=625 清洗后）**

| 实验 | 全量 MAPE | ≥100g MAPE | 结论 |
|------|-----------|------------|------|
| Oracle-1 (GT V × GT ρ) | 0.00% | — | sanity ✓ |
| Oracle-2 (GT V × LOO类别均值ρ) | **25.0%** | **23.4%** | **超过 20% gate** |
| Oracle-2b (LOO类别中位ρ) | 23.8% | 21.1% | 同上 |
| Oracle-3 (全局均值ρ) | 42.2% | 38.5% | 类别信息价值 ~17pp |
| Oracle-5 ε=10% | 28.1% | 26.0% | 体积误差代价仅 +2.6pp |
| Oracle-5 ε=20% | 33.4% | 31.2% | 仍以密度误差为主 |

**核心发现（误差预算分解）：**
1. **密度主导**：类别内密度离散度（log 空间）= 21.8% ≈ 整个单目问题难度（22%）。**问题不是几何，是密度**。
2. **体积几乎不重要**：±10% 体积误差只增加 2.6pp——多视角几何投入回报有界。
3. **范式双峰**：61/108 类（358 物体，57%）低离散（<15%，Avocado 1.4%/Banana 1.9%/Apple 2.4%）→ 仅这些类别 Oracle-2 = **10.5% / ≥100g 9.4%**（优于 18.4% 单目上限 2 倍）；15 类（94 物体）高离散（≥30%，Lasagna 79%/Hashbrown 64%）→ 密度不可查表
4. 蓝莓悖论：Blueberry(bowl) 2.2% vs Blueberries(散装) 33% —— **堆积孔隙率**是密度方差的物理来源

**判定**：
- ❌ 全局"体积×密度查表"范式证伪（23.4% > 18.4%，即使体积完美）
- ✅ 刚体食物子域可行（9.4%）
- **Phase 2' 重定义（剩余唯一路线）：学习式密度**——CNN 外观→密度能否打赢类别均值（21.8% floor）？

**原 Gate 规则（存档）**：Oracle-2 <15% 双路线；15-20% 主攻 P2；>20% 证伪。实测 25.0% 落证伪区，heterogeneity 修正为"查表版证伪、学习版待验"。

### Gate 2 — **已判定（2026-08-23，学习式密度证伪）**

**实验**：`mf3d_phase2_learned_density.py`（resnet18, log-密度, 5-fold 物体级 CV, 12,440 帧 Blender 渲染）+ **诚实重评 `mf3d_phase2_reeval.py`**

> ⚠️ **事故记录**：训练脚本最终聚合有 zip 错位 bug（每 fold 实际只评了 7 个物体，基线数字全是小样本噪声）。
> 训练 epoch 日志的帧级 MAPE 正确；用保存的权重重推理修复（`mf3d_phase2_reeval.py`，正确对齐 + 逐物体中位聚合）。

**诚实结果（5-fold 物体级，n=622）**：

| 方法 | 密度 MAPE | 说明 |
|------|-----------|------|
| 全局均值 | 42.3% ± 3.8 | 无信息下界 |
| **CNN 外观→密度** | **30.3% ± 2.5** | 赢全局 12.0pp = **能从外观恢复类别级信息** |
| **类别均值查表** | **26.2% ± 3.5** | CNN 仍输 4.1pp = **类内密度方差外观不可解** |

**Gate 2 判定：❌ 证伪**（CNN 30.3% > 查表 26.2%）

**信息层级结论（论文级发现）**：外观 → 类别身份 → 密度。CNN 赢全局 12pp 证明它学到的是类别等价信息；输查表 4pp 证明**类内密度离散（21.8%，Phase 1）不可从单视角外观预测**——密度方差源于堆积孔隙率/制备方式等外观不可见因素。

**范式全面闭合的最终账**（体积完美假设下）：
- 查表密度：25.0% 重量 MAPE（Phase 1）
- 学习密度：30.3%（Phase 2'）
- 均远超单目上限 18.4% → **体积×密度范式在 MetaFood3D 上彻底证伪，SAM 3D（Oracle-6）成为唯一剩余实验**（且其攻击的体积项已证明非瓶颈）

### 3.5 SAM 3D Objects 基线（2026-08-21 决定加入，Oracle-6）

**是什么**：Meta 2025-11 发布的单图 3D 重建基础模型（[facebookresearch/sam-3d-objects](https://github.com/facebookresearch/sam-3d-objects)）。输入单张 RGB + 分割 mask，输出带几何/纹理的 Gaussian splat（PLY）。在 3.14M 真实尺度 mesh 上训练 → **自带真实世界尺度先验**。

**为什么值得一个实验**：我们的根本瓶颈是尺度歧义（R²=0.39）。SAM 3D 的尺度先验正是数据驱动的"这个物体该多大"知识——如果它能从**单张图**恢复食物体积（volume MAPE ≤15%），等于零样本打平/超过多视角 TSDF 路线，直接冲击 18.4% 单目天花板；如果失败（食物非刚体、OOD），则强化论文 Limitations——"连商业级单图 3D 基础模型都无法处理非刚体食物"。**两个结果都有论文价值。**

**实验设计**（与 P2 正交，互为基线）：
1. 637 物体各选 1 帧标准位 RGB（数据集自带 mask，无需 SAM 3 分割）
2. SAM 3D → splat PLY → 高斯加权体积（或 poisson mesh 后体积）
3. 报告：volume MAPE（vs GT Volume）、按刚体/非刚体类别分层、log-space bias（尺度先验系统性偏大/偏小？）
4. 若 volume MAPE ≤15%：延伸到 Nutrition5K（单帧→体积→×密度→重量，对比 18.4%）

**已知风险**：
| 风险 | 等级 | 对策 |
|------|------|------|
| VRAM 32GB 需求 vs 3090 24GB | 高 | compile=False 推理试跑；OOM 则 CPU offload（376GB RAM，H3 项目有动态卸载经验）；再不行记录为环境限制 |
| 食物 OOD（训练集=产品/家具） | 高 | 这本身就是实验要回答的问题；分层报告刚体 vs 糊状 |
| 权重 gated（HF 需申请+token） | 中 | 用户 HF 账号申请（通常很快批）；huggingface.co 被墙但 hf-mirror.com 可达（已验证 200），token 下载走镜像 |
| 依赖重（pytorch3d/kaolin/hydra patch） | 中 | 独立 conda env `sam3d`（项目惯例：环境隔离） |
| 许可 SAM License（非商业研究） | 无 | 学术论文 OK |

**时间安排**：Blender 包下载完（约 4h）后再下权重（避免抢带宽）；期间用户先完成 HF 申请。代码现在就可通过 gh-proxy 克隆。

### Phase 2A：sim2real 体积预训练（P2 路线，~3-5 天，GPU）

1. **渲染集深度统计**：每个物体的 Blender 深度图 + 相机内参 → 反投影点云 → 凸包/mesh 体积 → 与标签 Volume 对齐校准（`mf3d_renders_volume.py`）
   - 这一步同时验证了"我们的体积测量管线是对的"（GT 可校准）
2. **单视角体积头训练**：输入 RGB-D（Nutrition5K 同款输入格式），输出体积（log 空间）
   - 训练集：Blender 渲染（按类别划分 train/val，防止类别泄漏）
   - 微调集：637 真实物体（类别划分 LOO）
   - 骨干：复用 resnet18 + efficientnet 双骨干（项目惯例），GBML 不需要（单一输出）
3. **迁移 Nutrition5K**：体积头预测 → × 物理密度（哪来？见 Phase 3）→ 重量；或体积头作为 v2 的辅助输出联合微调
4. **判定**：Nutrition5K hold-out PMAE < 18.4%（≥100g）/ < 22.0%（全集）才算突破

### Phase 2B：多视角融合（P1 路线，~3-5 天，CPU 为主）

1. **Clean 条件**：Blender 渲染多视角深度 + 已知位姿 → Open3D TSDF 融合 → watertight mesh → 体积 MAPE（应 <5%）
2. **Real 条件**：真实视频帧 + 基准点标记位姿（OpenCV aruco，数据集自带标记图）→ RGB-D 配准 → TSDF → 体积 MAPE
3. **判定**：Real 条件体积 MAPE ≤ 12%（对标挑战赛冠军 10.97%）→ P1 可行
4. **产出脚本**：`mf3d_phase2b_multiview.py`

### Phase 3：整合回营养管线（P2 成功后，~2 天）

| 整合点 | 现有资产 | 动作 |
|--------|----------|------|
| 重量模型 | `models/deep_weight_*_final.pth` (v2) | 体积头替换/融合，重训 hold-out 协议不变 |
| 物理密度 | 637 个 (图像, 真实体积) 对 | 新任务：单视角物理密度预测（v4 架构复用，2608+637 混合训练） |
| 营养素区间 | `nutrient_conformal_final.py` 的 Propagate 候选 | 重量不确定性收缩 → Propagate 变强 → 重跑 10-trial 协议，看 nWinkler 能否 < 2.42 |
| 生产脚本 | `test_100_samples_final.py` | 体积×密度模式作为可选推理路径 |
| 备份 | `models/backup_0821/` | 动 v2 前先备份（0817/0820 惯例） |

### Phase 4：论文整合

- **如果 P2 成功**：新增 Section "Geometric pretraining breaks the monocular ceiling"——Limitations 第 1 条从"未来工作"变成"已验证的突破路径"；R²=0.39 的叙事升级为"单目极限 + 突破方法"
- **如果只有 P1 成功**：Limitations 补一段"多视角范式验证：volume MAPE X% × density 查表 → 重量 MAPE Y%，证明尺度歧义确实是单目瓶颈的根因"
- **如果都失败**：把 Gate 判定数据写进 Limitations，证伪文化的一致性反而强化论文可信度
- MetaFood3D 引用 [8] 已在 paper.md 参考文献里，无需改

---

## 4. 评估协议（诚实性设计，延续项目铁律）

1. **类别查表必须 LOO**：密度查表/类别均值一律 leave-one-object-out，杜绝"用测试物体自己的类别密度"式泄漏
2. **渲染集按类别划分** train/val，类别泄漏 = 假泛化
3. **真实物体微调与测试分离**：131 类 → ~100 类训练 / ~31 类测试（类级 hold-out，比随机物体切分更严格）
4. **双条件报告**：clean（Blender）与 real（RGB-D 视频）分开报，不混
5. **跨数据集结论必须带域差距声明**：MetaFood3D 中位重量 93g vs Nutrition5K 分布不同，18.4% 与新数字的比较只作范式对比，不宣称同一测试集提升（除非在 Nutrition5K hold-out 上实测）
6. **失败照常记入** `历史方案_final.md`（失败 19、20...）

---

## 5. 环境准备

```bash
# nutri 环境缺的包（沙箱装不了 site-packages，需用户终端执行）：
conda activate nutri
pip install open3d              # TSDF 融合 + mesh 体积
pip install opencv-contrib-python  # aruco 基准点检测（若现有 opencv 无 contrib）

# GPU 训练照旧走用户终端；Phase 0/1/2B 均可沙箱跑
```

---

## 6. 风险与备选

| 风险 | 概率 | 备选 |
|------|------|------|
| 真实视频深度流缺失（只有 RGB 帧） | 中 | 用 Blender 深度训练单目深度估计 → 或降级只做 P2 |
| 基准点位姿估计不稳 | 中 | COLMAP / pixel-perfect-sfm（VolETA 同款）；或只用帧间 RGB-D IC P |
| 637 物体太少，体积头过拟合 | 高（v5-v7 同款教训） | 参数量压到最小（resnet18 单骨干）；Blender 渲染做主力训练数据 |
| sim2real 域差距过大 | 中 | 渲染域随机化（背景/光照）重渲不现实 → 接受差距，报告真实数字 |
| 类别密度方差过大（Oracle-2 > 20%） | ? | 这本身就是重要发现：写进论文，"多视角的收益边界" |

---

## 7. 下载完成后的第一小时清单

```bash
# 1. 检查下载完整性
tail -5 /home/metafood3d/download.log        # 应有 ALL_DOWNLOADS_DONE
# 2. 解压（nohup，19GB 解压要几分钟）
cd /home/metafood3d && nohup unzip -q metafood3d_rgbd_videos.zip -d data/ &
# 3. 跑 Phase 0 验证脚本（等我写好 mf3d_phase0_validate.py）
# 4. Phase 1 Oracle（半天内出 Gate 1 判定）
```

**当前状态**：RGB-D 视频包下载中（监控：`python /home/metafood3d/monitor.py`），Blender 包排队。标签已清洗待用。
