# 执行状态（STATUS）

> 供 GitHub 同步：服务器跑完实验后，请在本文件更新。`notes/` 历史周报已删除，见 [`notes/README.md`](../notes/README.md)。执行助手据此继续改代码或方案。聊天不是 runbook。
>
> **日期**：2026-11-13（Asia/Shanghai）。
>
> **方法真源**：Canvas「**ICDL 官方方案（2026-11-13）：从增量组成变化学习食物营养估计**」（https://cursor.com/canvas/store-745bdf38-f3d7-4313-8d66-08e4b6903e93/47e6af29-51f8-443c-954d-56b0c453b602）。工作名 **ICDL**（Incremental Composition-Difference Learning）。候选英文标题（未证实）：Learning Food Nutrition from Incremental Composition Changes。旧 VDR 页只作存档，不是现行工作项：https://cursor.com/canvas/store-745bdf38-f3d7-4313-8d66-08e4b6903e93/d7be617d-d37e-4fe1-a4b5-9da2c86b27a2 。
>
> **ICDL 尚未被证明。** 会刊仍瞄准 ACM MM 2027。MM 2026 Regular Paper 截稿已过。不编造 2027 截稿日期。
>
> **仓库约束**：不要动 `main`。`main` 停在空的说明性初始提交 `d43cb084eba7693a45c114dfdf06b258611463f4`（短 `d43cb08`）。本文件在工作分支 `icdl` 上。该分支从备份提交 `28bd3c9cae9795c822dc0af68f612ec1fd811e15`（短 `28bd3c9c`，已含合入的 #73）拉出。可行性被证明之前，不要把新方案合进 `main`。不要从空的 `main` 起步，也不要另开一套顶层分支。备份分支 `archive/main-before-new-scheme-2026-11-13` 与 `archive/vdr-old-plan-2026-11-13` 都指向 `28bd3c9c`。
>
> **第一阶段数据可行性审计已完成（2026-11-13），到此停止。** 这不再是开工许可。有效相邻阶段对 **1400**，独立餐盘序列 **692**，覆盖 **2094** / 5,006 个 dish_id。下一步是等 Triste 点头；点头之后，本拉取请求可以 squash 进 `icdl`，不得进 `main`。不要把后文改写成新的代码项，也不要解锁第二阶段。
>
> **数据分工**（外测与备用方向不开工）：
>
> - **Nutrition5k**：审计已经做完。官方发布规模是 5,006 个 dish_id。增量序列已经数完：692 条独立序列，1400 个有效相邻阶段对，覆盖 2094 个 dish_id。不要把这个计数改回未知，也不要另编一套。
> - **OmniFood8K**：留到第三阶段做跨菜系外部泛化验证。不得提前当开发集，不得在外部盲测前用它调参或泄漏标签，也不得在第一阶段下载后拿来训练或选超参。论文写出的规模是 8,036 个真实食物场景；那不是本次审计的序列计数。
> - **ACETADA**：只作辅助外测，且须区分实际摄入量与餐前供应量，不能混合不同评价目标。本阶段不启动该外测。
> - **505 与 Cafe2**：只作历史证据与开发诊断，不再当独立验证集。不要在这两个集上调新方法，不要覆盖其结果目录。
>
> **已锁定**：第二阶段（三个轻量模型，以及跨盘随机配对负对照）；第三阶段（论文价值与外部迁移，含 OmniFood8K 与 ACETADA 外测）；完整 ICDL 网络；Agent、检索增强与强化学习；不完整食材监督。不完整食材监督只是备用方向：依据是 Sterling 等对 Nutrition5k 食材标签遗漏的复核（bioRxiv 2026.07.26.740845，2026-07-29）。仅当 ICDL 因序列不可恢复或差异学习没有稳定收益而停止之后再议。第一阶段已经做完。现在不开工、不解锁第二阶段及以后。停止规则已写在官方方案里，但这并不解锁第三阶段，也不解锁外测。不要训练 DINOv2。不要重训质量估计、动作价值或阈值。检索重排仍锁定。不开 fast Cloud Agent。不用测试标签调参。不做私有采集。推理必须仍是单张 RGB：主推理协议不加入第二视角，也不加入深度。
>
> **可执行的下一步只在本文件的「下一步（立即执行）」。** 后文 Cafe2、#63、#65、#68、#57、#60、已取消的早间跨域诊断，以及仍在叙述双视角实验的段落，都是冻结历史，不是现行门禁。不要按那些段落重跑，不要覆盖结果目录，也不要据此开工第二阶段、第三阶段或新网络。
>
> **#73 已在备份提交 `28bd3c9c` 里。** 下一节只留档双视角实验的停止结论。不要把合入 #73 写成还没做的步骤，也不要再合一次。

## 双视角实验停止结论（历史留档，不是开工）

来源：#73，以及备份提交 `28bd3c9c` 中的 `results/experiment_a_n5k_dual_view/report.md` 与 `comparison.json`。不要覆盖该目录，不要重跑，不要再合入 #73。

评价范围是可按 dish_id 对齐、同时有俯视与侧视 RGB、侧视帧可解码、质量大于 0 克的盘子。主表只有 RGB。可用配对 3,289 盘，其中官方训练清单 2,638 盘，官方 rgb_test 交集 474 盘进入主表。这是已经做完的双视角对齐规模，不是增量扫描的阶段对条数，也不是独立物理餐盘序列条数。

五个设置共用 ImageNet ViT-B/16，在 2,638 对训练清单上新训，不是历史上冻结的只估质量头，因此不能与 Cafe2 的 50.9 g 直接对比。

| 设置 | 质量 MAE（g） | 相对单张俯视（g） | 95% 配对区间 | 是否稳定低于单张俯视 |
| --- | ---: | ---: | --- | --- |
| 单张俯视 | 30.5054 | 基线 | — | 基线 |
| 单张侧视 | 33.8340 | +3.3287 | [1.2231, 5.4225] | 否 |
| 双视角简单特征融合 | 29.8239 | −0.6815 | [−2.0066, 0.6269] | 否；区间跨零 |
| 双视角几何融合 | 31.8187 | +1.3134 | [−0.3355, 2.9750] | 否 |
| 双视角加食物语义约束 | 31.6681 | +1.1627 | [−0.5364, 2.8562] | 否 |

「稳定低于」的事先规则是：该方法与单张俯视的质量 MAE 之差，其配对 bootstrap 10,000 次、种子 0、2.5%–97.5% 分位区间整体落在 0 以下。简单融合只有微弱的点估计优势，区间跨过零；侧视单独更差；几何融合与食物语义约束的点估计都高于单张俯视。简单融合与几何融合都没有稳定胜过单张俯视。**停止。** 不再把多视角融合当作下一阶段的主要算法创新。

这次配对 bootstrap 是对 474 个官方测试 dish_id 做独立有放回重抽样，没有按物理餐盘序列分组。若相关 dish_id 来自同一餐盘，区间可能偏窄。该项不挡第一阶段开工。若序列得以恢复，只用已有预测补做分组 bootstrap，不重训。这是统计补充，不是新实验，也不解锁第二阶段。

## VDR 为何退役（历史判断，不是新实验）

研究假设与公开数据能提供的监督不匹配，不是再缺一个更强的网络。不要在 VDR 上再叠融合模块，也不要推倒已有的质量估计。

- **双视角实验**：简单融合与几何融合都没有稳定胜过单张俯视。数字见上一节，不要改结果文件。
- **Cafe2（n=228，冻结）**：完整方法质量 MAE **50.9483** g，低于直接视觉模型 **73.7456** g 约 **22.80** g。相对只估质量 **51.7107** g 只差 **0.76** g，不能写成优越性。能量 MAE **116.8494** kcal，落后只估质量 **75.3870** kcal，也落后直接视觉模型 **105.6157** kcal。即使用标签密度乘以我方预测克数，能量 MAE 仍是 **98.6201** kcal。质量有用，营养组成这条路径没有站住。后文冻结段的原句保留，不要覆盖 `results/heldout_plan/`、`results/heldout_eval/`、`results/heldout_energy_audit/`、`results/heldout_massonly_energy_path/`。
- **原三个模块**：缺少独立的体积真值、烹饪状态真值，以及外部混合餐上真实可请求的第二视角。完整的联合网络不再搭建。

505 与 Cafe2 只作历史证据与开发诊断，不再当独立验证集。

## 下一步（立即执行）

**第一阶段审计已完成（2026-11-13）。到此停止。** 报告在 [`results/icdl_phase1_sequence_audit/report.md`](../results/icdl_phase1_sequence_audit/report.md)，脚本是 `scripts/icdl_phase1_sequence_audit.py`。同一物理餐盘的相邻阶段可以按官方配料记录恢复：有效相邻阶段对 **1400**，独立序列 **692**，覆盖 **2094** 个 dish_id。官方规模仍是 5,006 个 dish_id。这 1400 对是增量阶段对，不是实验 A 的 3,289 个双视角配对。有效序列里有 **1** 条跨了官方 RGB 训练和测试（`cafe1-0168`，`dish_1561662926` → `dish_1561662948`，新增 green beans 17 克），另有 **92** 组重复拍摄跨划分。整图像素差还不能把新增食物和重复拍摄分开。实验 A 的 474 盘按序列分组重抽样之后，区间变宽，原先“没有一种设置稳定低于单张俯视”的判断不变。预测文件没有改。

**下一步：等 Triste。** 点头之后，这个拉取请求可以 squash 合并进 `icdl`，不得合并进 `main`。不要把下面六步改写成新的代码项。

**第二阶段、第三阶段、完整 ICDL 网络、Agent、强化学习、OmniFood8K、ACETADA 和不完整食材监督仍然锁定。** 不要因为这份报告自动开工，不要改 `main`，不要覆盖 `results/experiment_a_n5k_dual_view/`。下面六步是这次审计的工作说明，已经做完，不是新的开工许可。

**当时的开工范围是第一阶段数据可行性审计。** 本文件在分支 `icdl` 上只解锁过这一阶段。Nutrition5k 是这一阶段的唯一工作对象。

要回答的问题：能否可靠恢复 Nutrition5k 里属于同一物理餐盘的增量扫描序列，并核对组成变化、标签变化（质量、能量、宏量营养素）、相邻俯视 RGB 是否有可学习的视觉差异、有多少有效阶段对和独立餐盘序列，以及同一物理餐盘的每个阶段是否都留在同一个官方划分里。不要写新网络，不要训练 DINOv2，不要把第二阶段写成已开工。

审计已经做完，序列已经恢复，计数见本节开头：有效相邻阶段对 1400，独立序列 692，覆盖 2094 / 5,006 个 dish_id。不要把这个计数改回未知。当时的停止规则是：若序列不能可靠恢复，停止并建议放弃 ICDL，等待判断，不要另起一个网络，也不要改去做不完整食材监督。不要假定各阶段严格单调增加。不要只凭相近时间戳推断序列。必须依据原始采集记录与标签核验。报告必须可复核（脚本、计数规则、失败样例）。

**附带的分组 bootstrap 已经补做完，不解锁第二阶段。** 双视角实验原先对 474 个官方 rgb_test 配对 dish_id 做独立重抽样。审计用已有预测按物理餐盘序列重抽样，没有重训。预测在 `results/experiment_a_n5k_dual_view/predictions/`，不要覆盖。这是统计补充，不是新实验。

**代码范围**：只写审计脚本和可复核报告。可以复用已有 Nutrition5k 加载器。GitHub 推送可以攒批。不要把 `main` 当成已证明方法的去处，也不要开准备合进 `main` 的拉取请求。

**本阶段不要做**：不要下载 OmniFood8K 或 ACETADA 来训练、调参或外测。OmniFood8K 留到第三阶段，且不得提前当开发集，不得在外部盲测前调参或泄漏标签。ACETADA 只作将来的辅助外测，且须区分实际摄入量与餐前供应量。不要在 505 或 Cafe2 上调新方法。不要开工不完整食材监督。

**营养论文代码编写 — 严格清单（与官方方案一致；未通过的阶段不得自行解锁下一阶段）**

1. 从官方归档与元数据审计序列能否恢复，不只凭时间戳邻近。
2. 核对相邻阶段实际新增了哪些食物项，以及质量、能量、宏量差是否与这些组成变化相符。
3. 检查相邻俯视图像是否有可学习的视觉差异。
4. 统计有效阶段对与独立餐盘序列；核验同一条链是否始终在同一个官方划分。
5. 若序列恢复成功，在双视角实验已有预测上补做分组 bootstrap（不重训）。
6. 写完可复核报告后停止。不要开始 ICDL 网络，也不要开始第二阶段。

**锁定，不得因本清单顺手打开**：第二阶段、第三阶段、完整 ICDL 网络、Agent、强化学习、OmniFood8K 外测、ACETADA 外测、不完整食材监督。停止规则已写入官方方案，不等于解锁第三阶段。

## 历史冻结（不是现行门禁）

自本节以下至文末，是已落盘的历史记录，包括 Cafe2、#63、#65、#68、#57、#60、已取消的早间跨域诊断，以及 2026-11-12 晚的双视角实验开工段。数字保留。不要重跑，不要覆盖结果目录，不要删除 `results/` 下已有目录。

其中凡是把双视角实验写成「当时的下一步」或「曾经的开工许可」的句子，都是 2026-11-12 的历史句，2026-11-13 起失效。#73 已经在备份提交 `28bd3c9c` 里，不要再合一次。上面的「下一步（立即执行）」已经写成审计完成并等待 Triste，不是新的开工许可。


**已完成（降为 Ours-old 对照，不是 Ours-new 方法证明）**：P0=A 弃权校准（#31）；P1=B on-policy rgb_test 主表（#34）；#24 闭环 BC、#26 FixRate、#28 offline MAE/AURC。数字见「助手进度」；**禁止**声称 on-policy Full MAE 赢 B5；**禁止**混 offline **n=356** 与 on-policy **n=488**。

**营养论文代码编写 — 历史清单（2026-11-12 及更早；不是现行清单。现行六步只在文首「下一步」）**

**归因与卫生（2026-11-06 外部评审；先读这段）**

- **#53 冻结为 Ours-new 开发检查点。** 配对 **505** 盘 rgb_test 标为**开发 / 诊断集**：误差审计 → 码表轻修 → 覆盖约束，已经用测试集误差分析决定下一轮修改（标签没有直接输入选码、克重或阈值）。**禁止**再把这 505 盘当独立最终泛化主表，也**禁止**在同一 505 上继续调方法冒充最终主表。
- **#53 数字保留**，不删、不覆盖冻结结果。开发集上超过 Direct-VLM **不等于**方法创新已经成立，也**不等于** #53 证明了方法。
- **#51 / #53 增益 ≠ Q 学习改进。** 两步都没有重训 Mass、没有改 Q、没有重拟合 τ。#51 是确定性食物码 / 密度轻修。#53 是 **Coverage-Constrained Active Specialist Selection**：未写完组件时 **A_legal** 去掉 **STOP / ABSTAIN**，再在合法集上按 **Q_φ / Cost** 选动作。覆盖只保证初始清单内的组件被处理，**不能**证明 VLM 没有漏掉整道食物。
- **方法中心（2026-11-12 晚）**：候选定位是面向物理约束营养估计的主动多模态证据获取，尚未证实。VLM 食物理解是语义分支的基础。Mass Estimator 是预训练的几何-质量专家，冻结结果作历史证据。FNDDS 是营养密度知识来源。Coverage Constraint 是可选的输出完整性机制，保留为辅助。Nutrient Verifier 保留为辅助。Q_φ 不再是主方法，降为对照 / 消融，停止继续训练。同头密度换码是对照，停止继续改 food-id。分层少样本质量校准是普通基线，不再是创新点二。两路径能量融合 / Energy Router 仅作对照，不预定为贡献。蛋白质 / 碳水 / 脂肪作辅助。当时的主声称留在后文冻结段，不是已经成立的 ICDL 贡献。停止声称能量与质量双重提升。

**food-id / 密度轻修 — 已完成（#51，squash `6c8fda02`，2026-11-06）**

- **范围**：finished 多组件上，在**已有检索窗**内做**同头 FNDDS code swap**（food-table 能量 **≥15%** 更靠近 mass-estimator calorie head）；**克重与动作保持**。**无** Mass 重训、**无** Q 改、**无** P1。**#51 未改本文件**。
- paired **n=505**：能量 MAE **98.5521 → 92.6963**（质量不变 **58.9952**）；vs Direct-VLM **ΔE +1.42 → −4.43**。
- 多组件全写 **n=50**：food-table 能量 **190.79 → 131.65**；vs Direct **+10.01 → −49.13**；换码 **37/50**（**30↓ / 7↑**），**68** codes；calorie head 仍 **77.75**（目标；记分 commit 仍是 food-table）。
- Mass RE_PORTION **827** 成功、**0** hollow。全文见「助手进度」。
- **归因**：确定性码表 / 密度，**不是** Q 变强。换码 **37/50** 的语义审计已由 **②（#60，squash `c52ffaa2`）** 落盘。自动 Exact / Near / Wrong 只是建议；人审包仍开放。**kcal 变好 ≠ 身份变对。** **不**把本条能量下降写成身份成功。

**覆盖约束 — 已完成（#53，squash `2a406df4`，2026-11-06；Ours-new 开发检查点）**

- **当时顺序（已做完；food-id / 覆盖 / P1 那套编号退役）**：先 food-id / 密度（#51），再覆盖未写组件 + 早停（#53），P1 暂缓。下文 **①–⑦** 是另一套门禁。**P1 仍锁定。禁止自动开 P1。**
- **方法**：冻结 Q + fence 之后，未写组件没有 `estimate_mass` 写则 **拒绝 STOP/ABSTAIN**（defer 到 rule utility）；已回到检索 top code 的 swap 改为 **portion**，不开第二轮 top-code swap。**#51 同头 swap 保留**。**无** Mass / Q / τ 重训。**#53 未改本文件**。
- **paired n=505**（相对 **#51**）：能量 MAE **92.6963 → 64.2792**；质量 MAE **58.9952 → 32.9908**；tool mean **4.6970 → 6.0257**（max **17**，**505/505** 在 K=32 内）。
- **vs Direct-VLM**：ΔE **−4.43 → −32.85**；ΔM **−17.35 → −43.36**。
- **覆盖**：全部组件 mass 已写 **199/505 → 503/505**；STOP-unwritten **78 → 0**；Mass 成功 **827 → 1479**；hollow **0**。
- **残留 2 盘**（`dish_1565020895`、`dish_1565020926`）：`honeycomb` 的 FNDDS 检索失败 → 卡在 **RETRIEVE**；**不编造克重**；两盘都 **未 STOP**。全文见「助手进度」。
- **归因**：上列开发集变化来自覆盖约束（**A_legal** 在未写完时去掉 STOP/ABSTAIN），**不是** Q 学得更好，**不是**最终独立泛化主表。

**① 反事实 Q 动作价值 + weak-step dish_id 分组泄漏审计 — 已完成（#57，squash `0aa30537`，2026-11-06）**

- **产物**：`results/qphi_counterfactual/`（`report.md`、`report.txt`、`summary.json`、`decisions.jsonl`、`leakage_audit.md`、`leakage_audit.json`）。**#57 未改本文件。** 冻结 n5k / weak_steps 以及 #46 / #48 / #51 / #53 **未改**。**无** Mass / Q / τ 重训。**无 RL。**
- **泄漏**：fit / hold-out **按 dish_id 分组**（`int(sha256(dish_id).hexdigest()[:8], 16) % 10 == 0` → hold-out；不是把同一道菜的决策步拆进两侧）。fit **3679** 盘 / **25864** 步；hold-out **380** 盘 / **2771** 步。两份 dish 交集 **0**。结论 `no_dish_leakage`。审计之后**没有**重训。
- **头条**（n=**1914**，coverage-choice 且该盘有冻结 RGB mass forward；人群 **rgb_train hold-out**；与 #53 配对 **505** 交集 **0**）：
  - **Q_φ** Best-ID **0.6165**，ΔE improve **24.14**，E regret **49.96**
  - Rule Utility **0.6102** / **23.03** / **50.91**
  - Fixed Pipeline **0.5914** / **20.62** / **52.99**
  - Random **0.4338** / **4.04** / **78.15**
  - Oracle：regret **0**，improve **82.19**
- Q 略领先规则（能量 regret 约 **0.96** kcal）；两者都大约 **50** kcal 落后 oracle。
- **241/1914** 的能量最优合法动作是 **RE_DECOMPOSE** 空操作（冻结 Q **不能**发出该动作）。去掉该 no-op 再记分：Q regret **~17.33**，规则 **~18.28**。**不**把 no-op 当成改 Q 的理由。
- **价值校准**（n=**3475**）：MAE(Q, ΔL) **0.0226**，Pearson **0.747**。按动作的 Pearson：RE_PORTION **0.805**，CORRECT_CODE **0.016**。
- **归因**：这一步评测上 **Q ≈ 规则 ≪ oracle**。它**加固**「**#51 / #53 增益 ≠ Q 学习改进**」。本表不是 #53 开发集 MAE，也不是全程 rollout。**不**证明方法。**不**解锁重训。#41 的 hold-out 回归 MAE **0.0368** / Pearson **0.7799** 只说明有预测力，**不**改写上面的动作价值结论。

**② #51 food-id 语义审计 — 已完成（#60，squash `c52ffaa2`，2026-11-07）**

- **产物**：`results/food_id_semantic_audit/`（`report.md`、`report.txt`、`summary.json`、`human_review_components.csv`、`human_review_dishes.csv`、`human_review.md`、`README.md`）。脚本 `scripts/food_id_semantic_audit.py`。**#60 未改本文件。** **只读** #51 `results/food_id_density/`（squash `6c8fda02`）。冻结 #53 / #57 **未改**。**无** Mass / Q / τ 重训。**无 RL。** **不要重跑，不要改该目录。**
- **范围**：**68** 个同头换码，覆盖 **37/50** 多组件盘。
- **身份（自动标签，仅建议）**：换码后 Exact **0** / Near **50**（**73.53%**）/ Wrong **18**（**26.47%**）。换码前同一 cafe 行：Exact **27** / Near **23** / Wrong **18**。Exact→Near **27**。自动身份变差 **28**、变好 **1**、不变 **39**。换码后自动 Exact 为 **0**。
- **宏量（food-table × 存克 vs cafe 盘总量）**：finished 多组件 **n=50**：蛋白质 **10.57→10.07**，碳水 **13.00→16.17**，脂肪 **17.56→11.97**，能量 **190.79→131.65**（能量与冻结 #51 一致）。换码 **37** 盘：碳水 MAE **+4.28 g**，能量 MAE **−79.92 kcal**。
- **kcal 变好 ≠ 身份变对**：能量绝对误差下降的 **30** 盘换码盘里，全自动 Exact **0**；至少一码 Wrong **12**；仅 Near **18**。这 **30** 盘上的 **50** 个换码全部是 Near 或 Wrong。**不**把 #51 的 kcal 下降写成身份成功。
- **人审**：全部 **68** 个组件行 `needs_human_review=yes`（并 `auto_label_is_suggestion_only=yes`）。自动标签**不**关闭审计。文件：`human_review_components.csv`、`human_review_dishes.csv`、`human_review.md`。人审包仍开放。这**不是**代码项。人审**不**改写已完成的 ③ 协议，也**不**改写已完成的 Cafe2 能量审计（#65，squash `ecf859f5`），也**不**改写已完成的 Mass-only 能量路径诊断（#68，squash `f578e8f4`），也**不**把 #51 的 kcal 下降写成身份成功。

**③ 外部 held-out — 已完成（#63，squash `27d1dbfa`，2026-11-09 Asia/Shanghai）**

- **产物**：划分 `results/heldout_plan/`；主表与轨迹 `results/heldout_eval/`。**#63 未改本文件。** **不要重跑，不要覆盖这两个目录。** **无** Mass / Q / τ 重训。**无 RL。** **无 P1。**
- **划分**：Cafe2 held-out，**n=228**，与配对 **505** 的 dish_id **不相交**，也与 fit（**3679** 官方 rgb_train 且 `pool_of_dish==fit`）、rgb_train、rgb_test、cafe1 **不相交**。
- **主表（三法皆有限，n=228）**。开发 505 的能量 MAE **64.2792**、质量 MAE **32.9908** **不是**本表的行。
  - **Coverage-Constrained + density/same-head**：能量 MAE **116.8494** kcal，质量 MAE **50.9483** g，覆盖 **0.9737**（222/228），步数均值 **6.2193**，工具调用均值 **5.2456**
  - **Direct-VLM**：能量 **105.6157**，质量 **73.7456**
  - **Mass-only**（冻结 Mass）：能量 **75.3870**，质量 **51.7107**
- 能量 MAE 比 Direct-VLM **高 11.23** kcal，比 Mass-only **高 41.46** kcal。质量 MAE 比 Direct-VLM **低 22.80** g，比 Mass-only **低 0.76** g。
- STOP 且仍有未写组件：**0**。**6** 盘停在 **RETRIEVE**，组件仍未写完；**没有编造份量**。同头密度换码 **106** 盘。空壳 RE_PORTION **0**。工具调用：中位 **4**，最大 **14**，**228/228** 在 K=32 内。Mass 写：**575**。
- Cafe2 盘级能量列每行都是 **0**，所以能量目标是配料卡路里之和。质量是盘级克重总量。
- **诚实限制（短，不是重跑请求）**：这次 Direct-VLM 行记下 `placement_max_memory=8GiB`（共享 GPU）。同一 bf16 checkpoint 与 prompt，**不是**历史上满 GPU 的 `cuda:0` 放置。**不要重跑。不要据此解锁重训。**
- **#53 不证明方法。** #51 是密度 / kcal 启发式（自动 Exact=**0**）。另一次一步审计上 **Q ≈ 规则 ≪ oracle**。
- **主声称：质量稳赢可外推；能量开发集赢、外测未稳。** **不**声称 held-out 上能量与质量双赢。**不**把开发 505 的双赢数字写进本表。

**Cafe2 能量审计 — 已完成（#65，squash `ecf859f5`，2026-11-09 Asia/Shanghai）**

- **产物**：`results/heldout_energy_audit/`。脚本 `scripts/heldout_energy_audit.py`。**#65 未改本文件。** 只读冻结 Cafe2 **n=228**。**没有**重跑 #63 闭环、Direct-VLM 或 Mass-only。**没有**覆盖 `results/heldout_plan/`、`results/heldout_eval/`，也**没有**覆盖其他冻结结果目录。**无** Mass / Q / τ 重训。**无 RL。** **不要重跑，不要改该目录。**
- **#63 主表不变**：Coverage-Constrained 能量 MAE **116.8494** kcal，质量 MAE **50.9483** g；Direct-VLM **105.6157** / **73.7456**；Mass-only **75.3870** / **51.7107**。能量相对 Direct **+11.23**，相对 Mass-only **+41.46**。质量相对 Direct **−22.80**，相对 Mass-only **−0.76**。覆盖 **0.9737**（222/228）。开发 505 的能量 MAE **64.2792**、质量 MAE **32.9908** **仍不是**该表的行。
- **222**/228 盘上，已提交克重与 Mass-only 克重相差小于 0.001 g。整段 **−0.76** g 质量优势就是停在 RETRIEVE 的 **6** 盘（这 6 盘对 +41.46 均值的能量贡献是 **−0.52** kcal）。
- **+41.46** kcal 的差距是食物表密度打在已经与 Mass-only 对齐的克重上。Mass-only 用冻结质量估计器的 calorie head 给同一克重记分。
- 码保持固定，相对 100 g 网格的 mass 重写：能量 MAE **195.6564 → 125.2060** kcal，质量 MAE **116.9123 → 50.9483** g。汇合下降来自多组件（**216.37 → 115.21**）。在 **58** 盘单组件上，同一重写把能量 MAE 从 **134.93** 抬到 **154.51**，质量仍近 51 g。这 58 盘贡献 +41.46 均值里的 **+29.62** kcal。
- Oracle 标签密度 × 我方克数：能量 MAE **98.6201** kcal，低于食物表 commit，也低于 Direct-VLM，仍高于 Mass-only **75.3870**。不加权 kcal/g MAE：commit **0.631**，calorie head **0.753**；head 的抵消更大（**77.8** vs **58.2** kcal）。
- #51 密度换码开 vs 关：在 **106** 盘上触发。全表能量 MAE 关 **125.2060** / 开 **116.8494**（**−8.3566** kcal）。质量不变。**67** 盘更靠近 cafe 标签，**39** 盘更远。未换码的 **122** 盘贡献 +41.46 里的 **+38.30** kcal。#51 仍是密度 / kcal 启发式（开发集自动 Exact 为 **0**）。
- Bootstrap **10,000** 次，种子 **0**：相对 Mass-only 的能量差 **[+25.35, +58.28]** kcal（每次抽样均为正）。相对 Direct-VLM 的质量差 **[−29.80, −15.66]** g（每次抽样均为负）。相对 Direct 的能量差 **[−6.22, +29.16]** 含两种符号。相对 Mass-only 的质量差 **[−2.33, +0.10]** 含两种符号。
- **主声称仍是：质量稳赢可外推；能量开发集赢、外测未稳。** **不**声称能量与质量双赢。**不**从这些标签开工改方法。
- **已执行的协议（不是新开工）**：只读 `results/heldout_plan/` 与 `results/heldout_eval/`，新产物在 `results/heldout_energy_audit/`。只在冻结的 Cafe2 n=228 名单上做诊断。不做方法选择，不早停，不调阈值 / τ，不换方法，不用 GT 改码或份量。Diagnostic only on the frozen Cafe2 n=228 list. No method selection, no early stopping, no threshold/τ tuning, no swapping the method, no GT-driven code or portion edits. 与 #48 同一组切分：固定 food-id 时，mass 重写是否同时降低 energy（hold identity/codes fixed, then ask whether the mass rewrite also drops energy）；oracle 标签密度 + 我方克数（oracle label density with our grams）；另拆 #51 density swap 开/关，重点看换过码的 106 盘。基线仍是 Direct-VLM 与 Mass-only。头条问题已答：相对 Mass-only 的能量损失 +41.46 kcal（质量几乎打平，−0.76 g）来自食物表密度，不是来自克重分叉。

**Mass-only 能量路径便宜诊断 — 已完成（#68，squash `f578e8f4`，2026-11-10 Asia/Shanghai）**

- **产物**：`results/heldout_massonly_energy_path/`（`report.md`、`report.txt`、`summary.json`、`dishes.jsonl`、`paths.csv`、`slices.csv`、两张图 `fig_density_vs_mae.svg` 与 `fig_gap_split.svg`）。脚本 `scripts/heldout_massonly_energy_path.py`。**#68 未改本文件。** 只读冻结 Cafe2 held-out，**n=228**。**没有**重跑 #63 闭环、Direct-VLM 或 Mass-only，也**没有**重跑 #65。**没有**覆盖 `results/heldout_plan/`、`results/heldout_eval/`、`results/heldout_energy_audit/`，也**没有**覆盖其他冻结结果目录。**无** Mass / Q / τ 重训。**无 RL。** **不要重跑，不要改该目录。**
- **记分路径**：Mass-only 的能量是冻结质量估计器的盘级 `energy_kcal` 头，来自 `vit_b_16` checkpoint，目标 `mass_g`、`energy_kcal`、`protein_g`、`carb_g`、`fat_g`。记分器存的是 `estimate_mass(...).nutrients.energy_kcal`。它不读 food-id，也不把 FNDDS kcal/100 g 乘以克重。
- 在 **222 / 228** 盘上，质量头与已提交克重相差在 0.001 g 以内（测得最大 **0.0002** g）。另外 **6** 盘停在 RETRIEVE；Mass-only 仍给整盘头记分。
- **98.6201** 对 **75.3870**（结论：更大的有符号抵消，不是更准的密度或食物码；Mass-only 不用 food-id）：Oracle 标签密度 × 我方克数的能量 MAE **98.6201**。Mass-only 能量头 **75.3870**。差距 **23.2332** kcal。不加权 kcal/g MAE：标签密度 **0**，食物表 commit **0.631**，calorie head **0.753**。#65 通道抵消（kcal）：oracle **0**，食物表 **58.2**，head **77.8**。保持真密度、换成 Mass-only 克重，能量 MAE 从 **98.6201** 升到 **100.3152**（**+1.6951**）。在这些克重上，把真密度换成能量头，能量 MAE 从 **100.3152** 降到 **75.3870**（**−24.9283**）。净差是 **23.2332** kcal。在头自己的克重上，把头的密度换成标签密度，会把能量 MAE 从 **75.3870** 抬到 **100.3152**。食物表在 kcal/g 上更近的有 **139** 盘；头更近的有 **89** 盘。没有一盘的头密度误差低于 **0.0001** kcal/g。**58** 盘单组件贡献 **+23.23** 均值差距里的 **+19.85**。这一片上头的 kcal/g MAE 是 **1.023**，食物表是 **0.811**；能量 MAE 是 **38.06**，对照食物表 **154.51** / oracle **116.08**。**6** 盘 RETRIEVE 贡献 **−1.56**。按盘 bootstrap **10,000** 次，种子 **0**，把 **98.6201 − 75.3870** 放在 **[+7.92, +39.06]** kcal。**99.83%** 的抽样保持正均值。这个区间没有用来选方法。
- **冻结数字保持原样。** #63 / #65 已写下的数不改，包括 **116.8494** / **50.9483**，**105.6157** / **73.7456**，**75.3870** / **51.7107**，**+11.23**，**+41.46**，**−22.80**，**−0.76**，**0.9737**（222/228），**98.6201**，**0.631** vs **0.753**，**77.8** vs **58.2**，开发 505 的 **64.2792** / **32.9908**（**仍不是** held-out 表的行），码固定重写 **195.6564 → 125.2060**，**58** 盘单组件 **134.93 → 154.51** / **+29.62**，换码 **−8.3566**，未换码 **122** 盘 **+38.30**，以及 #65 的 bootstrap 区间 **[+25.35, +58.28]**、**[−29.80, −15.66]**、**[−6.22, +29.16]**、**[−2.33, +0.10]**。相对 Mass-only 的 **−0.76** g 仍只是停在 RETRIEVE 的 **6** 盘（#65 / #66 已记下）。**不**把它写成质量赢。**不**把只修 food-id 写成能量双赢。
- **主声称仍是：质量稳赢可外推；能量开发集赢、外测未稳。** **不**声称能量与质量双赢。**不**从这些标签开工改方法。
- **已执行的协议（不是新开工）**：
  1. 已只做这一项便宜诊断。没有改方法，没有调超参，没有在 Cafe2 上做早停或模型选择。No method changes, no hyperparameter tuning, and no early stopping or model selection on Cafe2.
  2. 已写明 Mass-only 的能量是怎么记分的：冻结质量估计器的盘级 `energy_kcal` 头，来自 `vit_b_16` checkpoint，目标 `mass_g`、`energy_kcal`、`protein_g`、`carb_g`、`fat_g`。记分器存 `estimate_mass(...).nutrients.energy_kcal`。它不读 food-id，也不把 FNDDS kcal/100 g 乘以克重。这和食物表 commit（FNDDS kcal/100 g × 克重）、和 oracle 标签密度 × 我方克数不同。The diagnostic documented how Mass-only energy is scored: which food-id / density / calorie head it uses, and how that differs from our food-table commit and from oracle label density × our grams.
  3. 已回答：oracle × 我方克数 **98.6201** 相对 Mass-only **75.3870**，是更大的有符号抵消，不是更准的密度或食物码（Mass-only 不用 food-id）。#65 已冻结：不加权 kcal/g MAE calorie head **0.753** vs 食物表 **0.631**；抵消 **77.8** vs **58.2** kcal。It answered whether oracle×our-grams 98.6201 vs Mass-only 75.3870 is mainly greater signed cancellation (already frozen in #65: unweighted kcal/g MAE 0.753 head vs 0.631 food-table; cancellation 77.8 vs 58.2) or whether Mass-only density/codes are actually more accurate: greater signed cancellation, not more accurate density or food codes.
  4. 落盘之后没有自动开下一道门。P1 / Mass·Q·τ 重训 / ④–⑦ 仍锁定。代码再次空闲。冻结后写作（「Coverage+Mass，能量外测未稳」）对比只在 505 上改 food-id、再重测 Cafe2，仍要等以后的调研阅读 + Triste 点头 + 再一次 STATUS 改写。After this diagnostic is filed, do not auto-open the next gate. Freeze-and-write vs 505-only food-id waits for a later research reading + Triste nod + a later STATUS rewrite.

**### 2026-11-12 晚双视角实验开工段（历史，已退役）

下面这一段是 2026-11-12 晚写下的双视角实验开工记录。该实验已在备份提交 `28bd3c9c`（含已合入的 #73）做完并停止。停止结论在文首，不要重跑，不要再合入，不要把本段当成现行门禁。

（历史，不是现行门禁。2026-11-12 晚曾把下一步写成 Nutrition5k 双视角实验。该实验已在备份提交 28bd3c9c 中做完并停止，不要重跑，不要再合入。现行下一步只在文首：第一阶段数据可行性审计。）**

- **评价范围的收缩（这是「若继续，先明确评价范围的收缩」的那一笔，不是数据补救）**：
  - 本轮不补采、也不新造外部混合餐的配对第二视角；用收缩评价范围代替数据补救。
  - 实验 A 仅在 Nutrition5k 同时具备顶视 RGB 与侧视 RGB、可按 dish_id 对齐的子集上进行；先核可用规模，不编造配对条数；侧视不作度量双相机。
  - ACETADA、VLM Food Benchmark 本轮不做双视角或主动获取主测（仍缺真实可请求的第二视图）。ECUSTFD 最多作辅助几何对照，不作混合餐主测试。Cafe2 仍是已探索诊断集。
  - 完整联合网络、实验 B、实验 C 仍锁。停止规则不变：若简单融合与几何融合不能稳定胜过单视图（主要是俯视）基线，停下来让 Triste 判断，不要强行第二视角，也不要自行换方案。官方方案页的原句一并有效：若几何/语义融合不能稳定胜过单视图基线，不要强行第二视图。主单视图基线是俯视图像。若简单融合与几何融合不能稳定胜过该俯视基线，不要为了多模态强行第二视角，也不要自行换方案。
  - 无 RL，不开 fast Cloud Agent，RGB-D 不进主表。深度若出现，只进附录。

- **方案真源（2026-11-12 晚，Asia/Shanghai）**：Canvas「**VDR 官方方案（2026-11-12 晚）：面向物理约束营养估计的主动多模态证据获取（候选）**」（https://cursor.com/canvas/store-745bdf38-f3d7-4313-8d66-08e4b6903e93/d7be617d-d37e-4fe1-a4b5-9da2c86b27a2）。本页是方法方案真源。它取代同日早间把 Evidence-aware Few-shot Mass Calibration 定为候选创新点二的定位，也取代以前把主方法写成「Coverage-Constrained Active Specialist Selection + Error-Aware Q（Route A 能量+质量双赢）」的主张写法。项目名仍是 **VDR**。输出仍是 \(y=(n, G, s, \tau)\)。
- **该句曾把本次改写写成双视角实验的开工许可，2026-11-13 起失效。** 解锁依据：Triste 于 2026-11-12 约 22:04（Asia/Shanghai）在群里说，按现行官方方案继续；只有结果不好、继续已无意义、或方案本身需要改时，才停下来由他判断。这次改写不是方案变更。官方方案页已经把实验 A 设计在 Nutrition5k 双视角子集上，并写明若继续，先明确第二视图的数据补救或收缩评价范围，再考虑解锁实验 A。本轮写明的是评价范围的收缩，不是补采。不要据此改 main，也不要重开已停止的双视角实验。不得据此派出实验 B 或实验 C，不得据此搭建完整联合网络，也不得据此重开 #70 的 Phase 1。少样本 / 分层质量校准仍只是普通基线，不是主实验。不重训 Q。不改 food-id。不在 505 或 Cafe2 上调新方法。不开 P1。无 RL。不开 fast Cloud Agent。门禁 ④–⑥ 仍是投稿前要做的事，不是本门。
- **#70 的 Phase 1 是历史，并已取消。** #70（squash `d622e6db`，2026-11-12）曾按同日早间方案把「Phase 1 跨域诊断（0/8）」写成当时的下一步，并写成那次文档的开工许可。该步没有开跑（仍是 0/8）。少样本 / 分层质量校准不再是创新点二，该诊断不再是现行下一步。不要下载外部集来跑 E1–E5，不要建 `results/phase1_cross_domain/`，不要实现校准网络的公式或分层结构。
- **#69 收口是历史**：#69（squash `ae0e73aa`，2026-11-10）把 Mass-only 能量路径诊断标成已完成，当时把状态收成空闲，并且那次改写不构成开工许可。#66（`344255f`，2026-11-09）对 Cafe2 能量审计的收口同样是历史。当时写成现行状态以那一节为准；2026-11-13 起现行状态只以文首为准。下文已归档段落里的冻结数字不改。
- **候选定位（未证实，不是已成立贡献）**：Active Multimodal Evidence Acquisition for Physically Grounded Food Nutrition Estimation（面向物理约束营养估计的主动多模态证据获取）。研究问题升级为：多模态模型如何理解食物图像中物理尺度、食物组分与营养密度的耦合，并主动获取最有价值的视觉证据。已有 VDR 冻结结果全部保留，作历史证据。输入协议是低成本多视角 RGB；主表仍不要求 RGB-D。若只能得到单张 RGB、没有第二视图，这条路线严重受限。
- **会刊**：ACM MM 2027 或更晚。MM 2026 Regular Paper 截稿为 2026-04-01 AoE；会议 2026-11-10 至 2026-11-14（里约热内卢）。MM 2027 在香港；日期与截稿尚未公布。不虚构 2027 截稿。
- **硬约束**：无 RL。不开 fast Cloud Agent。主表只 RGB（depth 只进附录）。禁止用测试集 GT 调参。FNDDS 默认 **2019–2020**，换版就全基线重跑。不靠弃权制造质量优势。AURC / 弃权不作主故事。所有样本都要出预测。禁空壳 RE_PORTION。
- **现行主声称：质量对 Direct 可外推（bootstrap 区间均为负）；对 Mass-only 基本打平（0.76 g 不算赢，不要当作方法优越性证据）；能量外测未稳。** 停止声称能量与质量双重提升。新句子不补 bootstrap 区间端点，也不把开发 505 写进外测表。已归档的 #63 / #65 / #68 / #48 句子保持原样。
- **Cafe2** 仍是已探索诊断集。禁止用 Cafe2 给新方法调参，也禁止把它当作下一轮完全盲测。505 仍是开发 / 诊断集，禁止在上面调新方法。
- **模块（2026-11-12 晚）**：VLM 食物理解是语义分支的基础模块，保留。**Mass Estimator** 是预训练的几何-质量专家，保留；冻结结果作历史证据。**FNDDS** 是营养密度知识来源，保留。**Coverage Constraint** 是可选的输出完整性机制，保留为辅助。**Nutrient Verifier** 是可核验输出的辅助机制，保留。**Q_φ** 不再是主方法，降为对照 / 消融，停止继续训练（相对规则约 **0.96 kcal**、远落后 Oracle）。同头密度换码是对照，停止继续改 food-id（#60：**68** 个换码，Exact=**0**）。分层少样本质量校准是普通基线，不是创新点二。两路径能量融合 / Energy Router 仅作对照，不预定为贡献。
- **Q1：有条件支持。** Nutrition5k 侧视 RGB 在有顶视时可以按 dish_id 与顶视 RGB 对齐。官方说法：约 3.5k / 5,006 盘含顶视 RGB-D（论文约数，不是精确可用配对条数）。不编造可用配对条数。不把非官方目录条数收成官方可用配对数。侧视没有度量标定，也没有公布位姿。源域几何-质量实验可以在同时具备两路视图的子集上设计。这不足以宣称已有度量双相机几何。实验 A 就在同时具备两路 RGB、可按 dish_id 对齐的子集上开工；侧视不作度量双相机。
- **Q2：未完全通过。** 源域对身份、质量与知识库密度的监督较足；烹饪 / 隐性配料缺少独立真值。关键缺口：外部混合餐公开集（ACETADA、VLM Food Benchmark）基本是单视图，不能支撑在外部请求真实第二视图。因此先不要搭建完整联合网络。本轮不把主动证据获取做到外部集。
- **Q3：未解。** 当时的双视角实验现已做完并停止，不是再开工，不得写成已解。实验 C 仍锁；外部几乎没有真实可请求的第二视图。没有实验 C 时，不得把主动获取写成已胜过普通融合。
- **Q4：可描述，但偏薄。** 预期差距是尺度 / 身份 / 密度的联合不确定性，加上由该因子化驱动的证据获取。Open-KNEAD: Knowledge-grounded Nutrition Estimation via Agentic Decomposition（arXiv:2607.12911，2026-07-14）已把邻近方向列为未来工作。若只做少样本校准或单独的 3D 分支，这条差距不成立。
- **公开数据边界（不编造条数）**：Nutrition5k 官方 **5,006** 盘；顶视 RGB-D 只写论文约数「约 3.5k / 5k」。ACETADA（arXiv:2507.07048）：该实验使用 **806** 张餐前智能手机图像；配对餐后是随时间留下的剩余食物，不是几何侧视。本轮 ACETADA 项目主页返回 500，不从主页采集额外数字。VLM Food Benchmark 官方 README：200 张食堂餐图，870 项已核验逐项质量，6 名参与者，多为自由条件俯视，不是多视角集。ECUSTFD（Hugging Face）：2,978 张手机图，19 类，顶视与侧视，25 mm 硬币，逐对象质量与体积；只作辅助几何-质量校准，不是混合膳食主测试；不编造可用配对条数。ASA24 官方页：超过 17,000 张与 FNDDS 份量关联的图像。17,135 / 6,469 是 Single-Image Food Mass Estimators Do Not Transfer Their Gram Scale Across Datasets（arXiv:2610.29025，2026-10-21）的处理计数，不是 NCI 官方可用质量标签总数。
- **少样本校准降为普通基线的文献依据（那篇论文的结果，不是我们的结果）**：arXiv:2610.29025 在已知食物上，每类 1 张标注照片把 ASA24 测试 MAE 从 42.6 g 降到 28.3 g（2 张 27.3 g，3 张 26.7 g）。因此少样本 / 分层质量校准不再当创新点二。分割也不当算法创新：Effects of segmentation granularity on nutrient prediction from real-world food images（doi:10.3389/fnut.2026.1927530）在 1,056 张 Nutrition5K 子集上，未分割均值 PMAE 19.91%，最佳二值食物区域 20.18%，配料级 23.76%–25.62%。单独 3D 分支也不够：MFP3D（arXiv:2411.10492）、Food Portion Estimation via 3D Object Scaling（arXiv:2404.12257）、Implicit-Scale 3D Reconstruction（arXiv:2602.13041）；可选对照 Size Matters（arXiv:2601.20051）。已有指针保留：arXiv **2610.11144**、NutriVision arXiv **2609.33076**。不编造其他 DOI。
- **评价范围已收缩，当时因此写下双视角实验；该解锁已退役。** 这就是官方方案里「若继续，先明确评价范围的收缩」的那一笔，不是数据补救，也不是另起一套方案。周度论文追踪（Open-KNEAD / 3D 份量 / 跨域质量）仍只是建议的后续跟进，不是本文件已启动的承诺。
- **双视角实验当时写成已解锁、且只此一项；2026-11-13 起该许可退役，实验已停止，不要重开。** 范围是 Nutrition5k 里同时有俯视 RGB 与侧视 RGB、并能按 dish_id 对齐的盘。先从官方数据核可用规模。不编造可用配对条数。不采用 GitHub issue #12 的非官方约 3,490。官方只写约 3.5k / 5,006 盘含顶视 RGB-D（论文约数），那不是可用的俯视 RGB 与侧视 RGB 配对条数。官方 Nutrition5k 是 5,006 盘。然后对比单张俯视、单张侧视、双视角简单特征融合、双视角几何融合、双视角+食物语义约束。主指标是 Mass MAE，以及质量预测跨视角的一致性。主单视图基线是俯视图像。输入仍是低成本多视角 RGB。RGB-D 不进主表；深度若出现，只进附录。停止规则：若几何/语义融合不能稳定胜过单视图基线，不要强行第二视图。若简单融合与几何融合不能稳定胜过俯视这一单视图基线，停下来让 Triste 判断，不要为了多模态强行第二视角，也不要自行换方案。实验 B 仍锁：三类诊断子集（外观相近而密度不同；身份易混；体积相近而份量不同）。实验 C 仍锁：固定单图、固定双视图、随机下一证据、普通置信度、提议的因子化获取，以及 Oracle 的同预算比较；同预算下若不能胜过固定双视图与普通置信度，放弃主动模块。这两门都没有开工许可。完整联合网络仍锁。
- **早间 Phase 2 门槛改为历史。** 同日早间曾记下：至少两个独立外部数据集上，质量 MAE 改善约 **5%–10%**，且配对 bootstrap 置信区间支持该改善，否则不要把分层校准写成方法贡献。Triste 原文审阅在此句被截断，上述是当时恢复后的工作门槛。该门槛随「少样本校准作为创新点二」一并撤回，不是实验 A 的通过线，也不是本文件的开工条件。
- **投稿前仍要做，但不是本门**：④ 真实工具成本；⑤ 忠实 DietAI24 / FoodAgent 式基线；⑥ Provenance / Semantic 审阅。⑦ 以及 P1、Mass/Q/τ 重训仍锁。相关工作对照仍保留已有指针：arXiv **2610.11144**、NutriVision arXiv **2609.33076**、DietAI24、CVPRW 2025 **Decomposing Food Images**。

**#70 Phase 1 清单（历史，已取消，0/8，不得执行）**

#70 把下面八步写成当时的 Phase 1。现已取消。不要按此开工，不要重排后拿去跑，也不要把它当成进入实验 A 的前置代码任务。

1. 当时拟冻结当前 VDR，并把新产物放到 `results/phase1_cross_domain/`。该目录不要为这项已取消的工作去建，也不要覆盖既有冻结结果目录。
2. 当时拟下载 ACETADA 与 VLM Food Benchmark 并核对许可。本审计的写法是：ACETADA 为 **806** 张餐前智能手机图，餐后不是几何侧视；VLM Food Benchmark 为 200 张食堂餐图 / 870 项 / 6 名参与者，且不是多视角集。
3. 当时拟统一 Nutrition5k 与外部集的质量标签单位和定义。
4. 当时拟按参与者或 dish_id 建立跨域划分。
5. 当时拟跑 E1 Direct-VLM、E2 Frozen Mass-only、E3 当前完整 VDR。三者都没有开跑。
6. 当时拟加 E4 全局仿射校准与 E5 类别级校准。没有开跑。那是对照，不是分层校准网络。
7. 当时拟记录 Mass MAE、相对误差与有符号偏差。没有测得的数，不预填。
8. 当时拟由该诊断决定是否进入当时的 Phase 2。Phase 2 没有解锁，并已随创新点二撤回。以上 1–8 都不要执行。

**其后仍锁定（不得自动开工）**

- **实验 B、实验 C** 与 **完整联合网络**。见上。禁止现在实现。双视角实验的开工许可已退役，旧协议不再执行。
- **已取消的 Phase 1** 少样本 / 跨域校准诊断，以及早间的 **Phase 2** 分层校准网络、**Phase 3** 能量专家融合对照。两路径能量融合只作对照。
- **P1** 检索重排。**Q / Mass / τ 重训。** **food-id 改写。** **505 或 Cafe2 上调新方法。**
- **④** 真实 Tool Cost：VLM 调用次数、延迟、token、单盘成本，以及能量 MAE–Cost Pareto。#53 的 tool mean **6.0257** 只是调用次数，不是真实成本。投稿前仍要做，但不是本门。
- **⑤** 忠实重做 B1，并做 B2（能完整复现就完整复现，否则诚实标明 approx）与相关工作。优先：arXiv **2610.11144**（食物身份验证 / 恢复）、CVPRW 2025 **Decomposing Food Images**、**NutriVision** arXiv **2609.33076**，以及上面 Q4 已点名的文献。不编造 DOI。历史 B1 **110.72** 非忠实；scaffold / dry-run 不算。投稿前仍要做，但不是本门。
- **⑥** 证据效度：Provenance（程序复算，自动，全量）与 Semantic（人工，提前到核心实验）。投稿前仍要做，但不是本门。
- **⑦** 再共决是否解锁 **P1** 或重训 Mass / Q。在此之前两项都锁定。

**Checklist（给代码 Agent）**

1. **P0a–P0d、Q 校准 #44、误差审计 #48、food-id #51、覆盖约束 #53、① #57、② #60、③ #63、Cafe2 能量审计 #65、Mass-only 能量路径便宜诊断 #68** — **已完成**。#53 = Ours-new **开发检查点**。**不**声称 #53 证明方法。**#51 / #53 ≠ Q 学习改进**（①：Q ≈ 规则 ≪ oracle）。**②** 的自动标签只是建议。**kcal 变好 ≠ 身份变对。不**把 #51 能量下降写成身份成功。#51 是密度 / kcal 启发式（换码后自动 Exact=**0**）。**③（#63，squash `27d1dbfa`）** 的 Cafe2 主表**不是**能量与质量双赢。**#65（squash `ecf859f5`）** 不把该表改写成双赢，也**不**开工改方法。**#68（squash `f578e8f4`）** 记下 **98.6201** 对 **75.3870** 是更大的有符号抵消，不是更准的密度或食物码，**不**把外测改写成双赢，也**不**开工改方法。**主声称：质量对 Direct 可外推（bootstrap 区间均为负）；对 Mass-only 基本打平（0.76 g 不算赢，不要当作方法优越性证据）；能量外测未稳。**

2. **#70 Phase 1 跨域诊断 — 历史，已取消，0/8。** 不是当前项。没有开跑。本文件这次改写**不是**它的开工许可，也**不是**实验 B、实验 C 或完整联合网络的开工许可。双视角实验的开工许可已退役；现行许可在文首的第一阶段审计，不在本条。不要用 Cafe2 调新方法，也不要把它当下一轮盲测。不要重跑 #48 / #51 / #53 / #57 / #60 / #63 / #65 / #68。**505** 仍是开发 / 诊断集，禁止在上面继续调。**禁止**重训 Mass / Q / τ；**禁止**开 P1；**禁止**自动开实验 B、实验 C、完整联合网络与 **④–⑦**。不要再按旧协议重开双视角实验；**无 RL**；**禁止** fast Cloud Agent。**主声称：质量对 Direct 可外推（bootstrap 区间均为负）；对 Mass-only 基本打平（0.76 g 不算赢，不要当作方法优越性证据）；能量外测未稳。**
3. **双视角实验** — **历史，已停止，不是现行下一步。** 只在 Nutrition5k 同时具备顶视 RGB 与侧视 RGB、可按 dish_id 对齐的子集上，先核可用规模，再对比单张俯视、单张侧视、双视角简单特征融合、双视角几何融合、双视角+食物语义约束。**实验 B、实验 C 与完整联合网络仍锁。禁止自动开。** Q2 未完全通过：外部混合餐缺少配对第二视图。本轮不补采、也不新造外部混合餐的配对第二视角；用收缩评价范围代替数据补救。Q3 不得写成已解。
4. **早间 Phase 2 分层校准网络与 Phase 3 能量融合对照** — **历史，锁定。禁止自动开。** 少样本校准已降为普通基线。Phase 3 的融合只作对照，不预定为贡献。
5. **④ 真实 Tool Cost** — **锁定。禁止自动开。** 投稿前仍要做，但不是本门。
6. **⑤ 忠实 B1/B2 + 相关工作** — **锁定。禁止自动开。** 投稿前仍要做，但不是本门。
7. **⑥ Provenance + Semantic** — **锁定。禁止自动开。** 投稿前仍要做，但不是本门。
8. **⑦ 共决是否解锁 P1 / 重训** — **锁定。P1 仍暂缓。禁止自动开 P1。禁止重训 Mass / Q / τ。禁止 food-id 改写。禁止 505 调参。**

**保留的 #48 诊断背景（下面两段原文不改。这是 #51 之前的对照；#53 的开发数字见上，仍不是最终主表）**

- **审计结论（背景；方法标签仍用 B0 / Ours-old）**：paired **n=505** 上 Ours-new 相对 **B0** 的 **ΔM −17.3513 g**（bootstrap 95% **[−24.94, −10.13]**，**万次 resample 均值均为负**）；**ΔE +1.4210 kcal**（95% **[−8.10, +11.05]** **跨 0**）；**median ΔE −4.54 kcal**；去掉 **2** 盘最大能量回归（`dish_1562172643`、`dish_1564675411`）后 **503** 盘 mean **−0.14 kcal**（敏感性，**主表 MAE 仍含该两盘**）。能量误差可拆 **克重项 vs 食物密度（kcal/g）项**；B0 上两项**更常反向抵消**，Mass 写克重后部分盘**打破抵消** → mass↓ 但 energy↑ 共存。**未完成 mass 写**的 **306** 盘对 pooled 能量均值 **+5.74 kcal**；其中 **78** 盘 **STOP 且仍有未写组件**，mean signed kcal err **−87.6**，对均值 **+6.06 kcal**。**481** 盘 **hold food-id 固定**对比写前克重：ΔM **−20.24 g**、ΔE **−17.90 kcal** — **不是**「用能量换质量」相对 prior grams。**Q fence 重放 2858/2858**；505 盘 **committing action 均为 Q/Cost**（本审计**未**去掉 fence 重跑轨迹）。本段 paired **ΔE +1.4210** 是 **#51 之前**；**#51** 后同池 vs Direct **ΔE −4.43**（能量 MAE **92.6963**）。**#53** 后同池能量 MAE **64.2792**、质量 MAE **32.9908**；vs Direct **ΔE −32.85** / **ΔM −43.36**（见上与「助手进度」）。
- **分层要点（#48 报告）**：**199** 盘 **全部组件 mass 已写**：ΔE **−10.96** / ΔM **−39.17** kcal·g；**149** 盘 **单组件** ΔE **−17.99**；**50** 盘 **多组件全写** ΔE **+10.01** / ΔM **−91.80** — food-table commit 能量 MAE **190.79** vs 同盘 stored calorie head **77.75**，**grams 与 head 一致**（commit 用 FNDDS kcal/100g×克重，**非** head）。相对 **Ours-old**（closed-loop BC）点估计 ΔE **−1.8588**、ΔM **−3.4147**，bootstrap 区间**仍跨 0**。本段是 **#48 写前**基线；**#51** 后同一切片 food-table 能量 MAE 为 **131.65**。**#53** 后全部组件 mass 已写 **503/505**、STOP-unwritten **0**（见上与「助手进度」）。

- **背景 — Q 保守性门禁（#41；选 B 的原因）**：P0c hold-out 上 Q/Cost **STOP-heavy**（2771 行：STOP **1360** / RE_PORTION **608** / RETRIEVE **499** / ABSTAIN **266** / CORRECT_CODE **38**）；Q/Cost 与规则 heuristic **一致率 0.4489**；stub **首步 STOP**。**#44 已缓解**（一致率 **0.7712**；stub **RETRIEVE→RE_PORTION→STOP**，1 次 mass，`portion_src=mass_estimator`）。

1. **P0a — 审计门禁（已完成 #37，squash `2e0e180`）**  
   - 脚本：`scripts/closed_loop_tool_state_audit.py`；定义 `vdr/eval/tool_state_audit.py`（ε=**5e-5**，非 hold-out / rgb_test 拟合；与 FixRate ε=1 kcal **不同**）。  
   - 数据源：hold-out logged（`results/weak_steps/rgb_train_holdout_all.jsonl` 链）+ on-policy `results/closed_loop/on_policy_rgb_test.jsonl`（#34）。  
   - 产物：`results/closed_loop/tool_state_audit.json`、`report_tool_state_audit.txt`、`report_tool_state_audit.md`。  
   - **改码/改份量 edit 动作**（CORRECT_CODE / RE_PORTION；hold-out **无** RE_DECOMPOSE）：hold-out logged **变更率 1.0000（1106/1106）**。  
   - on-policy edit：**变更率 0.9175（1647/1795）**、无变化 **0.0825**；分层 **CORRECT_CODE 0.7822**、**RE_PORTION 0.9476**。  
   - **结论**：「工具跑了但 n 不变」**不是**多数；**门禁通过**（仍禁空壳 RE_PORTION）。**P0b #39**、**P0c #41**、**Q 校准 #44**、**P0d #46**、**误差审计 #48**、**food-id #51**、**覆盖+早停 #53**、**① #57**、**② #60**、**③ #63** 已落盘 → **Cafe2 能量审计 #65 已完成**；**Mass-only 能量路径诊断 #68 已完成**；**（历史，不是现行门禁。2026-11-12 晚曾把下一步写成 Nutrition5k 双视角实验。该实验已在备份提交 28bd3c9c 中做完并停止，不要重跑，不要再合入。现行下一步只在文首：第一阶段数据可行性审计。）已取消的 Phase 1 没有开跑**（**不是门禁 ④**；**Phase 2、Phase 3、④–⑦ 与 P1 仍锁**；**不**重训 Mass/Q/τ）。

2. **P0b — Mass Estimator（已完成 #39，squash `4b6ccec`）**  
   - **Backbone**：**ViT-B/16**（fit-internal probe 优于 ResNet101：probe-val mass MAE **52.48** vs **57.52**）；ImageNet-1K → Nutrition5K **rgb_train fit** 微调，**12 epoch**，**last checkpoint**。  
   - **主表 RGB-only**（不读 depth；**禁止** rgb_test GT 用于选择/调参）。  
   - 多任务头 **[M,E,P,C,F] + Laplace σ**；工具 **`estimate_mass(image, candidates)`** 已接入闭环 **RE_PORTION**（`portion_src=mass_estimator`；**无**空壳 RE_PORTION / **无** gram-grid 在 RE_PORTION 上重打分）。  
   - **Hold-out**（rgb_train hold-out 俯拍 RGB，**n=257**，save 后单次评分）：mass MAE **29.6147 g**（median AE **18.0219**）；energy MAE **43.2161 kcal**（median AE **23.0434**）；protein **4.7922** / carb **4.5685** / fat **3.2113 g**；同盘 fit-mean 基线 mass **130.6992 g** / energy **176.8194 kcal**。  
   - **产物**：`results/mass_estimator/`（`weights-part-00..02.npz`、`train_meta.json`、`backbone_probe.json`、`eval_holdout.json`、`report_holdout.txt`）。冻结 n5k / weak_steps **未改**。

3. **P0c — U_* + Q_φ（已完成 #41，squash `54c32e8`）**  
   - **状态 z**：**98-d** = **U_id / U_mass / U_nutrient / U_evidence** + 模型分歧 + 先验 **o/b**（**禁**弱 kcal-proxy 进 z 作主叙事）；**Mass Laplace σ**（`estimate_mass`）馈入 **U_mass / U_nutrient**。  
   - **选择**：先规则 utility，再监督 **Q_φ(z,a)=E[ΔL|z,a]**；**ΔL** 目标 **L=min(|kcal err|,2500)/2500**（与 rule_verify 能量上限一致，**未**在 hold-out / rgb_test 上拟合）；**a*=argmax Q/Cost**。**无 RL**；#24 **State-aware BC 仅消融**。  
   - **Hold-out**（`rgb_train` weak steps logged，**2771** 行）：Q_φ 回归 MAE **0.0368** vs 零基线 **0.0518**（Pearson **0.7799**，**n=1960** 可测工具步）；与 logged 一致率 — heuristic **0.7968**、BC 消融 **0.9607**；Q/Cost vs heuristic **0.4489**。Q/Cost 动作计数见上「Q 保守性门禁」。  
   - **产物**：`results/qphi/`（`q_phi.npz`、`metrics_holdout.json`、`report_holdout.txt`、`train_meta.json` 等）。**不是 P0d 主表**；冻结 n5k / weak_steps **未改**（#41 未 touch STATUS；哈希与 #34 一致）。

4. **Q 校准 — Q/Cost terminal-margin（已完成 #44，squash `3fc611d`）**  
   - **方法**：fit weak steps 上，未校准 Q/Cost 选 STOP/ABSTAIN 且专家合法时，excess = 最佳终端 Q/Cost − 最佳合法专家 Q/Cost；**τ = median(excess) + 3·MAD(excess)**（**τ≈0.001645**；乘数 3 **未**在 hold-out / rgb_test 上搜）。终端领先 margin **< τ** → **defer 到 rule utility**；否则保持 argmax Q/Cost。**`q_phi.npz` 权重未改**。**无 RL**；**未**读 rgb_test GT；**未**用 hold-out 选 τ。  
   - **Hold-out**（`rgb_train` weak steps logged，**2771** 行，**n=2771**）：Q/Cost vs rule utility（P0c 同口径 heuristic）一致率 **0.4489 → 0.7712**。动作计数（校准后）：STOP **526**（未校准 **1360**）；RETRIEVE **1096**（**499**）；RE_PORTION **842**（**608**）；ABSTAIN **207**（**266**）；CORRECT_CODE **100**（**38**）。非 STOP 步：**专家调用率 0.811 → 0.908**；Mass|检索占比 **0.785 → 0.863**。决策：**保留 Q/Cost 1878**；**defer rule utility 893**。  
   - **Stub** `dish_1550704903`：未校准 **首步 STOP** → 校准 **RETRIEVE→RE_PORTION→STOP**（**mass_calls=1**，`portion_src=mass_estimator`）。  
   - **校准门禁**：调研门槛（一致率 **≫ 0.45**；非 STOP **实际** Mass/检索）→ **通过**；**P0d #46 已落盘**（`b9c4c5e7`）。**不是** P0d Full-set 主表的历史解读段（主表见下条 5）。  
   - **产物**：`results/qphi/calibration.json`、`calibration_holdout.json`、`report_calibration.txt`。冻结 n5k / weak_steps **未改**。

5. **P0d — 同预算评测（已完成 #46，squash `b9c4c5e7`）**  
   - **Ours-new**（Mass + **校准后** Q/Cost）vs **Ours-old**（closed-loop BC）/ **B0**（Direct-VLM）/ 开环 BC；定义 `p0d_same_budget_overhead_force_answer`；官方 rgb_test 有俯拍 **507** 盘；paired 四法皆可记分 **n=505**。  
   - **Ours-new（paired n=505）**：Energy MAE **98.5521** kcal；Mass MAE **58.9952** g；Tool cost mean **4.6970**（median **2**，max **32**，505/505 在 K=32 内）。  
   - **vs Ours-old（同 paired 池）**：ΔE **−1.8588** kcal，ΔM **−3.4147** g，Δtool **−1.8812**。  
   - **vs Direct-VLM（B0，同 paired 池）**：ΔM **−17.3513** g，ΔE **+1.4210** kcal — **能量主 claim 未成立**（质量赢、能量未稳赢 B0）。  
   - **Mass RE_PORTION**：**827** 次成功，**0** hollow（变更 dish 总营养 **827/827**）。  
   - **Coverage 0.5921**；**ABSTAIN 206** — **仅辅助**，**不是**通过门禁。  
   - **本池锚点复现（own-pool，非 paired 行）**：Direct-VLM **n=507** E **97.0256**（≈历史 **97.03**）/ M **76.4556**（≈**76.46**）；closed-loop STOP 子集 **n=488** E **96.2769**（与 #34 一致）。  
   - **开环 BC** 读 closed-loop 同一 logged trace：**E/M/tool 与 closed-loop 相同**（report 说明非独立 VLM rollout）。  
   - **产物**：`results/p0d/report_same_budget.txt`、`same_budget.json`、`same_budget_dishes.jsonl`、`ours_new_rgb_test.jsonl`、`ours_new_rgb_test_summary.json`。冻结 n5k / weak_steps **未改**。

6. **误差审计（已完成 #48，squash `91a05d8`）**  
   - **输入（只读 #46）**：`results/p0d/same_budget.json`、`same_budget_dishes.jsonl`、`ours_new_rgb_test.jsonl`、`ours_new_rgb_test_summary.json`、`report_same_budget.txt`；冻结 B0 jsonl（宏量）；`q_phi.npz` 仅 **Q fence 重放**。  
   - **脚本**：`scripts/p0d_error_audit.py`（**无新训练**；**未**用 rgb_test 标签选阈值/码/克重）。  
   - **产物**：`results/audit/report_error_audit.md`、`report_error_audit.txt`、`error_audit.json`、`paired_residuals.jsonl`、`fig_*.svg`。  
   - **结论摘要**：见「助手进度」#48 段。**#53** 已处理覆盖与早停（见该项）。**不**自动解锁 Mass 重训 / Q 改 / P1。

7. **food-id / 密度轻修（多组件优先；已完成 #51，squash `6c8fda02`）**：同头码替换（已有检索窗，≥15% 更靠近 calorie head）；克重与动作保持。paired **n=505** 能量 **98.5521 → 92.6963**（质量 **58.9952** 不变）；多组件全写 **n=50** food-table **190.79 → 131.65**。**827** 成功 mass、**0** hollow。**#51 未改 STATUS**。详见「助手进度」。**不得**据此开 Mass 重训 / Q 改 / **P1**。语义审计是门禁 **②**，**已由 #60（squash `c52ffaa2`）完成**（见文首）。自动标签只是建议。**kcal 变好 ≠ 身份变对。不**在本条重跑 #51，也**不**把能量下降写成身份成功。

8. **覆盖未写组件 + 早停处理（已完成 #53，squash `2a406df4`）**：冻结 Q+fence 后，未写组件无 `estimate_mass` 写则拒绝 STOP/ABSTAIN（defer rule utility）；回到检索 top code 则 portion，不开第二轮 top-code swap；**#51 同头 swap 保留**。paired **n=505** 能量 **92.6963 → 64.2792**、质量 **58.9952 → 32.9908**、tool mean **4.6970 → 6.0257**（max **17**，均在 K=32）。全部组件已写 **199/505 → 503/505**；STOP-unwritten **78 → 0**；Mass 成功 **827 → 1479**；hollow **0**。残留 **2** 盘（`dish_1565020895`、`dish_1565020926`）honeycomb 检索失败，卡 RETRIEVE，不编造克重、未 STOP。**无** Mass/Q/τ 重训。**#53 未改 STATUS**。这是开发检查点，**不**证明方法；增益 **≠** Q 学习改进。**① 已由 #57 完成，② 已由 #60 完成**，不在本条重开。**不得**据此重开本条或开 **P1**。**③ 已由 #63（`27d1dbfa`）完成**，不在本条重开，也**不**在这 505 盘上再调方法。Cafe2 能量审计已由 #65（`ecf859f5`）完成。Mass-only 能量路径便宜诊断已由 #68（`f578e8f4`）完成。**（历史，不是现行门禁。2026-11-12 晚曾把下一步写成 Nutrition5k 双视角实验。该实验已在备份提交 28bd3c9c 中做完并停止，不要重跑，不要再合入。现行下一步只在文首：第一阶段数据可行性审计。）**（见文首；**不是门禁 ④**，**禁止**自动开 **④–⑦**）。

9. **① 反事实 Q 动作价值 + dish_id 分组泄漏审计（已完成 #57，squash `0aa30537`）**：产物 `results/qphi_counterfactual/`。fit/hold-out **按 dish_id 分组**，交集 **0**，**无**重训。头条 n=**1914**（coverage-choice + mass forward；rgb_train hold-out；与 #53 **505** 交集 **0**）：Q Best-ID **0.6165** / ΔE improve **24.14** / E regret **49.96**；规则 **0.6102** / **23.03** / **50.91**；固定 **0.5914** / **20.62** / **52.99**；随机 **0.4338** / **4.04** / **78.15**；oracle improve **82.19**、regret **0**。Q 约 **0.96** kcal 优于规则，约 **50** kcal 落后 oracle。**241/1914** 的 oracle 最优是 RE_DECOMPOSE no-op（冻结 Q 不能发出）；去掉后 Q regret **~17.33** vs 规则 **~18.28**。价值校准 n=**3475**：MAE(Q, ΔL) **0.0226**，Pearson **0.747**；RE_PORTION **0.805**，CORRECT_CODE **0.016**。预算曲线在该目录 `report.md`（一步 cost cap，不是全程 rollout）。**Q ≈ 规则 ≪ oracle**。**② 已完成**，不在本条重开。**③ 已由 #63 完成**，不在本条重跑，也不用本条数字重开 held-out、重开 Cafe2 能量审计，或开工改方法。

10. **④–⑦ 与 P1（锁定）**：**①②③、Cafe2 能量审计与 Mass-only 能量路径便宜诊断已完成。** **（历史，不是现行门禁。2026-11-12 晚曾把下一步写成 Nutrition5k 双视角实验。该实验已在备份提交 28bd3c9c 中做完并停止，不要重跑，不要再合入。现行下一步只在文首：第一阶段数据可行性审计。）除文首第一阶段数据可行性审计以外，没有别的当前可写代码项。** #65（squash `ecf859f5`）与 #68（squash `f578e8f4`）已记下，**不是门禁 ④**，也**不**解锁后面各项，也**不是** Phase 2、Phase 3 或分层校准网络。不要据此改 main，也不要重开已停止的双视角实验；其余代码仍空闲。该句曾把本次改写写成双视角实验的开工许可，2026-11-13 起失效。实验 B、实验 C 与完整联合网络、已取消的 Phase 1 与 ④–⑦ 都没有开工许可。不要重跑，不要改 `results/heldout_energy_audit/` 或 `results/heldout_massonly_energy_path/`，不覆盖 `results/heldout_plan/`、`results/heldout_eval/`、#51 / #53 / #57 / #60。**禁止**用该审计做方法选择、早停、换方法、阈值 / τ，或 GT 驱动的码 / 份量修改，也**禁止**从这些标签开工改方法。基线仍是 **Direct-VLM** 与 **Mass-only**。其后仍锁定：**④** 真实 Tool Cost（VLM 次数、延迟、token、单盘成本、能量 MAE–Cost Pareto）→ **⑤** 忠实 DietAI24 / FoodAgent 式基线 + 相关工作（优先 arXiv **2610.11144**、CVPRW 2025 Decomposing Food Images、NutriVision arXiv **2609.33076**；不编 DOI）→ **⑥** Provenance + Semantic → **⑦** 再共决是否解锁 **P1** 或重训 Mass/Q。**禁止自动开 ④–⑦。P1 仍锁定 / 暂缓。禁止自动开 P1。禁止**重训 Mass / Q / τ。人审包仍开放，自动标签只是建议，**不**改写已完成的 ③ 协议，也**不**改写已完成的 Cafe2 能量审计（#65），也**不**改写已完成的 Mass-only 能量路径诊断（#68），也**不**把 #51 的 kcal 下降写成身份成功。忠实 DietAI24 / FoodAgent（[`docs/B1_B2_SCAFFOLD.md`](B1_B2_SCAFFOLD.md)）**不得**在 **⑤** 解锁前开工。

**禁令（仍有效）**：**不得**重训 Mass；**不得**重训 Q；**不得**重拟合 τ；**禁止自动开 P1**；**①②③、Cafe2 能量审计（#65，`ecf859f5`）与 Mass-only 能量路径便宜诊断（#68，`f578e8f4`）已完成，不是开工项**；**（历史，不是现行门禁。2026-11-12 晚曾把下一步写成 Nutrition5k 双视角实验。该实验已在备份提交 28bd3c9c 中做完并停止，不要重跑，不要再合入。现行下一步只在文首：第一阶段数据可行性审计。）除文首第一阶段数据可行性审计以外，没有别的当前可写代码项。不是门禁 ④。Phase 2、Phase 3 与新网络仍锁**（不要重跑、不要改 `results/heldout_energy_audit/` 与 `results/heldout_massonly_energy_path/`；不要覆盖 `results/heldout_plan/` 与 `results/heldout_eval/`）；**禁止**在 Cafe2 或 505 上改方法、调超参、早停或做模型选择；**禁止自动开 ④–⑦**；**禁止**用 held-out 做方法选择、早停、阈值 / τ 或 GT 驱动的码 / 份量修改；不上 RL；不开 fast Cloud Agent；禁止 rgb_test GT 调参；勿覆盖冻结 n5k / weak_steps / #46 / #48 / #51 / #53 / #57（`results/qphi_counterfactual/`）/ #60（`results/food_id_semantic_audit/`）/ #63（`results/heldout_plan/`、`results/heldout_eval/`）/ #65（`results/heldout_energy_audit/`）/ #68（`results/heldout_massonly_energy_path/`）结果；**禁**空壳 RE_PORTION；kcal-proxy AURC 作主 claim；**禁止**把 #51/#53 或 ① 写成 Q 学得更好；**禁止**把 #51 的 kcal 下降写成身份成功；**禁止**把开发 505 当独立最终测试；**禁止**声称 #53 已证明方法；**禁止**重跑 #48 / #51 / #53 / ① / ② / ③（#63）/ Cafe2 能量审计（#65）/ Mass-only 能量路径诊断（#68）；本机 Qwen closed-loop **≠** 忠实 DietAI24 / FoodAgent。

> 真源：[`PLAN.md`](../PLAN.md)、Canvas「**VDR 官方方案（2026-11-12 晚）：面向物理约束营养估计的主动多模态证据获取（候选）**」（https://cursor.com/canvas/store-745bdf38-f3d7-4313-8d66-08e4b6903e93/d7be617d-d37e-4fe1-a4b5-9da2c86b27a2）、[`docs/VDR_P0.md`](VDR_P0.md)、[`docs/STATE_FEATURE_DESIGN.md`](STATE_FEATURE_DESIGN.md)。`CLOSED_LOOP_READY=true` 指 **Ours-old** 脚手架就绪，**不是** Ours-new 已证明。**不上 RL。不开 fast Cloud Agent。** 冻结结果不重写。#53 是开发检查点，不是方法证明。

### 服务器

**没有新的 507 / 弱标签重跑命令。** **#68（squash `f578e8f4`）、#65（squash `ecf859f5`）、#63（squash `27d1dbfa`）、#60（squash `c52ffaa2`）、#57（squash `0aa30537`）、#53（squash `2a406df4`）与 #51（squash `6c8fda02`）已落盘，不要重跑，也不要在同一 505 开发池上再调方法。不要改 `results/food_id_semantic_audit/`、`results/heldout_energy_audit/` 或 `results/heldout_massonly_energy_path/`，也不要覆盖 `results/heldout_plan/` 或 `results/heldout_eval/`。** **Mass-only 能量路径便宜诊断已完成（#68，squash `f578e8f4`）。除文首第一阶段数据可行性审计以外，没有别的当前可写代码项。不是门禁 ④。Phase 2、Phase 3 与新网络仍锁。该句曾把本次改写写成双视角实验的开工许可，2026-11-13 起失效。不要据此改 main，也不要重开已停止的双视角实验；其余代码仍空闲。实验 B、实验 C、完整联合网络与 ④–⑦ 都没有开工许可。不要改 `results/heldout_massonly_energy_path/`。不在 Cafe2 或 505 上改方法、调超参、做早停或模型选择。禁止**自动开 **④–⑦**。**不要**重跑 #48，也**不要**重跑 #63 闭环主表，也**不要**重跑 #65，也**不要**重跑 #68。**#44 的 Q 权重不再重训。不得**重训 Mass。**不得**重拟合 τ。**禁止** Cloud Agent **fast**。**P1 仍锁定 / 暂缓。禁止自动开 P1。** `results/qphi_counterfactual/` 不要覆盖。冻结文件勿覆盖，也不要重跑：

- `results/n5k_rgb_test_overhead_b0_b1.json` 与 `.jsonl`（sha256 前缀 `a4dfdd82e3a595f4…`）
- 默认的 `results/n5k_subset_b0_b1.json` 与 `.jsonl`（现在就是上面这一对）
- n=50 基线 `results/n5k_subset_b0_b1_n50_seed42.json` 与 `.jsonl`
- `results/weak_steps/rgb_train_fit_all.jsonl`（sha256 前缀 `ed47f9239bfecb3c…`）与 `results/weak_steps/rgb_train_holdout_all.jsonl`

不要用这 507 盘的 test GT 调阈值、检索、克重或碎片合并。体积工具仍关。旧回归权重仍不接入。弱标签不改 B1 先验。

历史 B0 能量 MAE **97.03**（n=507）可作 Direct-VLM 锚点。历史 B1 能量 MAE **110.72**（n=506）**不是忠实 DietAI24**，主表须按官方协议重做后再进。

### 助手进度

> **解读**：以下 #24–#34 段均为 **Ours-old（纯闭环控制器 / 弱弃权）对照**；支撑「#34 MAE 打平 → 需要 Mass + Q_φ」叙事，**不是** Ours-new 方法证明。**#46–#53** 的数字是**开发结果**。**#51/#53 增益 ≠ Q 学习改进**（① #57：rgb_train hold-out 一步评测上 Q ≈ 规则 ≪ oracle）。#53（Coverage-Constrained Active Specialist Selection）**不**证明方法已在独立测试上成立。**#63（`27d1dbfa`）** 的 Cafe2 外测**不是**能量与质量双赢。**#65（`ecf859f5`）** 的能量审计已完成，**不**改写该声称，也**不**开工改方法。**#68（`f578e8f4`）** 的 Mass-only 能量路径诊断已完成，**不**改写该声称，也**不**开工改方法。**（历史，不是现行门禁。2026-11-12 晚曾把下一步写成 Nutrition5k 双视角实验。该实验已在备份提交 28bd3c9c 中做完并停止，不要重跑，不要再合入。现行下一步只在文首：第一阶段数据可行性审计。）除文首第一阶段数据可行性审计以外，没有别的当前可写代码项。** **主声称：质量对 Direct 可外推（bootstrap 区间均为负）；对 Mass-only 基本打平（0.76 g 不算赢，不要当作方法优越性证据）；能量外测未稳。**主指标见 [`PLAN.md`](../PLAN.md)：能量 MAE + mass MAE + 真实 Tool cost 优先；宏量辅助。

**P0a 工具态审计门禁已通过（Ours-new 前置）**（2026-11-05 Asia/Shanghai，#37，squash **`2e0e180`**）。

- 定义：`p0a_tool_pre_post_dish_n`（dish 级 energy + mass；ε=**5e-5**，**未**在 hold-out / rgb_test 上拟合）。
- hold-out logged **edit**（CORRECT_CODE / RE_PORTION）：**变更率 1.0000（1106/1106）**；RE_DECOMPOSE **0** 步。  
- on-policy **edit**：**变更 0.9175（1647/1795）**、无变化 **0.0825**；**CORRECT_CODE 0.7822**、**RE_PORTION 0.9476**。  
- **门禁结论**：「工具执行但 n 未变」**非**主导模式（接口仍须 `estimate_mass`，禁空壳 RE_PORTION）。**P0b Mass 已通过（#39）**。  
- 产物：`results/closed_loop/tool_state_audit.json`、`report_tool_state_audit.txt`、`report_tool_state_audit.md`。

**P0b Mass Estimator 已通过（Ours-new 组件）**（2026-11-05 Asia/Shanghai，#39，squash **`4b6ccec`**）。

- **Backbone / 训练**：ViT-B/16（probe-val mass MAE **52.48** vs ResNet101 **57.52**）；ImageNet-1K → **rgb_train fit** 微调 **12 epoch**（last ckpt）；**RGB-only**，不读 depth；**未**用 rgb_test GT 选择/调参。  
- **头与工具**：**[M,E,P,C,F] + Laplace σ**；**`estimate_mass(image, candidates)`** → 闭环 **RE_PORTION**（`portion_src=mass_estimator`；禁空壳 RE_PORTION / 禁 RE_PORTION gram-grid 重打分）。  
- **Hold-out**（rgb_train hold-out 俯拍 RGB，**n=257**，单次评分）：mass MAE **29.6147 g**（median **18.0219**）；energy **43.2161 kcal**（median **23.0434**）；protein **4.7922** / carb **4.5685** / fat **3.2113 g**；fit-mean 基线（同盘）mass **130.6992 g** / energy **176.8194 kcal**。  
- **产物**：`results/mass_estimator/`（`weights-part-00..02.npz`、`train_meta.json`、`backbone_probe.json`、`eval_holdout.json`、`report_holdout.txt`）。**无 RL**；冻结 n5k / weak_steps **未改**。

**P0c U_* + 监督 Q_φ 已落地（Ours-new 组件）**（2026-11-05 Asia/Shanghai，#41，squash **`54c32e8`**）。

- **U_* / z**：**U_id、U_mass、U_nutrient、U_evidence** + 分歧 + 先验 **o/b**；**z=98-d**；Mass **Laplace σ**（P0b `estimate_mass`）接入 **U_mass / U_nutrient**。**无 RL**；弱 kcal-proxy **不进 z** 作主叙事。  
- **Q_φ**：规则 utility → 监督 **Q_φ**；**L=min(|Δenergy|,2500)/2500** 上的 **ΔL** 标签；**a*=argmax Q/Cost**。#24 **State-aware BC 仅消融**（hold-out 与 logged Acc **0.9607**）。  
- **Hold-out**（rgb_train weak steps logged）：Q 回归 MAE **0.0368**（零基线 **0.0518**，Pearson **0.7799**，**n=1960** 可测工具步）；Q/Cost vs heuristic **0.4489**；heuristic vs logged **0.7968**。Q/Cost 动作 **STOP 1360** / RE_PORTION **608** / RETRIEVE **499** / ABSTAIN **266** / CORRECT_CODE **38**（**2771** 行）。stub：heuristic **RETRIEVE→RE_PORTION→STOP**；Q/Cost **首步 STOP**。  
- **门禁解读**：Q **偏保守（STOP-heavy）** — **P0c 交付已完成**；Triste + 调研 **2026-11-05 选 B** → **#44 Q 校准已完成**（`3fc611d`）；**校准门禁通过** → **P0d #46 已完成**（`b9c4c5e7`）。**勿**把本段当 P0d 同预算 Full MAE 主表。  
- **产物**：`results/qphi/`。**未**读 rgb_test GT；冻结 n5k / weak_steps **未改**。

**Q/Cost terminal-margin 校准已通过（Ours-new 组件）**（2026-11-05 Asia/Shanghai，#44，squash **`3fc611d`**）。

- **方法**：fit-only **median+3·MAD** terminal-margin fence（**τ≈0.001645**）；margin 不足 → rule utility；**Learned Q 权重未改**。**无 RL**；**未**用 hold-out / rgb_test 选 τ。脚本：`scripts/qphi_calibrate.py`；`vdr/controller/q_calibration.py`。  
- **Hold-out**（rgb_train weak steps logged，**n=2771**）：Q/Cost vs rule utility **0.4489 → 0.7712**。STOP **1360→526**；RETRIEVE **499→1096**；RE_PORTION **608→842**；ABSTAIN **266→207**；CORRECT_CODE **38→100**。非 STOP：**专家调用率 0.811→0.908**；Mass|检索 **0.785→0.863**。决策：**Q/Cost 1878** / **rule utility 893**。  
- **Stub** `dish_1550704903`：未校准首步 **STOP** → 校准 **RETRIEVE→RE_PORTION→STOP**（**1** mass call，`portion_src=mass_estimator`）。  
- **门禁结论**：调研门槛 **通过** → **P0d #46 已落盘**。**不是** P0d Full-set 主表（主表见下段 P0d）。  
- **产物**：`results/qphi/calibration.json`、`calibration_holdout.json`、`report_calibration.txt`。冻结 n5k / weak_steps **未改**（#44 未 touch 弱标签 jsonl 内容）。

**P0d 同预算 Full-set 已评测（当时称 Ours-new 主表；现标开发结果，不是最终独立泛化主表）**

- **定义**：`p0d_same_budget_overhead_force_answer`（`scripts/p0d_same_budget.py`；`vdr/eval/same_budget.py`）。官方 rgb_test **507** 有俯拍；paired 四法皆可记分 **n=505**；Cafe 标签只计误差；**未**在 rgb_test 上拟合阈值。  
- **Ours-new**（Mass + **校准后** Q/Cost）：Energy MAE **98.5521** / Mass MAE **58.9952** / Tool cost mean **4.6970**。  
- **vs Ours-old**（closed-loop BC，同 paired）：ΔE **−1.8588**，ΔM **−3.4147**，Δtool **−1.8812**。  
- **vs Direct-VLM（B0）**：ΔM **−17.3513**，ΔE **+1.4210** — **能量未稳赢 B0**（质量主赢、能量主 claim 未成立）。  
- **Mass RE_PORTION**：**827** success，**0** hollow。Coverage **0.5921**；ABSTAIN **206** — **辅助**，非 pass 门禁。  
- **锚点**：B0 own-pool **n=507** E **97.0256** / M **76.4556**；closed-loop STOP **n=488** E **96.2769**。开环 BC 读同一 closed-loop trace → **E/M/tool 与 closed-loop 相同**。  
- **诚实解读**：P0d 证明 Mass + 校准 Q 在同预算下 **降 mass、降 tool、略优于 Ours-old 能量**，但 **未**在 paired Full-set 上 **稳定击败直接视觉估计的能量**；**#48 误差审计** 已解释 pooled **+1.42 kcal** 机制。多组件 food-table 一侧由 **food-id #51（`6c8fda02`）** 落盘；覆盖与早停由 **#53（`2a406df4`）** 落盘（见「助手进度」）。该表是开发结果，**不是**最终独立主表。**①②③ 已完成（#57 / #60 / #63）。Cafe2 能量审计已完成（#65，`ecf859f5`）。Mass-only 能量路径便宜诊断已完成（#68，`f578e8f4`）。（历史，不是现行门禁。2026-11-12 晚曾把下一步写成 Nutrition5k 双视角实验。该实验已在备份提交 28bd3c9c 中做完并停止，不要重跑，不要再合入。现行下一步只在文首：第一阶段数据可行性审计。）④–⑦ 仍锁。** **P1 仍暂缓。**
- **产物**：`results/p0d/report_same_budget.txt`、`same_budget.json`、`same_budget_dishes.jsonl`、`ours_new_rgb_test.jsonl`、`ours_new_rgb_test_summary.json`。冻结 n5k / weak_steps **未改**。

**P0d 误差审计已落盘（只读 #46；Ours-new 诊断）**（2026-11-05 Asia/Shanghai，#48，squash **`91a05d8`**）。

- **脚本**：`scripts/p0d_error_audit.py`；**无新训练**；**未**用 rgb_test 标签拟合阈值/码/克重。  
- **paired n=505 vs B0**：ΔM **−17.3513 g**（bootstrap 95% **[−24.94, −10.13]**，**全 draw 为负**）；ΔE **+1.4210 kcal**（95% **[−8.10, +11.05]** **跨 0**）；**median ΔE −4.54 kcal**；去掉 **2** 盘最大能量回归后 **503** 盘 mean **−0.14 kcal**（敏感性）。  
- **finished all-mass-write 199**：ΔE **−10.96** / ΔM **−39.17**；**one-comp 149** ΔE **−17.99**；**multi-comp 50** ΔE **+10.01** / ΔM **−91.80** — food-table commit E MAE **190.79** vs stored calorie head **77.75**（**grams 与 head 一致**）。此为 **#48 写前**；**#51** 后同切片 food-table 能量 MAE **131.65**。**#53** 后全部组件 mass 已写 **503/505**（见 #53 段）。  
- **unfinished 306**：对 pooled 能量均值 **+5.74 kcal**；**78 STOP unfinished** mean signed kcal err **−87.6**，对均值 **+6.06 kcal**。  
- **hold food-id fixed（481）**：mass rewrite ΔM **−20.24** / ΔE **−17.90** — **非**相对 prior grams 用能量换质量。  
- **Q fence replay 2858/2858**；505 盘 **committing actions 均为 Q/Cost**。  
- **产物**：`results/audit/report_error_audit.md`、`report_error_audit.txt`、`error_audit.json`、`paired_residuals.jsonl`、`fig_energy_gap_by_quadrant.svg`、`fig_error_channels.svg`、`fig_mass_energy_delta.svg`。冻结 n5k / weak_steps **未改**。  
- **门禁解读**：审计 **不**启动 Mass 重训 / Q 改 / P1。**2026-11-06 共决**后 **food-id #51** 与 **覆盖+早停 #53** 已完成。**现行门禁见文首：（历史，不是现行门禁。2026-11-12 晚曾把下一步写成 Nutrition5k 双视角实验。该实验已在备份提交 28bd3c9c 中做完并停止，不要重跑，不要再合入。现行下一步只在文首：第一阶段数据可行性审计。）#70 的 Phase 1 已取消；实验 B、实验 C、完整联合网络与 ④–⑦ 仍锁。** **P1 仍暂缓。禁止自动开 P1。禁止**重训 Mass/Q。

**food-id / 密度轻修已完成（多组件优先）**（2026-11-06 Asia/Shanghai，#51，squash **`6c8fda02`**）。

- **范围**：finished multi-component 上，在**已有检索窗**内做**同头 FNDDS code swap**（food-table 能量 **≥15%** 更靠近 mass-estimator calorie head；15% 为既有 density-gain cutoff，**未**在本测试集上选取）。**克重与动作保持**（不重问 selector、不重训 Mass）。标签只用于事后记分，不用于选码、阈值或克重。
- **paired n=505**：能量 MAE **98.5521 → 92.6963** kcal；质量 MAE **不变 58.9952 g**。vs Direct-VLM **ΔE +1.42 → −4.43** kcal。
- **多组件全写 n=50**：food-table 能量 MAE **190.79 → 131.65**；vs Direct **+10.01 → −49.13**。换码 **37/50**（绝对能量误差 **30↓ / 7↑**），**68** component codes。stored calorie head 仍 **77.75**（swap 的目标；记分 commit 仍是 food-table）。
- **Mass RE_PORTION**：**827** 成功，**0** hollow（成功且非 mass estimator 的 portion = 0；下一步营养不变 = 0）。swap 不新增 portion 步。
- **未做**：Mass 重训；Q 权重与 terminal-margin fence 未改、未重拟合；无检索重排、无新候选列表；覆盖与早停未动；**本文件在 #51 未改**。
- **产物**：`results/food_id_density/report.md`、`report.txt`、`summary.json`；脚本 `scripts/food_id_density_align.py`。冻结 n5k / weak_steps **未改**。
- **门禁解读**：**不**解锁 Mass 重训 / Q 改 / **P1**。Triste 顺序的下一项 **覆盖未写组件 + 早停** 已由 **#53（`2a406df4`）** 完成。**现行门禁见文首：①②③、Cafe2 能量审计与 Mass-only 能量路径诊断已完成；（历史，不是现行门禁。2026-11-12 晚曾把下一步写成 Nutrition5k 双视角实验。该实验已在备份提交 28bd3c9c 中做完并停止，不要重跑，不要再合入。现行下一步只在文首：第一阶段数据可行性审计。）禁止自动开 ④–⑦。** 语义审计已由 **#60** 完成。**kcal 变好 ≠ 身份变对。不**在本段重跑或覆盖 `results/food_id_density/`。

**覆盖约束已完成（Coverage-Constrained Active Specialist Selection；Ours-new 开发检查点，不证明方法）**

- **方法**：冻结 Q 权重与 fit-set terminal-margin fence **不变**。在该选择之后，未写组件若没有 `estimate_mass` 写，**不能**提交 STOP/ABSTAIN，改为已有 rule utility。码替换若已回到检索 top code，则做 **portion**，不再开下一轮 swap；离开 top code 的一次 swap 保持。克重仍来自 `estimate_mass`。**#51 同头 swap** 在多组件盘写完后仍运行。**无** Mass 重训、**无** Q 重训、**无** τ 重拟合。标签只用于事后记分。**#53 未改本文件**。
- **paired n=505**（相对 **#51** 基线）：能量 MAE **92.6963 → 64.2792** kcal；质量 MAE **58.9952 → 32.9908** g；tool cost mean **4.6970 → 6.0257**（max **17**；**505/505** 在 K=32 内）。
- **vs Direct-VLM**：ΔE **−4.43 → −32.85** kcal；ΔM **−17.35 → −43.36** g。
- **覆盖**：全部组件 mass 已写 **199/505 → 503/505**；STOP 且仍有未写组件 **78 → 0**；Mass RE_PORTION 成功 **827 → 1479**；hollow **0**。
- **残留 2 盘**：`dish_1565020895`（7 个组件里 5 个已写）、`dish_1565020926`（8 个里 7 个已写）。`honeycomb` 的 FNDDS 检索失败，状态不变，既有三步 stuck guard 停在 **RETRIEVE**。没有 code 时 RE_PORTION 不合法，**不编造克重**。两盘都 **没有 STOP**。
- **未做**：Mass checkpoint、Q 权重、τ **未改**；无 P1 检索重排；无 RL；**未**用 rgb_test 标签选阈值。冻结 n5k / weak_steps **未改**。
- **产物**：`results/coverage_early_stop/report.md`、`report.txt`、`summary.json`、`continued_steps.jsonl`、`paired_scores.jsonl`。脚本 `scripts/coverage_early_stop.py`；`vdr/controller/coverage.py`。
- **门禁解读**：**不**解锁 Mass 重训 / Q 改 / **P1**。增益来自覆盖约束，**不是** Q 变强，**不**证明方法。**①（#57）** 在另一次一步评测上给出 **Q ≈ 规则 ≪ oracle**，加固该归因。**现行门禁见文首：①②③、Cafe2 能量审计与 Mass-only 能量路径诊断已完成；（历史，不是现行门禁。2026-11-12 晚曾把下一步写成 Nutrition5k 双视角实验。该实验已在备份提交 28bd3c9c 中做完并停止，不要重跑，不要再合入。现行下一步只在文首：第一阶段数据可行性审计。）禁止自动开 ④–⑦。** **禁止**自动开 **④–⑦** / **P1** / Mass 或 Q 重训。**不**在本段开工，也**不**在这 505 盘上再调方法。

**① 反事实 Q + dish_id 泄漏审计已完成（一步评测；不重训）**（2026-11-06 Asia/Shanghai，#57，squash **`0aa305379a7cc02045d07ec9c513bb9c2a3d3061`**，短 **`0aa30537`**）。

- **协议**：冻结 rgb_train hold-out 弱步骤状态上做一步分支。这些盘没有驱动 #51 换码或 #53 覆盖门（那两项读的是 rgb_test 开发 **505**）。也不是训练 Q 或设定 τ 的 fit 行。未写完且没有 mass-estimator 写时，**A_legal** 去掉 STOP/ABSTAIN。同一状态上每个合法动作执行一次。Oracle 最小化动作后的能量绝对误差。质量只并行记录，不是选择目标。Q 是冻结 checkpoint + 既有 terminal-margin fence + 覆盖门。**τ 不重拟合。无 RL。不读 rgb_test 标签。**
- **泄漏**：fit / hold-out **按 dish_id 分组**（`int(sha256(dish_id).hexdigest()[:8], 16) % 10 == 0` → hold-out，否则 fit）。fit **3679** 盘 / **25864** 步；hold-out **380** 盘 / **2771** 步。两侧 dish 交集 **0**，落在两侧的行 **0**。结论 `no_dish_leakage`。**没有**在审计之后重训 Q、Mass 或 τ。
- **头条** n=**1914**（coverage-choice 且有冻结 RGB mass forward，RE_PORTION 才算真实专家调用；与 #53 **505** 交集 **0**；样本里 rgb_test id **0**）：
  - Q Best-ID **0.6165**，ΔE improve **24.14**，E regret **49.96**
  - Rule **0.6102** / **23.03** / **50.91**
  - Fixed **0.5914** / **20.62** / **52.99**
  - Random **0.4338** / **4.04** / **78.15**
  - Oracle regret **0**，improve **82.19**
- Q 略领先规则（约 **0.96** kcal regret），约 **50** kcal 落后 oracle。**241/1914** 的 oracle 最优是 **RE_DECOMPOSE** 空操作：冻结 Q **不能**发出该动作（fit 弱步骤没记录过，该头被掩掉）。去掉 no-op 的敏感性：Q regret **~17.33** vs 规则 **~18.28**。**不**因此改 Q。
- **价值校准** n=**3475**：MAE(Q, ΔL) **0.0226**，Pearson **0.747**。RE_PORTION Pearson **0.805**，CORRECT_CODE **0.016**。斜率没有写回 fence 或权重。
- **归因**：**Q ≈ 规则 ≪ oracle**。这**加固** #53 的能量下降来自覆盖约束，**不是** Q 学得更好。一步分支，不是全程 rollout，也不是开发 505 的 MAE。**不**证明方法。
- **未做**：Mass checkpoint、Q 权重、τ **未改**；无 P1；无 RL；**未**用 rgb_test 标签选阈值。冻结 n5k / weak_steps / #46 / #48 / #51 / #53 **未改**。**#57 未改本文件。**
- **产物**：`results/qphi_counterfactual/report.md`、`report.txt`、`summary.json`、`decisions.jsonl`、`leakage_audit.md`、`leakage_audit.json`。
- **门禁解读**：本条数字**不**解锁 **④–⑦**，**不**解锁 Mass 重训 / Q 改 / **P1**，也**不**定义 held-out。**② 已由 #60 完成**，不在本段重开。**③ 已由 #63（`27d1dbfa`）完成。** Cafe2 能量审计已由 #65（`ecf859f5`）完成。Mass-only 能量路径便宜诊断已由 #68（`f578e8f4`）完成。**（历史，不是现行门禁。2026-11-12 晚曾把下一步写成 Nutrition5k 双视角实验。该实验已在备份提交 28bd3c9c 中做完并停止，不要重跑，不要再合入。现行下一步只在文首：第一阶段数据可行性审计。）**（见文首；**不是门禁 ④**，**禁止**自动开 **④–⑦**）。**不**在本段重跑或覆盖 `results/qphi_counterfactual/`。

**② food-id 语义审计已完成（只读 #51；自动标签仅建议）**（2026-11-07 Asia/Shanghai，#60，squash **`c52ffaa2`**）。

- **来源**：只读 #51 `results/food_id_density/`（squash **`6c8fda02`**）。从冻结轨迹收回 **68** 个同头换码，覆盖 **37/50** 多组件盘。克重与动作保持。**无** Mass / Q / τ 重训。**无 RL。** 不重跑 #51 / #53 / #57。**#60 未改本文件。**
- **身份（自动标签，仅建议）**：换码后 Exact **0** / Near **50**（**73.53%**）/ Wrong **18**（**26.47%**）。换码前同一 cafe 行：Exact **27** / Near **23** / Wrong **18**。Exact→Near **27**。自动身份变差 **28**、变好 **1**、不变 **39**。
- **宏量（food-table × 存克 vs cafe）**：finished 多组件 **n=50**：蛋白质 **10.57→10.07**，碳水 **13.00→16.17**，脂肪 **17.56→11.97**，能量 **190.79→131.65**（能量与冻结 #51 一致）。换码 **37** 盘：碳水 MAE **+4.28 g**，能量 MAE **−79.92 kcal**。
- **kcal 变好 ≠ 身份变对**：能量绝对误差下降的 **30** 盘换码盘里，全自动 Exact **0**；至少一码 Wrong **12**；仅 Near **18**。这 **30** 盘上的 **50** 个换码全部是 Near 或 Wrong。**不**把 #51 的 kcal 下降写成身份成功。
- **人审**：全部 **68** 个组件行 `needs_human_review=yes`。自动标签**不**关闭审计。文件：`human_review_components.csv`、`human_review_dishes.csv`、`human_review.md`。人审包仍开放。人审**不**改写已完成的 ③ 协议，也**不**改写已完成的 Cafe2 能量审计（#65），也**不**改写已完成的 Mass-only 能量路径诊断（#68）。
- **产物**：`results/food_id_semantic_audit/`；脚本 `scripts/food_id_semantic_audit.py`。不要重跑，不要改该目录。
- **门禁解读**：本条**不**解锁 **④–⑦**，**不**解锁 Mass 重训 / Q 改 / **P1**，也**不**定义 held-out。**③ 已由 #63（`27d1dbfa`）完成。** Cafe2 能量审计已由 #65（`ecf859f5`）完成。Mass-only 能量路径便宜诊断已由 #68（`f578e8f4`）完成。**（历史，不是现行门禁。2026-11-12 晚曾把下一步写成 Nutrition5k 双视角实验。该实验已在备份提交 28bd3c9c 中做完并停止，不要重跑，不要再合入。现行下一步只在文首：第一阶段数据可行性审计。）**（见文首；**不是门禁 ④**，**禁止**自动开 **④–⑦**）。

**③ Cafe2 外部 held-out 已完成（冻结 #53+#51；不证明双赢）**（2026-11-09 Asia/Shanghai，#63，squash **`27d1dbfa`**）。

- **划分**：`results/heldout_plan/`。Cafe2，**n=228**，与配对 **505**、fit（**3679** 官方 rgb_train 且 `pool_of_dish==fit`）、rgb_train、rgb_test、cafe1 的 dish_id **不相交**。
- **主表与轨迹**：`results/heldout_eval/`。三法皆有限，**n=228**。开发 505 能量 MAE **64.2792**、质量 MAE **32.9908** **不是**本表的行。
- **Coverage-Constrained + density/same-head**：能量 MAE **116.8494** kcal，质量 MAE **50.9483** g，覆盖 **0.9737**（222/228），步数均值 **6.2193**，工具调用均值 **5.2456**。
- **Direct-VLM**：能量 **105.6157**，质量 **73.7456**。**Mass-only**（冻结 Mass）：能量 **75.3870**，质量 **51.7107**。
- 能量 MAE 比 Direct-VLM **高 11.23** kcal，比 Mass-only **高 41.46** kcal。质量 MAE 比 Direct-VLM **低 22.80** g，比 Mass-only **低 0.76** g。
- STOP 且仍有未写组件：**0**。**6** 盘停在 **RETRIEVE**，组件仍未写完，**没有编造份量**。同头密度换码 **106** 盘。空壳 RE_PORTION **0**。工具调用中位 **4**、最大 **14**，**228/228** 在 K=32 内。Mass 写：**575**。
- Cafe2 盘级能量列每行都是 **0**，能量目标是配料卡路里之和。质量是盘级克重总量。
- **诚实限制**：这次 Direct-VLM 行记下 `placement_max_memory=8GiB`（共享 GPU）。同一 bf16 checkpoint 与 prompt，**不是**历史上满 GPU 的 `cuda:0` 放置。**不要重跑。不要据此解锁重训。**
- **主声称：质量稳赢可外推；能量开发集赢、外测未稳。** **不**声称 held-out 上能量与质量双赢。**#53 不证明方法。** #51 是密度 / kcal 启发式（自动 Exact=**0**）。一步审计上 **Q ≈ 规则 ≪ oracle**。
- **门禁解读**：本条**不**解锁 **④–⑦**，**不**解锁 Mass / Q / τ 重训，**不**开 **P1**。**不要重跑，不要覆盖** `results/heldout_plan/` 与 `results/heldout_eval/`。Cafe2 能量审计已由 **#65**（`ecf859f5`）完成。Mass-only 能量路径便宜诊断已由 **#68**（`f578e8f4`）完成。**（历史，不是现行门禁。2026-11-12 晚曾把下一步写成 Nutrition5k 双视角实验。该实验已在备份提交 28bd3c9c 中做完并停止，不要重跑，不要再合入。现行下一步只在文首：第一阶段数据可行性审计。）**（见文首；**不是门禁 ④**）。

**Cafe2 能量审计已完成（只读冻结 #63；不证明双赢，不开方法）**（2026-11-09 Asia/Shanghai，#65，squash **`ecf859f5`**）。

- **产物**：`results/heldout_energy_audit/`。脚本 `scripts/heldout_energy_audit.py`。**#65 未改本文件。** 只读冻结 Cafe2 **n=228**。**没有**重跑 #63 闭环、Direct-VLM 或 Mass-only。**没有**覆盖 `results/heldout_plan/`、`results/heldout_eval/` 或其他冻结结果目录。**无** Mass / Q / τ 重训。**无 RL。** **不要重跑，不要改该目录。**
- **#63 主表不变**：Coverage-Constrained 能量 MAE **116.8494** kcal，质量 MAE **50.9483** g；Direct-VLM **105.6157** / **73.7456**；Mass-only **75.3870** / **51.7107**。能量相对 Direct **+11.23**，相对 Mass-only **+41.46**。质量相对 Direct **−22.80**，相对 Mass-only **−0.76**。覆盖 **0.9737**（222/228）。开发 505 的 **64.2792** / **32.9908** **仍不是**该表的行。
- **222**/228 盘上，已提交克重与 Mass-only 克重相差小于 0.001 g。整段 **−0.76** g 质量优势就是停在 RETRIEVE 的 **6** 盘（这 6 盘对 +41.46 均值的能量贡献是 **−0.52** kcal）。
- **+41.46** kcal 的差距是食物表密度打在已经与 Mass-only 对齐的克重上。Mass-only 用冻结质量估计器的 calorie head 给同一克重记分。
- 码保持固定，相对 100 g 网格的 mass 重写：能量 MAE **195.6564 → 125.2060** kcal，质量 MAE **116.9123 → 50.9483** g。汇合下降来自多组件（**216.37 → 115.21**）。在 **58** 盘单组件上，同一重写把能量 MAE 从 **134.93** 抬到 **154.51**，质量仍近 51 g。这 58 盘贡献 +41.46 均值里的 **+29.62** kcal。
- Oracle 标签密度 × 我方克数：能量 MAE **98.6201** kcal，低于食物表 commit，也低于 Direct-VLM，仍高于 Mass-only **75.3870**。不加权 kcal/g MAE：commit **0.631**，calorie head **0.753**；head 的抵消更大（**77.8** vs **58.2** kcal）。
- #51 密度换码开 vs 关：在 **106** 盘上触发。全表能量 MAE 关 **125.2060** / 开 **116.8494**（**−8.3566** kcal）。质量不变。**67** 盘更靠近 cafe 标签，**39** 盘更远。未换码的 **122** 盘贡献 +41.46 里的 **+38.30** kcal。#51 仍是密度 / kcal 启发式（开发集自动 Exact 为 **0**）。
- Bootstrap **10,000** 次，种子 **0**：相对 Mass-only 的能量差 **[+25.35, +58.28]** kcal（每次抽样均为正）。相对 Direct-VLM 的质量差 **[−29.80, −15.66]** g（每次抽样均为负）。相对 Direct 的能量差 **[−6.22, +29.16]** 含两种符号。相对 Mass-only 的质量差 **[−2.33, +0.10]** 含两种符号。
- **主声称仍是：质量稳赢可外推；能量开发集赢、外测未稳。** **不**声称能量与质量双赢。**不**从这些标签开工改方法。
- **门禁解读**：本条**不**解锁 **④–⑦**，**不**解锁 Mass / Q / τ 重训，**不**开 **P1**。Mass-only 能量路径便宜诊断已由 #68（squash `f578e8f4`）完成。**（历史，不是现行门禁。2026-11-12 晚曾把下一步写成 Nutrition5k 双视角实验。该实验已在备份提交 28bd3c9c 中做完并停止，不要重跑，不要再合入。现行下一步只在文首：第一阶段数据可行性审计。） 不是门禁 ④，也不是重开本条诊断。** 双视角实验的开工许可已退役；现行许可在文首的第一阶段审计，不在本条。本文件这次改写不是本条诊断的重开许可。

**Mass-only 能量路径便宜诊断已完成（只读冻结 Cafe2；更大的有符号抵消，不开方法）**（2026-11-10 Asia/Shanghai，#68，squash **`f578e8f4`**）。

- **产物**：`results/heldout_massonly_energy_path/`（`report.md`、`report.txt`、`summary.json`、`dishes.jsonl`、`paths.csv`、`slices.csv`、两张图 `fig_density_vs_mae.svg` 与 `fig_gap_split.svg`）。脚本 `scripts/heldout_massonly_energy_path.py`。**#68 未改本文件。** 只读冻结 Cafe2 held-out，**n=228**。**没有**重跑 #63 闭环、Direct-VLM 或 Mass-only，也**没有**重跑 #65。**没有**覆盖 `results/heldout_plan/`、`results/heldout_eval/`、`results/heldout_energy_audit/`，也**没有**覆盖其他冻结结果目录。**无** Mass / Q / τ 重训。**无 RL。** **不要重跑，不要改该目录。**
- **记分路径**：Mass-only 的能量是冻结质量估计器的盘级 `energy_kcal` 头，来自 `vit_b_16` checkpoint，目标 `mass_g`、`energy_kcal`、`protein_g`、`carb_g`、`fat_g`。记分器存的是 `estimate_mass(...).nutrients.energy_kcal`。它不读 food-id，也不把 FNDDS kcal/100 g 乘以克重。
- 在 **222 / 228** 盘上，质量头与已提交克重相差在 0.001 g 以内（测得最大 **0.0002** g）。另外 **6** 盘停在 RETRIEVE；Mass-only 仍给整盘头记分。
- **98.6201** 对 **75.3870**（结论：更大的有符号抵消，不是更准的密度或食物码；Mass-only 不用 food-id）：Oracle 标签密度 × 我方克数的能量 MAE **98.6201**。Mass-only 能量头 **75.3870**。差距 **23.2332** kcal。不加权 kcal/g MAE：标签密度 **0**，食物表 commit **0.631**，calorie head **0.753**。#65 通道抵消（kcal）：oracle **0**，食物表 **58.2**，head **77.8**。保持真密度、换成 Mass-only 克重，能量 MAE 从 **98.6201** 升到 **100.3152**（**+1.6951**）。在这些克重上，把真密度换成能量头，能量 MAE 从 **100.3152** 降到 **75.3870**（**−24.9283**）。净差是 **23.2332** kcal。在头自己的克重上，把头的密度换成标签密度，会把能量 MAE 从 **75.3870** 抬到 **100.3152**。食物表在 kcal/g 上更近的有 **139** 盘；头更近的有 **89** 盘。没有一盘的头密度误差低于 **0.0001** kcal/g。**58** 盘单组件贡献 **+23.23** 均值差距里的 **+19.85**。这一片上头的 kcal/g MAE 是 **1.023**，食物表是 **0.811**；能量 MAE 是 **38.06**，对照食物表 **154.51** / oracle **116.08**。**6** 盘 RETRIEVE 贡献 **−1.56**。按盘 bootstrap **10,000** 次，种子 **0**，把 **98.6201 − 75.3870** 放在 **[+7.92, +39.06]** kcal。**99.83%** 的抽样保持正均值。这个区间没有用来选方法。
- **冻结数字保持原样。** #63 / #65 已写下的数不改，包括 **116.8494** / **50.9483**，**105.6157** / **73.7456**，**75.3870** / **51.7107**，**+11.23**，**+41.46**，**−22.80**，**−0.76**，**0.9737**（222/228），**98.6201**，**0.631** vs **0.753**，**77.8** vs **58.2**，开发 505 的 **64.2792** / **32.9908**（**仍不是** held-out 表的行），码固定重写 **195.6564 → 125.2060**，**58** 盘单组件 **134.93 → 154.51** / **+29.62**，换码 **−8.3566**，未换码 **122** 盘 **+38.30**，以及 #65 的 bootstrap 区间 **[+25.35, +58.28]**、**[−29.80, −15.66]**、**[−6.22, +29.16]**、**[−2.33, +0.10]**。相对 Mass-only 的 **−0.76** g 仍只是停在 RETRIEVE 的 **6** 盘（#65 / #66 已记下）。**不**把它写成质量赢。**不**把只修 food-id 写成能量双赢。
- **主声称仍是：质量稳赢可外推；能量开发集赢、外测未稳。** **不**声称能量与质量双赢。**不**从这些标签开工改方法。
- **门禁解读**：本条**不**解锁 **④–⑦**，**不**解锁 Mass / Q / τ 重训，**不**开 **P1**。**（历史，不是现行门禁。2026-11-12 晚曾把下一步写成 Nutrition5k 双视角实验。该实验已在备份提交 28bd3c9c 中做完并停止，不要重跑，不要再合入。现行下一步只在文首：第一阶段数据可行性审计。） 不是门禁 ④，也不是重开本条诊断。** 双视角实验的开工许可已退役；现行许可在文首的第一阶段审计，不在本条。本文件这次改写不是本条诊断的重开许可。已执行的协议见文首，不是新开工。冻结后写作（「Coverage+Mass，能量外测未稳」）对比只在 505 上改 food-id、再重测 Cafe2，仍要等以后的调研阅读 + Triste 点头 + 再一次 STATUS 改写。

**闭环 State-aware BC 已落地（Ours-old 组件）**（2026-11-04 Asia/Shanghai，自托管 `gpu-12-container`，分支 `cursor/closed-loop-bc-20261104`）。

- 数据：`vdr/controller/closed_loop_data.py`（fit / hold-out 弱步骤；`o_prev` 为上一步 IR；`b_t.tools_used=step_i`）。
- 策略：`vdr/controller/policy_state_mlp.py`（79-d → 64 → 64 → 6，与 B5 容量对齐）。
- 训练/评测：`scripts/closed_loop_bc_train.py`；冒烟 `scripts/closed_loop_smoke.py` + `scripts/bc_mlp_smoke.py`（B5 路径仍 31-d）。
- 产物（**新路径**，未改冻结 weak_steps / n5k）：`results/closed_loop/metrics_holdout.json`、`report_holdout.txt`、`policy_state_mlp.npz`。
- **Full fit** 25864 行 / hold-out 2771 行，CPU ~14s fit。Hold-out（mask none）：overall Acc **0.9607**（次要）；**ABSTAIN F1=0.5091**（14/24）；per-class PR/F1 见 json。**H2 mask-zero**：Acc **0.7037**，**ABSTAIN F1=0.1212**（obs 块有关）。mask-shuffle ABSTAIN F1 **0.0870**。这不是 Full-set MAE / AURC 主表证明。
- `CLOSED_LOOP_READY=true`。B5 开环 MLP hold-out Acc **0.9700** / ABSTAIN F1=0 仍是消融对照，不是主方法。

**FixRate 主定义已评测**（2026-11-05 Asia/Shanghai，#26，squash `7e3084f`）。

- 主定义：`offline_policy_matched_terminal_counterfactual`（`scripts/closed_loop_fixrate.py`；冒烟 `scripts/closed_loop_fixrate_smoke.py`）。
- Hold-out：FixRate **0.7731**，WorsenRate **0.2177**，Δ **0.5554**，95% CI **[0.4818, 0.6280]**（ε=1 kcal，按 dish 聚类 bootstrap B=10000）。
- 工具成本 K=32：mean **6.29** / median **5**；**380/380** 在预算内。
- 产物：`results/closed_loop/fixrate_holdout.json`、`report_fixrate.txt`。
- 说明：FixRate 不是 Full-set 能量 MAE / Coverage·Selective MAE / AURC 主表；consecutive-IR 诊断不是主指标。FixRate 为正仍**不能**单独当完整方法证明（须与 MAE/AURC 等一起看；MAE/AURC 主表见下段 #28）。

**主表 MAE/AURC（离线 logged）已评测**（2026-11-05 Asia/Shanghai，#28，squash `e88afc8`）。

- 主定义：`offline_logged_horizon_force_answer`（`vdr/eval/energy_selective.py`；脚本 `scripts/closed_loop_mae_aurc.py`，冒烟 `scripts/closed_loop_mae_aurc_smoke.py`）。
- 人群：`rgb_train` hold-out 上 logged **STOP** 且终态 `s∈{pass,corrected}`，**n=356**（cafe **train** 标签能量；**不是** `rgb_test`）。**不是** on-policy VLM rollout。
- 产物：`results/closed_loop/mae_aurc_holdout.json`、`report_mae_aurc.txt`。
- 四行主表（kcal；**AURC 越低越好**）：
  - **Ours（mask none）**：Full MAE **30.7085**，Coverage **0.9522**（339/356），Selective MAE **31.0032**，AURC **58.3726**
  - **B5 开环**：Full MAE **30.7085**，Coverage **1.0**，Selective MAE **30.7085**，AURC **32.1474**
  - **Ours mask-zero**：Full MAE **30.7085**，Coverage **0.0225**，Selective MAE **55.0052**，AURC **41.1197**
  - **Ours mask-shuffle**：Full MAE **80.3964**（n=354），Coverage **0.6949**，Selective MAE **100.9028**，AURC **101.2660**（sanity 退化）
- Force-answer kcal 在 Ours / B5 / mask-zero **相同**（同一 logged 终端总热量；0 次提前 STOP/ABSTAIN 决策）。老师终端 audit MAE 亦 **30.7085**。
- **诚实解读（#28 落盘快照）**：Full-set MAE **打平**——本离线定义**不能**声称 Ours 在能量上优于 B5。该快照中 Ours AURC **58.3726 劣于** B5（pre-A）；Selective MAE 略高于 Full-set。**#31 A 后 Ours AURC 见下段**。**B0 97.03 / 非忠实 B1 110.72 不是本表行**（那是 rgb_test 冻结锚点）。overall Acc / FixRate **不是**本表指标；勿把 `offline_logged_horizon_force_answer` 写成 on-policy rollout。

**弃权校准 A 已落地**（2026-11-05 Asia/Shanghai，#31，squash `aec2006`）。

- 仍用主定义 `offline_logged_horizon_force_answer`；primary hold-out logged STOP pass/corrected **n=356**（与 #28 相同人群；**offline logged**，**不是** on-policy `rgb_test`）。
- 动作头未重训：仍是 `results/closed_loop/policy_state_mlp.npz`（#24）。FixRate（#26）**未**重跑（动作命中不变）。
- Selective 排序：`abstain_score=clip(committed_kcal/2500,0,1)`，仅在 fit-internal val 上选定（**禁止** hold-out / rgb_test GT 调参）。Softmax P(ABSTAIN) 参照仍复现 #28 数字。
- 相对 #28：**Full MAE 仍 30.7085**（tie；0 early STOP）；Coverage / Selective MAE **不变**；**Ours AURC 13.5311**（#28 前 **58.3726**）**≤ B5 32.1474**。
- 产物：`results/closed_loop/abstain_calibration.json`；更新 `results/closed_loop/mae_aurc_holdout.json`、`report_mae_aurc.txt`。
- **诚实解读**：这是 hold-out 离线 logged 表上 selective **AURC 对齐** B5，**不是** Full-set MAE 优于 B5，**不是** on-policy 能量主表证明；on-policy 主表见下段 **P1=B**。

**P1=B on-policy rgb_test 能量主表已评测**（2026-11-05 Asia/Shanghai，#34，merge `75b80a1`）。

- 主定义：`on_policy_vlm_horizon_force_answer`（与 hold-out `offline_logged_horizon_force_answer` **不同**；勿混表）。
- 池：`rgb_test` 有俯拍 `rgb.png` 的盘，**507/507**；官方 709 id 中 **202** 无图、**未** impute。
- VLM：本机 **Qwen3.8-27B-VL**，`llama-server` **:18080**（GPU0）；closed-loop 骨干 only — **不是** 忠实 DietAI24 / FoodAgent（**C**）。
- 策略：`results/closed_loop/policy_state_mlp.npz` 驱动每一步；**3822** steps；终端 **STOP 488** / **ABSTAIN 9** / stuck **RETRIEVE 10**；**0** VLM failures。
- Primary 人群：on-policy 终端 **STOP** 且 `s∈{pass,corrected}` 且 finite `nutri_dish`，**n=488**。
- 主表（kcal；**AURC 越低越好**）：
  - **Ours mask-none**：Full MAE **96.2769**，Coverage **1.0**（488/488），Selective MAE **96.2769**，AURC **47.1737**
  - **B5 开环 counterfactual（同 trace）**：Full MAE **96.2769**，Coverage **1.0**，Selective MAE **96.2769**，AURC **81.5857**
  - **Ours mask-zero**：Full MAE **96.2769**，Coverage **0.0**，AURC **47.1737**
  - **Ours mask-shuffle**：Full MAE **115.4178**（n=487），Coverage **0.6448**，Selective MAE **129.1560**，AURC **71.5459**
- Force-answer kcal 在 Ours / B5 / mask-zero **相同**（early STOP **0**）。**Full MAE 与 B5 打平** — **勿声称** on-policy 能量赢 B5。
- **Ours AURC 优于 B5**：selective 排序仍用 #31 fit-internal 锁定的 `abstain_score=clip(committed_kcal/2500,0,1)`（**未**用 rgb_test GT）。
- 产物：`results/closed_loop/on_policy_rgb_test.jsonl`、`on_policy_rgb_test_summary.json`、`mae_aurc_testsplit.json`、`report_mae_aurc_testsplit.txt`。
- **与 offline hold-out 分开**：#28/#31 数字仍在 `mae_aurc_holdout.json`（**n=356**，MAE **30.7085** / Ours AURC **13.5311**）。**B0 97.03**（n=507 Direct-VLM 锚点）**不是** 与 96.2769（488 STOP）同人群对照行。历史 B1 **110.72** 仍 **不是** 忠实 DietAI24。冻结 n5k jsonl sha256 前缀 **`a4dfdd82e3a595f4…`**、fit weak_steps **`ed47f9239bfecb3c…`** — **#34 未改**。

**P1=B VLM 授权**（2026-11-05 Asia/Shanghai，Triste 群聊）：闭环 rollout 已用本机 **千问3.8 27B 视觉**（Qwen3.8-27B-VL）；见上段 #34 产物。

**P0 冒烟 / B1-B2 脚手架**（#19 / #22）仍有效：`CLOSED_LOOP_READY` 以前为 false 的记录保留在历史段。B1 scaffold / B2 approx 仍等 live API。

### 助手 / 本地（优先）

1. **双视角实验已停止，不是现行代码项（#70 Phase 1 已取消。Mass-only 能量路径诊断 #68，squash `f578e8f4` 与 Cafe2 能量审计 #65，squash `ecf859f5` 已完成）**：不要据此改 main，也不要重开已停止的双视角实验；其余代码仍空闲。该句曾把本次改写写成双视角实验的开工许可，2026-11-13 起失效。实验 B、实验 C 与完整联合网络、已取消的 Phase 1、Q/Mass/τ 重训、food-id 改写、505 或 Cafe2 调新方法、P1 与门禁 ④–⑦ 都没有开工许可。不是门禁 ④。Phase 2、Phase 3 与新网络仍锁。①②③ 已完成（#57 `0aa30537`、#60 `c52ffaa2`、#63 `27d1dbfa`）。**④–⑦ 锁定，禁止自动开。** **不要**重跑 #48 / #51 / #53 / #57 / #60 / #63 / #65 / #68，也不要改 `results/food_id_semantic_audit/`、`results/heldout_energy_audit/` 或 `results/heldout_massonly_energy_path/`，也不要覆盖 `results/heldout_plan/` 或 `results/heldout_eval/`。自动标签只是建议；**68** 行 `needs_human_review=yes`，人审包仍开放。**kcal 变好 ≠ 身份变对。** #51 是密度 / kcal 启发式（自动 Exact=**0**）。**Q ≈ 规则 ≪ oracle**。**主声称：质量对 Direct 可外推（bootstrap 区间均为负）；对 Mass-only 基本打平（0.76 g 不算赢，不要当作方法优越性证据）；能量外测未稳。** **不**声称双赢。**不**从 #65 或 #68 的标签开工改方法。**不**把只修 food-id 写成能量双赢。**不**把相对 Mass-only 的 **−0.76** g 写成质量赢（那一点只在 6 盘 RETRIEVE；#65 / #66 已记下）。**P1 仍未解锁 / 暂缓。禁止自动开 P1。禁止**重训 Mass/Q/τ。见文首与 Checklist。**#48 `results/audit/`、#51 `results/food_id_density/`、#53 `results/coverage_early_stop/`、#57 `results/qphi_counterfactual/`、#60 `results/food_id_semantic_audit/`、#63 `results/heldout_plan/` 与 `results/heldout_eval/`、#65 `results/heldout_energy_audit/`、#68 `results/heldout_massonly_energy_path/` 只读**。**禁止** Cloud Agent **fast**；**禁**空壳 RE_PORTION。**无 RL。**
2. **覆盖未写组件 + 早停 #53 已完成**（squash `2a406df4`）。paired n=505 能量 **64.2792** / 质量 **32.9908** / tool mean **6.0257**（max **17**，均在 K=32）。全部组件已写 **503/505**；STOP-unwritten **0**；Mass 成功 **1479**；hollow **0**。残留 2 盘 honeycomb 检索失败，不编造克重、未 STOP。
3. **food-id / 密度轻修 #51 已完成**（squash `6c8fda02`）。calorie head **77.75** 是目标读数；记分 commit 仍是 food-table。
4. **#34 / #31 / #28 已冻结为 Ours-old 对照**；勿声称 on-policy 能量赢 B5。
5. **P1 未解锁 / 暂缓**：B1/B2 忠实 API **不得**跳步开 retrieval-rerank。
6. **禁止：** 自动进 **P1** 或 Mass/Q/τ 训练；自动开 **④–⑦**（**Cafe2 能量审计已完成**；**Mass-only 能量路径便宜诊断已完成**；**（历史，不是现行门禁。2026-11-12 晚曾把下一步写成 Nutrition5k 双视角实验。该实验已在备份提交 28bd3c9c 中做完并停止，不要重跑，不要再合入。现行下一步只在文首：第一阶段数据可行性审计。）不是门禁 ④。Phase 2、Phase 3 与新网络仍锁**，不得据此开 ④–⑦，也不得用该审计或该诊断做方法选择、调超参、早停、模型选择、换方法、阈值 / τ、GT 驱动的码 / 份量修改，或开工改方法）；重跑 Cafe2 能量审计（#65 已落盘）；覆盖 `results/heldout_energy_audit/`；重跑 Mass-only 能量路径诊断（#68 已落盘）；覆盖 `results/heldout_massonly_energy_path/`；把 FixRate / AURC / #44 hold-out 动作表当 Ours-new Full-set 主证明；混 offline n=356 与 on-policy n=488；空壳 RE_PORTION；RL；fast Cloud Agent；test GT 调参；覆盖冻结文件；**110.72** 当忠实 DietAI24；**勿**重训 Q 权重覆盖 #44 校准产物；重跑 507 / P0d Full-set（#46 已落盘）；重跑误差审计（#48 已落盘）；重跑 food-id 对齐（#51 已落盘）；重跑覆盖+早停（#53 已落盘）；重跑反事实 Q（#57 已落盘）；重跑 food-id 语义审计（#60 已落盘）；重跑 ③ 闭环主表（#63 已落盘）；覆盖 `results/heldout_plan/` 或 `results/heldout_eval/`。

### 协作

docs-first：改下一步先改本文件并推仓。Cloud Agent **永远不开 fast**。**2026-11-09**：**③（#63，`27d1dbfa`）已完成**。同日晚，Triste 点头「继续吧」解锁了 Cafe2 能量审计；该诊断已由 **#65（`ecf859f5`）** 做完。**该审计已完成。禁止**自动开 **④–⑦**。**2026-11-10** 早（Asia/Shanghai），Triste 点头「继续吧」+ 调研建议，解锁了 Mass-only 能量路径便宜诊断（记录在 #67 / `7d5e034d`）；该诊断已由 **#68（`f578e8f4`）** 做完。**该诊断已完成。** #69 那次收口不构成开工许可（历史）。**2026-11-12 晚**（Asia/Shanghai）官方方案改为面向物理约束营养估计的主动多模态证据获取（候选）。同日早间群指令曾把 Phase 1 写成下一步；#70（squash `d622e6db`）的该许可现已取消。**（历史，不是现行门禁。2026-11-12 晚曾把下一步写成 Nutrition5k 双视角实验。该实验已在备份提交 28bd3c9c 中做完并停止，不要重跑，不要再合入。现行下一步只在文首：第一阶段数据可行性审计。）** 不要据此改 main，也不要重开已停止的双视角实验；其余代码仍空闲。该句曾把本次改写写成双视角实验的开工许可，2026-11-13 起失效。**禁止**自动开实验 B、实验 C、完整联合网络与 **④–⑦**。不要再按旧协议重开双视角实验。**主声称：质量对 Direct 可外推（bootstrap 区间均为负）；对 Mass-only 基本打平（0.76 g 不算赢，不要当作方法优越性证据）；能量外测未稳。** **2026-11-07**：**②（#60，`c52ffaa2`）已完成**；**①（#57，`0aa30537`）已完成**；**#53（`2a406df4`）仍是 Ours-new 开发检查点**；**#51（`6c8fda02`）已完成**。**④–⑦ 锁定。禁止自动开 ④–⑦。P1 仍未解锁 / 暂缓。禁止自动开 P1。禁止**重训 Mass/Q/τ。早先解锁 ③ 的依据是 Triste 点头（群聊 **2026-11-09**）+ 调研建议（505 开发集隔离 held-out；别急开 P1/重训），协议冻结为 **#53+#51**，该门禁现已由 #63 做完。自动标签只是建议，人审包仍开放，**不**把 #51 的 kcal 下降写成身份成功。**Triste 2026-11-06**：代码/结果可以分批推；派工前仍须先过 STATUS「下一步」门禁。**不得**助手用 fast 代跑 Full-set、审计或未解锁训练。

## 历史跑数（开环 v2 / 非主方法）

> 以下各段是 2026-10-27 已完成记录，命令都不要再执行。**开环 31 维 BC（线性 0.9206、MLP 0.9700、DAgger r1 0.9592）只是消融 B5**，不读 `o_{t-1}`，不是主方法，也不是方法证明。**旧 Ours-BC 已停。暂不上 RL。** 后文「不要把 `obs_t` 放进特征」是当时开环约束的原话，不是现行指令；现行主方法必须读 `o_{t-1}`。历史 B1 **110.72 不是忠实 DietAI24**。hold-out **ABSTAIN 0/24 是红警**。冻结文件引用保留，勿覆盖。

**n=8 冒烟已跑完**（2026-10-27 16:55–16:56 Asia/Shanghai，自托管 worker `gpu-12-container`，机器 `hostname` 为 `n012`）。CPU，进程里没有 `CUDA_VISIBLE_DEVICES`。解释器是仓库 `.venv` 的 Python 3.12（`jsonschema` 4.26）。`scripts/weak_steps_smoke.py` 通过（约 8 秒）。当时生成的是 fit 池官方顺序的前 8 盘，约 15 秒，61 步。**这次中间 jsonl 与 summary 已删除，不要再执行生成命令。** 弱步骤只保留全池 `results/weak_steps/rgb_train_fit_all` 与 `results/weak_steps/rgb_train_holdout_all`（各自含 `.summary.json`）。

当时 summary（teacher `0.1.0`）的数字仍有效：

- split / pool / n：`rgb_train` / `fit` / 8（无 seed）。有元数据的 fit 池共 **3679** 盘，hold-out **380** 盘。
- 动作：`RETRIEVE` 29、`RE_PORTION` 22、`STOP` 7、`CORRECT_CODE` 2、`ABSTAIN` 1。
- 终态：`corrected` 7、`abstain` 1。
- train 能量审计（pass/corrected 行对照 **train** 盘标签，不是 test 分）：n_scored **7**，`train_energy_mae` **35.0188**。`dish_1565799617` 为 abstain，不进这个 MAE。
- `uses_test_gt` false，`vlm` false，`volume_tool` false。当时 JSONL 61 行均过 `ir_step` schema，没有 `VOLUME_ESTIMATE`。
- 冻结文件未改：`results/n5k_rgb_test_overhead_b0_b1.jsonl` 的 sha256 仍是 `a4dfdd82e3a595f4…`。同目录 json 与 n=50 备份的 sha256 也与跑前一致。

**n=256 已跑完**（2026-10-27 17:00:09–17:07:05 Asia/Shanghai，同一台 `gpu-12-container` / `hostname` `n012`）。CPU，进程里没有 `CUDA_VISIBLE_DEVICES`。解释器仍是仓库 `.venv`。墙钟 **416** 秒（约 7 分钟），1809 步。含上面那 8 盘。**这次中间 jsonl 与 summary 已删除。** 不要按前缀盘数重跑。保留文件仍只有上面两份全池。

当时 summary（teacher `0.1.0`）的数字仍有效：

- split / pool / n：`rgb_train` / `fit` / 256（无 seed）。
- 动作：`RETRIEVE` 850、`RE_PORTION` 611、`STOP` 238、`CORRECT_CODE` 92、`ABSTAIN` 18。
- 终态：`corrected` 216、`pass` 22、`abstain` 18。
- train 能量审计（pass/corrected 对照 **train** 标签，不是 test 分）：n_scored **238**，`train_energy_mae` **25.3065**。18 盘 abstain 不进这个 MAE。
- `uses_test_gt` false，`vlm` false，`volume_tool` false。当时 JSONL 1809 行均过 `ir_step` schema，没有 `VOLUME_ESTIMATE`。
- 冻结文件未改：`results/n5k_rgb_test_overhead_b0_b1.jsonl` 的 sha256 仍是 `a4dfdd82e3a595f4…`。同目录 json 与 n=50 备份的 sha256 也与跑前一致。冻结能量 MAE 仍是 B0 **97.03** / B1 **110.72**（B0 可作 Direct-VLM 锚点；B1 **不是忠实 DietAI24**）。

**n=1024 已跑完**（2026-10-27 17:09:21–17:36:23 Asia/Shanghai，同一台 `gpu-12-container` / `hostname` `n012`）。CPU，进程里没有 `CUDA_VISIBLE_DEVICES`。解释器仍是仓库 `.venv`。墙钟 **1622** 秒（约 27 分钟），7128 步。前 256 个 dish_id 与 n=256 那次相同。**这次中间 jsonl 与 summary 已删除。** 不要按前缀盘数重跑。保留文件仍只有上面两份全池。

当时 summary（teacher `0.1.0`）的数字仍有效：

- split / pool / n：`rgb_train` / `fit` / 1024（无 seed）。
- 动作：`RETRIEVE` 3315、`RE_PORTION` 2396、`STOP` 955、`CORRECT_CODE` 393、`ABSTAIN` 69。
- 终态：`corrected` 859、`pass` 96、`abstain` 69。弃权比例 69/1024，与 n=256 的 18/256 接近。
- train 能量审计（pass/corrected 对照 **train** 标签，不是 test 分）：n_scored **955**，`train_energy_mae` **29.7984**。n=256 同一审计是 **25.3065**（n_scored 238）。前缀变长后 MAE 仍在三十上下，没有崩。69 盘 abstain 不进这个 MAE。
- `uses_test_gt` false，`vlm` false，`volume_tool` false。当时 JSONL 7128 行均过 `ir_step` schema，没有 `VOLUME_ESTIMATE`。
- 冻结文件未改。冻结能量 MAE 仍是 B0 **97.03** / B1 **110.72**（B0 可作 Direct-VLM 锚点；B1 **不是忠实 DietAI24**）。`results/n5k_rgb_test_overhead_b0_b1.jsonl` 的 sha256 仍是 `a4dfdd82e3a595f4…`。

**fit 全量已跑完**（2026-10-27 17:38:33–19:17:23 Asia/Shanghai，同一台 `gpu-12-container` / `hostname` `n012`）。CPU，进程里没有 `CUDA_VISIBLE_DEVICES`。解释器仍是仓库 `.venv`。墙钟 **5930** 秒（约 1 小时 39 分钟），25864 步。前 1024 个 dish_id 与 n=1024 那次相同（那次中间 jsonl 已删除）。命令已完成，不要再执行：

```bash
# 已完成，不要再执行
python scripts/gen_weak_steps_train.py --split rgb_train --pool fit --out results/weak_steps/rgb_train_fit_all.jsonl
```

summary（`results/weak_steps/rgb_train_fit_all.summary.json`，teacher `0.1.0`）：

- split / pool / n：`rgb_train` / `fit` / 全池 **3679**（`n_requested` 为空，无 seed）。
- 动作：`RETRIEVE` 12063、`RE_PORTION` 8654、`STOP` 3443、`CORRECT_CODE` 1468、`ABSTAIN` 236。
- 终态：`corrected` 3108、`pass` 335、`abstain` 236。弃权比例 236/3679，与 n=1024 的 69/1024 接近。
- train 能量审计（pass/corrected 对照 **train** 标签，不是 test 分）：n_scored **3443**，`train_energy_mae` **29.5844**。n=1024 同一审计是 **29.7984**（n_scored 955）。236 盘 abstain 不进这个 MAE。
- `uses_test_gt` false，`vlm` false，`volume_tool` false。JSONL 25864 行均过 `ir_step` schema，没有 `VOLUME_ESTIMATE`。
- 产物：`results/weak_steps/rgb_train_fit_all.jsonl`（25864 行，28,877,178 字节，约 27.5MB）与同名 `.summary.json`（约 1.6MB）。
- 速率约 **1.61 秒/盘**。冻结文件未改。冻结能量 MAE 仍是 B0 **97.03** / B1 **110.72**（B0 可作 Direct-VLM 锚点；B1 **不是忠实 DietAI24**）。`results/n5k_rgb_test_overhead_b0_b1.jsonl` 的 sha256 仍是 `a4dfdd82e3a595f4…`。

**hold-out 已跑完**（2026-10-27 19:20:14–19:30:45 Asia/Shanghai，同一台 `gpu-12-container` / `hostname` `n012`）。CPU，进程里没有 `CUDA_VISIBLE_DEVICES`。解释器仍是仓库 `.venv`。墙钟 **631** 秒，2771 步。380 个 dish_id 与 fit 全池不相交，并集是官方 `rgb_train` 的 **4059** 盘。命令已完成，不要再执行：

```bash
# 已完成，不要再执行
python scripts/gen_weak_steps_train.py --split rgb_train --pool holdout --out results/weak_steps/rgb_train_holdout_all.jsonl
```

summary（`results/weak_steps/rgb_train_holdout_all.summary.json`，teacher `0.1.0`）：

- split / pool / n：`rgb_train` / `holdout` / 全池 **380**（`n_requested` 为空，无 seed）。
- 动作：`RETRIEVE` 1285、`RE_PORTION` 949、`STOP` 356、`CORRECT_CODE` 157、`ABSTAIN` 24。
- 终态：`corrected` 328、`pass` 28、`abstain` 24。弃权比例 24/380，与 fit 全池的 236/3679 接近。
- train 能量审计（pass/corrected 对照 **train** 标签，不是 test 分）：n_scored **356**，`train_energy_mae` **30.7085**。fit 全池同一审计是 **29.5844**（n_scored 3443）。24 盘 abstain 不进这个 MAE。
- `uses_test_gt` false，`vlm` false，`volume_tool` false。JSONL 2771 行均过 `ir_step` schema，没有 `VOLUME_ESTIMATE`。
- 产物：`results/weak_steps/rgb_train_holdout_all.jsonl`（2771 行，3,105,355 字节，约 3.0MB）与同名 `.summary.json`（约 179KB）。
- 冻结文件未改。`results/weak_steps/rgb_train_fit_all.jsonl` 的 sha256 仍是 `ed47f9239bfecb3c…`，summary 仍是 `d11e6a3e547b38d0…`。冻结能量 MAE 仍是 B0 **97.03** / B1 **110.72**（B0 可作 Direct-VLM 锚点；B1 **不是忠实 DietAI24**）。`results/n5k_rgb_test_overhead_b0_b1.jsonl` 的 sha256 仍是 `a4dfdd82e3a595f4…`。

**train 弱步骤标签已经齐**：fit 3679 盘 + hold-out 380 盘。不要重跑这两份 jsonl，不要跑 rgb_test，不要开体积工具，不要用 507 盘的 test GT 改阈值、特征或这一节的准确率。

**BC 特征已落地（助手，CPU，无 GPU；开环 v2 / 消融 B5，非主方法）。** `vdr/controller/features.py` 的 `build_features` 只读动作前字段 `hyp_i`、`code_i`、`cands_i`、`portion_g_i`、`verify_codes`、`step_i`。标签是 `action_t`。`obs_t` 不进特征：`CORRECT_CODE` 和 `RE_PORTION` 的老师选定码或克重留在 `obs_t`。`vdr/controller/bc_data.py` 从 fit jsonl 组 `(h, a)`，动作准确率只在 hold-out jsonl 上算。`PolicyBC.fit` 不再是 `NotImplementedError`，但是 **numpy 多项逻辑回归**（80 步、学习率 2、L2 `1e-4`，零初始化，固定常数，没有在 hold-out 上搜）。这是线性基线，不是 Transformer，不回传 7B，不开体积工具。小 MLP 在下一段。动作头 6 个：`RETRIEVE`、`RE_PORTION`、`RE_DECOMPOSE`、`ABSTAIN`、`STOP`、`CORRECT_CODE`。`VOLUME_ESTIMATE` 不在头里。老师没有发出 `RE_DECOMPOSE`，这一列样本数是 0。

特征 **31** 维：假设是否出现、字数和词数；码是否出现、是不是 Top-1、在不在候选里、名次；候选条数、Top-1/Top-2 分数、分数差、均分；克重是否出现、克重/1500；9 个 `verify_codes` 一位有效码加一位未知码；规则触发数和 `1-Top1` 作为无 GT 的误差代理；`step_i` 相对固定步数上限 32 的比例、剩余比例、是否到顶。这些尺度没有在 hold-out 或 rgb_test 上拟合。

冒烟已在本机跑完（CPU，约 1 秒，不写结果文件，不读 rgb_test）。命令已完成，不要把它当成新的服务器任务：

```bash
# 已完成，不要再执行
python scripts/bc_features_smoke.py
```

- fit：3679 盘、25864 步。hold-out：380 盘、2771 步。两份 dish_id 不相交。
- hold-out 动作准确率 **0.9206**（2551/2771）。fit 集多数类 `RETRIEVE` 拿到 hold-out 上是 **0.4637**（1285/2771）。训练集上的拟合准确率是 0.9194（23780/25864），这不是要报的数。
- hold-out recall：`RETRIEVE` 1246/1285，`RE_PORTION` 949/949，`STOP` 356/356，`CORRECT_CODE` 0/157，`ABSTAIN` 0/24。后两类和二次 `RETRIEVE` 的差别在 `obs_t` 里，线性玩具把它们收成多数类。**ABSTAIN 0/24 是红警。** 当时写过「不要为了抬这两类把 `obs_t` 放进特征」；那是开环 v2 的约束，不是现行指令。31 维路径只作消融 B5。现行主方法必须读 `o_{t-1}`，见文首。
- 冻结文件未改。冻结能量 MAE 仍是 B0 **97.03** / B1 **110.72**（B0 可作 Direct-VLM 锚点；B1 **不是忠实 DietAI24**）。

**小 MLP BC 已落地（助手，CPU，无 GPU；开环 v2 / 消融 B5，非主方法）。** `PolicyMLP`（`vdr/controller/policy_mlp.py`）是 31→64→64→6，ReLU，dropout 0，L2 `1e-4`，Adam 学习率 `1e-2`（β1 0.9，β2 0.999），batch 512，80 epoch，种子 0。隐藏层 He 初始化，输出层用 `1/64` 的方差。类别权重是 fit 标签的逆平方根频数，再缩放到平均样本权重为 1：`RETRIEVE` 0.7471、`RE_PORTION` 0.8820、`STOP` 1.3984、`CORRECT_CODE` 2.1416、`ABSTAIN` 5.3412、`RE_DECOMPOSE` 0（fit 里没有这一类，预测时掩掉）。这些常数没有在 hold-out 或 rgb_test 上搜。`PolicyBC` 的超参没改。两边都只吃 `build_features`。不读 `obs_t`。没有 `VOLUME_ESTIMATE`。不回传 7B。不开体积工具。

冒烟已在本机跑完（CPU，数秒，只打 stdout，不写结果文件，不读 rgb_test）。命令已完成，不要把它当成新的服务器任务：

```bash
# 已完成，不要再执行
python scripts/bc_mlp_smoke.py
```

- fit：3679 盘、25864 步。hold-out：380 盘、2771 步。两份 dish_id 不相交。
- 线性 hold-out 动作准确率仍是 **0.9206**（2551/2771）。MLP hold-out 动作准确率 **0.9700**（2688/2771）。fit 集上 MLP 的拟合准确率是 0.9660（24984/25864），这不是要报的数。
- hold-out recall（命中/该类行数）：`RETRIEVE` 1273/1285（线性 1246/1285），`RE_PORTION` 917/949（线性 949/949），`STOP` 356/356（线性 356/356），`CORRECT_CODE` 142/157（线性 0/157），`ABSTAIN` 0/24（线性 0/24）。`RE_DECOMPOSE` 两边都是 0 行。
- 相对线性，MLP 把 142 个 `CORRECT_CODE` 从 `RE_PORTION` 里收回来，同时把 31 个 `RE_PORTION` 判成 `CORRECT_CODE`、1 个 `RE_PORTION` 判成 `RETRIEVE`，另有 27 个线性判成 `RE_PORTION` 的 `RETRIEVE` 被改对。净增 137 步。`RETRIEVE` / `RE_PORTION` / `STOP` 没有塌。
- `ABSTAIN` 仍然是 0。fit 上 236 行里 235 行和 `STOP` 的 31 维向量在四舍五入到 1e-6 后相同：老师写的都是空的 `hyp_i` / `code_i` / `cands_i` / `portion_g_i`，差别在 `obs_t` 和 `abstain_reason`，这两项不进特征。MLP 把 hold-out 的 24 个弃权步全部判成 `STOP`，和线性一样。**ABSTAIN 0/24 是红警**，overall Acc（含 **0.9700**）不能当方法证明。当时写过「不要为了抬 `ABSTAIN` 把 `obs_t` 放进特征」；那是开环 v2 的约束，不是现行指令。
- 弱步骤 summary 仍是 `uses_test_gt` false，`vlm` false，`volume_tool` false。冻结文件未改。冻结能量 MAE 仍是 B0 **97.03** / B1 **110.72**（B0 可作 Direct-VLM 锚点；B1 **不是忠实 DietAI24**）。

**DAgger 第 1 轮已落地（助手，CPU，无 GPU；开环 v2。旧 Ours-BC 已停，暂不上 RL）。** 专家是现有的 train-only 规则老师（`vdr/weak/teacher.py`，`0.1.0`）。学习者是 `PolicyMLP`，超参与上一节相同（31→64→64→6，dropout 0，L2 `1e-4`，Adam 学习率 `1e-2`，batch 512，80 epoch，种子 0），没有在 hold-out 或 rgb_test 上搜。先在 `results/weak_steps/rgb_train_fit_all.jsonl` 上拟合纯 BC MLP，再只在 fit 池上展开。环境走老师同一套工具：FNDDS 检索、盲 100 g 克重网格、`choose_code`、按老师规则的 `RE_PORTION` 和终态。学习者的动作驱动转移，标签始终是该状态上老师的 `action_t`。特征仍只来自 `build_features` 的动作前字段。`obs_t` 不进特征。没有 `VOLUME_ESTIMATE`。不开体积工具。不读 rgb_test / depth_test。不改 fit / hold-out jsonl，不改 507 文件。不需要 7B。

fit 行上预测已经和老师一致的盘不再展开：确定性的学习者和老师每步都一致时，会留在老师轨迹上，没有新状态。展开的盘里，访问到的状态在两种情况下追加进聚合：31 维特征键（四舍五入到 1e-6）不在 fit 矩阵里，或者学习者动作 ≠ 老师动作。已经在 BC 里、而且学习者判对的状态不重复追加。步数上限 48，连续 3 步除 `step_i` 外公开字段不变就停。这两个上限是固定护栏，不是 hold-out 搜出来的。老师轨迹最多 32 步。train 食材克重和 kcal 只留在老师查询里，用来决定 `a*` 和内部记账；下一步的 `portion_g_i` 不写入老师贴上去的克重。

冒烟不写结果文件。全池这一轮已在助手 CPU 上跑完，墙钟 **535** 秒（含前 8 盘专家轨迹与 `generate_dish_steps` 的对照，以及纯 BC 与聚合后各一次 MLP 拟合）。命令已完成，不要把它当成新的服务器任务，也不要设 `CUDA_VISIBLE_DEVICES`：

```bash
# 已完成，不要再执行
python scripts/dagger_smoke.py
python scripts/dagger_smoke.py --n 32
python scripts/dagger_smoke.py --all-fit
```

全池（这一轮要报的数）：

- 盘：fit **3679**。其中 **2913** 盘与老师一致，跳过。展开 **766** 盘，访问 **6061** 步，学习者动作 ≠ 老师 **1401** 步。
- 新行 **1490**（聚合 25864+1490=**27354**）。特征键是新的 **117** 行；其余 **1373** 行是学习者犯错的状态，特征键已经在 BC 里，再记一份老师标签。其中 **4** 行的旧标签集合里没有这次的老师动作。
- 新行的动作：`RETRIEVE` 145、`RE_PORTION` 945、`CORRECT_CODE` 201、`ABSTAIN` 196、`STOP` 3。没有 `RE_DECOMPOSE`，没有 `VOLUME_ESTIMATE`。
- hold-out 动作准确率：纯 BC MLP **0.9700**（2688/2771）→ DAgger **0.9592**（2658/2771）。线性基线仍是 **0.9206**（2551/2771）。少了 30 步。
- hold-out recall（命中/该类行数）：`RETRIEVE` 1285/1285（纯 BC 1273/1285），`RE_PORTION` 900/949（纯 BC 917/949），`STOP` 356/356，`CORRECT_CODE` 117/157（纯 BC 142/157），`ABSTAIN` 仍是 0/24。`RE_DECOMPOSE` 仍是 0 行。
- 边界往 `RETRIEVE` 和再记进去的 `RE_PORTION` 靠，`CORRECT_CODE` 少了 25 个命中。这不是改规则的依据。不要为了把准确率拉回去改特征、改 MLP 常数，或在 hold-out 上挑聚合方式。
- `ABSTAIN` 新行有 196 条，hold-out 仍然 **0/24（红警）**。动作前特征和 `STOP` 撞在一起，差别在 `obs_t` 和 `abstain_reason`。当时写过「不要为了抬 `ABSTAIN` 把 `obs_t` 放进特征」；那是开环 v2 的约束，不是现行指令。旧 Ours-BC 已停，不要为了把这轮准确率拉回去再开一轮。

同一条代码的前缀（不是另一套超参）：

- `--n 8`：展开 3 盘，新行 5（全是犯错副本，novel 0）。hold-out **2677/2771**。`CORRECT_CODE` 145/157，`RE_PORTION` 896/949，`RETRIEVE` 1280/1285，`ABSTAIN` 0/24。
- `--n 32`：展开 8 盘，新行 11（novel 1 + 副本 10）。hold-out **2671/2771**。`CORRECT_CODE` 148/157，`RE_PORTION` 889/949，`RETRIEVE` 1278/1285，`ABSTAIN` 0/24。

`uses_test_gt` false，`vlm` false，`volume_tool` false。冻结文件未改。`results/weak_steps/rgb_train_fit_all.jsonl` 的 sha256 仍是 `ed47f9239bfecb3c…`，`results/n5k_rgb_test_overhead_b0_b1.jsonl` 仍是 `a4dfdd82e3a595f4…`。冻结能量 MAE 仍是 B0 **97.03** / B1 **110.72**（B0 可作 Direct-VLM 锚点；B1 **不是忠实 DietAI24**）。

**DAgger 第 1 轮已落地（开环 v2，到此停止）。** 全池 hold-out 动作准确率 **0.9592**（2658/2771），纯 BC MLP 是 **0.9700**（2688/2771），线性基线 **0.9206**（2551/2771）。新行 **1490**。**没有新的服务器命令。** **旧 Ours-BC 已停。暂不上 RL。** 不要再开 DAgger 轮，也不要把 0.97 Acc 当方法证明；hold-out **ABSTAIN 0/24 是红警**。冻结能量 MAE 仍是 B0 **97.03** / B1 **110.72**（B0 可作 Direct-VLM 锚点；B1 **不是忠实 DietAI24**）。体积工具仍关。不要为了再跑一轮开 GPU，也不要重跑 `--n 507`。

### 俯拍 RGB 池结果（2026-10-27）

> 开环 v2 冻结跑数，文件引用保留，勿覆盖，勿重跑。B0 **97.03** 可作 Direct-VLM 锚点。B1 **110.72** **不是忠实 DietAI24**，主表须官方协议重做后再进。

- 代码：推理进程启动时日志为 `CODE 260c4b0`。`vdr/` 与 `scripts/` 相对 `1d4da37` 无差异；`9594f57` 只改了文档。祖先含 `1d4da37`。
- 时间：`2026-10-27T11:18:02+08:00` → `2026-10-27T12:14:52+08:00`（约 57 分钟），物理 GPU 0。GPU 1 上的 llama-server 未停。
- 解释器：仓库 `.venv` 没有 torch。按文首旧命令 `source .venv` 会每盘 `ModuleNotFoundError`，并立刻覆盖默认 json/jsonl。那次空跑没有拷到稳定文件名，默认文件已从 n=50 备份恢复后再跑。实际命令是 `CUDA_VISIBLE_DEVICES=0 /var/tmp/nutri-venv/bin/python -u scripts/eval_n5k_subset.py --n 507 --seed 42 --split rgb_test`。`CUDA_VISIBLE_DEVICES=0` 让加载器的 `cuda:0` 落在物理 GPU 0，没有绑到 GPU 1。
- 首行日志：`pool=507 subset=507 seed=42 split=rgb_test`
- n_b0_ok / n_b1_ok / n_b1_abstain：`507` / `507` / `0`
- 能量 MAE（summary 原值）：B0 **97.03**（中位 **63.51**，MAPE 65.17%，n=507），B1 **110.72**（中位 **63.57**，MAPE 71.61%，n=506；**不是忠实 DietAI24**）
- 同一次 summary 的其余 MAE：B0 蛋白质 8.79 g、碳水 10.39 g、脂肪 6.63 g、质量 76.46 g；B1 蛋白质 8.9 g、碳水 9.3 g、脂肪 8.84 g、质量 74.85 g。B1 这四项的 n 都是 506。
- 产物：`results/n5k_rgb_test_overhead_b0_b1.json` 与 `.jsonl`。成功跑完后，默认的 `results/n5k_subset_b0_b1.json` 与 `.jsonl` 与这一对相同。n=50 基线只在 `results/n5k_subset_b0_b1_n50_seed42.json` 与 `.jsonl`（能量 MAE 仍是 B0 98.29、B1 88.21）。
- 范围：官方 rgb_test 中有俯拍 RGB 且有标签的 507 盘。不是 709 盘全表。
- B1 能量 n=506：`dish_1557861216`（标签 0 kcal）VLM 返回空列表，`s=pass`，`n` 全是 null，MAE 不计这盘。弃权数仍是 0。
- 506 盘里 B1 能量绝对误差大于 B0 的有 269 盘。中位几乎相同，MAE 差距在尾部。例如 `dish_1562963289`（标签 350 kcal）B1 870.5 / B0 350.0；`dish_1565811139`（标签 902 kcal）B1 226.5 / B0 250.0。没有据此改检索或克重。
- 与 n=50 备份相比，这 50 个 dish_id 都在本池里：B1 的 `n` 与 VLM 原文都相同的 40/50，B0 的 `n` 相同的 37/50。其余按重跑抖动看。

```bash
# 已完成，不要执行
CUDA_VISIBLE_DEVICES=0 /var/tmp/nutri-venv/bin/python -u scripts/eval_n5k_subset.py --n 507 --seed 42 --split rgb_test
```

仍不要做：把这 507 盘写成 709 盘全表；打开体积工具；接入旧回归权重；用 test GT 调阈值、检索、克重或碎片合并；重跑或覆盖上面的 507 文件。弱标签只许用 train。生成命令在「历史跑数」里，都已跑完，不要在这 507 盘上跑，也不要重跑。

### 507 分析（助手）

只读 `results/n5k_rgb_test_overhead_b0_b1.jsonl`（与默认 `results/n5k_subset_b0_b1.jsonl` 的 sha256 同为 `a4dfdd82e3a595f4…`，492261 字节）。没有改结果文件。配对明细就在本节。当时的周报已删除，不要再到 `notes/` 找那份列表。

`dish_1557861216`（标签 0 kcal）B1 返回空列表，`n` 全是 null。能量和宏量的胜负只在 **506** 盘上比。summary 里 B0 能量 MAE **97.03** 含这盘（绝对误差 0，n=507）。同一 506 盘上 B0 能量 MAE 是 **97.22**。下面的差都用这 506 盘。

**能量。** 配对 MAE：B0 **97.22**、B1 **110.72**，差 **+13.50** kcal。中位绝对误差 63.59 / 63.57。绝对误差之差（B1−B0）的中位数只有 **+3.5** kcal。B1 更好 / 更差 / 相同：**233 / 269 / 4**。平均有符号误差（预测−标签）B1 **+23.2** kcal，B0 **−3.9**。右尾更重：正差距之和 23242 kcal，负差距之和 −16412，净差 6830。差距最大的 10 盘占净差的 56%，20 盘占 94%；去掉差距最大的 25 盘后，平均差距从 +13.5 变为 **−1.6**。两边绝对误差都截到 150 kcal 时，MAE 差只剩 **+1.8**。两端各去掉 10% 的修剪均值仍是 B0 79.7、B1 88.6。中位持平；MAE 差距由右尾抬起，单独几盘解释不完。

**宏量与质量（同一 506 盘）。** 碳水 MAE B1 **9.30**、B0 **10.41**（更好 279 / 更差 224 / 平 3）。蛋白质 B1 **8.90**、B0 **8.81**（261 / 245）。脂肪 B1 **8.84**、B0 **6.65**，中位绝对误差 5.25 / 3.69（221 / 284 / 1）；脂肪的差距不是几盘尾部。质量 MAE B1 **74.85**、B0 **76.60**，中位绝对误差反而 52.5 / 48.5，有符号误差 B1 **+17.8 g**、B0 **−5.9 g**。能量变差不是质量 MAE 变差。

**尾部假设（不是改规则的依据）。** jsonl 只有 VLM 假设原文和最终 `n`，没有检索命中的 FNDDS 食物名。能量差距 >100 kcal 的有 **82** 盘（B1 高估超过 100 kcal 的 63 盘，低估超过 100 kcal 的 19 盘）。按互斥顺序：

1. **少计整盘（19）**。`dish_1563551220`：标签 562 kcal / 558 g，B1 148 / 210 g，B0 600。`dish_1563551194`：标签 472，B1 138，B0 500。
2. **披萨碎片，或旁边再加一整个披萨（18）**。`dish_1562963289`：标签 350，B1 870.5（Pasta Salad 150 g + Cheese Pizza 200 g），B0 350。`dish_1560203400`：标签 293，B1 756.5（Pepperoni 100 g + Cheese 50 g + Crust 25 g，三条都留下），B0 250。两条假设的 Cheese+Crust 合并仍然有效：`dish_1563305513` 标签 606，B1 548，B0 300。三条及以上现有代码不合并。这里只记录，不据此改合并。
3. **培根或香肠的克重（13）**。`dish_1564675380`：标签 87 kcal / 23 g，B1 494 / 100 g（假设只有 Bacon 100 g），B0 150。`dish_1559578579`：标签 143，B1 512（bacon 100 g + strawberry 50 g），B0 150。
4. **质量接近、能量密度偏高（10）**。`dish_1564588859`：标签 184 kcal / 257 g，B1 492 / 255 g（Feta + peach + chia pudding），B0 200。
5. **预测质量明显偏大（6）**。`dish_1566501575`：标签 252 kcal / 54 g，B1 764 / 420 g。
6. **其余 16**，形态类似，没跨过上面的切分。`dish_1559233445`：标签 24.5 kcal，B1 186（假设 Butter 25 g），B0 10。

`CODE_CONF_GAP` 在 506 盘里出现 447 次；B1 更差的 269 盘里 245 次，B1 更好的 233 盘里 199 次。这个码分不开胜负。

**和 n=50。** n=50 文件没有算错：能量 MAE B0 **98.29**、B1 **88.21**，B1 更好 29 盘、更差 21 盘。这 50 个 id 都在本池里。用 507 这次的预测重算这 50 盘，仍是 B1 **85.75**、B0 **98.22**（更好 27 / 更差 23）。B1 的营养向量和 VLM 原文与 n=50 文件相同的是 40/50，B0 营养向量相同的是 37/50。重跑抖动没有把子集结论翻过来。其余 **456** 盘：MAE B0 **97.11**、B1 **113.45**（更好 206 / 更差 246 / 平 4）。从 506 盘随机抽 50 盘、重复 20000 次（种子 0）：MAE 差（B1−B0）均值 **+13.5**、标准差 **15.3**；B1 的 MAE 更低的抽样占 **18.9%**。n=50 文件的差 **−10.1** 落在约 **5.9%** 分位。这 50 盘里只有 2 个落在全池差距最大的 50 盘，4 个落在 B1 更好最多的 50 盘。子集优势是真实的，也是偏有利的一抽。
## 一句话

W1 基础设施在这台 2×4090D 上已打通：**schema + 骨架 + toy 冒烟 + 真实 FNDDS catalog（5624）+ N5K 元数据/官方 split + 俯拍 RGB + Qwen2.5-VL-7B 单盘冒烟**。  
B1 第二轮先验（菜品形态、未点名的仿肉、100 g 以上的高密度克重、Cheese+Crust 碎片合并）后，同一 50 盘子集已重跑：能量 MAE 为 B0 **98.3**、B1 **88.2**。上一轮是 97.4 / 107.9，硬化前是 98.2 / 112.1。  
有俯拍 RGB 的官方 `rgb_test` 池已跑完：能量 MAE 为 B0 **97.03**（Direct-VLM 锚点）、B1 **110.72**（**不是忠实 DietAI24**；中位 63.51 / 63.57）。配对 506 盘上中位差距约 +3.5 kcal，MAE 差距在右尾。50 盘子集仍是 B0 **98.3**、B1 **88.2**，是偏有利的一抽，不能代替这 507 盘。见「507 分析（助手）」。  
**历史链（现行缺口不在这里，而在文首第一阶段审计）：****P0a #37**、**P0b #39**、**P0c #41（`54c32e8`）**、**Q 校准 #44（`3fc611d`）**、**P0d #46（`b9c4c5e7`）**、**误差审计 #48（`91a05d8`）**、**2026-11-06 排序（先 #51 再 #53，P1 暂缓）**、**food-id #51（`6c8fda02`）**、**覆盖约束 #53（`2a406df4`，开发检查点）**、**① 反事实 Q #57（`0aa30537`）**、**② food-id 语义审计 #60（`c52ffaa2`）**、**③ 外部 held-out #63（`27d1dbfa`）**、**Cafe2 能量审计 #65（`ecf859f5`）**、**Mass-only 能量路径便宜诊断 #68（`f578e8f4`）** 已完成 → **#69 收口为历史**（当时收成空闲，不构成开工许可）。**（历史，不是现行门禁。2026-11-12 晚曾把下一步写成 Nutrition5k 双视角实验。该实验已在备份提交 28bd3c9c 中做完并停止，不要重跑，不要再合入。现行下一步只在文首：第一阶段数据可行性审计。）** **④–⑦、Phase 2、Phase 3、新网络、P1、Mass/Q/τ 重训、food-id、505 调参锁定。禁止自动开。** 不要据此改 main，也不要重开已停止的双视角实验；其余代码仍空闲。该句曾把本次改写写成双视角实验的开工许可，2026-11-13 起失效。实验 B、实验 C 与完整联合网络、已取消的 Phase 1、Q/Mass/τ 重训、food-id 改写、505 或 Cafe2 调新方法、P1 与门禁 ④–⑦ 都没有开工许可。方案真源是 2026-11-12 晚 Canvas。Triste 群指令「按照以下建议调整方案，然后接着做」（2026-11-12，Asia/Shanghai）。#68 的解锁仍是 Triste 点头「继续吧」，群聊 2026-11-10 早 Asia/Shanghai，记录在 #67 / `7d5e034d`；那次只记下结果。Cafe2 不再用于新方法调参，也不再当下一轮盲测。**主声称：质量对 Direct 可外推（bootstrap 区间均为负）；对 Mass-only 基本打平（0.76 g 不算赢，不要当作方法优越性证据）；能量外测未稳。** 自动标签只是建议，人审包仍开放，**不**把 #51 的 kcal 下降写成身份成功）。Ours-old（#24–#34）已落盘作对照。**P1 未解锁 / 暂缓；禁止自动开 P1；禁止重训 Mass/Q。** #53 **不**证明方法。① 一步评测 **Q ≈ 规则 ≪ oracle**，加固该归因。试标 30（kappa **0.7–0.8**）仍为 P2。**暂不上 RL。**

## 已完成（box）

| 项 | 说明 |
|----|------|
| Schema 冻结 | `schemas/vdr_output.schema.json`、`ir_step.schema.json`、`actions.json`；`docs/schema.md` |
| 项目骨架 | `vdr/{schemas,ir,tools,baselines,controller,eval}`；`examples/`；`requirements.txt` / `pyproject.toml` |
| Toy 评测 | `vdr/eval/metrics.py`（MAE/MAPE/FixRate/WorsenRate/Evidence Acc）；`scripts/eval_toy_smoke.py` **通过** |
| B1 冒烟（无 VLM） | `b1_rag_once.predict_from_hypotheses`；`scripts/b1_toy_smoke.py` **5/5 schema-valid** |
| RuleVerify 骨架 | 宏量界、Atwater 粗检、UNGROUNDED、CODE_CONF_GAP、PORTION_EXTREME；`scripts/rule_verify_toy_smoke.py` **通过** |
| FNDDS catalog | `data/fndds/2019-2020/foods.json`：**5624** 条，~4MB；默认优先加载；`scripts/build_fndds_catalog.py` / `fndds_retrieve_smoke.py` |
| N5K 元数据+split | `data/nutrition5k/{dish_ids,metadata}/`；rgb_train **4059** / rgb_test **709**；metadata unique **5006** |
| N5K 俯拍 RGB | `data/nutrition5k/imagery/realsense_overhead/*/rgb.png`：**3490** 张，1.35GB。覆盖官方 rgb_train **2755/4059**、rgb_test **507/709**；另有 **228** 张不在官方 RGB split。未解 depth / 侧视视频。摘要：`results/n5k_overhead_rgb.json` |
| Qwen2.5-VL-7B | 权重在 `/var/tmp/vdr-models/Qwen2.5-VL-7B-Instruct`（不进 Git）。B0/B1 图像入口已接。单盘冒烟：`results/b0_b1_image_smoke.json`。50 盘基线在 `results/n5k_subset_b0_b1_n50_seed42.json`（能量 MAE：B0 **98.3**，B1 **88.2**）。俯拍池 507 盘在 `results/n5k_rgb_test_overhead_b0_b1.json`（能量 MAE：B0 **97.03**，B1 **110.72**）。跑在 GPU 0；解释器是 `/var/tmp/nutri-venv`，不是没有 torch 的 `.venv`。GPU 1 上原有 llama-server 未动 |
| B1 硬化（CPU，无 VLM） | 同名假设取中位数、不累加；`portion_grid` 默认克重网格接到查表；plain 果蔬查询压低 dip/soup/juice 等并偏向 raw；`CODE_CONF_GAP` 时先二次 `RETRIEVE`（仍是蘸酱/汤类）或确认性 `RE_PORTION` 再 `STOP`；终态 `n` 不含 `water_g`。冒烟：`scripts/b1_harden_smoke.py` |
| B1 第二轮（CPU，无 VLM） | 查询里的 pudding/salad/pizza 等形态词不再落到短食材名；未写 meatless/vegetarian 时，培根这类肉类头不选仿肉。能量密度 ≥300 kcal/100g 且克重超过 100 g 才加连续惩罚，100 g 的一份坚果不动。`Cheese`+`Crust` 在检索前并成一次查询。体积仍关，不接旧回归权重。冒烟仍是 `scripts/b1_harden_smoke.py` |
| 步骤弱标签老师（CPU，无 VLM） | `vdr/weak/teacher.py` + `scripts/gen_weak_steps_train.py`。只读 `rgb_train`。fit 全池在 `results/weak_steps/rgb_train_fit_all.jsonl`（3679 盘、25864 步、5930 秒；train 能量审计 MAE **29.5844**）。hold-out 在 `rgb_train_holdout_all.jsonl`（380 盘、2771 步、631 秒；train 能量审计 MAE **30.7085**）。两份 dish_id 不相交，并集 4059。n=8 / n=256 / n=1024 的中间 jsonl **已删除** |
| BC 特征 + 线性基线（CPU） | **开环 v2 / 消融 B5，非主方法。** `build_features` **31** 维，只读动作前字段，不读 `obs_t`。`load_bc_rows` 从 fit jsonl 组 `(h, a)`。`PolicyBC.fit` 是 numpy 多项逻辑回归。hold-out 动作准确率 **0.9206**（2551/2771）；多数类基线 0.4637。`CORRECT_CODE` 与 `ABSTAIN` 的 recall 是 0（**0/24 是红警**）。冒烟：`scripts/bc_features_smoke.py` |
| BC 小 MLP（CPU） | **开环 v2 / 消融 B5。** `PolicyMLP` 31→64→64→6，numpy，只在 fit 行上训练。hold-out 动作准确率 **0.9700**（2688/2771），不能当方法证明。`CORRECT_CODE` 142/157，`RE_PORTION` 917/949，`ABSTAIN` 仍是 **0/24（红警）**。不读 `obs_t`，没有 `VOLUME_ESTIMATE`。冒烟：`scripts/bc_mlp_smoke.py` |
| DAgger 第 1 轮（CPU） | **开环 v2。旧 Ours-BC 已停，暂不上 RL。** 规则老师当专家，`PolicyMLP` 当学习者。只在 fit 池展开。全池新行 **1490**（novel 117 + 犯错副本 1373）。hold-out 动作准确率 **0.9592**（2658/2771），低于纯 BC **0.9700**。`CORRECT_CODE` 117/157，`RETRIEVE` 1285/1285，`ABSTAIN` 仍是 **0/24（红警）**。不读 `obs_t`，没有 `VOLUME_ESTIMATE`。冒烟：`scripts/dagger_smoke.py` |
| 标注指南 v0 | `docs/annotation_guide_v0.md`。文首横幅：字段是 v0，相对 PLAN P2（Visibility / Code Match / …，kappa 0.7–0.8）部分过时；试标可用，不是最终协议 |
| 方案 MD | 根目录 `PLAN.md`（2026-11-12 晚起与 Canvas「面向物理约束营养估计的主动多模态证据获取（候选）」对齐。早间少样本校准创新点二，以及更早的 Coverage-Constrained Active Specialist Selection，不再作现行主张） |
| P0 离线脚手架（未训练） | 设计 `docs/STATE_FEATURE_DESIGN.md`；混淆 / Provenance / Selective 脚本在 `scripts/`；mock `tests/fixtures/fndds_mock_foods.json`；草图 `vdr/controller/state_features.py`（`CLOSED_LOOP_READY` 假）。开环 31 维仍是 B5。**2026-11-04 离线冒烟已跑完**（数字在文首「助手进度」；不是方法证明，不是训练）。索引 [`docs/VDR_P0.md`](VDR_P0.md) |
| Q/Cost 校准 #44 | fit-only terminal-margin **τ≈0.001645**；hold-out 一致率 **0.4489→0.7712**（**n=2771**）；产物 `results/qphi/calibration*.json`、`report_calibration.txt`；**无 RL**；**未**改 `q_phi.npz` 权重；冻结 weak_steps / n5k **未改** |
| P0d 同预算 #46 | paired **n=505** Ours-new E **98.5521** / M **58.9952** / tool **4.6970**；vs B0 ΔE **+1.4210**（能量未稳赢）；产物 `results/p0d/`；squash **`b9c4c5e7`** |
| P0d 误差审计 #48 | 只读 #46；paired ΔM **−17.3513** / ΔE **+1.4210**；产物 `results/audit/report_error_audit.md`（+ json/jsonl/figs）；脚本 `scripts/p0d_error_audit.py`；squash **`91a05d8`** |
| 共决下一步（Triste + 调研） | **2026-11-06** 已执行：先 food-id/密度（#51），再覆盖+早停（#53），暂缓 P1。**同日外部评审**把 #53 标为开发检查点，下一门禁改为 **①–⑦**。**① 已由 #57 完成。** 当时 **②** 尚未写入文首 |
| 解锁 ②（Triste + 调研） | **2026-11-07** 群聊点头 + 调研建议（CORRECT_CODE 几乎无校准 → 身份审计马上做）。该解锁随后由 **#60**（squash `c52ffaa2`）做完。当时约定：代码只在本文件进入主分支之后才许开工；**③–⑦** 与 P1 / Mass·Q·τ 重训仍锁。不重跑 #51 / #53 / #57 |
| food-id / 密度轻修 #51 | 同头 FNDDS 码替换（已有检索窗，≥15% 靠近 calorie head）；克重与动作保持。paired **n=505** 能量 **98.5521 → 92.6963**（质量 **58.9952** 不变）；vs Direct **ΔE +1.42 → −4.43**。多组件全写 **n=50** food-table **190.79 → 131.65**；**37/50**（**30↓ / 7↑**），**68** codes；head **77.75** 为目标，记分仍是 food-table。**827** mass / **0** hollow。squash **`6c8fda02`**；**#51 未改本文件** |
| 覆盖未写组件 + 早停 #53 | 冻结 Q+fence 后，未写组件无 `estimate_mass` 写则拒绝 STOP/ABSTAIN（defer rule utility）；回到检索 top code 则 portion，不开第二轮 top-code swap；**#51 同头 swap 保留**。paired **n=505** vs #51：能量 **92.6963 → 64.2792**；质量 **58.9952 → 32.9908**；tool mean **4.6970 → 6.0257**（max **17**，均在 K=32）。vs Direct **ΔE −4.43 → −32.85**；**ΔM −17.35 → −43.36**。全部组件已写 **199/505 → 503/505**；STOP-unwritten **78 → 0**；Mass 成功 **827 → 1479**；hollow **0**。残留 **2** 盘（`dish_1565020895`、`dish_1565020926`）honeycomb 检索失败，卡 RETRIEVE，不编造克重、未 STOP。**无** Mass/Q/τ 重训。squash **`2a406df4`**；**#53 未改本文件**。**开发检查点，不证明方法；增益 ≠ Q 学习改进** |
| ① 反事实 Q + dish_id 泄漏审计 #57 | 产物 `results/qphi_counterfactual/`。fit/hold-out 按 dish_id 分组，交集 **0**，无重训。头条 n=**1914**（rgb_train hold-out；coverage-choice + mass forward；与 #53 **505** 交集 **0**）：Q **0.6165** / **24.14** / **49.96**；规则 **0.6102** / **23.03** / **50.91**；固定 **0.5914** / **20.62** / **52.99**；随机 **0.4338** / **4.04** / **78.15**；oracle improve **82.19**、regret **0**。Q 约 **0.96** kcal 优于规则，约 **50** kcal 落后 oracle。**241/1914** 为 RE_DECOMPOSE no-op（冻结 Q 不能发出）；敏感性 Q **~17.33** vs 规则 **~18.28**。价值校准 n=**3475**：MAE **0.0226**，Pearson **0.747**；RE_PORTION **0.805**，CORRECT_CODE **0.016**。**Q ≈ 规则 ≪ oracle**。squash **`0aa305379a7cc02045d07ec9c513bb9c2a3d3061`**（短 **`0aa30537`**）。**#57 未改本文件** |
| ② food-id 语义审计 #60 | 只读 #51 `results/food_id_density/`（squash `6c8fda02`）。**68** 个同头换码，覆盖 **37/50** 多组件盘。自动标签（仅建议）：换码后 Exact **0** / Near **50**（**73.53%**）/ Wrong **18**（**26.47%**）；换码前同一 cafe 行 Exact **27** / Near **23** / Wrong **18**；Exact→Near **27**；变差 **28** / 变好 **1** / 不变 **39**。宏量 food-table vs cafe，finished 多组件 **n=50**：P **10.57→10.07**，C **13.00→16.17**，F **17.56→11.97**，E **190.79→131.65**（能量与冻结 #51 一致）。换码 **37**：碳水 MAE **+4.28 g**，能量 MAE **−79.92 kcal**。能量变好的 **30** 盘：全 Exact **0**，至少一码 Wrong **12**，仅 Near **18**；这 **30** 盘上 **50** 个码全是 Near 或 Wrong。**kcal 变好 ≠ 身份变对。** 人审：68 行全部 `needs_human_review=yes`；自动标签不关闭审计。文件 `human_review_components.csv`、`human_review_dishes.csv`、`human_review.md`。产物 `results/food_id_semantic_audit/`；脚本 `scripts/food_id_semantic_audit.py`。squash **`c52ffaa2`**。**#60 未改本文件**。不要重跑，不要改该目录。本条不定义 held-out。③ 已由 #63 完成（见「③ 外部 held-out #63」行）。Cafe2 能量审计已由 #65 完成（见「Cafe2 能量审计 #65」）。Mass-only 能量路径便宜诊断已由 #68 完成（见「Mass-only 能量路径便宜诊断 #68」）。**（历史，不是现行门禁。2026-11-12 晚曾把下一步写成 Nutrition5k 双视角实验。该实验已在备份提交 28bd3c9c 中做完并停止，不要重跑，不要再合入。现行下一步只在文首：第一阶段数据可行性审计。）** 历史解锁见「解锁 Cafe2 能量审计」、「解锁 Mass-only 能量路径便宜诊断」与「解锁 ③」。 |
| 解锁 ③（Triste + 调研） | **2026-11-09** 群聊点头 + 调研建议（配对 **505** 仍是开发 / 诊断集；held-out 必须按 dish_id 隔离；别急开 P1 / 重训）。该解锁随后由 **#63**（squash `27d1dbfa`）做完。当时约定：代码只在本文件进入主分支之后才许开工。协议冻结为 **#53+#51** 同预算主路径（Coverage-Constrained + 密度 / 同头换码）。先划分脚本和 dish 列表（`results/heldout_plan/`），再跑该冻结协议，再交主表（**Energy MAE + Mass MAE**；基线 **Direct-VLM** 与 **Mass-only**；覆盖率与步数 / 工具次数）。**禁止**用 held-out 做方法选择、早停、阈值 / τ 或 GT 驱动的码 / 份量修改。当时 **④–⑦** 与 P1 / Mass·Q·τ 重训仍锁。不覆盖 #51 / #53 / #57 / #60。本行不是当前开工项 |
| ③ 外部 held-out #63 | Cafe2，**n=228**，与配对 **505**、fit（**3679** 官方 rgb_train 且 `pool_of_dish==fit`）、rgb_train、rgb_test、cafe1 的 dish_id **不相交**。划分 `results/heldout_plan/`；主表与轨迹 `results/heldout_eval/`。主表（三法皆有限，n=228）：Coverage-Constrained + density/same-head 能量 MAE **116.8494** kcal，质量 MAE **50.9483** g，覆盖 **0.9737**（222/228），步数均值 **6.2193**，工具调用均值 **5.2456**；Direct-VLM 能量 **105.6157**、质量 **73.7456**；Mass-only（冻结 Mass）能量 **75.3870**、质量 **51.7107**。能量 MAE 比 Direct-VLM **高 11.23** kcal，比 Mass-only **高 41.46** kcal；质量 MAE 比 Direct-VLM **低 22.80** g，比 Mass-only **低 0.76** g。STOP 且仍有未写组件 **0**。**6** 盘停在 RETRIEVE，组件仍未写完，没有编造份量。同头密度换码 **106** 盘。空壳 RE_PORTION **0**。工具调用中位 **4**、最大 **14**，**228/228** 在 K=32 内。Mass 写 **575**。Cafe2 盘级能量列每行都是 **0**，能量目标是配料卡路里之和；质量是盘级克重总量。这次 Direct-VLM 行记下 `placement_max_memory=8GiB`（共享 GPU）；同一 bf16 checkpoint 与 prompt，不是历史上满 GPU 的 `cuda:0` 放置。不要重跑，不要据此解锁重训。开发 505 的能量 MAE **64.2792**、质量 MAE **32.9908** **不是**本表的行。**#53 不证明方法。** #51 是密度 / kcal 启发式（自动 Exact=**0**）。一步审计上 **Q ≈ 规则 ≪ oracle**。**主声称：质量稳赢可外推；能量开发集赢、外测未稳。** **不**声称 held-out 上能量与质量双赢。squash **`27d1dbfa`**。**#63 未改本文件**。不要重跑，不要覆盖这两个目录 |
| 解锁 Cafe2 能量审计（Triste + 调研） | **2026-11-09** 晚（Asia/Shanghai）群聊点头「继续吧」，在调研建议之后。该解锁随后由 **#65**（squash `ecf859f5`）做完。当时约定：代码只在本文件进入主分支之后才许开工；**不是门禁 ④**。当时这是 ③ 的诊断后续，与 #48 同口径，不是真实 Tool Cost，不是 ⑤–⑦，不是 P1，不是 Mass / Q / τ 重训。**505** 仍是开发 / 诊断集。只读冻结 Cafe2 **n=228**，不重画划分；只读 `results/heldout_plan/` 与 `results/heldout_eval/`。当时新产物写到 `results/heldout_energy_audit/`。不要覆盖 `results/heldout_plan/`、`results/heldout_eval/`、`results/food_id_density/`、`results/coverage_early_stop/`、`results/qphi_counterfactual/`、`results/food_id_semantic_audit/`。不要重跑 #63 闭环主表，也不要重跑 #48 / #51 / #53 / #57 / #60。**主声称：质量稳赢可外推；能量开发集赢、外测未稳。** 当时协议（已由 #65 做完，不是当前开工）：（1）只在冻结的 Cafe2 n=228 名单上做诊断。不做方法选择，不早停，不调阈值 / τ，不换方法，不用 GT 改码或份量。Diagnostic only on the frozen Cafe2 n=228 list. No method selection, no early stopping, no threshold/τ tuning, no swapping the method, no GT-driven code or portion edits.（2）与 #48 同一组切分，写明白。Same cuts as #48, written plainly: 固定 food-id 时，mass 重写是否同时降低 energy（hold identity/codes fixed, then ask whether the mass rewrite also drops energy）。oracle 标签密度 + 我方克数（oracle label density with our grams）。另拆 #51 density swap 开/关，重点看换过码的 106 盘。（3）基线仍是 Direct-VLM 与 Mass-only。头条问题：相对 Mass-only 的能量损失 +41.46 kcal（质量几乎打平，−0.76 g）是否来自 food-id / 密度。Baselines remain Direct-VLM and Mass-only. The headline question is whether the energy loss vs Mass-only of +41.46 kcal (mass almost tied, −0.76 g) comes from food-id / density。当时 **④–⑦** 与 P1 / Mass·Q·τ 重训仍锁。本行不是当前开工项。结果见「Cafe2 能量审计 #65」。Mass-only 能量路径便宜诊断已由 #68（squash `f578e8f4`）完成。**（历史，不是现行门禁。2026-11-12 晚曾把下一步写成 Nutrition5k 双视角实验。该实验已在备份提交 28bd3c9c 中做完并停止，不要重跑，不要再合入。现行下一步只在文首：第一阶段数据可行性审计。） 禁止**自动开 **④–⑦**。 |
| Cafe2 能量审计 #65 | 只读冻结 Cafe2 **n=228**。产物 `results/heldout_energy_audit/`；脚本 `scripts/heldout_energy_audit.py`。**没有**重跑 #63 闭环、Direct-VLM 或 Mass-only。**没有**覆盖 `results/heldout_plan/`、`results/heldout_eval/` 或其他冻结结果目录。**#65 未改本文件**。**#63 主表不变**：Coverage-Constrained 能量 MAE **116.8494** kcal、质量 MAE **50.9483** g；Direct-VLM **105.6157** / **73.7456**；Mass-only **75.3870** / **51.7107**。能量相对 Direct **+11.23**、相对 Mass-only **+41.46**。质量相对 Direct **−22.80**、相对 Mass-only **−0.76**。覆盖 **0.9737**（222/228）。开发 505 的 **64.2792** / **32.9908** **仍不是**该表的行。**222**/228 盘上已提交克重与 Mass-only 克重相差小于 0.001 g；整段 **−0.76** g 质量优势是停在 RETRIEVE 的 **6** 盘（这 6 盘对 +41.46 均值的能量贡献 **−0.52** kcal）。**+41.46** kcal 是食物表密度打在已经与 Mass-only 对齐的克重上；Mass-only 用冻结质量估计器的 calorie head 给同一克重记分。码固定，相对 100 g 网格的 mass 重写：能量 MAE **195.6564 → 125.2060** kcal，质量 MAE **116.9123 → 50.9483** g。汇合下降来自多组件（**216.37 → 115.21**）。**58** 盘单组件上同一重写把能量 MAE **134.93 → 154.51**，质量仍近 51 g；这 58 盘贡献 **+29.62** kcal。Oracle 标签密度 × 我方克数：能量 MAE **98.6201** kcal，低于食物表 commit，低于 Direct-VLM，仍高于 Mass-only **75.3870**。不加权 kcal/g MAE：commit **0.631**，calorie head **0.753**；head 抵消更大（**77.8** vs **58.2** kcal）。#51 密度换码在 **106** 盘触发。全表能量 MAE 关 **125.2060** / 开 **116.8494**（**−8.3566** kcal），质量不变。**67** 盘更靠近 cafe 标签，**39** 盘更远。未换码的 **122** 盘贡献 **+38.30** kcal。#51 仍是密度 / kcal 启发式（开发集自动 Exact 为 **0**）。Bootstrap **10,000** 次、种子 **0**：能量差 vs Mass-only **[+25.35, +58.28]** kcal（每次均为正）；质量差 vs Direct-VLM **[−29.80, −15.66]** g（每次均为负）；能量差 vs Direct **[−6.22, +29.16]** 含两种符号；质量差 vs Mass-only **[−2.33, +0.10]** 含两种符号。**主声称仍是：质量稳赢可外推；能量开发集赢、外测未稳。** **不**声称双赢。**不**从这些标签开工改方法。squash **`ecf859f5`**。不要重跑，不要改该目录。本条**不**解锁 **④–⑦** / **P1** / Mass·Q·τ 重训。本条不是当前开工项。Mass-only 能量路径便宜诊断已由 #68 完成。**（历史，不是现行门禁。2026-11-12 晚曾把下一步写成 Nutrition5k 双视角实验。该实验已在备份提交 28bd3c9c 中做完并停止，不要重跑，不要再合入。现行下一步只在文首：第一阶段数据可行性审计。）** |
| 解锁 Mass-only 能量路径便宜诊断（Triste + 调研） | **2026-11-10** 早（Asia/Shanghai）群聊点头「继续吧」，在调研建议之后。该解锁记录在 #67 / `7d5e034d`，随后由 **#68**（squash `f578e8f4`）做完。当时约定：代码只在本文件进入主分支之后才许开工；**不是门禁 ④**。当时这是说清 98 vs 75 的便宜诊断，不是在 505 上改 food-id，不是 Cafe2 重测，不是 ⑤–⑦，不是 P1，不是 Mass / Q / τ 重训。**505** 仍是开发 / 诊断集。只读冻结 Cafe2 **n=228**，不重画划分；只读 `results/heldout_plan/`、`results/heldout_eval/`、`results/heldout_energy_audit/`。当时新产物写到 `results/heldout_massonly_energy_path/`。不要覆盖冻结结果目录。不要重跑 #63 闭环主表、Direct-VLM 或 Mass-only，也不要重跑 #65 / #48 / #51 / #53 / #57 / #60。**主声称仍是：质量稳赢可外推；能量开发集赢、外测未稳。** 当时协议（已由 #68 做完，不是当前开工）：（1）只做这一项便宜诊断。不改方法，不调超参，不在 Cafe2 上做早停或模型选择。No method changes, no hyperparameter tuning, and no early stopping or model selection on Cafe2。（2）写明 Mass-only 能量怎么记分：哪一个 food-id / 密度 / calorie head，以及它和食物表 commit、和 oracle 标签密度 × 我方克数的差别。冻结质量估计器 calorie head × 已在 222/228 盘上对齐的同一克重。（3）回答 oracle × 我方克数 **98.6201** vs Mass-only **75.3870** 主要是更大的有符号抵消（#65 已冻结：不加权 kcal/g MAE **0.753** head vs **0.631** 食物表；抵消 **77.8** vs **58.2**），还是 Mass-only 的密度 / 食物码其实更准。（4）本诊断落盘之后不要自动开下一道门。P1 / Mass·Q·τ 重训 / ④–⑦ 仍锁定。冻结后写作（「Coverage+Mass，能量外测未稳」）对比只在 505 上改 food-id 再重测 Cafe2，要等以后的调研阅读 + Triste 点头 + 再一次 STATUS 改写。基线仍是 Direct-VLM 与 Mass-only。冻结 #63 / #65 数字不改。当时 **④–⑦** 与 P1 / Mass·Q·τ 重训仍锁。本行不是当前开工项。结果见「Mass-only 能量路径便宜诊断 #68」。**（历史，不是现行门禁。2026-11-12 晚曾把下一步写成 Nutrition5k 双视角实验。该实验已在备份提交 28bd3c9c 中做完并停止，不要重跑，不要再合入。现行下一步只在文首：第一阶段数据可行性审计。） 禁止**自动开 **④–⑦**。 |
| Mass-only 能量路径便宜诊断 #68 | 只读冻结 Cafe2 **n=228**。产物 `results/heldout_massonly_energy_path/`（`report.md`、`report.txt`、`summary.json`、`dishes.jsonl`、`paths.csv`、`slices.csv`、两张图）；脚本 `scripts/heldout_massonly_energy_path.py`。**没有**重跑 #63 闭环、Direct-VLM 或 Mass-only，也**没有**重跑 #65。**没有**覆盖 `results/heldout_plan/`、`results/heldout_eval/`、`results/heldout_energy_audit/` 或其他冻结结果目录。**#68 未改本文件**。Mass-only 能量是冻结质量估计器的盘级 `energy_kcal` 头，`vit_b_16` checkpoint，目标 `mass_g`、`energy_kcal`、`protein_g`、`carb_g`、`fat_g`。记分器存 `estimate_mass(...).nutrients.energy_kcal`。不读 food-id，也不把 FNDDS kcal/100 g 乘以克重。**222 / 228** 盘上质量头与已提交克重相差在 0.001 g 以内（测得最大 **0.0002** g）。另外 **6** 盘停在 RETRIEVE；Mass-only 仍给整盘头记分。Oracle 标签密度 × 我方克数能量 MAE **98.6201**，Mass-only 能量头 **75.3870**，差距 **23.2332** kcal：更大的有符号抵消，不是更准的密度或食物码。不加权 kcal/g MAE：标签密度 **0**，食物表 commit **0.631**，calorie head **0.753**。#65 通道抵消（kcal）：oracle **0**，食物表 **58.2**，head **77.8**。保持真密度、换成 Mass-only 克重：**98.6201 → 100.3152**（**+1.6951**）。在这些克重上把真密度换成能量头：**100.3152 → 75.3870**（**−24.9283**）。净差 **23.2332** kcal。在头自己的克重上，把头的密度换成标签密度，会把能量 MAE 从 **75.3870** 抬到 **100.3152**。食物表 kcal/g 更近 **139** 盘，头更近 **89** 盘。没有一盘的头密度误差低于 **0.0001** kcal/g。**58** 盘单组件贡献 **+23.23** 里的 **+19.85**；该片 kcal/g MAE 头 **1.023** vs 食物表 **0.811**，能量 MAE **38.06** vs 食物表 **154.51** / oracle **116.08**。**6** 盘 RETRIEVE 贡献 **−1.56**。按盘 bootstrap **10,000** 次、种子 **0**：**98.6201 − 75.3870** 在 **[+7.92, +39.06]** kcal，**99.83%** 为正均值。这个区间没有用来选方法。**#63 主表句子不改。** 冻结 #63 / #65 数字不改，包括 **116.8494** / **50.9483**、**105.6157** / **73.7456**、**75.3870** / **51.7107**、**+11.23**、**+41.46**、**−22.80**、**−0.76**、**0.9737**（222/228）、**98.6201**、**0.631** vs **0.753**、**77.8** vs **58.2**、开发 505 **64.2792** / **32.9908**（**仍不是** held-out 表的行）、**195.6564 → 125.2060**、**134.93 → 154.51** / **+29.62**、**−8.3566**、未换码 **122** 盘 **+38.30**，以及 #65 bootstrap **[+25.35, +58.28]**、**[−29.80, −15.66]**、**[−6.22, +29.16]**、**[−2.33, +0.10]**。相对 Mass-only 的 **−0.76** g 仍只是 **6** 盘 RETRIEVE（#65 / #66 已记下）。**不**把它写成质量赢。**不**把只修 food-id 写成能量双赢。**主声称仍是：质量稳赢可外推；能量开发集赢、外测未稳。** **不**声称双赢。**不**从这些标签开工改方法。squash **`f578e8f4`**。不要重跑，不要改该目录。本条**不**解锁 **④–⑦** / **P1** / Mass·Q·τ 重训。本条不是当前开工项。**（历史，不是现行门禁。2026-11-12 晚曾把下一步写成 Nutrition5k 双视角实验。该实验已在备份提交 28bd3c9c 中做完并停止，不要重跑，不要再合入。现行下一步只在文首：第一阶段数据可行性审计。）** |

| #66 收口（历史，不是现行） | **2026-11-09**（Asia/Shanghai）文档 #66（`344255f`）把 Cafe2 能量审计标成已完成，当时把状态收成空闲，并且那次改写不构成开工许可。本行不是现行状态。 |
| #69 收口（历史，不是现行） | **2026-11-10**（Asia/Shanghai）文档 #69（squash `ae0e73aa`）把 Mass-only 能量路径诊断标成已完成，当时把状态收成空闲，并且那次改写不构成开工许可。本行不是现行状态。 |
| #70 解锁 Phase 1（历史，已取消） | 这是 #70 的历史授权记录。同日晚已取消，不是现行开工许可。**2026-11-12**（Asia/Shanghai）早间群指令「按照以下建议调整方案，然后接着做」。方案真源：Canvas「VDR 官方方案（2026-11-12 晚）：面向物理约束营养估计的主动多模态证据获取（候选）」（https://cursor.com/canvas/store-745bdf38-f3d7-4313-8d66-08e4b6903e93/d7be617d-d37e-4fe1-a4b5-9da2c86b27a2）。**（历史，不是现行门禁。2026-11-12 晚曾把下一步写成 Nutrition5k 双视角实验。该实验已在备份提交 28bd3c9c 中做完并停止，不要重跑，不要再合入。现行下一步只在文首：第一阶段数据可行性审计。）** 本行已降为历史，该授权已取消。不要据此改 main，也不要重开已停止的双视角实验；其余代码仍空闲。该句曾把本次改写写成双视角实验的开工许可，2026-11-13 起失效。实验 B、实验 C 与完整联合网络、已取消的 Phase 1、Q/Mass/τ 重训、food-id 改写、505 或 Cafe2 调新方法、P1 与门禁 ④–⑦ 都没有开工许可。不把实验 B、实验 C 或完整联合网络写进主分支。不解锁 Phase 2、Phase 3、新网络、Q/Mass/τ 重训、food-id、505 调参、P1 或 ④–⑦。**主声称：质量对 Direct 可外推（bootstrap 区间均为负）；对 Mass-only 基本打平（0.76 g 不算赢，不要当作方法优越性证据）；能量外测未稳。** Cafe2 只作已探索诊断集。 |

## 明确未完成

| 项 | 阻塞 |
|----|------|
| 第一阶段数据可行性审计 | **现行下一步。本文件在分支 `icdl` 上是开工许可，只此一项。** 只审计 Nutrition5k 增量扫描。不写新网络，不训练 DINOv2，不把第二阶段写成已开工。 |
| 第二阶段、第三阶段、完整网络、外测与备用方向 | **锁定。** 三个轻量模型与跨盘随机配对负对照、论文价值与外部迁移、完整网络、Agent、强化学习、OmniFood8K、ACETADA、不完整食材监督，都不要开工。停止规则已写入方案，不等于解锁第三阶段。OmniFood8K 不得提前当开发集，也不得在外部盲测前调参或泄漏标签。ACETADA 只作辅助外测，且须区分实际摄入量与餐前供应量。505 与 Cafe2 不再当独立验证集。 |
| 四问可行性审计 | **已完成，不再是现行代码项。** Q1 有条件支持；Q2 未完全通过；Q3 不得写成已解；Q4 可描述，但偏薄。评价范围的收缩当时写在双视角开工段，并因此写下该实验；该解锁已退役。周度论文追踪不是本行已启动的承诺。 |
| 双视角实验（#73，历史） | **已停止，不是现行下一步。** 已在备份提交 `28bd3c9c` 中。停止结论见文首。不要重跑，不要再合入，不要覆盖 `results/experiment_a_n5k_dual_view/`。 |
| 实验 B、实验 C 与完整联合网络 | **仍锁。禁止自动开。** Q2 未完全通过：外部混合餐仍缺真实可请求的第二视图。本轮不补采、也不新造外部混合餐的配对第二视角。 |
| #70 Phase 1（已取消） | **历史，已取消。0/8，没有开跑，不是现行下一步。** Mass-only 能量路径便宜诊断已完成（#68，squash `f578e8f4`，2026-11-10 Asia/Shanghai）。Cafe2 能量审计已完成（#65，squash `ecf859f5`）。**这一历史行不是现行代码项。** 这一历史行不是门禁 ④，也不是现行下一步。Phase 2、Phase 3 与新网络仍锁。**禁止**自动开 **④–⑦** / **P1** / Mass·Q·τ 重训。不要据此改 main，也不要重开已停止的双视角实验（该句曾把本次改写写成双视角实验的开工许可，2026-11-13 起失效；不解锁实验 B、实验 C、完整联合网络或 **④**）。不在 505 上改 food-id，也不重测 Cafe2。2026-11-12 早间曾把下一步定为 Phase 1；同日晚已取消该步。**主声称：质量对 Direct 可外推（bootstrap 区间均为负）；对 Mass-only 基本打平（0.76 g 不算赢，不要当作方法优越性证据）；能量外测未稳。** **不**声称双赢。**不**从 #65 或 #68 的标签开工改方法。**不**把只修 food-id 写成能量双赢。**不**把相对 Mass-only 的 **−0.76** g 写成质量赢（那一点只在 6 盘 RETRIEVE；#65 / #66 已记下）。不要重跑，不要改 `results/heldout_massonly_energy_path/` 或 `results/heldout_energy_audit/`。 |
| Phase 2 分层校准网络 | **锁定。禁止自动开。** 不是本文件的开工项。早间写入门槛现已随创新点二撤回，不是实验 A 的通过线，也不是另推的新标准：至少两个独立外部数据集上质量 MAE 改善约 5%–10%，且配对 bootstrap 置信区间支持改善；否则不要写成创新。Triste 原文审阅在此句被截断，上述是恢复后的工作门槛。禁止现在实现公式或分层结构。 |
| Phase 3 能量专家融合对照 | **锁定。禁止自动开。** FNDDS 组分合成、Mass Estimator 盘级能量头、以及二者的自适应融合，只是对照，不是预先定下的第二创新。Open-KNEAD: Knowledge-grounded Nutrition Estimation via Agentic Decomposition（arXiv:2607.12911，2026-07-14）。不编 DOI。 |
| ④ 真实 Tool Cost | **锁定。禁止自动开。** VLM 次数、延迟、token、单盘成本、能量 MAE–Cost Pareto |
| ⑤ 忠实 B1/B2 + 相关工作 | **锁定。禁止自动开。** 优先 arXiv 2610.11144、CVPRW 2025 Decomposing Food Images、NutriVision arXiv 2609.33076 |
| ⑥ Provenance + Semantic | **锁定。禁止自动开。** |
| ⑦ 共决 P1 / 重训 | **锁定。** ①–⑥ 有结论后再共决。在此之前禁止开 P1、禁止重训 Mass/Q/τ |
| P1 检索重排 | **锁定 / 暂缓。禁止自动开 P1。** 须等门禁 **⑦**。不得重训 Mass / Q / τ |
| 有俯拍 RGB 的官方 rgb_test（507/709） | **已完成（历史锚点）。** 能量 MAE：B0 **97.03**（Direct-VLM），B1 **110.72**（**非忠实 DietAI24**，主表须重做）。不要重跑。50 盘基线仍是 B0 **98.3**、B1 **88.2** |
| 无俯拍 RGB 的官方 test 盘 | 不进入这次命令，也不要写成 709 盘全表 |
| 步骤弱标签全量（仅 train） | **已完成。** fit 3679 + hold-out 380。没有新的生成命令。阈值仍只许在 hold-out 上看，禁止用 test GT |
| 试标 30 + 新标签 / kappa 0.7–0.8 | **人工，P2。** 指南仍是 v0，文首已标明相对 PLAN P2 部分过时。正式协议仍要扩到 Visibility / Code Match 等。不作 0.6 凑合 |
| 200–500 金标准 | **人工**，可分期；协议与 kappa 门禁不能省。若其中一部分再充当外部 held-out，仍须按菜品 / 来源隔离，并与配对 **505** 的 dish_id 不相交；**禁止**用于方法选择、早停或调 τ。**③ 已由 #63 完成**（Cafe2 **n=228**）。开标**不是**重训许可，也**不**自动开 **④–⑦** 或 P1，也**不**把已完成的 Cafe2 能量审计（#65）或已完成的 Mass-only 能量路径诊断（#68）改叫门禁 ④，也**不**据此新开代码项。**（历史，不是现行门禁。2026-11-12 晚曾把下一步写成 Nutrition5k 双视角实验。该实验已在备份提交 28bd3c9c 中做完并停止，不要重跑，不要再合入。现行下一步只在文首：第一阶段数据可行性审计。）**（见文首；不是本行） |
| 旧开环 Ours-BC / RL | **已停 / 暂不上。** 31 维开环只作消融 B5：线性 **0.9206**，MLP **0.9700**，DAgger r1 **0.9592**（新行 1490）。`ABSTAIN` **0/24 是红警**。没有新的服务器训练命令。P0 脚本与未训练草图已进仓库（文首）；闭环训练仍要等额度 |
| 体积工具 | **保持默认关闭**（刻意不做） |
| 旧回归权重 | **禁止接入** |

## 建议服务器本周顺序

> 2026-10-27 的完成记录，不是现行任务单。下一步在文首。这里没有新命令。

1. 按 [`SERVER_SETUP.md`](SERVER_SETUP.md) clone、装依赖、挂数据。已完成（依赖装在 `.venv`，PyPI 主站不可达，用了清华镜像）。  
2. 优先下 **overhead RGB**。已从本机 tar 抽出 `rgb.png`，未解侧视视频。  
3. 冒烟已通过：`validate_schema` / `fndds_retrieve_smoke`（5624）/ `n5k_metadata_summary` / `rule_verify_toy_smoke` / `b1_toy_smoke`（5/5）/ `eval_toy_smoke`。  
4. Qwen2.5-VL-7B 已装上，B0/B1 图像入口和单盘冒烟已通过。  
5. 覆盖摘要在 `results/n5k_overhead_rgb.json`，单盘冒烟在 `results/b0_b1_image_smoke.json`。  
6. 有图的官方 test 小子集在 B1 硬化后重跑完毕（2026-10-27 10:02 Asia/Shanghai，GPU 0，约 6 分钟）：seed=42 的 50/507 盘，50/50 schema-valid，弃权 0。能量 MAE：B0 97.4 kcal（中位 60.9），B1 107.9 kcal（中位 48.8，MAPE 74%）。B1 质量 MAE 65.6 g，碳水 MAE 7.8 g。不是全表。  
7. 相对硬化前：B1 能量 MAE 112.1 → 107.9。22/50 盘的 B1 数字变了，其中 16 盘 VLM 原文相同。`dish_1557937335`（标签 47 kcal）五行重复红薯从 550 kcal / 500 g 收到 110 kcal / 100 g。`dish_1563898084`（标签 419 kcal）从 530 kcal 升到 845 kcal，质量仍是 380 g：`Chia Seed Pudding` 绑到了 Chia seeds。B0 有 16/50 盘与上次不同，能量 MAE 98.2 → 97.4，按重跑抖动看。体积工具仍关，没有接入旧回归权重。无俯拍 RGB 的官方 test 盘不在当时的命令里。弱标签仍只许用 train。  
8. 第二轮先验的同一协议已重跑（2026-10-27 10:40 Asia/Shanghai，GPU 0，约 6 分钟，代码 `1d4da37`）：50/50 schema-valid，弃权 0。能量 MAE：B0 98.3 kcal（中位 64.5），B1 88.2 kcal（中位 51.6，MAPE 72%）。B1 脂肪 MAE 5.9 g，蛋白质 MAE 7.4 g，质量 MAE 64.8 g，碳水 MAE 8.1 g。9/50 盘的 B1 数字变了，VLM 原文相同的 2 盘都变好。`dish_1563305513`（标签 606 kcal）Cheese+Crust 并成 stuffed-crust 芝士披萨，1100 kcal / 350 g → 548 kcal / 200 g。`dish_1559849497`（标签 521 kcal）高密度克重从 990 kcal / 300 g 收到 692 kcal / 250 g。`dish_1563898084` 在同一假设列下不再绑 Chia seeds，改为 Pudding, rice（845 → 522 kcal）；这次 VLM 把布丁写成 75 g，最终 495 kcal。B0 有 13/50 盘与上一轮不同，能量 MAE 97.4 → 98.3，按重跑抖动看。  
9. 俯拍 RGB 池已跑完（2026-10-27 11:18–12:15 Asia/Shanghai，GPU 0，约 57 分钟，代码 `1d4da37` / 启动日志 `260c4b0`）：507/507 schema-valid，弃权 0。能量 MAE：B0 97.03（中位 63.51），B1 110.72（中位 63.57，n=506；**不是忠实 DietAI24**）。无俯拍 RGB 的官方 test 盘没有补进这次。没有用这些数字改检索或克重。
10. 助手已把 507 盘的配对分析写入「507 分析（助手）」。当时的周报已删除，以该节为准。没有用这些数字改 B1。弱标签老师已落地。`gpu-12-container` 已跑完 fit 全池（2026-10-27 19:17）和 hold-out 全池（2026-10-27 19:30，631 秒，380 盘）。BC 特征与 numpy 逻辑回归基线已落地，hold-out 动作准确率 0.9206。小 MLP BC 已在助手侧用 CPU 跑完，hold-out 动作准确率 0.9700。DAgger 第 1 轮也已在助手 CPU 上跑完（全池 535 秒，新行 1490，hold-out 动作准确率 0.9592）。没有新的服务器命令。这几项是开环 v2：31 维只作消融 B5，`ABSTAIN` 0/24 是红警。**旧 Ours-BC 已停。暂不上 RL。** 下一步见文首，不是 507，也不是再开一轮 DAgger 或 RL。

## 50 盘子集重跑（2026-10-27）

协议已跑完。代码为 `9687a8c`，数字留在提交 `a14f6a3`。下面这条命令只作记录，**不要再跑**。

```bash
# 已完成，不要执行
python scripts/eval_n5k_subset.py --n 50 --seed 42 --split rgb_test
```

## 第二轮 50 盘（2026-10-27 10:40）

协议已跑完。代码为 `1d4da37`。能量 MAE：B0 **98.3**，B1 **88.2**。这一对现在只在 `results/n5k_subset_b0_b1_n50_seed42.json` 与 `.jsonl`。默认的 `results/n5k_subset_b0_b1.json` 与 `.jsonl` 已是俯拍池 507 盘。同一次拉代码后，CPU 回归已通过：`b1_harden_smoke`、`validate_schema_examples`、`fndds_retrieve_smoke`、`rule_verify_toy_smoke`、`b1_toy_smoke`（5/5）、`eval_toy_smoke`。

```bash
# 已完成，不要执行
python scripts/eval_n5k_subset.py --n 50 --seed 42 --split rgb_test
```

这一轮在该子集上 B1 能量 MAE 低于 B0。俯拍池 507 盘已经另跑完，能量 MAE 是 B0 97.03、B1 110.72（B1 **不是忠实 DietAI24**）。体积工具仍关。弱标签仍只许用 train。现行下一步见文首。

协作协议：[`COLLAB_GITHUB.md`](COLLAB_GITHUB.md)。人工清单：[`needs_human.md`](needs_human.md)。W1–W4 看板已删除。进度以本文件、[`PLAN.md`](../PLAN.md) 与 [`VDR_P0.md`](VDR_P0.md) 为准。
