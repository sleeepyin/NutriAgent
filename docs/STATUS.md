# 执行状态（STATUS）

> 供 GitHub 同步：服务器跑完实验后，请在本文件更新。`notes/` 历史周报已删除，见 [`notes/README.md`](../notes/README.md)。执行助手据此继续改代码/方案。  
> **截止日期（文档写入时）**：2026-11-09（Asia/Shanghai）。**方法真源**：Canvas「**Coverage-Constrained Active Specialist Selection**」（https://cursor.com/canvas/store-745bdf38-f3d7-4313-8d66-08e4b6903e93/d7be617d-d37e-4fe1-a4b5-9da2c86b27a2）；Triste **2026-11-06** 按外部评审改方案。**技术中心**：**Coverage-Constrained Active Specialist Selection**（覆盖约束下的主动专家选择）+ **Error-Aware Q**。**Route A**：论文主张收窄为误差感知的**能量 + 质量**决策系统；蛋白质 / 碳水 / 脂肪作辅助。**任务框架**仍是 Evidence-Guided Active Multimodal Nutrition Estimation。旧「Evidence-Aware Closed-Loop Tool Control」→ **Ours-old / 纯控制器对照**，不作方法证明。  
> **#53 冻结为 Ours-new 开发检查点。** 配对 **505** 盘 rgb_test 是**开发 / 诊断集**（**#48→#51→#53** 用了测试集误差分析反馈；GT 未作选码 / 克重 / 阈值的程序输入），**不是**最终独立泛化主表。**#53 数字保留**为有价值的开发结果，不删、不覆盖冻结结果。**禁止**声称 #53 已证明方法。**#51 / #53 增益 ≠ Q 学习改进**（Mass / Q / τ 冻结：#51 是确定性食物码 / 密度轻修；#53 是覆盖约束，未写完组件时 **A_legal** 去掉 **STOP / ABSTAIN**）。**①（#57，squash `0aa30537`）** 在 rgb_train hold-out 一步评测上 **Q ≈ 规则 ≪ oracle**，加固上一句；**不**因此解锁重训。  
> **#24/#26/#28/#31/#34 数字保留**：offline **n=356** Full MAE **30.7085** 打平、Ours AURC **13.5311 ≤ B5 32.1474**；on-policy **n=488** Full MAE **96.2769** 打平 B5、Ours AURC **47.1737 ≤ B5 81.5857**（**勿声称** on-policy 能量赢；**勿混** offline 与 on-policy 人群）。弱 `abstain_score=clip(kcal/2500)` 已降权。  
> **硬锁**：**P1 检索重排锁定 / 暂缓**；**不重训 Mass**；**不重训 Q**；**不重拟合 τ**；**无 RL**；**不开 fast Cloud Agent**；**禁止** rgb_test GT 调参；**禁止**用 held-out 做方法选择、早停、阈值 / τ 或 GT 驱动的码 / 份量修改；**禁**空壳 RE_PORTION。  
> **下一步（严格顺序，不可跳步）**：**P0a 已完成（#37，`2e0e180`）** → **P0b 已完成（#39，`4b6ccec`）** → **P0c 已完成（#41，`54c32e8`）** → **Q 校准已完成（#44，`3fc611d`）** → **P0d 已完成（#46，`b9c4c5e7`）** → **误差审计已完成（#48，`91a05d8`）** → **food-id / 密度轻修已完成（#51，squash `6c8fda02`）** → **覆盖约束已完成（#53，squash `2a406df4`；开发检查点）** → **① 反事实 Q + dish_id 泄漏审计已完成（#57，squash `0aa30537`）** → **② food-id 语义审计已完成（#60，squash `c52ffaa2`）** → **③ 外部 held-out 已完成（#63，squash `27d1dbfa`）** → **当前第一项 = Cafe2 能量审计（已解锁；唯一允许写代码的项；不是门禁 ④）**。其后仍锁定：**④** 真实 Tool Cost → **⑤** 忠实 B1/B2 + 相关工作 → **⑥** Provenance + Semantic → **⑦** 再共决是否解锁 P1 / 重训。**禁止自动开 ④–⑦、P1、Mass/Q/τ 重训。** **Cafe2 能量审计的代码只在本文件进入主分支之后才许开工**（合并前的 PR 不算开工许可）。解锁依据：Triste 点头「继续吧」（群聊 2026-11-09 晚，Asia/Shanghai）+ 调研建议。**主声称：质量稳赢可外推；能量开发集赢、外测未稳。** 自动 Exact / Near / Wrong 只是建议，**不**把 #51 的 kcal 下降写成身份成功。详见下文「下一步（立即执行）」。  
> **可执行的下一步只在本文件。** 聊天不是 runbook。助手改下一步、跑法或方案时，先更新文档并推到 GitHub；服务器上的人或 AI 先 `git pull --ff-only`，再读「下一步」执行。  
> **服务器执行看「下一步」。** 这一节没有新的弱标签 / 507 命令。**#53 不要再跑，也不要在同一 505 盘上再调方法冒充最终主表。** **①（#57）、②（#60）与 ③（#63，squash `27d1dbfa`）已完成，不要重跑。#48 / #51 / #53 / #57 / #60 / #63 都不要重跑。不要改 `results/food_id_semantic_audit/`，也不要覆盖 `results/heldout_plan/`、`results/heldout_eval/`、#51 / #53 / #57 / #60。** **当前第一项 = Cafe2 能量审计（已解锁；不是门禁 ④）。代码只在本文件进入主分支之后才许开工。P1 未解锁，不要开。④–⑦ 未解锁，不要自动开。不要重训 Mass / Q / τ。** 历史 `--n 50`、俯拍池 `--n 507`、弱标签 jsonl、开环 BC 数字都是已完成记录，不要再执行。

## 下一步（立即执行）

**已完成（降为 Ours-old 对照，不是 Ours-new 方法证明）**：P0=A 弃权校准（#31）；P1=B on-policy rgb_test 主表（#34）；#24 闭环 BC、#26 FixRate、#28 offline MAE/AURC。数字见「助手进度」；**禁止**声称 on-policy Full MAE 赢 B5；**禁止**混 offline **n=356** 与 on-policy **n=488**。

**营养论文代码编写 — 严格顺序（前一步未落盘/未通过门禁，不得开始下一步）**

**归因与卫生（2026-11-06 外部评审；先读这段）**

- **#53 冻结为 Ours-new 开发检查点。** 配对 **505** 盘 rgb_test 标为**开发 / 诊断集**：误差审计 → 码表轻修 → 覆盖约束，已经用测试集误差分析决定下一轮修改（标签没有直接输入选码、克重或阈值）。**禁止**再把这 505 盘当独立最终泛化主表，也**禁止**在同一 505 上继续调方法冒充最终主表。
- **#53 数字保留**，不删、不覆盖冻结结果。开发集上超过 Direct-VLM **不等于**方法创新已经成立，也**不等于** #53 证明了方法。
- **#51 / #53 增益 ≠ Q 学习改进。** 两步都没有重训 Mass、没有改 Q、没有重拟合 τ。#51 是确定性食物码 / 密度轻修。#53 是 **Coverage-Constrained Active Specialist Selection**：未写完组件时 **A_legal** 去掉 **STOP / ABSTAIN**，再在合法集上按 **Q_φ / Cost** 选动作。覆盖只保证初始清单内的组件被处理，**不能**证明 VLM 没有漏掉整道食物。
- **方法中心**：Coverage-Constrained Active Specialist Selection + **Error-Aware Q**。**Route A**：论文主张是误差感知的**能量 + 质量**决策系统；蛋白质 / 碳水 / 脂肪作辅助。不把 Q 扩成多目标向量。

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
- **人审**：全部 **68** 个组件行 `needs_human_review=yes`（并 `auto_label_is_suggestion_only=yes`）。自动标签**不**关闭审计。文件：`human_review_components.csv`、`human_review_dishes.csv`、`human_review.md`。人审包仍开放。这**不是**代码项。人审**不**改写已完成的 ③ 协议，也**不**改写 Cafe2 能量审计的协议，也**不**把 #51 的 kcal 下降写成身份成功。

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

**当前第一项 = Cafe2 能量审计（已解锁；唯一允许写代码的项；不是门禁 ④）**

- **① 已完成（#57，squash `0aa30537`）。不要重跑。② 已完成（#60，squash `c52ffaa2`）。不要重跑，不要改 `results/food_id_semantic_audit/`。③ 已完成（#63，squash `27d1dbfa`）。不要重跑，不要覆盖 `results/heldout_plan/` 或 `results/heldout_eval/`。**
- **这是 ③ 的诊断后续，与 #48 同口径，不是门禁 ④（真实 Tool Cost），不是 ⑤–⑦，不是 P1，不是 Mass / Q / τ 重训。**
- **解锁依据**：Triste 点头「继续吧」（群聊 **2026-11-09** 晚，Asia/Shanghai），在调研建议之后。
- **开工条件**：**代码只在本文件进入主分支之后才许开工**（合并前的 PR 不算开工许可）。**禁止**自动开 **④–⑦**。**P1 锁定 / 暂缓。禁止**重训 Mass。**禁止**重训 Q。**禁止**重拟合 τ。**禁止** fast Cloud Agent。**无 RL。禁**空壳 RE_PORTION。
- **505 仍是开发 / 诊断集。** **禁止**声称 #53 已证明方法。**不**重画 Cafe2 划分。
- **协议（#48 同口径）**。只读 `results/heldout_plan/` 与 `results/heldout_eval/`。新产物写到新路径，例如 `results/heldout_energy_audit/`。**不要**覆盖 `results/heldout_plan/`、`results/heldout_eval/`、`results/food_id_density/`、`results/coverage_early_stop/`、`results/qphi_counterfactual/`、`results/food_id_semantic_audit/`。**不要**重跑 #63 闭环主表。**不要**重跑 #48 / #51 / #53 / #57 / #60。
   1. 只在冻结的 Cafe2 n=228 名单上做诊断。不做方法选择，不早停，不调阈值 / τ，不换方法，不用 GT 改码或份量。Diagnostic only on the frozen Cafe2 n=228 list. No method selection, no early stopping, no threshold/τ tuning, no swapping the method, no GT-driven code or portion edits.
   2. 与 #48 同一组切分，写明白。Same cuts as #48, written plainly:
      - 固定 food-id 时，mass 重写是否同时降低 energy（hold identity/codes fixed, then ask whether the mass rewrite also drops energy）。
      - oracle 标签密度 + 我方克数（oracle label density with our grams）。
      - 另拆 #51 density swap 开/关，重点看换过码的 106 盘。
   3. 基线仍是 Direct-VLM 与 Mass-only。头条问题：相对 Mass-only 的能量损失 +41.46 kcal（质量几乎打平，−0.76 g）是否来自 food-id / 密度。Baselines remain Direct-VLM and Mass-only. The headline question is whether the energy loss vs Mass-only of +41.46 kcal (mass almost tied, −0.76 g) comes from food-id / density.
- **主声称：质量稳赢可外推；能量开发集赢、外测未稳。**

**其后仍锁定（本文件逐项改写之前不得自动开工；①②③ 已完成，Cafe2 能量审计已解锁且不是门禁 ④，不在下面）**

- **④** 真实 Tool Cost：VLM 调用次数、延迟、token、单盘成本，以及能量 MAE–Cost Pareto。#53 的 tool mean **6.0257** 只是调用次数，不是真实成本。
- **⑤** 忠实重做 B1，并做 B2（能完整复现就完整复现，否则诚实标明 approx）与相关工作。优先：arXiv **2610.11144**（食物身份验证 / 恢复）、CVPRW 2025 **Decomposing Food Images**、**NutriVision** arXiv **2609.33076**。不编造 DOI。历史 B1 **110.72** 非忠实；scaffold / dry-run 不算。
- **⑥** 证据效度：Provenance（程序复算，自动，全量）与 Semantic（人工，提前到核心实验）。
- **⑦** ①–⑥ 有结论之后，再共决是否解锁 **P1** 或重训 Mass / Q。在此之前两项都锁定。

**Checklist（给代码 Agent）**

1. **P0a–P0d、Q 校准 #44、误差审计 #48、food-id #51、覆盖约束 #53、① #57、② #60、③ #63** — **已完成**。#53 = Ours-new **开发检查点**。**不**声称 #53 证明方法。**#51 / #53 ≠ Q 学习改进**（①：Q ≈ 规则 ≪ oracle）。**②** 的自动标签只是建议。**kcal 变好 ≠ 身份变对。不**把 #51 能量下降写成身份成功。#51 是密度 / kcal 启发式（换码后自动 Exact=**0**）。**③（#63，squash `27d1dbfa`）** 的 Cafe2 主表**不是**能量与质量双赢。**主声称：质量稳赢可外推；能量开发集赢、外测未稳。**
2. **当前第一项 = Cafe2 能量审计** — **已解锁；唯一允许写代码的项；不是门禁 ④。** 代码只在本文件进入主分支之后才许开工。只读冻结 Cafe2 **n=228**（`results/heldout_plan/` 与 `results/heldout_eval/`），**不**重画划分。新产物例如 `results/heldout_energy_audit/`。**不要**覆盖 `results/heldout_plan/`、`results/heldout_eval/`、`results/food_id_density/`、`results/coverage_early_stop/`、`results/qphi_counterfactual/`、`results/food_id_semantic_audit/`。**不要**重跑 #63 闭环主表，也**不要**重跑 #48 / #51 / #53 / #57 / #60。**505** 仍是开发 / 诊断集。**禁止**重训 Mass / Q / τ；**禁止**开 P1；**禁止**自动开 **④–⑦**；**无 RL**；**禁止** fast Cloud Agent。协议：
    1. 只在冻结的 Cafe2 n=228 名单上做诊断。不做方法选择，不早停，不调阈值 / τ，不换方法，不用 GT 改码或份量。Diagnostic only on the frozen Cafe2 n=228 list. No method selection, no early stopping, no threshold/τ tuning, no swapping the method, no GT-driven code or portion edits.
    2. 与 #48 同一组切分，写明白。Same cuts as #48, written plainly:
       - 固定 food-id 时，mass 重写是否同时降低 energy（hold identity/codes fixed, then ask whether the mass rewrite also drops energy）。
       - oracle 标签密度 + 我方克数（oracle label density with our grams）。
       - 另拆 #51 density swap 开/关，重点看换过码的 106 盘。
    3. 基线仍是 Direct-VLM 与 Mass-only。头条问题：相对 Mass-only 的能量损失 +41.46 kcal（质量几乎打平，−0.76 g）是否来自 food-id / 密度。Baselines remain Direct-VLM and Mass-only. The headline question is whether the energy loss vs Mass-only of +41.46 kcal (mass almost tied, −0.76 g) comes from food-id / density.
3. **④ 真实 Tool Cost** — **锁定。禁止自动开。**
4. **⑤ 忠实 B1/B2 + 相关工作** — **锁定。禁止自动开。**
5. **⑥ Provenance + Semantic** — **锁定。禁止自动开。**
6. **⑦ 共决是否解锁 P1 / 重训** — **锁定。P1 仍暂缓。禁止自动开 P1。禁止重训 Mass / Q / τ。**

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
   - **结论**：「工具跑了但 n 不变」**不是**多数；**门禁通过**（仍禁空壳 RE_PORTION）。**P0b #39**、**P0c #41**、**Q 校准 #44**、**P0d #46**、**误差审计 #48**、**food-id #51**、**覆盖+早停 #53**、**① #57**、**② #60**、**③ #63** 已落盘 → **当前第一项 = Cafe2 能量审计**（**已解锁**；**不是门禁 ④**；**④–⑦ 与 P1 仍锁**；**不**重训 Mass/Q/τ）。

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

8. **覆盖未写组件 + 早停处理（已完成 #53，squash `2a406df4`）**：冻结 Q+fence 后，未写组件无 `estimate_mass` 写则拒绝 STOP/ABSTAIN（defer rule utility）；回到检索 top code 则 portion，不开第二轮 top-code swap；**#51 同头 swap 保留**。paired **n=505** 能量 **92.6963 → 64.2792**、质量 **58.9952 → 32.9908**、tool mean **4.6970 → 6.0257**（max **17**，均在 K=32）。全部组件已写 **199/505 → 503/505**；STOP-unwritten **78 → 0**；Mass 成功 **827 → 1479**；hollow **0**。残留 **2** 盘（`dish_1565020895`、`dish_1565020926`）honeycomb 检索失败，卡 RETRIEVE，不编造克重、未 STOP。**无** Mass/Q/τ 重训。**#53 未改 STATUS**。这是开发检查点，**不**证明方法；增益 **≠** Q 学习改进。**① 已由 #57 完成，② 已由 #60 完成**，不在本条重开。**不得**据此重开本条或开 **P1**。**③ 已由 #63（`27d1dbfa`）完成**，不在本条重开，也**不**在这 505 盘上再调方法。现行第一项是 Cafe2 能量审计（见文首；**不是门禁 ④**）。

9. **① 反事实 Q 动作价值 + dish_id 分组泄漏审计（已完成 #57，squash `0aa30537`）**：产物 `results/qphi_counterfactual/`。fit/hold-out **按 dish_id 分组**，交集 **0**，**无**重训。头条 n=**1914**（coverage-choice + mass forward；rgb_train hold-out；与 #53 **505** 交集 **0**）：Q Best-ID **0.6165** / ΔE improve **24.14** / E regret **49.96**；规则 **0.6102** / **23.03** / **50.91**；固定 **0.5914** / **20.62** / **52.99**；随机 **0.4338** / **4.04** / **78.15**；oracle improve **82.19**、regret **0**。Q 约 **0.96** kcal 优于规则，约 **50** kcal 落后 oracle。**241/1914** 的 oracle 最优是 RE_DECOMPOSE no-op（冻结 Q 不能发出）；去掉后 Q regret **~17.33** vs 规则 **~18.28**。价值校准 n=**3475**：MAE(Q, ΔL) **0.0226**，Pearson **0.747**；RE_PORTION **0.805**，CORRECT_CODE **0.016**。预算曲线在该目录 `report.md`（一步 cost cap，不是全程 rollout）。**Q ≈ 规则 ≪ oracle**。**② 已完成**，不在本条重开。**③ 已由 #63 完成**，不在本条重跑，也不用本条数字选 held-out 或 Cafe2 能量审计。

10. **④–⑦ 与 P1（锁定）**：**①②③ 已完成。** 当前第一项是 **Cafe2 能量审计**（已解锁；唯一允许写代码的项；**不是门禁 ④**；见文首；代码只在本文件进入主分支之后才许开工）。只读冻结 Cafe2 **n=228**。新产物例如 `results/heldout_energy_audit/`，不覆盖 `results/heldout_plan/`、`results/heldout_eval/`、#51 / #53 / #57 / #60。**禁止**用该审计做方法选择、早停、换方法、阈值 / τ，或 GT 驱动的码 / 份量修改。基线仍是 **Direct-VLM** 与 **Mass-only**。其后仍锁定：**④** 真实 Tool Cost（VLM 次数、延迟、token、单盘成本、能量 MAE–Cost Pareto）→ **⑤** 忠实 B1/B2 + 相关工作（优先 arXiv **2610.11144**、CVPRW 2025 Decomposing Food Images、NutriVision arXiv **2609.33076**；不编 DOI）→ **⑥** Provenance + Semantic → **⑦** 再共决是否解锁 **P1** 或重训 Mass/Q。**禁止自动开 ④–⑦。P1 仍锁定 / 暂缓。禁止自动开 P1。禁止**重训 Mass / Q / τ。人审包仍开放，自动标签只是建议，**不**改写已完成的 ③ 协议，也**不**改写 Cafe2 能量审计的协议，也**不**把 #51 的 kcal 下降写成身份成功。忠实 B1/B2（[`docs/B1_B2_SCAFFOLD.md`](B1_B2_SCAFFOLD.md)）**不得**在 **⑤** 解锁前开工。

**禁令（仍有效）**：**不得**重训 Mass；**不得**重训 Q；**不得**重拟合 τ；**禁止自动开 P1**；**①②③ 已完成，不是开工项**；**当前第一项 = Cafe2 能量审计，已解锁，是唯一允许写代码的项，不是门禁 ④**（代码只在本文件进入主分支之后；只读 `results/heldout_plan/` 与 `results/heldout_eval/`；新产物例如 `results/heldout_energy_audit/`）；**禁止自动开 ④–⑦**；**禁止**用 held-out 做方法选择、早停、阈值 / τ 或 GT 驱动的码 / 份量修改；不上 RL；不开 fast Cloud Agent；禁止 rgb_test GT 调参；勿覆盖冻结 n5k / weak_steps / #46 / #48 / #51 / #53 / #57（`results/qphi_counterfactual/`）/ #60（`results/food_id_semantic_audit/`）/ #63（`results/heldout_plan/`、`results/heldout_eval/`）结果；**禁**空壳 RE_PORTION；kcal-proxy AURC 作主 claim；**禁止**把 #51/#53 或 ① 写成 Q 学得更好；**禁止**把 #51 的 kcal 下降写成身份成功；**禁止**把开发 505 当独立最终测试；**禁止**声称 #53 已证明方法；**禁止**重跑 #48 / #51 / #53 / ① / ② / ③（#63）；本机 Qwen closed-loop **≠** 忠实 DietAI24 / FoodAgent。

> 真源：[`PLAN.md`](../PLAN.md)、Canvas「**Coverage-Constrained Active Specialist Selection**」（https://cursor.com/canvas/store-745bdf38-f3d7-4313-8d66-08e4b6903e93/d7be617d-d37e-4fe1-a4b5-9da2c86b27a2）、[`docs/VDR_P0.md`](VDR_P0.md)、[`docs/STATE_FEATURE_DESIGN.md`](STATE_FEATURE_DESIGN.md)。`CLOSED_LOOP_READY=true` 指 **Ours-old** 脚手架就绪，**不是** Ours-new 已证明。**不上 RL。不开 fast Cloud Agent。** 冻结结果不重写。#53 是开发检查点，不是方法证明。

### 服务器

**没有新的 507 / 弱标签重跑命令。** **#63（squash `27d1dbfa`）、#60（squash `c52ffaa2`）、#57（squash `0aa30537`）、#53（squash `2a406df4`）与 #51（squash `6c8fda02`）已落盘，不要重跑，也不要在同一 505 开发池上再调方法。不要改 `results/food_id_semantic_audit/`，也不要覆盖 `results/heldout_plan/` 或 `results/heldout_eval/`。** **当前第一项 = Cafe2 能量审计（不是门禁 ④）。代码只在本文件进入主分支之后才许开工**（只读上述 held-out 目录，新产物例如 `results/heldout_energy_audit/`）。**禁止**自动开 **④–⑦**。**不要**重跑 #48，也**不要**重跑 #63 闭环主表。**#44 的 Q 权重不再重训。不得**重训 Mass。**不得**重拟合 τ。**禁止** Cloud Agent **fast**。**P1 仍锁定 / 暂缓。禁止自动开 P1。** `results/qphi_counterfactual/` 不要覆盖。冻结文件勿覆盖，也不要重跑：

- `results/n5k_rgb_test_overhead_b0_b1.json` 与 `.jsonl`（sha256 前缀 `a4dfdd82e3a595f4…`）
- 默认的 `results/n5k_subset_b0_b1.json` 与 `.jsonl`（现在就是上面这一对）
- n=50 基线 `results/n5k_subset_b0_b1_n50_seed42.json` 与 `.jsonl`
- `results/weak_steps/rgb_train_fit_all.jsonl`（sha256 前缀 `ed47f9239bfecb3c…`）与 `results/weak_steps/rgb_train_holdout_all.jsonl`

不要用这 507 盘的 test GT 调阈值、检索、克重或碎片合并。体积工具仍关。旧回归权重仍不接入。弱标签不改 B1 先验。

历史 B0 能量 MAE **97.03**（n=507）可作 Direct-VLM 锚点。历史 B1 能量 MAE **110.72**（n=506）**不是忠实 DietAI24**，主表须按官方协议重做后再进。

### 助手进度

> **解读**：以下 #24–#34 段均为 **Ours-old（纯闭环控制器 / 弱弃权）对照**；支撑「#34 MAE 打平 → 需要 Mass + Q_φ」叙事，**不是** Ours-new 方法证明。**#46–#53** 的数字是**开发结果**。**#51/#53 增益 ≠ Q 学习改进**（① #57：rgb_train hold-out 一步评测上 Q ≈ 规则 ≪ oracle）。#53（Coverage-Constrained Active Specialist Selection）**不**证明方法已在独立测试上成立。**#63（`27d1dbfa`）** 的 Cafe2 外测**不是**能量与质量双赢。**主声称：质量稳赢可外推；能量开发集赢、外测未稳。**主指标见 [`PLAN.md`](../PLAN.md)：能量 MAE + mass MAE + 真实 Tool cost 优先；宏量辅助。

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
- **诚实解读**：P0d 证明 Mass + 校准 Q 在同预算下 **降 mass、降 tool、略优于 Ours-old 能量**，但 **未**在 paired Full-set 上 **稳定击败 B0 能量**；**#48 误差审计** 已解释 pooled **+1.42 kcal** 机制。多组件 food-table 一侧由 **food-id #51（`6c8fda02`）** 落盘；覆盖与早停由 **#53（`2a406df4`）** 落盘（见「助手进度」）。该表是开发结果，**不是**最终独立主表。**①②③ 已完成（#57 / #60 / #63）。当前第一项 = Cafe2 能量审计（已解锁；不是门禁 ④）。④–⑦ 仍锁。** **P1 仍暂缓。**
- **产物**：`results/p0d/report_same_budget.txt`、`same_budget.json`、`same_budget_dishes.jsonl`、`ours_new_rgb_test.jsonl`、`ours_new_rgb_test_summary.json`。冻结 n5k / weak_steps **未改**。

**P0d 误差审计已落盘（只读 #46；Ours-new 诊断）**（2026-11-05 Asia/Shanghai，#48，squash **`91a05d8`**）。

- **脚本**：`scripts/p0d_error_audit.py`；**无新训练**；**未**用 rgb_test 标签拟合阈值/码/克重。  
- **paired n=505 vs B0**：ΔM **−17.3513 g**（bootstrap 95% **[−24.94, −10.13]**，**全 draw 为负**）；ΔE **+1.4210 kcal**（95% **[−8.10, +11.05]** **跨 0**）；**median ΔE −4.54 kcal**；去掉 **2** 盘最大能量回归后 **503** 盘 mean **−0.14 kcal**（敏感性）。  
- **finished all-mass-write 199**：ΔE **−10.96** / ΔM **−39.17**；**one-comp 149** ΔE **−17.99**；**multi-comp 50** ΔE **+10.01** / ΔM **−91.80** — food-table commit E MAE **190.79** vs stored calorie head **77.75**（**grams 与 head 一致**）。此为 **#48 写前**；**#51** 后同切片 food-table 能量 MAE **131.65**。**#53** 后全部组件 mass 已写 **503/505**（见 #53 段）。  
- **unfinished 306**：对 pooled 能量均值 **+5.74 kcal**；**78 STOP unfinished** mean signed kcal err **−87.6**，对均值 **+6.06 kcal**。  
- **hold food-id fixed（481）**：mass rewrite ΔM **−20.24** / ΔE **−17.90** — **非**相对 prior grams 用能量换质量。  
- **Q fence replay 2858/2858**；505 盘 **committing actions 均为 Q/Cost**。  
- **产物**：`results/audit/report_error_audit.md`、`report_error_audit.txt`、`error_audit.json`、`paired_residuals.jsonl`、`fig_energy_gap_by_quadrant.svg`、`fig_error_channels.svg`、`fig_mass_energy_delta.svg`。冻结 n5k / weak_steps **未改**。  
- **门禁解读**：审计 **不**启动 Mass 重训 / Q 改 / P1。**2026-11-06 共决**后 **food-id #51** 与 **覆盖+早停 #53** 已完成。**现行门禁见文首：①②③ 已完成；当前第一项 = Cafe2 能量审计（已解锁；不是门禁 ④）；④–⑦ 仍锁。** **P1 仍暂缓。禁止自动开 P1。禁止**重训 Mass/Q。

**food-id / 密度轻修已完成（多组件优先）**（2026-11-06 Asia/Shanghai，#51，squash **`6c8fda02`**）。

- **范围**：finished multi-component 上，在**已有检索窗**内做**同头 FNDDS code swap**（food-table 能量 **≥15%** 更靠近 mass-estimator calorie head；15% 为既有 density-gain cutoff，**未**在本测试集上选取）。**克重与动作保持**（不重问 selector、不重训 Mass）。标签只用于事后记分，不用于选码、阈值或克重。
- **paired n=505**：能量 MAE **98.5521 → 92.6963** kcal；质量 MAE **不变 58.9952 g**。vs Direct-VLM **ΔE +1.42 → −4.43** kcal。
- **多组件全写 n=50**：food-table 能量 MAE **190.79 → 131.65**；vs Direct **+10.01 → −49.13**。换码 **37/50**（绝对能量误差 **30↓ / 7↑**），**68** component codes。stored calorie head 仍 **77.75**（swap 的目标；记分 commit 仍是 food-table）。
- **Mass RE_PORTION**：**827** 成功，**0** hollow（成功且非 mass estimator 的 portion = 0；下一步营养不变 = 0）。swap 不新增 portion 步。
- **未做**：Mass 重训；Q 权重与 terminal-margin fence 未改、未重拟合；无检索重排、无新候选列表；覆盖与早停未动；**本文件在 #51 未改**。
- **产物**：`results/food_id_density/report.md`、`report.txt`、`summary.json`；脚本 `scripts/food_id_density_align.py`。冻结 n5k / weak_steps **未改**。
- **门禁解读**：**不**解锁 Mass 重训 / Q 改 / **P1**。Triste 顺序的下一项 **覆盖未写组件 + 早停** 已由 **#53（`2a406df4`）** 完成。**现行门禁见文首：①②③ 已完成；当前第一项 = Cafe2 能量审计（已解锁；不是门禁 ④）。** 语义审计已由 **#60** 完成。**kcal 变好 ≠ 身份变对。不**在本段重跑或覆盖 `results/food_id_density/`。

**覆盖约束已完成（Coverage-Constrained Active Specialist Selection；Ours-new 开发检查点，不证明方法）**

- **方法**：冻结 Q 权重与 fit-set terminal-margin fence **不变**。在该选择之后，未写组件若没有 `estimate_mass` 写，**不能**提交 STOP/ABSTAIN，改为已有 rule utility。码替换若已回到检索 top code，则做 **portion**，不再开下一轮 swap；离开 top code 的一次 swap 保持。克重仍来自 `estimate_mass`。**#51 同头 swap** 在多组件盘写完后仍运行。**无** Mass 重训、**无** Q 重训、**无** τ 重拟合。标签只用于事后记分。**#53 未改本文件**。
- **paired n=505**（相对 **#51** 基线）：能量 MAE **92.6963 → 64.2792** kcal；质量 MAE **58.9952 → 32.9908** g；tool cost mean **4.6970 → 6.0257**（max **17**；**505/505** 在 K=32 内）。
- **vs Direct-VLM**：ΔE **−4.43 → −32.85** kcal；ΔM **−17.35 → −43.36** g。
- **覆盖**：全部组件 mass 已写 **199/505 → 503/505**；STOP 且仍有未写组件 **78 → 0**；Mass RE_PORTION 成功 **827 → 1479**；hollow **0**。
- **残留 2 盘**：`dish_1565020895`（7 个组件里 5 个已写）、`dish_1565020926`（8 个里 7 个已写）。`honeycomb` 的 FNDDS 检索失败，状态不变，既有三步 stuck guard 停在 **RETRIEVE**。没有 code 时 RE_PORTION 不合法，**不编造克重**。两盘都 **没有 STOP**。
- **未做**：Mass checkpoint、Q 权重、τ **未改**；无 P1 检索重排；无 RL；**未**用 rgb_test 标签选阈值。冻结 n5k / weak_steps **未改**。
- **产物**：`results/coverage_early_stop/report.md`、`report.txt`、`summary.json`、`continued_steps.jsonl`、`paired_scores.jsonl`。脚本 `scripts/coverage_early_stop.py`；`vdr/controller/coverage.py`。
- **门禁解读**：**不**解锁 Mass 重训 / Q 改 / **P1**。增益来自覆盖约束，**不是** Q 变强，**不**证明方法。**①（#57）** 在另一次一步评测上给出 **Q ≈ 规则 ≪ oracle**，加固该归因。**现行门禁见文首：①②③ 已完成；当前第一项 = Cafe2 能量审计（已解锁；不是门禁 ④）。** **禁止**自动开 **④–⑦** / **P1** / Mass 或 Q 重训。**不**在本段开工，也**不**在这 505 盘上再调方法。

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
- **门禁解读**：本条数字**不**解锁 **④–⑦**，**不**解锁 Mass 重训 / Q 改 / **P1**，也**不**定义 held-out。**② 已由 #60 完成**，不在本段重开。**③ 已由 #63（`27d1dbfa`）完成。** 现行第一项是 Cafe2 能量审计（见文首；**不是门禁 ④**）。**不**在本段重跑或覆盖 `results/qphi_counterfactual/`。

**② food-id 语义审计已完成（只读 #51；自动标签仅建议）**（2026-11-07 Asia/Shanghai，#60，squash **`c52ffaa2`**）。

- **来源**：只读 #51 `results/food_id_density/`（squash **`6c8fda02`**）。从冻结轨迹收回 **68** 个同头换码，覆盖 **37/50** 多组件盘。克重与动作保持。**无** Mass / Q / τ 重训。**无 RL。** 不重跑 #51 / #53 / #57。**#60 未改本文件。**
- **身份（自动标签，仅建议）**：换码后 Exact **0** / Near **50**（**73.53%**）/ Wrong **18**（**26.47%**）。换码前同一 cafe 行：Exact **27** / Near **23** / Wrong **18**。Exact→Near **27**。自动身份变差 **28**、变好 **1**、不变 **39**。
- **宏量（food-table × 存克 vs cafe）**：finished 多组件 **n=50**：蛋白质 **10.57→10.07**，碳水 **13.00→16.17**，脂肪 **17.56→11.97**，能量 **190.79→131.65**（能量与冻结 #51 一致）。换码 **37** 盘：碳水 MAE **+4.28 g**，能量 MAE **−79.92 kcal**。
- **kcal 变好 ≠ 身份变对**：能量绝对误差下降的 **30** 盘换码盘里，全自动 Exact **0**；至少一码 Wrong **12**；仅 Near **18**。这 **30** 盘上的 **50** 个换码全部是 Near 或 Wrong。**不**把 #51 的 kcal 下降写成身份成功。
- **人审**：全部 **68** 个组件行 `needs_human_review=yes`。自动标签**不**关闭审计。文件：`human_review_components.csv`、`human_review_dishes.csv`、`human_review.md`。人审包仍开放。人审**不**改写已完成的 ③ 协议，也**不**改写 Cafe2 能量审计的协议。
- **产物**：`results/food_id_semantic_audit/`；脚本 `scripts/food_id_semantic_audit.py`。不要重跑，不要改该目录。
- **门禁解读**：本条**不**解锁 **④–⑦**，**不**解锁 Mass 重训 / Q 改 / **P1**，也**不**定义 held-out。**③ 已由 #63（`27d1dbfa`）完成。** 现行第一项是 Cafe2 能量审计（见文首；**不是门禁 ④**）。

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
- **门禁解读**：本条**不**解锁 **④–⑦**，**不**解锁 Mass / Q / τ 重训，**不**开 **P1**。**不要重跑，不要覆盖** `results/heldout_plan/` 与 `results/heldout_eval/`。现行第一项是 **Cafe2 能量审计**（见文首；**不是门禁 ④**）。

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

1. **当前第一项 = Cafe2 能量审计（已解锁；唯一允许写代码的项；不是门禁 ④）**：代码只在本文件进入主分支之后才许开工。①②③ 已完成（#57 `0aa30537`、#60 `c52ffaa2`、#63 `27d1dbfa`）。只读冻结 Cafe2 **n=228**，**不**重画划分。新产物例如 `results/heldout_energy_audit/`。**④–⑦ 锁定，禁止自动开。** **不要**重跑 #48 / #51 / #53 / #57 / #60 / #63，也不要改 `results/food_id_semantic_audit/`，也不要覆盖 `results/heldout_plan/` 或 `results/heldout_eval/`。自动标签只是建议；**68** 行 `needs_human_review=yes`，人审包仍开放。**kcal 变好 ≠ 身份变对。** #51 是密度 / kcal 启发式（自动 Exact=**0**）。**Q ≈ 规则 ≪ oracle**。**主声称：质量稳赢可外推；能量开发集赢、外测未稳。** **P1 仍未解锁 / 暂缓。禁止自动开 P1。禁止**重训 Mass/Q/τ。见文首与 Checklist。**#48 `results/audit/`、#51 `results/food_id_density/`、#53 `results/coverage_early_stop/`、#57 `results/qphi_counterfactual/`、#60 `results/food_id_semantic_audit/`、#63 `results/heldout_plan/` 与 `results/heldout_eval/` 只读**。**禁止** Cloud Agent **fast**；**禁**空壳 RE_PORTION。**无 RL。** 协议：
    1. 只在冻结的 Cafe2 n=228 名单上做诊断。不做方法选择，不早停，不调阈值 / τ，不换方法，不用 GT 改码或份量。Diagnostic only on the frozen Cafe2 n=228 list. No method selection, no early stopping, no threshold/τ tuning, no swapping the method, no GT-driven code or portion edits.
    2. 与 #48 同一组切分，写明白。Same cuts as #48, written plainly:
       - 固定 food-id 时，mass 重写是否同时降低 energy（hold identity/codes fixed, then ask whether the mass rewrite also drops energy）。
       - oracle 标签密度 + 我方克数（oracle label density with our grams）。
       - 另拆 #51 density swap 开/关，重点看换过码的 106 盘。
    3. 基线仍是 Direct-VLM 与 Mass-only。头条问题：相对 Mass-only 的能量损失 +41.46 kcal（质量几乎打平，−0.76 g）是否来自 food-id / 密度。Baselines remain Direct-VLM and Mass-only. The headline question is whether the energy loss vs Mass-only of +41.46 kcal (mass almost tied, −0.76 g) comes from food-id / density.
2. **覆盖未写组件 + 早停 #53 已完成**（squash `2a406df4`）。paired n=505 能量 **64.2792** / 质量 **32.9908** / tool mean **6.0257**（max **17**，均在 K=32）。全部组件已写 **503/505**；STOP-unwritten **0**；Mass 成功 **1479**；hollow **0**。残留 2 盘 honeycomb 检索失败，不编造克重、未 STOP。
3. **food-id / 密度轻修 #51 已完成**（squash `6c8fda02`）。calorie head **77.75** 是目标读数；记分 commit 仍是 food-table。
4. **#34 / #31 / #28 已冻结为 Ours-old 对照**；勿声称 on-policy 能量赢 B5。
5. **P1 未解锁 / 暂缓**：B1/B2 忠实 API **不得**跳步开 retrieval-rerank。
6. **禁止：** 自动进 **P1** 或 Mass/Q/τ 训练；自动开 **④–⑦**（**Cafe2 能量审计**只在本文件进入主分支之后，**不是门禁 ④**，且不得扩到 ④–⑦，也不得用该审计做方法选择、早停、换方法、阈值 / τ 或 GT 驱动的码 / 份量修改）；把 FixRate / AURC / #44 hold-out 动作表当 Ours-new Full-set 主证明；混 offline n=356 与 on-policy n=488；空壳 RE_PORTION；RL；fast Cloud Agent；test GT 调参；覆盖冻结文件；**110.72** 当忠实 DietAI24；**勿**重训 Q 权重覆盖 #44 校准产物；重跑 507 / P0d Full-set（#46 已落盘）；重跑误差审计（#48 已落盘）；重跑 food-id 对齐（#51 已落盘）；重跑覆盖+早停（#53 已落盘）；重跑反事实 Q（#57 已落盘）；重跑 food-id 语义审计（#60 已落盘）；重跑 ③ 闭环主表（#63 已落盘）；覆盖 `results/heldout_plan/` 或 `results/heldout_eval/`。

### 协作

docs-first：改下一步先改本文件并推仓。Cloud Agent **永远不开 fast**。**2026-11-09**：**③（#63，`27d1dbfa`）已完成**。同日晚，Triste 点头「继续吧」后，**Cafe2 能量审计已解锁**，为当前第一项、唯一允许写代码的项，**不是门禁 ④**；代码只在本文件进入主分支之后才许开工。**主声称：质量稳赢可外推；能量开发集赢、外测未稳。** **2026-11-07**：**②（#60，`c52ffaa2`）已完成**；**①（#57，`0aa30537`）已完成**；**#53（`2a406df4`）仍是 Ours-new 开发检查点**；**#51（`6c8fda02`）已完成**。**④–⑦ 锁定。禁止自动开 ④–⑦。P1 仍未解锁 / 暂缓。禁止自动开 P1。禁止**重训 Mass/Q/τ。早先解锁 ③ 的依据是 Triste 点头（群聊 **2026-11-09**）+ 调研建议（505 开发集隔离 held-out；别急开 P1/重训），协议冻结为 **#53+#51**，该门禁现已由 #63 做完。自动标签只是建议，人审包仍开放，**不**把 #51 的 kcal 下降写成身份成功。**Triste 2026-11-06**：代码/结果可以分批推；派工前仍须先过 STATUS「下一步」门禁。**不得**助手用 fast 代跑 Full-set、审计或未解锁训练。

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
**现行缺口**在文首「下一步」：**P0a #37**、**P0b #39**、**P0c #41（`54c32e8`）**、**Q 校准 #44（`3fc611d`）**、**P0d #46（`b9c4c5e7`）**、**误差审计 #48（`91a05d8`）**、**2026-11-06 排序（先 #51 再 #53，P1 暂缓）**、**food-id #51（`6c8fda02`）**、**覆盖约束 #53（`2a406df4`，开发检查点）**、**① 反事实 Q #57（`0aa30537`）**、**② food-id 语义审计 #60（`c52ffaa2`）**、**③ 外部 held-out #63（`27d1dbfa`）** 已完成 → **当前第一项 = Cafe2 能量审计（已解锁；不是门禁 ④）**（**④–⑦ 锁定**；代码只在本文件进入主分支之后才许开工。解锁：Triste 点头「继续吧」，群聊 2026-11-09 晚 Asia/Shanghai，+ 调研建议。**主声称：质量稳赢可外推；能量开发集赢、外测未稳。** 自动标签只是建议，人审包仍开放，**不**把 #51 的 kcal 下降写成身份成功）。Ours-old（#24–#34）已落盘作对照。**P1 未解锁 / 暂缓；禁止自动开 P1；禁止重训 Mass/Q。** #53 **不**证明方法。① 一步评测 **Q ≈ 规则 ≪ oracle**，加固该归因。试标 30（kappa **0.7–0.8**）仍为 P2。**暂不上 RL。**

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
| 方案 MD | 根目录 `PLAN.md`（2026-11-06 起与 Canvas「Coverage-Constrained Active Specialist Selection」对齐；任务框架仍是 Evidence-Guided Active Multimodal Nutrition Estimation） |
| P0 离线脚手架（未训练） | 设计 `docs/STATE_FEATURE_DESIGN.md`；混淆 / Provenance / Selective 脚本在 `scripts/`；mock `tests/fixtures/fndds_mock_foods.json`；草图 `vdr/controller/state_features.py`（`CLOSED_LOOP_READY` 假）。开环 31 维仍是 B5。**2026-11-04 离线冒烟已跑完**（数字在文首「助手进度」；不是方法证明，不是训练）。索引 [`docs/VDR_P0.md`](VDR_P0.md) |
| Q/Cost 校准 #44 | fit-only terminal-margin **τ≈0.001645**；hold-out 一致率 **0.4489→0.7712**（**n=2771**）；产物 `results/qphi/calibration*.json`、`report_calibration.txt`；**无 RL**；**未**改 `q_phi.npz` 权重；冻结 weak_steps / n5k **未改** |
| P0d 同预算 #46 | paired **n=505** Ours-new E **98.5521** / M **58.9952** / tool **4.6970**；vs B0 ΔE **+1.4210**（能量未稳赢）；产物 `results/p0d/`；squash **`b9c4c5e7`** |
| P0d 误差审计 #48 | 只读 #46；paired ΔM **−17.3513** / ΔE **+1.4210**；产物 `results/audit/report_error_audit.md`（+ json/jsonl/figs）；脚本 `scripts/p0d_error_audit.py`；squash **`91a05d8`** |
| 共决下一步（Triste + 调研） | **2026-11-06** 已执行：先 food-id/密度（#51），再覆盖+早停（#53），暂缓 P1。**同日外部评审**把 #53 标为开发检查点，下一门禁改为 **①–⑦**。**① 已由 #57 完成。** 当时 **②** 尚未写入文首 |
| 解锁 ②（Triste + 调研） | **2026-11-07** 群聊点头 + 调研建议（CORRECT_CODE 几乎无校准 → 身份审计马上做）。该解锁随后由 **#60**（squash `c52ffaa2`）做完。当时约定：代码只在本文件进入主分支之后才许开工；**③–⑦** 与 P1 / Mass·Q·τ 重训仍锁。不重跑 #51 / #53 / #57 |
| food-id / 密度轻修 #51 | 同头 FNDDS 码替换（已有检索窗，≥15% 靠近 calorie head）；克重与动作保持。paired **n=505** 能量 **98.5521 → 92.6963**（质量 **58.9952** 不变）；vs Direct **ΔE +1.42 → −4.43**。多组件全写 **n=50** food-table **190.79 → 131.65**；**37/50**（**30↓ / 7↑**），**68** codes；head **77.75** 为目标，记分仍是 food-table。**827** mass / **0** hollow。squash **`6c8fda02`**；**#51 未改本文件** |
| 覆盖未写组件 + 早停 #53 | 冻结 Q+fence 后，未写组件无 `estimate_mass` 写则拒绝 STOP/ABSTAIN（defer rule utility）；回到检索 top code 则 portion，不开第二轮 top-code swap；**#51 同头 swap 保留**。paired **n=505** vs #51：能量 **92.6963 → 64.2792**；质量 **58.9952 → 32.9908**；tool mean **4.6970 → 6.0257**（max **17**，均在 K=32）。vs Direct **ΔE −4.43 → −32.85**；**ΔM −17.35 → −43.36**。全部组件已写 **199/505 → 503/505**；STOP-unwritten **78 → 0**；Mass 成功 **827 → 1479**；hollow **0**。残留 **2** 盘（`dish_1565020895`、`dish_1565020926`）honeycomb 检索失败，卡 RETRIEVE，不编造克重、未 STOP。**无** Mass/Q/τ 重训。squash **`2a406df4`**；**#53 未改本文件**。**开发检查点，不证明方法；增益 ≠ Q 学习改进** |
| ① 反事实 Q + dish_id 泄漏审计 #57 | 产物 `results/qphi_counterfactual/`。fit/hold-out 按 dish_id 分组，交集 **0**，无重训。头条 n=**1914**（rgb_train hold-out；coverage-choice + mass forward；与 #53 **505** 交集 **0**）：Q **0.6165** / **24.14** / **49.96**；规则 **0.6102** / **23.03** / **50.91**；固定 **0.5914** / **20.62** / **52.99**；随机 **0.4338** / **4.04** / **78.15**；oracle improve **82.19**、regret **0**。Q 约 **0.96** kcal 优于规则，约 **50** kcal 落后 oracle。**241/1914** 为 RE_DECOMPOSE no-op（冻结 Q 不能发出）；敏感性 Q **~17.33** vs 规则 **~18.28**。价值校准 n=**3475**：MAE **0.0226**，Pearson **0.747**；RE_PORTION **0.805**，CORRECT_CODE **0.016**。**Q ≈ 规则 ≪ oracle**。squash **`0aa305379a7cc02045d07ec9c513bb9c2a3d3061`**（短 **`0aa30537`**）。**#57 未改本文件** |
| ② food-id 语义审计 #60 | 只读 #51 `results/food_id_density/`（squash `6c8fda02`）。**68** 个同头换码，覆盖 **37/50** 多组件盘。自动标签（仅建议）：换码后 Exact **0** / Near **50**（**73.53%**）/ Wrong **18**（**26.47%**）；换码前同一 cafe 行 Exact **27** / Near **23** / Wrong **18**；Exact→Near **27**；变差 **28** / 变好 **1** / 不变 **39**。宏量 food-table vs cafe，finished 多组件 **n=50**：P **10.57→10.07**，C **13.00→16.17**，F **17.56→11.97**，E **190.79→131.65**（能量与冻结 #51 一致）。换码 **37**：碳水 MAE **+4.28 g**，能量 MAE **−79.92 kcal**。能量变好的 **30** 盘：全 Exact **0**，至少一码 Wrong **12**，仅 Near **18**；这 **30** 盘上 **50** 个码全是 Near 或 Wrong。**kcal 变好 ≠ 身份变对。** 人审：68 行全部 `needs_human_review=yes`；自动标签不关闭审计。文件 `human_review_components.csv`、`human_review_dishes.csv`、`human_review.md`。产物 `results/food_id_semantic_audit/`；脚本 `scripts/food_id_semantic_audit.py`。squash **`c52ffaa2`**。**#60 未改本文件**。不要重跑，不要改该目录。本条不定义 held-out。③ 已由 #63 完成（见「③ 外部 held-out #63」行）。当前第一项见「解锁 Cafe2 能量审计」。历史解锁见「解锁 ③」行 |
| 解锁 ③（Triste + 调研） | **2026-11-09** 群聊点头 + 调研建议（配对 **505** 仍是开发 / 诊断集；held-out 必须按 dish_id 隔离；别急开 P1 / 重训）。该解锁随后由 **#63**（squash `27d1dbfa`）做完。当时约定：代码只在本文件进入主分支之后才许开工。协议冻结为 **#53+#51** 同预算主路径（Coverage-Constrained + 密度 / 同头换码）。先划分脚本和 dish 列表（`results/heldout_plan/`），再跑该冻结协议，再交主表（**Energy MAE + Mass MAE**；基线 **Direct-VLM** 与 **Mass-only**；覆盖率与步数 / 工具次数）。**禁止**用 held-out 做方法选择、早停、阈值 / τ 或 GT 驱动的码 / 份量修改。当时 **④–⑦** 与 P1 / Mass·Q·τ 重训仍锁。不覆盖 #51 / #53 / #57 / #60。本行不是当前开工项 |
| ③ 外部 held-out #63 | Cafe2，**n=228**，与配对 **505**、fit（**3679** 官方 rgb_train 且 `pool_of_dish==fit`）、rgb_train、rgb_test、cafe1 的 dish_id **不相交**。划分 `results/heldout_plan/`；主表与轨迹 `results/heldout_eval/`。主表（三法皆有限，n=228）：Coverage-Constrained + density/same-head 能量 MAE **116.8494** kcal，质量 MAE **50.9483** g，覆盖 **0.9737**（222/228），步数均值 **6.2193**，工具调用均值 **5.2456**；Direct-VLM 能量 **105.6157**、质量 **73.7456**；Mass-only（冻结 Mass）能量 **75.3870**、质量 **51.7107**。能量 MAE 比 Direct-VLM **高 11.23** kcal，比 Mass-only **高 41.46** kcal；质量 MAE 比 Direct-VLM **低 22.80** g，比 Mass-only **低 0.76** g。STOP 且仍有未写组件 **0**。**6** 盘停在 RETRIEVE，组件仍未写完，没有编造份量。同头密度换码 **106** 盘。空壳 RE_PORTION **0**。工具调用中位 **4**、最大 **14**，**228/228** 在 K=32 内。Mass 写 **575**。Cafe2 盘级能量列每行都是 **0**，能量目标是配料卡路里之和；质量是盘级克重总量。这次 Direct-VLM 行记下 `placement_max_memory=8GiB`（共享 GPU）；同一 bf16 checkpoint 与 prompt，不是历史上满 GPU 的 `cuda:0` 放置。不要重跑，不要据此解锁重训。开发 505 的能量 MAE **64.2792**、质量 MAE **32.9908** **不是**本表的行。**#53 不证明方法。** #51 是密度 / kcal 启发式（自动 Exact=**0**）。一步审计上 **Q ≈ 规则 ≪ oracle**。**主声称：质量稳赢可外推；能量开发集赢、外测未稳。** **不**声称 held-out 上能量与质量双赢。squash **`27d1dbfa`**。**#63 未改本文件**。不要重跑，不要覆盖这两个目录 |
| 解锁 Cafe2 能量审计（Triste + 调研） | **2026-11-09** 晚（Asia/Shanghai）群聊点头「继续吧」，在调研建议之后。**当前第一项 = Cafe2 能量审计**（已解锁；唯一允许写代码的项；**不是门禁 ④**）。代码只在本文件进入主分支之后才许开工。这是 ③ 的诊断后续，与 #48 同口径，不是真实 Tool Cost，不是 ⑤–⑦，不是 P1，不是 Mass / Q / τ 重训。**505** 仍是开发 / 诊断集。只读冻结 Cafe2 **n=228**，不重画划分；只读 `results/heldout_plan/` 与 `results/heldout_eval/`。新产物例如 `results/heldout_energy_audit/`。不要覆盖 `results/heldout_plan/`、`results/heldout_eval/`、`results/food_id_density/`、`results/coverage_early_stop/`、`results/qphi_counterfactual/`、`results/food_id_semantic_audit/`。不要重跑 #63 闭环主表，也不要重跑 #48 / #51 / #53 / #57 / #60。**主声称：质量稳赢可外推；能量开发集赢、外测未稳。** 协议：（1）只在冻结的 Cafe2 n=228 名单上做诊断。不做方法选择，不早停，不调阈值 / τ，不换方法，不用 GT 改码或份量。Diagnostic only on the frozen Cafe2 n=228 list. No method selection, no early stopping, no threshold/τ tuning, no swapping the method, no GT-driven code or portion edits.（2）与 #48 同一组切分，写明白。Same cuts as #48, written plainly: 固定 food-id 时，mass 重写是否同时降低 energy（hold identity/codes fixed, then ask whether the mass rewrite also drops energy）。oracle 标签密度 + 我方克数（oracle label density with our grams）。另拆 #51 density swap 开/关，重点看换过码的 106 盘。（3）基线仍是 Direct-VLM 与 Mass-only。头条问题：相对 Mass-only 的能量损失 +41.46 kcal（质量几乎打平，−0.76 g）是否来自 food-id / 密度。Baselines remain Direct-VLM and Mass-only. The headline question is whether the energy loss vs Mass-only of +41.46 kcal (mass almost tied, −0.76 g) comes from food-id / density。**④–⑦** 与 P1 / Mass·Q·τ 重训仍锁 |

## 明确未完成

| 项 | 阻塞 |
|----|------|
| Cafe2 能量审计 | **当前第一项。已解锁；唯一允许写代码的项；不是门禁 ④。** 代码只在本文件进入主分支之后才许开工。③ 的诊断后续，与 #48 同口径，**不是**真实 Tool Cost，**不是** ⑤–⑦，**不是** P1，**不是** Mass / Q / τ 重训。配对 **505** 仍只是开发 / 诊断集。只读冻结 Cafe2 **n=228**，不重画划分。新产物例如 `results/heldout_energy_audit/`，不覆盖 `results/heldout_plan/`、`results/heldout_eval/`、#48 / #51 / #53 / #57 / #60。不要重跑 #63 闭环主表。**主声称：质量稳赢可外推；能量开发集赢、外测未稳。** 协议：（1）Diagnostic only on the frozen Cafe2 n=228 list. No method selection, no early stopping, no threshold/τ tuning, no swapping the method, no GT-driven code or portion edits.（2）Same cuts as #48, written plainly: 固定 food-id 时，mass 重写是否同时降低 energy（hold identity/codes fixed, then ask whether the mass rewrite also drops energy）。oracle 标签密度 + 我方克数（oracle label density with our grams）。另拆 #51 density swap 开/关，重点看换过码的 106 盘。（3）Baselines remain Direct-VLM and Mass-only. The headline question is whether the energy loss vs Mass-only of +41.46 kcal (mass almost tied, −0.76 g) comes from food-id / density. 解锁：Triste 点头「继续吧」（群聊 2026-11-09 晚，Asia/Shanghai）+ 调研建议 |
| ④ 真实 Tool Cost | **锁定。禁止自动开。** VLM 次数、延迟、token、单盘成本、能量 MAE–Cost Pareto |
| ⑤ 忠实 B1/B2 + 相关工作 | **锁定。禁止自动开。** 优先 arXiv 2610.11144、CVPRW 2025 Decomposing Food Images、NutriVision arXiv 2609.33076 |
| ⑥ Provenance + Semantic | **锁定。禁止自动开。** |
| ⑦ 共决 P1 / 重训 | **锁定。** ①–⑥ 有结论后再共决。在此之前禁止开 P1、禁止重训 Mass/Q/τ |
| P1 检索重排 | **锁定 / 暂缓。禁止自动开 P1。** 须等门禁 **⑦**。不得重训 Mass / Q / τ |
| 有俯拍 RGB 的官方 rgb_test（507/709） | **已完成（历史锚点）。** 能量 MAE：B0 **97.03**（Direct-VLM），B1 **110.72**（**非忠实 DietAI24**，主表须重做）。不要重跑。50 盘基线仍是 B0 **98.3**、B1 **88.2** |
| 无俯拍 RGB 的官方 test 盘 | 不进入这次命令，也不要写成 709 盘全表 |
| 步骤弱标签全量（仅 train） | **已完成。** fit 3679 + hold-out 380。没有新的生成命令。阈值仍只许在 hold-out 上看，禁止用 test GT |
| 试标 30 + 新标签 / kappa 0.7–0.8 | **人工，P2。** 指南仍是 v0，文首已标明相对 PLAN P2 部分过时。正式协议仍要扩到 Visibility / Code Match 等。不作 0.6 凑合 |
| 200–500 金标准 | **人工**，可分期；协议与 kappa 门禁不能省。若其中一部分再充当外部 held-out，仍须按菜品 / 来源隔离，并与配对 **505** 的 dish_id 不相交；**禁止**用于方法选择、早停或调 τ。**③ 已由 #63 完成**（Cafe2 **n=228**）。开标**不是**重训许可，也**不**自动开 **④–⑦** 或 P1，也**不**把 Cafe2 能量审计改叫门禁 ④ |
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
