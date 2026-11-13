# Verified Dietary Reasoning（VDR）实验方案

> **2026-11-13（Asia/Shanghai）导语：自本段以下的正文是已退役的 VDR 方案，只作历史对照，不是现行开工许可。**
>
> 现行方法真源是 Canvas「**ICDL 官方方案（2026-11-13）：从增量组成变化学习食物营养估计**」（https://cursor.com/canvas/store-745bdf38-f3d7-4313-8d66-08e4b6903e93/47e6af29-51f8-443c-954d-56b0c453b602）。工作名 ICDL（Incremental Composition-Difference Learning）。候选英文标题（未证实）：Learning Food Nutrition from Incremental Composition Changes。旧 VDR 页只作存档：https://cursor.com/canvas/store-745bdf38-f3d7-4313-8d66-08e4b6903e93/d7be617d-d37e-4fe1-a4b5-9da2c86b27a2 。ICDL 尚未被证明。会刊瞄准 2027；MM 2026 Regular Paper 截稿已过。不编造 2027 截稿日期。
>
> **第一阶段数据可行性审计已完成，到此停止。** 有效相邻阶段对 **1400**，独立餐盘序列 **692**，覆盖 **2094** / 5,006 个 dish_id。这是已经数完的增量序列，不要改回未知，也不要另编一套。下一步是等 Triste 点头；点头之后，审计拉取请求可以 squash 进 `icdl`，不得进 `main`。详情在 [`docs/STATUS.md`](docs/STATUS.md)。不要把这一段改写成新的代码项，也不要解锁第二阶段。
>
> **锁定**：第二阶段（三个轻量模型与跨盘随机配对负对照）、第三阶段（论文价值与外部迁移）、完整 ICDL 网络、Agent、强化学习、OmniFood8K 外测、ACETADA 外测、不完整食材监督。停止规则已写入方案，不等于解锁第三阶段，也不解锁外测。OmniFood8K 留到第三阶段做跨菜系外部泛化验证，不得提前当开发集，也不得在外部盲测前用它调参或泄漏标签。ACETADA 只作辅助外测，且须区分实际摄入量与餐前供应量，不能混合不同评价目标。505 与 Cafe2 只作历史证据与开发诊断，不再当独立验证集，不要在这两个集上调新方法。
>
> 不要改 `main`（空初始提交 `d43cb084eba7693a45c114dfdf06b258611463f4`，短 `d43cb08`）。工作分支 `icdl` 从备份提交 `28bd3c9c` 拉出，该提交已含 #73。不要把新方案合进 `main`，不要从空的 `main` 另开分支，不要再合一次 #73。推理必须仍是单张 RGB。
>
> 下文凡把双视角实验写成当时的下一步或开工许可，都是 2026-11-12 的历史句，2026-11-13 起失效。


> **自本段起是已退役的 VDR 历史正文，不是现行真源。** 方法方案以 Canvas 为准（2026-11-12 晚，Asia/Shanghai）：「**VDR 官方方案（2026-11-12 晚）：面向物理约束营养估计的主动多模态证据获取（候选）**」。同日早间把少样本质量校准当作候选创新点二的定位，以及 2026-11-06 的旧标题，都不再作现行主张。  
> Canvas：https://cursor.com/canvas/store-745bdf38-f3d7-4313-8d66-08e4b6903e93/d7be617d-d37e-4fe1-a4b5-9da2c86b27a2  
> 算力假设：**2×4090D 24GB**。数据：**公开 Nutrition5K + FNDDS 2019–2020**；人工 **200–500** 盘薄标注；**无私有大规模采集**。  
> **项目名仍是 VDR。** 输出仍是 \(y=(n,G,s,\tau)\)。**候选定位（未证实）**：Active Multimodal Evidence Acquisition for Physically Grounded Food Nutrition Estimation（面向物理约束营养估计的主动多模态证据获取）。研究问题升级为：多模态模型如何理解物理尺度、食物组分与营养密度的耦合，并主动获取最有价值的视觉证据。输入协议是低成本多视角 RGB；主表仍不要求 RGB-D。若只能得到单张 RGB，这条路线严重受限。**2026-11-12 晚模块**：VLM 食物理解是语义分支基础；Mass Estimator 是预训练几何-质量专家，冻结结果作历史证据；FNDDS 是营养密度知识来源；Coverage Constraint 是可选的输出完整性机制，保留为辅助；Nutrient Verifier 保留为辅助；Q_φ 与同头密度换码降为对照，停止继续训练 Q、停止继续改 food-id；分层少样本质量校准是普通基线，不再是创新点二；两路径能量融合 / Energy Router 仅作对照。该页取代同日早间「覆盖约束组分分解 + 少样本质量校准」定位，也取代以前「Coverage-Constrained Active Specialist Selection + Error-Aware Q（Route A 能量+质量双赢）」的主张写法。蛋白质 / 碳水 / 脂肪仍作辅助。**暂不上 RL。**  
> **历史对照**：旧中心「Evidence-Aware Closed-Loop Tool Control」仅作对照叙事，不再作主标题。更早的任务框架名 Evidence-Guided Active Multimodal Nutrition Estimation 不再作现行方法标题。#53 是开发检查点，不是本卖点的证明。2026-11-06 把技术中心写成 Coverage-Constrained Active Specialist Selection + Error-Aware Q 的那一版，自 2026-11-12 起改为上面的模块定位；同日晚再撤回少样本校准作为创新点二。  
> **会刊**：ACM MM 2027 或更晚。MM 2026 Regular Paper 截稿为 2026-04-01 AoE；会议 2026-11-10 至 2026-11-14（里约热内卢）。MM 2027 在香港；日期与截稿尚未公布。不虚构 2027 截稿。

---

## 0. Diff vs 旧 Closed-Loop（必须读）

| 诊断 | 内容 |
|------|------|
| **#34 on-policy rgb_test** | Primary **n=488 STOP**：Ours 与 B5 **Full MAE 打平 96.2769** kcal；Ours **AURC 47.17 ≤ B5 81.59**。改善了**风险排序**，**没有**改善预测能力。 |
| 根因 | 79→64→64→6 控制器主要学「选动作名」；缺真正提高证据质量的 **Specialist Models**（尤其 **RE_PORTION** 需独立 DL 份量估计，禁止空壳）。 |
| 弱弃权 | `abstain_score=clip(committed_kcal/2500,0,1)` **降权**为负面对照；须拆 **U_id / U_mass / U_nutrient / U_evidence**。 |
| FixRate | #26 **Δ0.5554**（CI≠0）**保留**；须审计 CORRECT_CODE / RE_PORTION 后**终态 n 是否真变**——纠错有效 ≠ 最终预测有效。 |
| 主创新迁移 | 从「弃权 / 开环 BC 小修 / AURC 补丁」→ **Mass + 主动专家选择 + Q_φ**；AURC / 弃权降为**辅助**。 |
| 开环 31 维 BC | 仍仅消融 **B5**；State-aware BC 仅**启动 / 消融**，不作主贡献标题。 |
| RL | **暂不上 RL**。相对旧闭环，当时的 MVP 以监督 **Q_φ(z,a)** 为中心；2026-11-12 晚起 Q_φ 只作对照，停止继续训练。 |

---

## 1. 一句话定位

提出 **Verified Dietary Reasoning（VDR）**：输出

\[
y = (n,\ G,\ s,\ \tau)
\]

- \(n\)：营养向量（至少 energy / protein / carb / fat / mass）
- \(G\)：证据图（链：Region → Component → FNDDS Code → Portion → Nutrient）
- \(s \in \{\mathrm{pass},\ \mathrm{corrected},\ \mathrm{abstain}\}\)（支持盘级 / 成分级弃权）
- \(\tau\)：有序纠错轨迹（动作 + **工具观测**）

**主投 ACM Multimedia（CCF-A）**；备选 AAAI/IJCAI。

### 三条主贡献（2026-11-12 起以 Canvas 为准）

1. **任务**：形式化 VDR；\(y=(n,G,s,\tau)\)；证据链可程序复算 + 可人工语义核；支持盘级/成分级弃权。
2. **已冻结控制器（2026-11-12 起不再当作全部贡献）**：Coverage 是辅助机制，Q_φ 是对照。已落地的选择式仍是：未写完组件时 \(A_{legal}\) 去掉 STOP/ABSTAIN，再在合法集上 \(a^*=\arg\max_{a\in A_{legal}} Q_\phi(z,a)/\mathrm{Cost}(a)\)，\(Q_\phi(z,a)=\mathbb{E}[\Delta L\mid z,a]\)。Mass Estimator 已落地并冻结；图文检索重排（P1）暂缓；Grounding 非主创新；Nutrient Verifier 为护栏。**State-aware BC 仅 bootstrap/消融。** #51/#53 在冻结 Mass 与 Q/τ 上的开发集变化是确定性策略，**不是** Q 学得更好。#57 一步反事实同向：Q ≈ 规则 ≪ oracle（约 0.96 kcal、远落后 Oracle）。**2026-11-12 晚**：Mass Estimator 是预训练几何-质量专家，冻结结果作历史证据。Evidence-aware Few-shot Mass Calibration 降为普通基线，不再是创新点二；#70（squash `d622e6db`）的 Phase 1 已取消，不实现其公式或分层结构。两路径能量融合 / Energy Router 只作对照，不预定为贡献。候选关系（未证实，本文件不据此写网络）是：组分质量 \(m_i=V_i\rho_i\)，盘级营养 \(N=\sum_i m_i D(c_i,h_i)\)；用尺度、身份、密度的联合不确定性驱动下一证据。这不是单独一个 3D 分支，也不是 Q 选择专家。
3. **证明（外测已记下，但不是能量与质量同时获胜，也不是 #53 或 #57 已完成该证明）**：#63（squash `27d1dbfa`）在 Cafe2 held-out 上记下能量与质量 MAE（见 §14，本句不重抄表内数字）。**主声称：质量对 Direct 可外推（bootstrap 区间均为负）；对 Mass-only 基本打平（0.76 g 不算赢，不要当作方法优越性证据）；能量外测未稳。** #57 已记下反事实动作价值（一步评测 Q ≈ 规则 ≪ oracle，见 §14；不单独构成该证明）。真实成本 Pareto、Provenance / Semantic、忠实 B1/B2，以及相对 arXiv 2610.11144、CVPRW 2025 Decomposing Food Images、NutriVision（arXiv 2609.33076）的差异仍在锁定门禁 **④–⑦**。开发集超过 Direct-VLM **不**等于该证明。Coverage / Selective / AURC 仍辅助。

---

## 2. 硬禁止

| 禁止 | 原因 |
|------|------|
| 把 **kcal-proxy AURC** 当主创新 | #34 只改善排序不改善预测；审稿质疑按规模降 Coverage |
| **空壳 RE_PORTION** | 无独立 DL 份量估计则无法改善终态 \(n\) |
| 工具**全串行**冒充 Agent | 选择必须由 \(A_{legal}\) 与 \(Q_\phi\) 驱动 |
| **RGB-D 混进 RGB 主表** | 主协议与现 rgb_test 一致 **仅 RGB** |
| 上 **RL** | 采纳边界：暂不上 RL |
| 用 **test GT** 调参 | 泄漏 |
| 把 **#51/#53** 写成 Q 学得更好 | Mass / Q / τ 冻结；增益是确定性策略 |
| 把开发 **505** 当独立最终测试 | #48→#51→#53 已形成测试集反馈 |
| 现在开 **P1** 或重训 Mass/Q | 须先走完 STATUS ①–⑥，再由 ⑦ 共决 |
| weak-step 按步随机拆 fit/hold-out | 同一 dish 泄漏 |
| **开 fast Cloud Agent** | 资源边界 |
| **体积工具默认开** | Tanabe 警示；默认关 |
| 把开环 31 维 BC 写回主方法 | 仍仅消融 **B5** |
| 加载旧营养回归权重 | 污染贡献 |
| 私有大规模采集替代公开主表 | 复现性 |
| **编造 DOI** / 未核对文献 | 写「待补正式引用」 |
| 把 B1 叫「弱 RAG-once」或 B2 叫 Fixed-MoE | 对照不忠实 |

**2026-11-12 追加（不改上表已有行）**：主表只 RGB，depth 只进附录。不靠弃权制造质量优势。AURC / 弃权不作主故事。FNDDS 默认 2019–2020，换版触发全基线重跑。禁止把相对 Mass-only 的 0.76 g 写成优越性。Cafe2 不再用于新方法调参，也不再当下一轮完全盲测。**2026-11-12 晚再追加**：禁止把少样本 / 分层质量校准写成创新点二；禁止把 #70 的 Phase 1 当现行下一步；在外部配对第二视图的缺口被正视之前禁止搭建完整联合网络；当时本文件只解锁双视角实验，不解锁后续实验或完整联合网络；2026-11-13 起该解锁失效；禁止编造 Nutrition5k 可用配对条数，也禁止把非官方目录条数写成官方配对数。官方只写约 3.5k / 5,006 盘含顶视 RGB-D（论文约数）。禁止编造 ACM MM 2027 截稿。外部集许可未核对，不得写成已可商用或已可投稿使用。

---

## 3. 研究问题与假设

- **RQ1（预测）**：相同 VLM / FNDDS / 工具预算下，接入 **Mass Specialist** 并以误差来源驱动的主动选择，能否显著降低 Full-set MAE@energy 与 **mass MAE**（相对 **Ours-old** / B5 / B0）？
- **RQ2（机制）**：按 **U_id / U_mass / U_nutrient / U_evidence** 选专家，是否优于固定流水线、随机调用与开环不读观测？掩蔽 U 或锁死 Mass 后增益是否消失？
- **RQ3（对标）**：相对忠实 B1/B2，优势是否来自主动专家选择 + 可验证证据链，而非再堆一条检索管线？

| 假设 | 内容 |
|------|------|
| **H1** | Mass + \(Q_\phi\) 后 Full-set MAE@energy 与 mass MAE 显著优于 Ours-old / B5；Selective / AURC 不显著变差或更好 |
| **H2** | 掩蔽 U_* 或锁死 Mass → MAE 回升；固定流水线 / 随机劣于 \(Q_\phi\) |
| **H3** | 相对忠实 B1/B2，优势在主动专家 + 证据；无法完全复现则诚实声明 approx |

**主张边界（2026-11-12）**：输出仍可含 \(n=(E,P,C,F,M)\)。冻结 Q 的监督核心仍是能量误差改善，但 Q 已降为对照，停止继续训练。论文现行主声称不是能量与质量同时获胜：质量对 Direct 可外推（bootstrap 区间均为负），对 Mass-only 基本打平（0.76 g 不算赢），能量外测未稳。蛋白质 / 碳水 / 脂肪作辅助。多目标 Q（Route B）不开。

**候选研究问题（2026-11-12 晚，未证实，本文件不据此开工）**：多模态模型如何理解食物图像中物理尺度、食物组分与营养密度的耦合，并主动获取最有价值的视觉证据。要展示的关系是质量、体积与密度不独立，能量也不是质量的简单放缩。预期差距是尺度、身份、密度的联合不确定性，以及由该因子化驱动的证据获取。Open-KNEAD（arXiv:2607.12911，2026-07-14）已把邻近方向列为未来工作。若只做少样本校准或单独 3D 分支，差距不成立。上表 RQ1–H3 是已冻结 VDR 控制器的历史问题，不是实验 B、实验 C 或完整联合网络的开工许可。双视角实验的旧开工许可已退役；现行许可只在 [`docs/STATUS.md`](docs/STATUS.md) 文首的第一阶段审计。Q1 有条件支持源域双视图设计；Q2 未完全通过，外部混合餐基本没有配对第二视图，因此先不要搭建完整联合网络，本轮也不把主动证据获取做到外部集；Q3 不得写成已解，当时的双视角实验现已做完并停止，不是再开工；Q4 可描述，但偏薄。

---

## 4. 四层架构（技术中心）

输入膳食图像 \(x\)（**主表 RGB-only**）。四层：

1. **Initial Visual Understanding** — Qwen 等 VLM → 假设 + 初始证据链  
2. **Deep Learning Specialist Tools** — Mass / Retrieval+rerank / Grounding / Nutrient Verifier  
3. **Coverage-Constrained Active Controller** — 未写完组件时 \(A_{legal}\) 去掉 STOP/ABSTAIN，再按 \(Q_\phi/\mathrm{Cost}\) 在合法集上选工具  
4. **Verified Nutrition Output** — \(\hat n=\sum \hat m_i f(c_i)\) + 证据链 + 轨迹  

| 层 | 要点 |
|----|------|
| Initial VLM | 冻结或轻量格式对齐；禁旧回归权重 |
| DL Specialists | **真正改善证据质量**；RE_PORTION 须接 Mass |
| Coverage-Constrained Controller | 先 \(A_{legal}\)（未写完则去掉 STOP/ABSTAIN），再监督 \(Q_\phi/\mathrm{Cost}\)；**非 RL**；BC 仅启动/消融 |
| Verified Output | 可程序复算；断链 → ungrounded |

---

## 5. 专家工具优先级

| 优先级 | 专家 | 骨干 / 接口 | 主创新? |
|--------|------|-------------|---------|
| **P0** | **Mass Estimator** | ViT-B/16 或 ResNet101；ImageNet → N5K train 微调；**RGB-only 主表**；多任务 **[M,E,P,C,F]+σ**；`estimate_mass(image, candidates)` | **保留为预训练专家**；冻结结果是历史证据，不是本晚候选贡献本身 |
| **P1** | 图文检索 + FNDDS 重排 | SigLIP/CLIP + 向量库 + cross-modal/VLM rerank | 次主，**现暂缓**（⑦ 之前不开） |
| **P2/P3** | 区域定位 | Grounding DINO / SAM2 + 局部再问 VLM | **否** — 不当主创新 |
| 配套 | Nutrient Verifier | FNDDS 查表 + 规则/一致性 | 护栏 |

Grounding 为 P2/P3：细粒度分割未必自动改善营养 MAE。Effects of segmentation granularity on nutrient prediction from real-world food images（doi:10.3389/fnut.2026.1927530）在 1,056 张 Nutrition5K 子集上：未分割均值 PMAE 19.91%，最佳二值食物区域 20.18%，配料级 23.76%–25.62%。本晚候选不是再加一个分割网络或单独的 3D 分支。

**P1 检索重排仍暂缓**，STATUS 门禁 ⑦ 之前不开，也不重训 Mass / Q。**Coverage Guard** 已在 #53 落地，应写成 \(A_{legal}\) 方法，不是空壳，也不是 Bug Fix。#51 同头换码已落地；语义审计（门禁 **②**）已由 #60 完成（见 §14）。**①②③ 已完成。Cafe2 能量审计已完成（#65，squash `ecf859f5`）。Mass-only 能量路径便宜诊断已完成（#68，squash `f578e8f4`）。**（历史，不是现行门禁。2026-11-12 晚曾把下一步写成 Nutrition5k 双视角实验。该实验已在备份提交 28bd3c9c 中做完并停止，不要重跑，不要再合入。现行下一步只在文首：第一阶段数据可行性审计。）** #70 的 Phase 1 已取消。实验 B、实验 C、完整联合网络与 ④–⑦ 仍锁。双视角实验曾按当时的 STATUS 解锁；该许可已退役，不要重开。**禁止**自动开实验 B、实验 C、完整联合网络与 **④–⑦**。

---

## 6. 非 RL 控制器

**状态** \(z_t = [U_{id}, U_{mass}, U_{nutrient}, U_{evidence}, D_{models}, o_{t-1}, b_t]\)

\[
Q_\phi(z,a) = \mathbb{E}[\Delta L \mid z,a]
\]

未写完组件时，终止动作不进合法集（#53 写成方法，不是 if-else 修补）：

\[
A_{legal}(s_t)=A(s_t)\setminus\{\mathrm{STOP},\mathrm{ABSTAIN}\}
\]

仍有未写完组件时用上面的合法集；否则 \(A_{legal}(s_t)=A(s_t)\)。

\[
a^*=\arg\max_{a\in A_{legal}(s_t)}\frac{Q_\phi(s_t,a)}{Cost(a)}
\]

覆盖约束只保证**初始清单内**未写完的组件会继续做合法质量估计，**不能**证明 VLM 没有漏掉整道食物。#51 的同头换码与 #53 的这道门都在冻结 Mass 与 Q/τ 上完成；开发集上的变化**不是** Q 学得更好。监督仍按 Route A 看能量误差改善；质量 MAE 是并列主指标；宏量是辅助。**非 RL。**

| 分量 | 含义 | 触发 |
|------|------|------|
| \(U_{id}\) | 身份/码不确定 | 检索重排 / CORRECT_CODE |
| \(U_{mass}\) | 份量/质量不确定（含 σ） | **Mass Estimator** |
| \(U_{nutrient}\) | 营养一致性冲突 | Nutrient Verifier |
| \(U_{evidence}\) | 证据链完整度/断链 | Grounding 或 PARTIAL ABSTAIN |

动作：调用 Mass / Retrieval / Grounding / Verifier / CORRECT_CODE / RE_PORTION（**须接 Mass**）/ PARTIAL ABSTAIN / STOP。**体积默认关。** 仍有未写完组件时，STOP 与 ABSTAIN 不在 \(A_{legal}\) 里。

训练：先在 train 上收集调用前后 \(\Delta L\) 监督；可先**规则启发式 utility**，再学 \(Q_\phi\)。**State-aware BC → 仅 bootstrap/消融。**

---

## 7. 证据链与两类效度

强制链：**Image Region → Component → FNDDS Code → Portion → Nutrient**。Portion 须可溯至 Mass 或声明源；**禁止空壳 RE_PORTION**。

| 类型 | 定义 | 谁评 |
|------|------|------|
| **Provenance Validity** | 程序复算：\(n\) 是否等于 FNDDS(code)×portion 聚合 | 自动脚本 |
| **Semantic Grounding** | 人判：码/份量/可见性是否与图像一致 | 人工金标准 |

**FNDDS release 钉死：2019–2020**（5624 codes，对齐 DietAI24）；换版全员重跑。

---

## 8. 人工金标准

| 字段 | 含义 |
|------|------|
| Visibility | Visible / Ambiguous / Hidden |
| Code Match | Exact / Near / Wrong / Unsure |
| Portion Confidence | High / Mid / Low |
| Evidence Completeness | Complete / Partial / Broken |
| Should Abstain | 盘级 + 成分级 partial abstain |
| Conflict | id / portion / db_gap / sauce / multi / other |

**两遍标注**；双人 + 第三人裁决。kappa 目标 **0.7–0.8**。规模 200–500；后移但不砍。

---

## 9. 主指标（顺序 = 优先级）

| 优先级 | 指标 | 角色 |
|--------|------|------|
| **主** | Full-set **MAE@energy** | Route A 主 claim。开发对照可以看；**最终主表必须来自隔离 held-out**，不能用当前开发 505 |
| **主** | **mass MAE** | 与能量并列的 Route A 主指标，用来看 Mass 与覆盖约束 |
| **主** | **真实 Tool cost** | 同预算调用次数不够。须记 VLM 调用次数、延迟、token、单盘成本，并报能量 MAE–Cost Pareto（门禁 ④）。调用次数 ≠ 真实成本 |
| 辅 | Coverage / Selective MAE / **AURC** | 可信曲线；**不作主创新** |
| 辅 | Evidence Validity | Provenance + Semantic |
| 行为 | FixRate / WorsenRate（ε + 95% CI） | 须审计终态 \(n\) 是否真变 |

要报的数仍是 **Full-set MAE@energy + mass MAE + 真实 Tool cost**。宏量 MAE 为辅助。禁止 kcal-proxy AURC 作主 claim。开发 505 不是最终独立主表。**2026-11-12 现行主声称**不是能量与质量同时获胜：质量对 Direct 可外推（bootstrap 区间均为负），对 Mass-only 的 0.76 g 不算赢，能量外测未稳。

---

## 10. 基线（统一 VLM / FNDDS / 工具预算 K / RGB）

| 代号 | 定义 | 注意 |
|------|------|------|
| B0 Direct-VLM | 单次 VLM 无工具 | 历史能量 MAE **97.03** 可作锚点 |
| **B1 DietAI24** | **忠实复现**官方协议/代码 | 历史 **110.72 非忠实**；须重做；**不挡 Mass P0** |
| **B2 FoodAgent** | **动态专家路由** | 无法完整则诚实 approx |
| B3 Rule Verify | 纯规则控制器 | 规则上限 |
| B4 Self-Refine/CRITIC | 同预算文本自纠 | 预算对齐 |
| **B5 Open-loop BC** | 原 31 维不读 obs | **仅消融**；#34 与 Ours-old MAE 打平 |
| **Ours-old** | 仅闭环控制器 / 弱 kcal-proxy / **无 Mass Specialist** | **对照**；不再写作主方法 |
| **Ours-new（dev）** | Mass + 校准 Q + #51 码表 + #53 覆盖约束 | **开发检查点**；非最终独立泛化；非 RL |

**B1 必须忠实重做**后再进主表。历史 B1 能量 MAE 不是忠实 DietAI24（数字在 §13，本节不改）；scaffold / dry-run 不算完成。

优先对照的相关工作（不编造未核对的 DOI。下面 Frontiers 的 DOI 是官方 Canvas 已核对的那一个；其余用标题 + arXiv。刊名、arXiv 号与日期来自既有评审和 2026-11-12 晚 Canvas）：

- **DietAI24**（简报所给刊名 Communications Medicine 2025；代码 https://github.com/Runz96/DietAI24）：忠实复现为 B1。数据库检索本身不作主创新。
- **Decomposing Food Images**（CVPRW 2025）：两步分解本身不构成新意。
- arXiv **2610.11144**（2026-10，食物身份验证 / 恢复；https://arxiv.org/abs/2610.11144）：与覆盖约束有交集，是重点对比。须划清「清单内写齐」和「身份验证 / 恢复」，并补人工身份审计。
- **NutriVision**，arXiv **2609.33076**（2026-09；https://arxiv.org/abs/2609.33076）：专用营养预测方向，不是同型 Agent。
- **Open-KNEAD**，arXiv **2607.12911**（2026-07-14；https://arxiv.org/abs/2607.12911）：agentic 分解、FNDDS、逐项质量与组分营养。文中把学习式盘级能量路由器，以及多视角或尺度参考下的视觉工具，列为未来工作。
- **Single-Image Food Mass Estimators Do Not Transfer Their Gram Scale Across Datasets**，arXiv **2610.29025**（2026-10-21；https://arxiv.org/abs/2610.29025）：与早间少样本质量校准重叠。该文已知食物设定下，每类 1 张标注照片把 ASA24 测试 MAE 从 42.6 g 降到 28.3 g（2 张 27.3 g，3 张 26.7 g）。这是该论文的结果，不是我们的结果。少样本校准因此只作普通基线。
- **Effects of segmentation granularity on nutrient prediction from real-world food images**，doi:10.3389/fnut.2026.1927530：1,056 张 Nutrition5K 子集上，未分割均值 PMAE 19.91%，最佳二值食物区域 20.18%，配料级 23.76%–25.62%。
- **MFP3D**，arXiv **2411.10492**；**Food Portion Estimation via 3D Object Scaling**，arXiv **2404.12257**；**Implicit-Scale 3D Reconstruction**，arXiv **2602.13041**；可选对照 **Size Matters**，arXiv **2601.20051**。单独 3D 分支不够。
- **ACETADA**，arXiv **2507.07048**（https://arxiv.org/abs/2507.07048）：实验使用 806 张餐前智能手机图像；配对餐后是剩余食物，不是几何侧视。

---

## 11. 必出实验

- **机制（STATUS ①，已完成 #57）**：反事实动作价值——同一状态下 oracle / \(Q_\phi\) / 规则 / 随机 / 固定流水线。一步评测上 Q ≈ 规则 ≪ oracle（数字在 STATUS）。开发集 MAE 仍然不要写成 Q 的证明。
- **M1 主**：Full-set MAE@energy + mass MAE（Ours-new vs B0/B5/Ours-old）+ Tool cost  
- M2 辅：Coverage / Selective / AURC  
- M3：Provenance 全量；Semantic 提前到核心实验（门禁 ⑥，仍排在 ① 之后）  
- M4：Fix/Worsen + **终态 n 变更审计**  
- A1 锁死 Mass → 增益消失  
- A2 掩蔽 U_* → MAE 回升  
- A3 固定流水线 / 随机 vs \(Q_\phi\)  
- A4 B5 / Ours-old 负面对照  
- A5 忠实 B1/B2（不挡 Mass P0）  
- F1 案例：Mass 纠正 / 检索重排 / PARTIAL ABSTAIN  

早期取舍是保住主表能量、质量、工具成本与机制消融。**现行顺序以 §14 为准（①②③、Cafe2 能量审计与 Mass-only 能量路径便宜诊断已完成；（历史，不是现行门禁。2026-11-12 晚曾把下一步写成 Nutrition5k 双视角实验。该实验已在备份提交 28bd3c9c 中做完并停止，不要重跑，不要再合入。现行下一步只在文首：第一阶段数据可行性审计。）不是门禁 ④；④–⑦ 仍锁）**；切勿再把 Policy Acc、kcal-proxy AURC 或开发 505 当最终主表。真实 Tool cost 仍是门禁 **④**，该诊断已完成，不在这里开。

---

## 12. 周程重排

| 周 | 内容 |
|----|------|
| **W1（门禁）** | **先**审计 on-policy/hold-out traces：CORRECT_CODE / RE_PORTION（及同类工具）后终态 \(n\) **变更率 / 无变化比例**；**审计通过前不得开训 Mass** |
| **W1–4（审计后）** | **再** Mass P0：训练 Mass Estimator（RGB-only）；接入 `estimate_mass` |
| **W3–6（Mass 接入后）** | **再** 收集 \(\Delta L\)；定义 **U_***；规则启发式 → **\(Q_\phi\)**；同预算 Ours-new vs B5/B0/Ours-old 主表初稿 |
| W5–8 | P1 检索+重排；消融锁 Mass / 掩蔽 U |
| W7–10 | 忠实 B1/B2；Grounding 可选附录 |
| W10–13 | 人工金标准 / 试标 kappa 0.7–0.8 |
| W14–16 | 写作投稿 ACM MM |

**B1/B2 并行，不挡 Mass P0。**

上表是早期周程。**现行严格顺序以 §14 与 [`docs/STATUS.md`](docs/STATUS.md) 为准**：**①②③、Cafe2 能量审计与 Mass-only 能量路径便宜诊断已完成**；**（历史，不是现行门禁。2026-11-12 晚曾把下一步写成 Nutrition5k 双视角实验。该实验已在备份提交 28bd3c9c 中做完并停止，不要重跑，不要再合入。现行下一步只在文首：第一阶段数据可行性审计。）**（**不是门禁 ④**，也不是 Phase 2、Phase 3 或新网络；**④–⑦** 仍锁定；**禁止**自动开）；P1 与 Mass/Q 重训仍锁。

---

## 13. 历史冻结数（保留，降权为对照）

| 项 | 数字 | 新解读 |
|----|------|--------|
| **#34 on-policy** | Full MAE **96.2769** tie B5；Ours AURC **47.17 ≤ 81.59**；n=488 STOP | **Ours-old** 对照；排序改善非预测 |
| **#31 offline** | n=356；Full MAE **30.7085** tie；Ours AURC **13.53 ≤ 32.15** | offline logged；**勿混** on-policy n=488 |
| **#26 FixRate** | Δ **0.5554** CI≠0 | 保留；须终态 \(n\) audit |
| B0 能量 MAE | **97.03**（rgb_test 507） | Direct-VLM 锚点 |
| 旧 B1 能量 MAE | **110.72** | **非忠实** DietAI24 |
| 开环 BC Acc | 0.9206 / **0.9700** | 仅 B5 |
| ABSTAIN | **0/24**（开环） | 红警 |
| DAgger r1 Acc | 0.9592 | 开环 B5 参照 |
| 弱 `abstain_score` | clip(kcal/2500) | 负面对照；拆 U_* |

---

## 14. MVP-for-A（严格顺序；前一项未完成不得跳步）

本节标题里的 A 是早期 MVP 代号，不是 Nutrition5k 双视角实验。该双视角实验的开工许可已退役。现行许可只在 [`docs/STATUS.md`](docs/STATUS.md) 文首的第一阶段审计，且只解锁那一项。实验 B、实验 C 与完整联合网络仍锁。

本轮不补采、也不新造外部混合餐的配对第二视角；用收缩评价范围代替数据补救。实验 A 仅在 Nutrition5k 同时具备顶视 RGB 与侧视 RGB、可按 dish_id 对齐的子集上进行；先核可用规模，不编造配对条数；侧视不作度量双相机。ACETADA、VLM Food Benchmark 本轮不做双视角或主动获取主测（仍缺真实可请求的第二视图）。ECUSTFD 最多作辅助几何对照，不作混合餐主测试。Cafe2 仍是已探索诊断集。完整联合网络、实验 B、实验 C 仍锁。停止规则不变：若简单融合与几何融合不能稳定胜过单视图（主要是俯视）基线，停下来让 Triste 判断，不要强行第二视角，也不要自行换方案。无 RL，不开 fast Cloud Agent，RGB-D 不进主表。Q3 不得写成已解。主指标是 Mass MAE，以及质量预测跨视角的一致性。主单视图基线是俯视图像。若几何/语义融合不能稳定胜过单视图基线，不要强行第二视图。

**① 已完成（#57，squash `0aa30537`）。② 已完成（#60，squash `c52ffaa2`）。③ 已完成（#63，squash `27d1dbfa`）。Cafe2 能量审计已完成（#65，squash `ecf859f5`）。Mass-only 能量路径便宜诊断已完成（#68，squash `f578e8f4`）。**（历史，不是现行门禁。2026-11-12 晚曾把下一步写成 Nutrition5k 双视角实验。该实验已在备份提交 28bd3c9c 中做完并停止，不要重跑，不要再合入。现行下一步只在文首：第一阶段数据可行性审计。）** 除文首第一阶段数据可行性审计以外，没有别的当前可写代码项。不是门禁 ④，也不是实验 B、实验 C 或完整联合网络。以 [`docs/STATUS.md`](docs/STATUS.md) 为准。**禁止**自动开 **④–⑦**、实验 B、实验 C 与完整联合网络。只按 [`docs/STATUS.md`](docs/STATUS.md) 文首的第一阶段审计执行。不要据此改 main，也不要重开已停止的双视角实验；其余代码仍空闲。该句曾把本次改写写成双视角实验的开工许可，2026-11-13 起失效。**P1 仍暂缓 / 锁定。禁止**重训 Mass。**禁止**重训 Q。**禁止**重拟合 τ。**无 RL。不开 fast Cloud Agent。** #65 的解锁依据（已做完）：Triste 点头「继续吧」（群聊 2026-11-09 晚，Asia/Shanghai）+ 调研建议。本诊断的解锁依据（已做完，记录在 #67 / `7d5e034d`，不是新的开工许可）：Triste 点头「继续吧」（群聊 2026-11-10 早，Asia/Shanghai）+ 调研建议。#65 与 #68 是已完成的诊断。**主声称：质量对 Direct 可外推（bootstrap 区间均为负）；对 Mass-only 基本打平（0.76 g 不算赢，不要当作方法优越性证据）；能量外测未稳。** **不**声称 held-out 上能量与质量双赢。开发 505 的能量 MAE **64.2792**、质量 MAE **32.9908** **不是** Cafe2 主表的行。自动 Exact / Near / Wrong 只是建议，人审包仍开放；**kcal 变好 ≠ 身份变对**，不把 #51 的能量下降写成身份成功。#51 是密度 / kcal 启发式（换码后自动 Exact=0）。① 上 Q ≈ 规则 ≪ oracle。#53 冻结为 Ours-new **开发检查点**（配对 505 是开发 / 诊断集，不是最终独立泛化主表）。#51/#53 的开发数字不在这里改写。2026-11-06 的旧排序（先 #51 再 #53，P1 暂缓）已经做完，那套编号退役。

**已完成**：P0a–P0d、Q 校准 #44、误差审计 #48、food-id #51、覆盖约束 #53、**① #57**、**② #60**、**③ #63**、**Cafe2 能量审计 #65**、**Mass-only 能量路径便宜诊断 #68**。**#51/#53 增益 ≠ Q 学习改进**（Mass / Q / τ 冻结；① 一步评测 Q ≈ 规则 ≪ oracle）。**②** 不把换码后的 kcal 下降写成身份正确。**③** 不把外测写成能量与质量双赢。**#65** 不把外测改写成双赢，也**不**开工改方法。**#68** 记下 **98.6201** 对 **75.3870** 是更大的有符号抵消，不是更准的密度或食物码，也**不**开工改方法。**主声称：质量对 Direct 可外推（bootstrap 区间均为负）；对 Mass-only 基本打平（0.76 g 不算赢，不要当作方法优越性证据）；能量外测未稳。**

**Checklist（给代码 Agent）**

1. **P0a 门禁 — 审计 traces — 已完成（#37，`2e0e180`）**：在现有 on-policy（#34 jsonl）与 hold-out logged traces 上统计 CORRECT_CODE / RE_PORTION（及同类改证据工具）执行后，终态营养向量 \(n\) **是否真的变化**——报变更率、无变化比例、按动作分层；审计摘要已落盘后才开训 Mass（该步已完成，见下条）。
2. **P0b — Mass Estimator — 已完成（#39，`4b6ccec`）**：ViT-B/16 或 ResNet101；ImageNet → N5K train；**RGB-only**；多任务 [M,E,P,C,F]+σ；工具 `estimate_mass` 接入（禁止空壳 RE_PORTION）。
3. **P0c — U_* + Q_φ — 已完成（#41，`54c32e8`）**；Q 校准 **已完成（#44，`3fc611d`）**：定义 U_id / U_mass / U_nutrient / U_evidence；先规则启发式 utility，再监督 **\(Q_\phi(z,a)=\mathbb{E}[\Delta L\mid z,a]\)**（**无 RL**）。

4. **P0d / #48 / #51 / #53 — 已完成。** P0d（#46，`b9c4c5e7`）→ 误差审计（#48，`91a05d8`）→ food-id / 密度轻修（#51，`6c8fda02`）→ 覆盖约束（#53，`2a406df4`）。#53 把「未写完就不许 STOP/ABSTAIN」写成 **Coverage-Constrained Active Specialist Selection** 的 \(A_{legal}\) 门，**不是** Bug Fix，也**不是** Q 变强，**不是**最终主表。
5. **① 已完成（#57，squash `0aa30537`）— Q 反事实 + dish_id 泄漏审计。** 产物 `results/qphi_counterfactual/`。fit/hold-out **按 dish_id 分组**，交集 0，**无**重训。头条 n=1914（coverage-choice + mass forward；rgb_train hold-out；与 #53 的 505 交集 0）：Q Best-ID 0.6165、ΔE improve 24.14、E regret 49.96；规则 0.6102 / 23.03 / 50.91；固定 0.5914 / 20.62 / 52.99；随机 0.4338 / 4.04 / 78.15；oracle regret 0、improve 82.19。Q 略优于规则（约 0.96 kcal），约 50 kcal 落后 oracle。241/1914 的 oracle 最优是 RE_DECOMPOSE no-op（冻结 Q 不能发出）；去掉后 Q regret ~17.33 vs 规则 ~18.28。价值校准 n=3475：MAE(Q, ΔL) 0.0226，Pearson 0.747；RE_PORTION 0.805，CORRECT_CODE 0.016。**Q ≈ 规则 ≪ oracle**，加固 #53 增益 ≠ Q 学习。**不**重训 Q 或 Mass。
6. **② 已完成（#60，squash `c52ffaa2`）— 不是当前开工项。** 只读 #51 同头换码（`results/food_id_density/`，squash `6c8fda02`）：**68** 码，**37/50** 多组件盘。自动标签（仅建议）：换码后 Exact **0** / Near **50**（73.53%）/ Wrong **18**（26.47%）；换码前同一 cafe 行 Exact **27** / Near **23** / Wrong **18**；Exact→Near **27**；变差 **28**、变好 **1**、不变 **39**。宏量 food-table vs cafe，finished 多组件 n=50：P 10.57→10.07，C 13.00→**16.17**，F 17.56→11.97，E 190.79→131.65（能量与冻结 #51 一致）。换码 37：碳水 MAE **+4.28 g**，能量 MAE **−79.92 kcal**。能量变好的 30 盘：全 Exact **0**，任一 Wrong **12**，仅 Near **18**；其上 **50** 码全是 Near 或 Wrong。**kcal 变好 ≠ 身份变对。** 人审包仍开放：68 行 `needs_human_review=yes`（`human_review_*.csv` / `human_review.md`）。产物 `results/food_id_semantic_audit/`；脚本 `scripts/food_id_semantic_audit.py`。**不**重跑，**不**改该目录，**不**重训 Mass / Q / τ，**不**开 P1。本条不定义 held-out。
7. **③ 已完成（#63，squash `27d1dbfa`）— 不是当前开工项。** Cafe2 held-out，**n=228**，与配对 **505**、fit（**3679** 官方 rgb_train 且 `pool_of_dish==fit`）、rgb_train、rgb_test、cafe1 的 dish_id 不相交。划分 `results/heldout_plan/`；主表与轨迹 `results/heldout_eval/`。主表（三法皆有限，n=228）：Coverage-Constrained + density/same-head 能量 MAE **116.8494** kcal、质量 MAE **50.9483** g、覆盖 **0.9737**（222/228）、步数均值 **6.2193**、工具调用均值 **5.2456**；Direct-VLM 能量 **105.6157**、质量 **73.7456**；Mass-only（冻结 Mass）能量 **75.3870**、质量 **51.7107**。能量 MAE 比 Direct-VLM **高 11.23** kcal，比 Mass-only **高 41.46** kcal。质量 MAE 比 Direct-VLM **低 22.80** g，比 Mass-only **低 0.76** g。STOP 且仍有未写组件 **0**。**6** 盘停在 RETRIEVE，组件仍未写完，没有编造份量。同头密度换码 **106** 盘。空壳 RE_PORTION **0**。工具调用中位 **4**、最大 **14**，**228/228** 在 K=32 内。Mass 写 **575**。Cafe2 盘级能量列每行都是 **0**，能量目标是配料卡路里之和；质量是盘级克重总量。这次 Direct-VLM 行记下 `placement_max_memory=8GiB`（共享 GPU）；同一 bf16 checkpoint 与 prompt，不是历史上满 GPU 的 `cuda:0` 放置。不要重跑，不要据此解锁重训。开发 505 的能量 MAE **64.2792**、质量 MAE **32.9908** **不是**本表的行。**不**声称 #53 证明方法。#51 是密度 / kcal 启发式（自动 Exact=0）。① 上 Q ≈ 规则 ≪ oracle。**主声称：质量稳赢可外推；能量开发集赢、外测未稳。** **不**声称 held-out 上能量与质量双赢。**不**重跑，**不**覆盖这两个目录，**不**重训 Mass / Q / τ，**不**开 P1。
8. **Cafe2 能量审计已完成（#65，squash `ecf859f5`，2026-11-09 Asia/Shanghai）— 不是当前开工项，不是门禁 ④。** ③ 的诊断后续，与 #48 同口径，已做完。不是真实 Tool Cost，不是 ⑤–⑦，不是 P1，不是 Mass / Q / τ 重训，也不是新的代码项。配对 **505** 仍是开发 / 诊断集。只读冻结 Cafe2 **n=228**，不重画划分。产物 `results/heldout_energy_audit/`；脚本 `scripts/heldout_energy_audit.py`。**没有**重跑 #63 闭环、Direct-VLM 或 Mass-only。**不**覆盖 `results/heldout_plan/`、`results/heldout_eval/`、`results/food_id_density/`、`results/coverage_early_stop/`、`results/qphi_counterfactual/`、`results/food_id_semantic_audit/`，也**不**改 `results/heldout_energy_audit/`。**不**重跑 #48 / #51 / #53 / #57 / #60 / #65。**#65 未改 STATUS。** **#63 主表不变**：Coverage-Constrained 能量 MAE **116.8494** kcal、质量 MAE **50.9483** g；Direct-VLM **105.6157** / **73.7456**；Mass-only **75.3870** / **51.7107**。能量相对 Direct **+11.23**、相对 Mass-only **+41.46**。质量相对 Direct **−22.80**、相对 Mass-only **−0.76**。覆盖 **0.9737**（222/228）。开发 505 的 **64.2792** / **32.9908** **仍不是**该表的行。**222**/228 盘上已提交克重与 Mass-only 克重相差小于 0.001 g；整段 **−0.76** g 质量优势是停在 RETRIEVE 的 **6** 盘（这 6 盘对 +41.46 均值的能量贡献 **−0.52** kcal）。**+41.46** kcal 是食物表密度打在已经与 Mass-only 对齐的克重上；Mass-only 用冻结质量估计器的 calorie head 给同一克重记分。码固定，相对 100 g 网格的 mass 重写：能量 MAE **195.6564 → 125.2060** kcal，质量 MAE **116.9123 → 50.9483** g。汇合下降来自多组件（**216.37 → 115.21**）。**58** 盘单组件上同一重写把能量 MAE **134.93 → 154.51**，质量仍近 51 g；这 58 盘贡献 **+29.62** kcal。Oracle 标签密度 × 我方克数：能量 MAE **98.6201** kcal，低于食物表 commit，低于 Direct-VLM，仍高于 Mass-only **75.3870**。不加权 kcal/g MAE：commit **0.631**，calorie head **0.753**；head 抵消更大（**77.8** vs **58.2** kcal）。#51 密度换码在 **106** 盘触发。全表能量 MAE 关 **125.2060** / 开 **116.8494**（**−8.3566** kcal），质量不变。**67** 盘更靠近 cafe 标签，**39** 盘更远。未换码的 **122** 盘贡献 **+38.30** kcal。#51 仍是密度 / kcal 启发式（开发集自动 Exact 为 **0**）。Bootstrap **10,000** 次、种子 **0**：能量差 vs Mass-only **[+25.35, +58.28]** kcal（每次均为正）；质量差 vs Direct-VLM **[−29.80, −15.66]** g（每次均为负）；能量差 vs Direct **[−6.22, +29.16]** 含两种符号；质量差 vs Mass-only **[−2.33, +0.10]** 含两种符号。**主声称仍是：质量稳赢可外推；能量开发集赢、外测未稳。** **不**声称双赢。**不**从这些标签开工改方法。**不**重训 Mass，**不**重训 Q，**不**开 P1，**不**改 τ。**禁止**自动开 **④–⑦**。已执行的协议（不是新开工）：Diagnostic only on the frozen Cafe2 n=228 list. No method selection, no early stopping, no threshold/τ tuning, no swapping the method, no GT-driven code or portion edits. 固定 food-id 的 mass 重写、oracle 标签密度 × 我方克数、#51 density swap 开/关（换过码的 106 盘）。基线仍是 Direct-VLM 与 Mass-only。
9. **Mass-only 能量路径便宜诊断已完成（#68，squash `f578e8f4`，2026-11-10 Asia/Shanghai）— 不是当前开工项，不是门禁 ④。** 说清 98 vs 75，已做完。不是在 505 上改 food-id，不是 Cafe2 重测，不是 ⑤–⑦，不是 P1，不是 Mass / Q / τ 重训，也不是新的代码项。配对 **505** 仍是开发 / 诊断集。只读冻结 Cafe2 **n=228**，不重画划分。产物 `results/heldout_massonly_energy_path/`（`report.md`、`report.txt`、`summary.json`、`dishes.jsonl`、`paths.csv`、`slices.csv`、两张图 `fig_density_vs_mae.svg` 与 `fig_gap_split.svg`）；脚本 `scripts/heldout_massonly_energy_path.py`。**没有**重跑 #63 闭环、Direct-VLM 或 Mass-only，也**没有**重跑 #65。**不**覆盖 `results/heldout_plan/`、`results/heldout_eval/`、`results/heldout_energy_audit/` 或其他冻结结果目录，也**不**改 `results/heldout_massonly_energy_path/`。**不**重跑 #48 / #51 / #53 / #57 / #60 / #68。**#68 未改 STATUS。** Mass-only 能量是冻结质量估计器的盘级 `energy_kcal` 头，`vit_b_16` checkpoint，目标 `mass_g`、`energy_kcal`、`protein_g`、`carb_g`、`fat_g`。记分器存 `estimate_mass(...).nutrients.energy_kcal`。不读 food-id，也不把 FNDDS kcal/100 g 乘以克重。**222 / 228** 盘上质量头与已提交克重相差在 0.001 g 以内（测得最大 **0.0002** g）。另外 **6** 盘停在 RETRIEVE；Mass-only 仍给整盘头记分。Oracle 标签密度 × 我方克数能量 MAE **98.6201**，Mass-only 能量头 **75.3870**，差距 **23.2332** kcal：更大的有符号抵消，不是更准的密度或食物码。不加权 kcal/g MAE：标签密度 **0**，食物表 commit **0.631**，calorie head **0.753**。#65 通道抵消（kcal）：oracle **0**，食物表 **58.2**，head **77.8**。保持真密度、换成 Mass-only 克重，能量 MAE **98.6201 → 100.3152**（**+1.6951**）。在这些克重上把真密度换成能量头，**100.3152 → 75.3870**（**−24.9283**）。净差 **23.2332** kcal。在头自己的克重上，把头的密度换成标签密度，会把能量 MAE 从 **75.3870** 抬到 **100.3152**。食物表在 kcal/g 上更近 **139** 盘，头更近 **89** 盘。没有一盘的头密度误差低于 **0.0001** kcal/g。**58** 盘单组件贡献 **+23.23** 里的 **+19.85**；这一片 kcal/g MAE 头 **1.023** vs 食物表 **0.811**，能量 MAE **38.06** vs 食物表 **154.51** / oracle **116.08**。**6** 盘 RETRIEVE 贡献 **−1.56**。按盘 bootstrap **10,000** 次、种子 **0**：**98.6201 − 75.3870** 在 **[+7.92, +39.06]** kcal，**99.83%** 的抽样为正均值。这个区间没有用来选方法。**#63 主表句子不改。** 冻结数字不改，包括 **116.8494** / **50.9483**、**105.6157** / **73.7456**、**75.3870** / **51.7107**、**+11.23**、**+41.46**、**−22.80**、**−0.76**、**0.9737**（222/228）、**98.6201**、**0.631** vs **0.753**、**77.8** vs **58.2**、开发 505 的 **64.2792** / **32.9908**（**仍不是** held-out 表的行）、**195.6564 → 125.2060**、**134.93 → 154.51** / **+29.62**、**−8.3566**、未换码 **122** 盘 **+38.30**，以及 #65 bootstrap **[+25.35, +58.28]**、**[−29.80, −15.66]**、**[−6.22, +29.16]**、**[−2.33, +0.10]**。相对 Mass-only 的 **−0.76** g 仍只是 **6** 盘 RETRIEVE（#65 / #66 已记下）。**不**把它写成质量赢。**不**把只修 food-id 写成能量双赢。**主声称仍是：质量稳赢可外推；能量开发集赢、外测未稳。** **不**声称双赢。**不**从这些标签开工改方法。**不**重训 Mass，**不**重训 Q，**不**开 P1，**不**改 τ。**禁止**自动开 **④–⑦**。已执行的协议（不是新开工）：（1）只做了这一项便宜诊断。不改方法，不调超参，不在 Cafe2 上做早停或模型选择。No method changes, no hyperparameter tuning, and no early stopping or model selection on Cafe2。（2）已写明记分路径：冻结质量估计器 calorie head，不读 food-id，不把 FNDDS kcal/100 g 乘以克重。The diagnostic documented how Mass-only energy is scored: which food-id / density / calorie head it uses, and how that differs from our food-table commit and from oracle label density × our grams。（3）**98.6201** vs **75.3870** 是更大的有符号抵消，不是更准的密度或食物码。It answered that 98.6201 vs 75.3870 is greater signed cancellation, not more accurate density or food codes。（4）落盘之后没有自动开下一道门。冻结后写作（「Coverage+Mass，能量外测未稳」）对比只在 505 上改 food-id、再重测 Cafe2，仍要等以后的调研阅读 + Triste 点头 + 再一次 STATUS 改写。After this diagnostic is filed, do not auto-open the next gate. Freeze-and-write vs 505-only food-id waits for a later research reading + Triste nod + a later STATUS rewrite。

**（历史，不是现行门禁。2026-11-12 晚曾把下一步写成 Nutrition5k 双视角实验。该实验已在备份提交 28bd3c9c 中做完并停止，不要重跑，不要再合入。现行下一步只在文首：第一阶段数据可行性审计。）** 上条诊断已完成，不再是开工项。不要据此改 main，也不要重开已停止的双视角实验；其余代码仍空闲。该句曾把本次改写写成双视角实验的开工许可，2026-11-13 起失效。实验 B、实验 C 与完整联合网络、已取消的 Phase 1、Q/Mass/τ 重训、food-id 改写、505 或 Cafe2 调新方法、P1 与门禁 ④–⑦ 都没有开工许可。不把实验 B、实验 C 或完整联合网络写进主分支。不实现候选校准网络。Cafe2 只作已探索诊断集，不再用于新方法调参，也不再当下一轮完全盲测。已取消的八步没有开跑。下面八句只复述 #70（squash `d622e6db`）当时写下、现已取消的清单，不是命令，也不是实验 A 的前置：

1. 当时拟冻结当前 VDR，并把新产物放到 `results/phase1_cross_domain/`。该目录不要为这项已取消的工作去建。
2. 当时拟下载 ACETADA 与 VLM Food Benchmark 并核对许可。早间把 ACETADA 写成 806 张餐前 / 餐后图像；本晚审计改为：实验使用 **806** 张餐前智能手机图像，配对餐后不是几何侧视。VLM Food Benchmark 是 **200** 张食堂餐图 / **870** 项 / **6** 名参与者，不是多视角集。
3. 当时拟统一质量标签的单位和定义。
4. 当时拟按参与者或 dish_id 做跨域划分。
5. 当时拟跑冻结的 E1 Direct-VLM、E2 Frozen Mass-only、E3 当前完整 VDR。三者都没有开跑。
6. 当时拟加 E4 全局仿射与 E5 类别级校准，作为对照，不是分层校准网络。没有开跑。
7. 当时拟记录 Mass MAE 等，看有没有可学习的跨域偏差。没有测得的数，不预填。
8. 当时拟由该诊断决定是否进入早间的 Phase 2。Phase 2 没有解锁，并已随创新点二撤回。以上 1–8 都不要执行。

**主声称：质量对 Direct 可外推（bootstrap 区间均为负）；对 Mass-only 基本打平（0.76 g 不算赢，不要当作方法优越性证据）；能量外测未稳。** 动机文献只写标题和 arXiv，不编 DOI：Open-KNEAD: Knowledge-grounded Nutrition Estimation via Agentic Decomposition（arXiv:2607.12911，2026-07-14）；Single-Image Food Mass Estimators Do Not Transfer Their Gram Scale Across Datasets（arXiv:2610.29025，2026-10-21）。早间 Phase 2 的写入门槛现已撤回，不解锁任何实验：至少两个独立外部数据集上质量 MAE 改善约 5%–10%，且配对 bootstrap 置信区间支持改善；否则不要把分层校准写成方法贡献。Triste 原文审阅在此句被截断，上述是恢复后的工作门槛，不是另推的新标准。Nutrition5k 官方 **5,006** 盘；顶视 RGB-D 只写论文约数约 3.5k / 5k，不编造可用配对条数。ACETADA（arXiv:2507.07048）实验使用 **806** 张餐前智能手机图像；配对餐后是剩余食物，不是几何侧视。本轮项目主页返回 500，不从主页采集额外数字。VLM Food Benchmark 官方 README：200 张食堂餐图，870 项已核验逐项质量，6 名参与者，多为自由条件俯视，不是多视角集。ECUSTFD（Hugging Face）：2,978 张手机图，19 类，顶视与侧视，25 mm 硬币，逐对象质量与体积；只作辅助几何-质量校准，不是混合膳食主测试，不编造可用配对条数。ASA24 官方页：超过 17,000 张与 FNDDS 份量关联的图像。17,135 / 6,469 是 arXiv:2610.29025 的处理计数，不是 NCI 官方可用质量标签总数。外部混合餐基本不能支撑请求真实第二视图，因此先不要搭建完整联合网络。


10. **④ 锁定 — 真实 Tool Cost。** VLM 调用次数、延迟、token、单盘成本，以及能量 MAE–Cost Pareto。调用次数不等于真实成本。
11. **⑤ 锁定 — 忠实 B1/B2 + 相关工作。** B1 必须按官方协议重做（历史非忠实数字见 §13，本节不改；scaffold / dry-run 不算）。B2 能完整则完整，否则诚实 approx。优先对照：arXiv **2610.11144**、CVPRW 2025 **Decomposing Food Images**、**NutriVision** arXiv **2609.33076**。不编造 DOI。
12. **⑥ 锁定 — Provenance + Semantic。** Provenance 自动全量；Semantic 人工提前到核心实验。
13. **⑦ 锁定 — 然后才共决**是否解锁 P1 或重训 Mass/Q。在此之前 **禁止自动开 P1**，**禁止**重训。

**可推迟**：RL；Route B 多目标 Q；体积；RGB-D 主表；VLM LoRA 营养头；kcal 代理弃权作主故事；在开发 505 上继续调方法冒充最终主表。完整金标准仍可分期。**③** 已完成，其 dish_id 隔离要求不因此取消。Cafe2 能量审计（#65，`ecf859f5`）与 Mass-only 能量路径便宜诊断（#68，`f578e8f4`）已完成，不重画该划分，也**不**叫门禁 ④，也**不**解锁 **④–⑦**。**（历史，不是现行门禁。2026-11-12 晚曾把下一步写成 Nutrition5k 双视角实验。该实验已在备份提交 28bd3c9c 中做完并停止，不要重跑，不要再合入。现行下一步只在文首：第一阶段数据可行性审计。）** 除文首第一阶段数据可行性审计以外，没有别的当前可写代码项。该句曾把本次改写写成双视角实验的开工许可，2026-11-13 起失效。不要据此改 main，也不要重开已停止的双视角实验；其余代码仍空闲。实验 B、实验 C、完整联合网络与 ④–⑦ 都没有开工许可。

Schema 仍冻结于 `schemas/`；说明见 `docs/schema.md`。执行状态见 [`docs/STATUS.md`](docs/STATUS.md)。

Canvas 真源：https://cursor.com/canvas/store-745bdf38-f3d7-4313-8d66-08e4b6903e93/d7be617d-d37e-4fe1-a4b5-9da2c86b27a2（**VDR 官方方案（2026-11-12 晚）：面向物理约束营养估计的主动多模态证据获取（候选）**）
