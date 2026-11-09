# Verified Dietary Reasoning（VDR）实验方案

> **本文档是投稿叙事与执行的 Markdown 真源。** 与 Cursor Canvas「**Coverage-Constrained Active Specialist Selection**」对齐（2026-11-06）。  
> Canvas：https://cursor.com/canvas/store-745bdf38-f3d7-4313-8d66-08e4b6903e93/d7be617d-d37e-4fe1-a4b5-9da2c86b27a2  
> 算力假设：**2×4090D 24GB**。数据：**公开 Nutrition5K + FNDDS 2019–2020**；人工 **200–500** 盘薄标注；**无私有大规模采集**。  
> **任务框架**：**Evidence-Guided Active Multimodal Nutrition Estimation**（证据引导的主动式多模态营养估计）。**技术中心 / 方法核心卖点**：**Coverage-Constrained Active Specialist Selection**（覆盖约束下的主动专家选择）+ **Error-Aware Q**。**Route A**：论文主张收窄为误差感知的**能量 + 质量**估计决策系统；蛋白质 / 碳水 / 脂肪作辅助。**暂不上 RL。**  
> **历史对照**：旧中心「Evidence-Aware Closed-Loop Tool Control」仅作对照叙事，不再作主标题。#53 是开发检查点，不是本卖点的证明。

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
| RL | **暂不上 RL**；MVP 以监督 **Q_φ(z,a)** 为中心。 |

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

### 三条主贡献（写作冻结）

1. **任务**：形式化 VDR；\(y=(n,G,s,\tau)\)；证据链可程序复算 + 可人工语义核；支持盘级/成分级弃权。
2. **方法**：**Coverage-Constrained Active Specialist Selection** + **Error-Aware Q**（Route A）。未写完组件时 \(A_{legal}\) 去掉 STOP/ABSTAIN，再在合法集上 \(a^*=\arg\max_{a\in A_{legal}} Q_\phi(z,a)/\mathrm{Cost}(a)\)，\(Q_\phi(z,a)=\mathbb{E}[\Delta L\mid z,a]\)。Mass Estimator 已落地并冻结；图文检索重排（P1）暂缓；Grounding 非主创新；Nutrient Verifier 为护栏。**State-aware BC 仅 bootstrap/消融。** #51/#53 在冻结 Mass 与 Q/τ 上的开发集变化是确定性策略，**不是** Q 学得更好。#57 一步反事实同向：Q ≈ 规则 ≪ oracle。
3. **证明（③ 已记下外测；不是能量与质量双赢，也不是 #53 或 #57 已完成该证明）**：#63（squash `27d1dbfa`）在 Cafe2 held-out 上记下能量与质量 MAE（见 §14）。**主声称：质量稳赢可外推；能量开发集赢、外测未稳。** #57 已记下反事实动作价值（一步评测 Q ≈ 规则 ≪ oracle，见 §14；不单独构成该证明）。真实成本 Pareto、Provenance / Semantic、忠实 B1/B2，以及相对 arXiv 2610.11144、CVPRW 2025 Decomposing Food Images、NutriVision（arXiv 2609.33076）的差异仍在锁定门禁 **④–⑦**。开发集超过 Direct-VLM **不**等于该证明。Coverage / Selective / AURC 仍辅助。

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

**主张边界（Route A）**：输出仍可含 \(n=(E,P,C,F,M)\)，但 \(Q_\phi\) 的监督核心是能量误差改善。论文主 claim 是误差感知的能量与质量估计决策系统；蛋白质 / 碳水 / 脂肪作辅助。多目标 Q（Route B）在反事实实验与外测完成前不开。

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
| **P0** | **Mass Estimator** | ViT-B/16 或 ResNet101；ImageNet → N5K train 微调；**RGB-only 主表**；多任务 **[M,E,P,C,F]+σ**；`estimate_mass(image, candidates)` | **是 — 主增益来源** |
| **P1** | 图文检索 + FNDDS 重排 | SigLIP/CLIP + 向量库 + cross-modal/VLM rerank | 次主，**现暂缓**（⑦ 之前不开） |
| **P2/P3** | 区域定位 | Grounding DINO / SAM2 + 局部再问 VLM | **否** — 不当主创新 |
| 配套 | Nutrient Verifier | FNDDS 查表 + 规则/一致性 | 护栏 |

Grounding 为 P2/P3：细粒度分割未必自动改善营养 MAE，**份量/质量**才是瓶颈（待补正式引用，勿编造 DOI）。

**P1 检索重排仍暂缓**，STATUS 门禁 ⑦ 之前不开，也不重训 Mass / Q。**Coverage Guard** 已在 #53 落地，应写成 \(A_{legal}\) 方法，不是空壳，也不是 Bug Fix。#51 同头换码已落地；语义审计（门禁 **②**）已由 #60 完成（见 §14）。**①②③ 已完成。当前第一项 = Cafe2 能量审计（已解锁；不是门禁 ④）。④–⑦** 仍锁。

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

Primary 叙事（Route A）：**Full-set MAE@energy + mass MAE + 真实 Tool cost**。宏量 MAE 为辅助。禁止 kcal-proxy AURC 作主 claim。开发 505 不是最终独立主表。

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

优先对照的相关工作（不编 DOI；刊名、arXiv 号与日期来自 2026-11-06 评审和 Canvas）：

- **DietAI24**（简报所给刊名 Communications Medicine 2025；代码 https://github.com/Runz96/DietAI24）：忠实复现为 B1。数据库检索本身不作主创新。
- **Decomposing Food Images**（CVPRW 2025）：两步分解本身不构成新意。
- arXiv **2610.11144**（2026-10，食物身份验证 / 恢复；https://arxiv.org/abs/2610.11144）：与覆盖约束有交集，是重点对比。须划清「清单内写齐」和「身份验证 / 恢复」，并补人工身份审计。
- **NutriVision**，arXiv **2609.33076**（2026-09；https://arxiv.org/abs/2609.33076）：专用营养预测方向，不是同型 Agent。

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

早期取舍是保住 **M1 + mass + Tool cost + A1/A2**。**现行顺序以 §14 为准（①②③ 已完成；当前第一项 = Cafe2 能量审计，已解锁，不是门禁 ④；④–⑦ 仍锁）**；切勿再把 Policy Acc、kcal-proxy AURC 或开发 505 当最终主表。真实 Tool cost 仍是门禁 **④**，不在该审计里开。

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

上表是早期周程。**现行严格顺序以 §14 与 [`docs/STATUS.md`](docs/STATUS.md) 为准**：**①②③ 已完成**；当前第一项 = **Cafe2 能量审计**（已解锁；**不是门禁 ④**；**④–⑦** 仍锁定）；P1 与 Mass/Q 重训仍锁。

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

**① 已完成（#57，squash `0aa30537`）。② 已完成（#60，squash `c52ffaa2`）。③ 已完成（#63，squash `27d1dbfa`）。当前第一项 = Cafe2 能量审计（已解锁；唯一允许写代码的项；不是门禁 ④）。** 以 [`docs/STATUS.md`](docs/STATUS.md) 为准：**该审计的代码只在这份解锁进入主分支之后才许开工。禁止**自动开 **④–⑦**。**P1 仍暂缓 / 锁定。禁止**重训 Mass。**禁止**重训 Q。**禁止**重拟合 τ。**无 RL。不开 fast Cloud Agent。** 解锁依据：Triste 点头「继续吧」（群聊 2026-11-09 晚，Asia/Shanghai）+ 调研建议。**主声称：质量稳赢可外推；能量开发集赢、外测未稳。** **不**声称 held-out 上能量与质量双赢。开发 505 的能量 MAE **64.2792**、质量 MAE **32.9908** **不是** Cafe2 主表的行。自动 Exact / Near / Wrong 只是建议，人审包仍开放；**kcal 变好 ≠ 身份变对**，不把 #51 的能量下降写成身份成功。#51 是密度 / kcal 启发式（换码后自动 Exact=0）。① 上 Q ≈ 规则 ≪ oracle。#53 冻结为 Ours-new **开发检查点**（配对 505 是开发 / 诊断集，不是最终独立泛化主表）。#51/#53 的开发数字不在这里改写。2026-11-06 的旧排序（先 #51 再 #53，P1 暂缓）已经做完，那套编号退役。

**已完成**：P0a–P0d、Q 校准 #44、误差审计 #48、food-id #51、覆盖约束 #53、**① #57**、**② #60**、**③ #63**。**#51/#53 增益 ≠ Q 学习改进**（Mass / Q / τ 冻结；① 一步评测 Q ≈ 规则 ≪ oracle）。**②** 不把换码后的 kcal 下降写成身份正确。**③** 不把外测写成能量与质量双赢。**主声称：质量稳赢可外推；能量开发集赢、外测未稳。**

**Checklist（给代码 Agent）**

1. **P0a 门禁 — 审计 traces — 已完成（#37，`2e0e180`）**：在现有 on-policy（#34 jsonl）与 hold-out logged traces 上统计 CORRECT_CODE / RE_PORTION（及同类改证据工具）执行后，终态营养向量 \(n\) **是否真的变化**——报变更率、无变化比例、按动作分层；审计摘要已落盘后才开训 Mass（该步已完成，见下条）。
2. **P0b — Mass Estimator — 已完成（#39，`4b6ccec`）**：ViT-B/16 或 ResNet101；ImageNet → N5K train；**RGB-only**；多任务 [M,E,P,C,F]+σ；工具 `estimate_mass` 接入（禁止空壳 RE_PORTION）。
3. **P0c — U_* + Q_φ — 已完成（#41，`54c32e8`）**；Q 校准 **已完成（#44，`3fc611d`）**：定义 U_id / U_mass / U_nutrient / U_evidence；先规则启发式 utility，再监督 **\(Q_\phi(z,a)=\mathbb{E}[\Delta L\mid z,a]\)**（**无 RL**）。

4. **P0d / #48 / #51 / #53 — 已完成。** P0d（#46，`b9c4c5e7`）→ 误差审计（#48，`91a05d8`）→ food-id / 密度轻修（#51，`6c8fda02`）→ 覆盖约束（#53，`2a406df4`）。#53 把「未写完就不许 STOP/ABSTAIN」写成 **Coverage-Constrained Active Specialist Selection** 的 \(A_{legal}\) 门，**不是** Bug Fix，也**不是** Q 变强，**不是**最终主表。
5. **① 已完成（#57，squash `0aa30537`）— Q 反事实 + dish_id 泄漏审计。** 产物 `results/qphi_counterfactual/`。fit/hold-out **按 dish_id 分组**，交集 0，**无**重训。头条 n=1914（coverage-choice + mass forward；rgb_train hold-out；与 #53 的 505 交集 0）：Q Best-ID 0.6165、ΔE improve 24.14、E regret 49.96；规则 0.6102 / 23.03 / 50.91；固定 0.5914 / 20.62 / 52.99；随机 0.4338 / 4.04 / 78.15；oracle regret 0、improve 82.19。Q 略优于规则（约 0.96 kcal），约 50 kcal 落后 oracle。241/1914 的 oracle 最优是 RE_DECOMPOSE no-op（冻结 Q 不能发出）；去掉后 Q regret ~17.33 vs 规则 ~18.28。价值校准 n=3475：MAE(Q, ΔL) 0.0226，Pearson 0.747；RE_PORTION 0.805，CORRECT_CODE 0.016。**Q ≈ 规则 ≪ oracle**，加固 #53 增益 ≠ Q 学习。**不**重训 Q 或 Mass。
6. **② 已完成（#60，squash `c52ffaa2`）— 不是当前开工项。** 只读 #51 同头换码（`results/food_id_density/`，squash `6c8fda02`）：**68** 码，**37/50** 多组件盘。自动标签（仅建议）：换码后 Exact **0** / Near **50**（73.53%）/ Wrong **18**（26.47%）；换码前同一 cafe 行 Exact **27** / Near **23** / Wrong **18**；Exact→Near **27**；变差 **28**、变好 **1**、不变 **39**。宏量 food-table vs cafe，finished 多组件 n=50：P 10.57→10.07，C 13.00→**16.17**，F 17.56→11.97，E 190.79→131.65（能量与冻结 #51 一致）。换码 37：碳水 MAE **+4.28 g**，能量 MAE **−79.92 kcal**。能量变好的 30 盘：全 Exact **0**，任一 Wrong **12**，仅 Near **18**；其上 **50** 码全是 Near 或 Wrong。**kcal 变好 ≠ 身份变对。** 人审包仍开放：68 行 `needs_human_review=yes`（`human_review_*.csv` / `human_review.md`）。产物 `results/food_id_semantic_audit/`；脚本 `scripts/food_id_semantic_audit.py`。**不**重跑，**不**改该目录，**不**重训 Mass / Q / τ，**不**开 P1。本条不定义 held-out。
7. **③ 已完成（#63，squash `27d1dbfa`）— 不是当前开工项。** Cafe2 held-out，**n=228**，与配对 **505**、fit（**3679** 官方 rgb_train 且 `pool_of_dish==fit`）、rgb_train、rgb_test、cafe1 的 dish_id 不相交。划分 `results/heldout_plan/`；主表与轨迹 `results/heldout_eval/`。主表（三法皆有限，n=228）：Coverage-Constrained + density/same-head 能量 MAE **116.8494** kcal、质量 MAE **50.9483** g、覆盖 **0.9737**（222/228）、步数均值 **6.2193**、工具调用均值 **5.2456**；Direct-VLM 能量 **105.6157**、质量 **73.7456**；Mass-only（冻结 Mass）能量 **75.3870**、质量 **51.7107**。能量 MAE 比 Direct-VLM **高 11.23** kcal，比 Mass-only **高 41.46** kcal。质量 MAE 比 Direct-VLM **低 22.80** g，比 Mass-only **低 0.76** g。STOP 且仍有未写组件 **0**。**6** 盘停在 RETRIEVE，组件仍未写完，没有编造份量。同头密度换码 **106** 盘。空壳 RE_PORTION **0**。工具调用中位 **4**、最大 **14**，**228/228** 在 K=32 内。Mass 写 **575**。Cafe2 盘级能量列每行都是 **0**，能量目标是配料卡路里之和；质量是盘级克重总量。这次 Direct-VLM 行记下 `placement_max_memory=8GiB`（共享 GPU）；同一 bf16 checkpoint 与 prompt，不是历史上满 GPU 的 `cuda:0` 放置。不要重跑，不要据此解锁重训。开发 505 的能量 MAE **64.2792**、质量 MAE **32.9908** **不是**本表的行。**不**声称 #53 证明方法。#51 是密度 / kcal 启发式（自动 Exact=0）。① 上 Q ≈ 规则 ≪ oracle。**主声称：质量稳赢可外推；能量开发集赢、外测未稳。** **不**声称 held-out 上能量与质量双赢。**不**重跑，**不**覆盖这两个目录，**不**重训 Mass / Q / τ，**不**开 P1。
8. **当前第一项 = Cafe2 能量审计 — 已解锁，唯一允许写代码的项。不是门禁 ④。** ③ 的诊断后续，与 #48 同口径。不是真实 Tool Cost，不是 ⑤–⑦，不是 P1，不是 Mass / Q / τ 重训。配对 **505** 仍是开发 / 诊断集。代码只在 STATUS 的这份解锁进入主分支之后才许开工。只读冻结 Cafe2 n=228，不重画划分；只读 `results/heldout_plan/` 与 `results/heldout_eval/`。新产物例如 `results/heldout_energy_audit/`。**不**覆盖 `results/heldout_plan/`、`results/heldout_eval/`、`results/food_id_density/`、`results/coverage_early_stop/`、`results/qphi_counterfactual/`、`results/food_id_semantic_audit/`。**不**重跑 #63 闭环主表。**不**重跑 #48 / #51 / #53 / #57 / #60。**不**重训 Mass，**不**重训 Q，**不**开 P1，**不**改 τ。**禁止**自动开 **④–⑦**。协议：
    1. 只在冻结的 Cafe2 n=228 名单上做诊断。不做方法选择，不早停，不调阈值 / τ，不换方法，不用 GT 改码或份量。Diagnostic only on the frozen Cafe2 n=228 list. No method selection, no early stopping, no threshold/τ tuning, no swapping the method, no GT-driven code or portion edits.
    2. 与 #48 同一组切分，写明白。Same cuts as #48, written plainly:
       - 固定 food-id 时，mass 重写是否同时降低 energy（hold identity/codes fixed, then ask whether the mass rewrite also drops energy）。
       - oracle 标签密度 + 我方克数（oracle label density with our grams）。
       - 另拆 #51 density swap 开/关，重点看换过码的 106 盘。
    3. 基线仍是 Direct-VLM 与 Mass-only。头条问题：相对 Mass-only 的能量损失 +41.46 kcal（质量几乎打平，−0.76 g）是否来自 food-id / 密度。Baselines remain Direct-VLM and Mass-only. The headline question is whether the energy loss vs Mass-only of +41.46 kcal (mass almost tied, −0.76 g) comes from food-id / density.
9. **④ 锁定 — 真实 Tool Cost。** VLM 调用次数、延迟、token、单盘成本，以及能量 MAE–Cost Pareto。调用次数不等于真实成本。
10. **⑤ 锁定 — 忠实 B1/B2 + 相关工作。** B1 必须按官方协议重做（历史非忠实数字见 §13，本节不改；scaffold / dry-run 不算）。B2 能完整则完整，否则诚实 approx。优先对照：arXiv **2610.11144**、CVPRW 2025 **Decomposing Food Images**、**NutriVision** arXiv **2609.33076**。不编造 DOI。
11. **⑥ 锁定 — Provenance + Semantic。** Provenance 自动全量；Semantic 人工提前到核心实验。
12. **⑦ 锁定 — 然后才共决**是否解锁 P1 或重训 Mass/Q。在此之前 **禁止自动开 P1**，**禁止**重训。

**可推迟**：RL；Route B 多目标 Q；体积；RGB-D 主表；VLM LoRA 营养头；kcal 代理弃权作主故事；在开发 505 上继续调方法冒充最终主表。完整金标准仍可分期。**③** 已完成，其 dish_id 隔离要求不因此取消。Cafe2 能量审计不重画该划分，也**不**叫门禁 ④。

Schema 仍冻结于 `schemas/`；说明见 `docs/schema.md`。执行状态见 [`docs/STATUS.md`](docs/STATUS.md)。

Canvas 真源：https://cursor.com/canvas/store-745bdf38-f3d7-4313-8d66-08e4b6903e93/d7be617d-d37e-4fe1-a4b5-9da2c86b27a2（**Coverage-Constrained Active Specialist Selection**；任务框架 **Evidence-Guided Active Multimodal Nutrition Estimation**）
