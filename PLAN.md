# Verified Dietary Reasoning（VDR）实验方案

> **本文档是投稿叙事与执行的 Markdown 真源。** 与 Cursor Canvas「VDR — Evidence-Aware Closed-Loop Tool Control」对齐。  
> 算力假设：**2×4090D 24GB**。数据：**公开 Nutrition5K + FNDDS 2019–2020**；人工 **200–500** 盘薄标注；**无私有大规模采集**。  
> **技术中心**：**Evidence-Aware Closed-Loop Tool Control**（证据反馈驱动的闭环工具控制）。**暂不上 RL。**

---

## 0. Diff vs 开环 v2（必须读）

| 修正 | 内容 |
|------|------|
| 主方法 | 改为真正读工具观测的状态闭环：`π(a_t \| h_t, o_{t-1}, b_t)` |
| 旧 31 维 BC | **降为消融 B5**，不是主方法、不是主贡献 |
| 0.97 Acc | **不作方法证明**；须显示：观测改变下一动作、纠错降误差、证据不足时弃权、增益不只来自多调工具 |
| 新颖性 | 相对 DietAI24 / FoodAgent：卖 **证据反馈决策 + 选择性输出（弃权）**，不是又一条营养管线 |
| 历史 B1=110.72 | **非忠实 DietAI24**；主表必须按官方协议重做后再进 |
| RL | 暂不优先；MVP 以 State-aware BC/DAgger 为中心 |

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

1. **任务**：形式化 VDR；证据链可程序复算 + 可人工语义核；支持盘级/成分级弃权。
2. **方法**：Evidence-Aware Closed-Loop Tool Control——`π(a_t | h_t, o_{t-1}, b_t)`；规则仅先验/护栏；开环 BC 仅消融。
3. **基准与证明**：VDR-Bench；证明观测改变动作、纠错降误差、弃权可校准，且相对忠实 DietAI24/FoodAgent 有选择性输出优势。

---

## 2. 硬禁止

| 禁止 | 原因 |
|------|------|
| 把开环 31 维 BC 当主方法 | 不读 `o_{t-1}` 不是闭环 |
| 用弱老师 overall Acc 证明方法 | ABSTAIN 0/24 已红警 |
| 把 B1 叫「弱 RAG-once」或把 FoodAgent 叫 Fixed-MoE | 对照不忠实 |
| 加载旧营养回归权重 | 污染贡献 |
| 体积工具默认开 | Tanabe 警示 |
| 大规模 RLHF / 优先 RL | 算力与 MVP 焦点不匹配 |
| 用 test GT 反标步骤弱监督 | 泄漏 |
| 私有大规模采集替代公开主表 | 复现性 |

---

## 3. 研究问题与假设

- **RQ1**：证据效度 + 选择性输出能否在 Coverage–Risk 帕累托上优于强制全答？
- **RQ2**：同 VLM / FNDDS / 工具预算下，读观测闭环是否显著优于开环 BC、规则、同预算 Self-Refine/CRITIC、忠实 DietAI24/FoodAgent？
- **RQ3**：下一动作是否随 `o_{t-1}` 改变？纠错是否系统降误差？弃权是否集中在证据不足样本？

| 假设 | 内容 |
|------|------|
| **H1** | State-aware 闭环相对 B1/B2/B4/B5：Full-set MAE 不显著变差（或更好），Selective MAE / AURC / Evidence Validity 显著更好；FixRate−WorsenRate > 0（ε 与 CI 预注册） |
| **H2** | 掩蔽/打乱 `o_{t-1}` 后动作与指标变差；去掉 ABSTAIN 后 AURC 崩溃；同预算下闭环仍优于「只加工具次数」开环 |
| **H3** | 相对忠实 B1/B2，优势在证据反馈决策与选择性输出；无法完全复现则诚实声明 approx |

---

## 4. 闭环方法（技术中心）

流水线：感知 VLM → 闭环控制器 ⇄ 工具；规则先验/硬护栏。

| 符号 | 内容 | 实现要点 |
|------|------|----------|
| `h_t` | 成分假设、Top-k、当前 code/份量、证据完整度、步数 | 固定维特征 + 可选短摘要 |
| `o_{t-1}` | 上一工具返回：候选、verify_codes、聚合营养、错误码 | **禁止再忽略**；掩蔽消融用于 H2 |
| `b_t` | 剩余工具预算、硬安全、弃权护栏 | 与基线同一预算 K |
| `a_t` | RETRIEVE / CORRECT_CODE / RE_PORTION / RE_DECOMPOSE / ABSTAIN / STOP | 无默认 VOLUME |

训练主路径：**State-aware BC → DAgger**（读观测老师轨迹）；暂缓 RL。

### 什么算证明了方法（替代 Policy Acc=0.97）

1. **Obs→Action**：掩蔽/打乱 `o_{t-1}` 导致动作分布与下游指标显著变差。
2. **Correction helps**：带 ε 的 FixRate > WorsenRate，并报配对 CI。
3. **Abstain when unsupported**：盘级 + 成分级；每类 PR/F1 + 混淆矩阵。
4. **Not just more tools**：同预算匹配；报 Tool cost。

---

## 5. 证据链与两类效度

强制链：**Image Region → Component → FNDDS Code → Portion → Nutrient**。断链 → ungrounded。

| 类型 | 定义 | 谁评 |
|------|------|------|
| **Provenance Validity** | 程序复算：`n` 是否等于 FNDDS(code)×portion 聚合；边是否齐全 | 自动脚本 |
| **Semantic Grounding** | 人判：码/份量/可见性是否与图像语义一致 | 人工金标准 |

**FNDDS release 钉死：2019–2020**（5624 codes，对齐 DietAI24）；换版全员重跑。

---

## 6. 人工金标准

| 字段 | 含义 |
|------|------|
| Visibility | Visible / Ambiguous / Hidden |
| Code Match | Exact / Near / Wrong / Unsure |
| Portion Confidence | High / Mid / Low |
| Evidence Completeness | Complete / Partial / Broken |
| Should Abstain | 盘级 + 成分级 partial abstain |
| Conflict | id / portion / db_gap / sauce / multi / other |

**两遍标注**：Pass A 仅看图；Pass B 给参考链。双人 + 第三人裁决。关键标签 kappa 目标 **0.7–0.8**（不作 0.6 凑合）。规模 200–500；金标准主要用于 held-out，不刷训练 Acc。

---

## 7. 主指标

| 指标 | 角色 |
|------|------|
| Full-set MAE/MAPE/r | 兼容旧文献 |
| Coverage / Selective MAE | 选择性预测 |
| Risk-Coverage / **AURC** | 主可信曲线 |
| Evidence Validity（Provenance + Semantic 分栏） | 新任务主指标 |
| Tool cost | 防「多调用刷分」 |
| FixRate / WorsenRate（\|Δerr\| > ε + 95% CI） | 闭环行为 |
| 每类 ABSTAIN PR/F1 + 混淆矩阵 | 红警协议 |

Primary 建议：N5K Full-set MAE@energy + AURC + Provenance；人工子集 Semantic + Abstain F1。

---

## 8. 基线（统一 VLM / FNDDS / 工具预算 K）

| 代号 | 定义 | 注意 |
|------|------|------|
| B0 Direct-VLM | 单次 VLM 无工具 | 历史能量 MAE 97.03 可作锚点 |
| **B1 DietAI24** | **忠实复现**官方协议/代码 | 禁止「弱 RAG-once」；历史 110.72 非忠实须重做 |
| **B2 FoodAgent** | **动态专家路由** | 禁止 Fixed-MoE；无法完整则诚实 approx |
| B3 Rule Verify | 纯规则控制器 | 规则上限 |
| B4 Self-Refine/CRITIC | 同预算文本自纠 | 预算对齐 |
| **B5 Open-loop BC** | 原 31 维不读 obs | **仅消融** |
| **Ours** | State-aware BC/DAgger | 主方法；RL 不进 MVP |

---

## 9. 必出实验

- M1 Full-set MAE；M2 Coverage/Selective MAE/AURC；M3 Provenance+Semantic；M4 Fix/Worsen + Tool cost + ABSTAIN 每类指标  
- A1 开环 B5 vs Ours + 掩蔽 o；A2 去掉 ABSTAIN；A3 预算匹配；A4 B1/B2 完整 vs approx  
时间不够保住 **M2/M3/M4/A1**；切勿再把 Policy Acc 当主表。

---

## 10. 12–16 周重排

| 周 | 内容 |
|----|------|
| W1–3 | 状态闭环脚手架：注入 `o_{t-1}`；State-aware BC；掩蔽消融；ABSTAIN 仪表 |
| W2–5 | 忠实 B1/B2；统一 VLM/FNDDS/K |
| W3–6 | Coverage/AURC/Provenance/Fix-Worsen；成分级弃权 |
| W5–8 | Ours vs 基线主表；闭环有效后再 DAgger 1–2 轮 |
| W7–11 | 两遍标注 + 试标 kappa→0.7–0.8；200–500 金标准 |
| W10–16 | 第二源、案例、写作投稿 |

---

## 11. 开环 v2 历史冻结数（保留，降权）

| 项 | 数字 | 新解读 |
|----|------|--------|
| B0 能量 MAE | 97.03（rgb_test 507） | Direct-VLM 历史锚点 |
| 旧 B1 能量 MAE | 110.72 | **非忠实** DietAI24；须重做 |
| 弱标签审计 MAE | 29.58 / 30.71 | 老师可造轨迹 ≠ 闭环证明 |
| 开环 BC Acc | 0.9206 / **0.9700** | 降为 B5 参照 |
| ABSTAIN | **0/24** | 红警；overall Acc 无效 |
| DAgger r1 Acc | 0.9592 | 开环下不支持盲目加轮 |

---

## 12. MVP-for-A

**必须**：读观测 State-aware BC/DAgger；B5+掩蔽 o；忠实 B1；B2 完整或诚实 approx；Full-set MAE + Coverage + Selective MAE + AURC；Provenance + ABSTAIN 每类指标；Fix/Worsen(ε,CI) + 同预算 Tool cost；小规模两遍试标。

**可推迟**：RL；完整 200–500 可分期（协议与 kappa 门禁不能省）；体积；第三数据集；VLM LoRA 营养头。

Schema 仍冻结于 `schemas/`；说明见 `docs/schema.md`。执行状态见 [`docs/STATUS.md`](docs/STATUS.md)。

Canvas 真源：https://cursor.com/canvas/store-745bdf38-f3d7-4313-8d66-08e4b6903e93/d7be617d-d37e-4fe1-a4b5-9da2c86b27a2
