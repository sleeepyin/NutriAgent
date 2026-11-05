# Verified Dietary Reasoning（VDR）实验方案

> **本文档是投稿叙事与执行的 Markdown 真源。** 与 Cursor Canvas「**VDR — Evidence-Guided Active Multimodal Nutrition Estimation**」对齐。  
> Canvas：https://cursor.com/canvas/store-745bdf38-f3d7-4313-8d66-08e4b6903e93/d7be617d-d37e-4fe1-a4b5-9da2c86b27a2  
> 算力假设：**2×4090D 24GB**。数据：**公开 Nutrition5K + FNDDS 2019–2020**；人工 **200–500** 盘薄标注；**无私有大规模采集**。  
> **技术中心**：**Evidence-Guided Active Multimodal Nutrition Estimation**（证据引导的主动式多模态营养估计）。**方法核心卖点**：**Error-Aware Active Specialist Selection**（误差感知的主动专家工具选择）。**暂不上 RL。**  
> **历史对照**：旧中心「Evidence-Aware Closed-Loop Tool Control」仅作对照叙事，不再作主标题。

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
2. **方法**：**Error-Aware Active Specialist Selection**——**Mass Estimator** / 图文检索+重排 / Grounding（非主创新）+ **Nutrient Verifier** 护栏；监督工具效用 **\(Q_\phi(z,a)=\mathbb{E}[\Delta L \mid z,a]\)**，\(a^*=\arg\max_a Q_\phi/Cost\)；**State-aware BC 仅 bootstrap/消融**。
3. **证明**：**主表优先** Full-set **MAE@energy** + **mass MAE** + **Tool cost**（同预算）；相对 B5/B0/**Ours-old** 显著改善预测；Coverage / Selective / **AURC 辅助**；相对忠实 B1/B2 优势在主动专家选择 + 可验证证据，而非又一条检索管线。

---

## 2. 硬禁止

| 禁止 | 原因 |
|------|------|
| 把 **kcal-proxy AURC** 当主创新 | #34 只改善排序不改善预测；审稿质疑按规模降 Coverage |
| **空壳 RE_PORTION** | 无独立 DL 份量估计则无法改善终态 \(n\) |
| 工具**全串行**冒充 Agent | 选择必须由 \(z_t\) / \(Q_\phi\) 驱动 |
| **RGB-D 混进 RGB 主表** | 主协议与现 rgb_test 一致 **仅 RGB** |
| 上 **RL** | 采纳边界：暂不上 RL |
| 用 **test GT** 调参 | 泄漏 |
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

---

## 4. 四层架构（技术中心）

输入膳食图像 \(x\)（**主表 RGB-only**）。四层：

1. **Initial Visual Understanding** — Qwen 等 VLM → 假设 + 初始证据链  
2. **Deep Learning Specialist Tools** — Mass / Retrieval+rerank / Grounding / Nutrient Verifier  
3. **Evidence-Aware Active Controller** — 按 \(z_t\) 选工具；STOP / PARTIAL ABSTAIN  
4. **Verified Nutrition Output** — \(\hat n=\sum \hat m_i f(c_i)\) + 证据链 + 轨迹  

| 层 | 要点 |
|----|------|
| Initial VLM | 冻结或轻量格式对齐；禁旧回归权重 |
| DL Specialists | **真正改善证据质量**；RE_PORTION 须接 Mass |
| Active Controller | 监督 \(Q_\phi\)；**非 RL**；BC 仅启动/消融 |
| Verified Output | 可程序复算；断链 → ungrounded |

---

## 5. 专家工具优先级

| 优先级 | 专家 | 骨干 / 接口 | 主创新? |
|--------|------|-------------|---------|
| **P0** | **Mass Estimator** | ViT-B/16 或 ResNet101；ImageNet → N5K train 微调；**RGB-only 主表**；多任务 **[M,E,P,C,F]+σ**；`estimate_mass(image, candidates)` | **是 — 主增益来源** |
| **P1** | 图文检索 + FNDDS 重排 | SigLIP/CLIP + 向量库 + cross-modal/VLM rerank | 次主 |
| **P2/P3** | 区域定位 | Grounding DINO / SAM2 + 局部再问 VLM | **否** — 不当主创新 |
| 配套 | Nutrient Verifier | FNDDS 查表 + 规则/一致性 | 护栏 |

Grounding 为 P2/P3：细粒度分割未必自动改善营养 MAE，**份量/质量**才是瓶颈（待补正式引用，勿编造 DOI）。

---

## 6. 非 RL 控制器

**状态** \(z_t = [U_{id}, U_{mass}, U_{nutrient}, U_{evidence}, D_{models}, o_{t-1}, b_t]\)

\[
Q_\phi(z,a) = \mathbb{E}[\Delta L \mid z,a],\quad a^* = \arg\max_a \frac{Q_\phi(z,a)}{Cost(a)}
\]

| 分量 | 含义 | 触发 |
|------|------|------|
| \(U_{id}\) | 身份/码不确定 | 检索重排 / CORRECT_CODE |
| \(U_{mass}\) | 份量/质量不确定（含 σ） | **Mass Estimator** |
| \(U_{nutrient}\) | 营养一致性冲突 | Nutrient Verifier |
| \(U_{evidence}\) | 证据链完整度/断链 | Grounding 或 PARTIAL ABSTAIN |

动作：调用 Mass / Retrieval / Grounding / Verifier / CORRECT_CODE / RE_PORTION（**须接 Mass**）/ PARTIAL ABSTAIN / STOP。**体积默认关。**

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
| **主** | Full-set **MAE@energy** | 必须先追相对 B5/B0/Ours-old 的改善 |
| **主** | **mass MAE** | 证明 Mass Specialist 真起作用 |
| **主** | **Tool cost** | 同预算；防多调用刷分 |
| 辅 | Coverage / Selective MAE / **AURC** | 可信曲线；**不作主创新** |
| 辅 | Evidence Validity | Provenance + Semantic |
| 行为 | FixRate / WorsenRate（ε + 95% CI） | 须审计终态 \(n\) 是否真变 |

Primary 叙事：**Full-set MAE@energy + mass MAE + Tool cost**。禁止 kcal-proxy AURC 作主 claim。

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
| **Ours-new** | Mass（+可选检索）+ **\(Q_\phi\)** 主动专家选择 | **主方法**；非 RL |

---

## 11. 必出实验

- **M1 主**：Full-set MAE@energy + mass MAE（Ours-new vs B0/B5/Ours-old）+ Tool cost  
- M2 辅：Coverage / Selective / AURC  
- M3：Provenance；Semantic 后移  
- M4：Fix/Worsen + **终态 n 变更审计**  
- A1 锁死 Mass → 增益消失  
- A2 掩蔽 U_* → MAE 回升  
- A3 固定流水线 / 随机 vs \(Q_\phi\)  
- A4 B5 / Ours-old 负面对照  
- A5 忠实 B1/B2（不挡 Mass P0）  
- F1 案例：Mass 纠正 / 检索重排 / PARTIAL ABSTAIN  

时间不够保住 **M1 + mass + Tool cost + A1/A2**；切勿再把 Policy Acc 或 kcal-proxy AURC 当主表。

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

**Checklist（给代码 Agent）**

1. **P0a 门禁 — 审计 traces（必须先做）**：在现有 on-policy（#34 jsonl）与 hold-out logged traces 上统计 CORRECT_CODE / RE_PORTION（及同类改证据工具）执行后，终态营养向量 \(n\) **是否真的变化**——报变更率、无变化比例、按动作分层；**未落盘审计摘要前禁止开训 Mass**。
2. **P0b — Mass Estimator（仅 P0a 通过后）**：ViT-B/16 或 ResNet101；ImageNet → N5K train；**RGB-only**；多任务 [M,E,P,C,F]+σ；工具 `estimate_mass` 接入（禁止空壳 RE_PORTION）。
3. **P0c — U_* + Q_φ（Mass  wired 后）**：定义 U_id / U_mass / U_nutrient / U_evidence；先规则启发式 utility，再监督 **\(Q_\phi(z,a)=\mathbb{E}[\Delta L\mid z,a]\)**（**无 RL**）。
4. **P0d — 同预算主表**（**已完成 #46 / `b9c4c5e7`**）→ **误差审计**（**已完成 #48 / `91a05d8`**，见 [`docs/STATUS.md`](docs/STATUS.md)）→ **当前首项：等 Triste + 调研共决下一步**（P1 **未自动解锁**）。
5. **P1  later**：检索+重排；忠实 B1/B2 **并行 OK，不挡、不替代 P0a 门禁**。

**可推迟**：RL；完整 200–500 金标准可分期；体积；RGB-D 主表；VLM LoRA 营养头；kcal 代理弃权作主故事。

Schema 仍冻结于 `schemas/`；说明见 `docs/schema.md`。执行状态见 [`docs/STATUS.md`](docs/STATUS.md)。

Canvas 真源：https://cursor.com/canvas/store-745bdf38-f3d7-4313-8d66-08e4b6903e93/d7be617d-d37e-4fe1-a4b5-9da2c86b27a2（**VDR — Evidence-Guided Active Multimodal Nutrition Estimation**）
