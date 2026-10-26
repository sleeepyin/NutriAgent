# Verified Dietary Reasoning（VDR）实验方案

> **本文档是投稿叙事与执行的 Markdown 真源。** 与早期 Cursor Canvas 方案对齐并落地；精读笔记见文末 Related。  
> 算力假设：**2×4090D 24GB**。数据：**公开 Nutrition5K + FNDDS**；人工 **200–500** 盘薄标注；**无私有大规模采集**。

---

## 0. 一句话定位

提出 **Verified Dietary Reasoning（VDR）**：把膳食图像理解做成「可验证中间状态的多模态工具调用 Agent」测试床——输出不是单点营养数字，而是

\[
y = (n,\ G,\ s,\ \tau)
\]

- \(n\)：营养向量（至少 energy / protein / carb / fat / mass）
- \(G\)：FNDDS/计算路径证据图
- \(s \in \{\mathrm{pass},\ \mathrm{corrected},\ \mathrm{abstain}\}\)：核验状态
- \(\tau\)：有序纠错轨迹（动作 + 工具观测）

方法核是**可学习控制器**；规则核验只作先验与安全护栏。  
**主投 ACM Multimedia（CCF-A）**；备选 AAAI/IJCAI。CVPR/ICCV 仅作加长愿景，不赌新视觉 backbone。Nature 医学刊不作主目标。

一般化叙事对标：Toolformer / ReAct / Self-Refine / CRITIC + 可信多模态；领域对标 DietAI24（单向 RAG）、FoodAgent（启发式 MoE）、NutriMLLM（反弃权训练哲学作对照）。

### 三条主贡献（写作时冻结）

1. **任务**：形式化 VDR，输出 \((n,G,s,\tau)\)，兼容旧 MAE，并引入证据与弃权协议。
2. **基准**：VDR-Bench（Nutrition5K 主干 + 步骤级弱监督 + 200–500 盘人工金标准子集），开放评测脚本与标注规范。
3. **方法**：可学习工具控制器（BC → DAgger → 可选小动作空间 RL）；证明相对纯规则与 Self-Refine 的必要性。

---

## 1. 硬禁止（Desk-reject / 自我打脸预防）

| 禁止 | 原因 |
|------|------|
| **加载/蒸馏旧营养回归权重** | 方案已锁定从零搭 Agent；回归专家会污染贡献叙事 |
| **把「纯对话营养 VLM」当主创新** | 缺工具状态与证据绑定，易被视作应用拼装 |
| **退回「只报 N5K MAE 的规则 Agent」** | 那是规则版 v1，不够 CCF-A |
| **体积工具默认开启** | Tanabe 2025：体积 Prompt 可灾难性拖垮开源 VLM；默认关，仅消融可选 |
| **大规模 RLHF 整模偏好训练** | 算力与贡献焦点都不匹配；控制器小网络即可 |
| **用 test GT 反标步骤弱监督** | 泄漏；弱标签仅 train |
| **私有大规模采集替代公开主表** | 许可/复现性差；第二源另议 |

---

## 2. 研究问题与假设

**RQ1（任务/基准）**：仅优化营养 MAE 是否掩盖「无证据、不可审计、不会弃权」的失败？引入证据绑定与校准弃权后，能否在不牺牲（或小幅牺牲）MAE 的前提下显著提升可信指标？

**RQ2（方法）**：在相同感知 VLM 与工具集下，可学习控制器是否显著优于规则映射、Self-Refine/CRITIC 式文本自纠，以及 DietAI24/FoodAgent 式单次或固定路由？

| 假设 | 内容 |
|------|------|
| **H1（主）** | LearnedController 相对 RuleOnly 与 Self-Refine，在 primary 指标合集上帕累托更优：能量 MAE 不显著变差（或更好），同时 Evidence-Grounded Acc 与 Abstention AUROC/ECE 显著更好，且 FixRate > WorsenRate（配对检验预注册）。 |
| **H2（机制）** | 去掉步骤级监督后性能显著下降；去掉证据约束只优化 MAE 时 Evidence Acc 崩溃。 |
| **H3（泛化）** | 第二源上同策略零样本/少样本迁移后，相对 B1 RAG-once 仍保持证据与弃权优势（允许 MAE 差距缩小）。 |

**Primary 预注册建议**：Nutrition5K test 上 MAE@energy；人工子集上 Evidence Acc 与 Abstention AUROC（主表双栏）。  
**不编造目标数值**。外部锚点（文献报告，引用时再核 PDF）：DietAI24 ASA24 能量 MAE 约 47.7；Tanabe N5K LLaVA LoRA MAE=64.3。

---

## 3. 任务形式化与 IR / 证据图

### 3.1 输出 \(y=(n,G,s,\tau)\)

输入：膳食图像 \(x\)（可选元数据）。  
流水线：感知 VLM → 可学习控制器 \(\pi(a\mid h)\) ⇄ 工具集；规则先验注入 \(h\) / 硬护栏；停止或弃权后发射 \(y\)。

Schema 已冻结：`schemas/vdr_output.schema.json`、`schemas/ir_step.schema.json`、`schemas/actions.json`。说明见 [`docs/schema.md`](docs/schema.md)。

### 3.2 逐步 IR（每步必留）

| 字段 | 含义 | 评测用途 |
|------|------|----------|
| `hyp_i` | 成分/区域假设 | 认错分析 |
| `code_i` / `cands_i` | FNDDS 码与 Top-k | Evidence Acc |
| `portion_g_i` / src | 克重与来源 | 份量错 |
| `nutri_i` / `nutri_dish` | 查表缩放 / 盘级 | MAE/MAPE |
| `verify_codes` | 触发规则码 | 控制器输入 |
| `evidence_edges` | `code<-query`、`n<-code×portion` 等 | G 完整性 |
| `action_t` / `obs_t` | 动作与工具返回 | 步骤监督 |
| `s` / `abstain_reason` | 终态 | 校准 |

### 3.3 证据绑定（操作定义）

**每个上报的营养分量必须能追溯到至少一条** `(food_code, portion_g, FNDDS_per_100g)` **计算路径**。  
缺失路径 → ungrounded（Evidence Acc 分子不计）。

边类型至少含：`code<-query`、`code<-hyp`、`portion<-code`、`n<-code×portion`、`n<-aggregate`、`rule_trigger`、`corrects`、`abstain_from`。

---

## 4. 基准：VDR-Bench

| 项 | 约定 |
|----|------|
| 主干 | Nutrition5K **官方 split**；营养硬标签；自动生成步骤级弱标签（仅 train） |
| 第二源 | 优先 ASA24（DietAI24 协议）或 SNAPMe；拿不到则降级自建 100–200 张（仅证据/弃权标，营养查表近似并声明） |
| 人工金标准 | **200–500** 盘分层抽样（简单/混菜/酱汁/高误差）；双人标 `evidence_ok` / `should_abstain` / `conflict_type∈{id,portion,db_gap,sauce,multi,other}`；第三人裁决 |
| 许可 | 遵守 N5K/FNDDS/ASA24/SNAPMe 原许可；人工子集默认只发 **dish_id + 标签** |

当前仓库内 N5K **元数据+split 已就绪**（rgb_train **4059** / rgb_test **709**）；**全量 imagery 需在服务器下载**。见 [`docs/STATUS.md`](docs/STATUS.md)、[`docs/SERVER_SETUP.md`](docs/SERVER_SETUP.md)。

### 4.1 步骤级监督（弱→强）

1. **自动弱标签（全量 train）**：规则老师 + 对照营养 GT 反推 RETRIEVE / RE_PORTION / ABSTAIN 等（**不用 test GT**）。
2. **DAgger**：控制器上线后每周约 50–100 盘纠偏。
3. **人工子集**：金标准证据/弃权/冲突类型。

### 4.2 主指标（预注册）

| 指标 | 角色 |
|------|------|
| MAE/MAPE/r | 兼容旧文献 |
| Evidence-Grounded Acc | 新任务主指标 |
| Abstention AUROC / ECE；AbstainRate；条件 MAE | 可信性 |
| FixRate / WorsenRate | 控制器行为 |
| Tool Efficiency（#Tools、Latency） | 实用约束 |
| Policy Acc | 学控制器必要 |

### 4.3 质量控制

- 标注指南约 2 页 + 试标 30 盘；Cohen **kappa ≥ 0.6**，否则修订指南。见 [`docs/annotation_guide_v0.md`](docs/annotation_guide_v0.md)。
- 规则阈值只在 train hold-out 调；测试与人工子集冻结。

---

## 5. 方法：感知 + 工具 + 可学习控制器 + 规则先验

与规则版 v1 关键差异：**规则不再是主方法**，只提供 `verify_codes` 与硬安全动作。

| 模块 | 约定 |
|------|------|
| 感知 VLM | 开源 7B–8B（如 Qwen2.5-VL）；输出成分假设；可冻结或轻量格式对齐。**禁止旧回归权重** |
| 工具 | FNDDS 检索、份量网格、查表聚合、规则核验、可选重分解；**体积默认关** |
| 状态 \(h\) | 感知摘要 + Top-k 分数差 + verify one-hot + 步数预算 + 误差代理 |
| 动作 \(a\) | `{RETRIEVE, RE_PORTION, RE_DECOMPOSE, ABSTAIN, STOP}`（可含 `CORRECT_CODE`） |

### 两阶段训练

1. **Stage A：BC / 模仿**（+ 可选 DAgger 2–3 轮）  
2. **Stage B：小动作空间 RL**（A 会强烈建议；冻结感知与工具，只训控制器；REINFORCE/PPO-lite/bandit；**禁止 RLHF 整模**）

控制器用小型 MLP/Transformer，**不必回传梯度进 7B VLM**。

---

## 6. 基线

| 代号 | 定义 | 回答什么 |
|------|------|----------|
| B0 Direct-VLM | 单次 VLM 直接报营养 | 查表必要性 |
| B1 RAG-once | DietAI24 式单次检索聚合 | 领域强基线 |
| B2 Fixed-MoE | FoodAgent 式固定路由（或 approx） | 启发式分治 |
| B3 Self-Refine | 文本自反思，无结构化工具状态 | 对标 Self-Refine |
| B4 CRITIC-style | 外部批评者再生成，弱工具 | 对标 CRITIC |
| B5 RuleVerify | v1 纯规则控制器 | 规则上限 |
| B6 Toolformer-FT | 轨迹上指令微调选工具，无显式核验特征 | 对标 Toolformer |
| Ours-BC | 可学习控制器仅 BC/DAgger | 模仿是否够 |
| Ours-BC+RL | BC 初始化 + 小动作 RL | 完整主方法 |

---

## 7. 必出实验表（时间不够也要保 M2 + A1）

| 表/图 | 内容 |
|-------|------|
| **M1** | MAE/MAPE/r on Nutrition5K（B0–B6 vs Ours） |
| **M2** | Evidence Acc、Abstention AUROC/ECE、AbstainRate、条件 MAE（人工子集） |
| **M3** | FixRate、WorsenRate、#Tools、Latency、Policy Acc |
| **A1** | RuleVerify vs Ours-BC vs Ours-BC+RL；去 verify 特征；去证据奖励 |
| A2 | 监督必要：无步骤弱标签；仅最终奖励 RL；+DAgger |
| A3 | 动作空间消融 |
| F1–F3 | 案例证据图；覆盖–误差帕累托；按 conflict_type 分层 |

审稿人必看：若 M2/A1 不显著，易被打成工程拼装。宁可砍体积与第三数据集，也要保住 **M2 与 A1**。

---

## 8. 第二数据集与泛化

- **G1**：ASA24（优先，对齐 DietAI24）  
- **G2**：SNAPMe  
- **G3**：自建 100–200（降级）  
报告零样本迁移表；可选 50 盘子样本微调控制器（禁止测试泄漏）。

---

## 9. 12–16 周里程碑

| 周 | 内容 |
|----|------|
| W1–2 | N5K+评测脚本；FNDDS 工具；IR/证据图 schema；B0/B1 冒烟 |
| W3–4 | RuleVerify；train 弱标签；标注指南 + 试标 30；kappa |
| W5–7 | 200–500 双人标注与裁决；冻结人工测试子集 |
| W6–8 | Ours-BC；接 B3/B4/B5；M1/M3 初稿 |
| W9–11 | Ours-BC+RL；A1–A3 |
| W10–12 | 第二源三选一；泛化表 |
| W13–14 | 冻结数字；案例图；开源清单 |
| W15–16 | ACM MM 稿；备用 AAAI/IJCAI 叙事 |

倒推以当年 ACM MM CfP 为准；至少 16 周执行 + 2 周缓冲。

看板：[`docs/w1_w4_checklist.md`](docs/w1_w4_checklist.md)。当前进度：[`docs/STATUS.md`](docs/STATUS.md)。

---

## 10. 算力与人力

- **算力**：1–2× 24–48GB GPU；7B–8B 推理；控制器训练单卡数小时到数天。API 仅附录敏感性。
- **人力**：1 主研全职；标注 2 人兼职，约 200–500 盘 × 5–8 分钟/盘 + 培训。
- **禁止预算**：不投入旧回归重训；不投入大规模 RLHF；体积仅可选消融。

---

## 11. MVP-for-A vs Full

**MVP-for-A（仍冲 MM/AAAI）**

- VDR 任务形式化 + IR/证据图  
- N5K 主表 MAE + 至少 200 盘人工 Evidence/Abstain  
- 步骤弱标签 + Ours-BC（DAgger ≥1 轮）  
- 基线：B1、B3、B5、Ours；表 M1/M2/M3/A1  
- 开源评测与标注规范  

**可推迟到 Full / 附录**：Stage B RL；SNAPMe/大规模第二源；体积工具；65 微营养；完整 FoodAgent 复现；VLM 端到端 LoRA。

若 RL 与第二源未完成：主投仍写 VDR 任务+基准+BC，强调相对 RuleVerify/Self-Refine 的可信指标。**禁止**退回只报 MAE 的规则 Agent。

---

## 12. Diff from 规则版 v1（简表）

| 维度 | v1 规则 Verify-then-Correct | 本方案 VDR |
|------|------------------------------|------------|
| 输出 | 营养数字 (+轨迹) | 营养 + 证据图 + 核验状态 + 轨迹 |
| 贡献叙事 | 工程自纠错提升 MAE | 新任务 + 新基准 + 可学习控制器 |
| 主方法 | 规则控制器 | 学习控制器为主，规则仅先验 |
| 指标 | MAE + FixRate 为主 | MAE + Evidence Acc + 弃权校准 + Fix/Worsen + Policy Acc |
| 数据 | N5K + 人工错误分析 | 弱监督全量 + 200–500 金标准 |
| 基线 | 缺 Self-Refine/CRITIC/Toolformer | 补齐 Agent/自纠错对照 |
| 会刊 | JBHI/Nutrients/Workshop 口味 | 主投 ACM MM，备选 AAAI/IJCAI |
| 工期 | ~8 周 | 12–16 周（含标注） |

---

## 13. 前 4 周立刻动手

1. 冻结任务 schema：`(n,G,s,τ)` 与证据边类型。✅（见 `schemas/`）  
2. N5K 官方 split 评测 + FNDDS 工具 + B1 RAG-once 冒烟。🔄（元数据/FNDDS/B1-toy 已通；imagery+VLM 待服务器）  
3. 实现 RuleVerify 老师与步骤弱标签生成器（仅 train）。🔄（RuleVerify 骨架已有；弱标签全量待做）  
4. 写出 2 页标注指南；完成 30 盘试标并算初版 kappa。🔄（指南 v0 已有；试标待人工）  
5. 搭控制器 BC 数据管线；不碰旧回归、不开体积。⏳  

协作方式（内网服务器 ↔ GitHub ↔ 执行助手）：[`docs/COLLAB_GITHUB.md`](docs/COLLAB_GITHUB.md)。

---

## 14. 风险、伦理（摘要）

- Desk-reject「应用拼装」→ Intro 前两段必须出现新任务与三贡献；Related Work 先 Agent/verification 再 food。  
- 「人工子集太小」→ 200–500 + CI；弱标签全量 + 人工校准。  
- 「弃权刷分」→ 同时报强制回答 MAE 与允许弃权的条件 MAE + AUROC。  
- 伦理：非医疗器械；膳食建议免责；数据许可；无临床决策主张。

---

## Related

- 精读笔记（FoodAgent / DietAI24 / Tanabe / VLMs Ready / NutriMLLM）：原 Cursor Canvas「营养 VLM/Agent 五篇精读笔记」（仓库外）；写作 Related Work 时以原文 PDF/HTML 再核数字。  
- 实现状态：[`docs/STATUS.md`](docs/STATUS.md)  
- 服务器：[`docs/SERVER_SETUP.md`](docs/SERVER_SETUP.md)  
- Schema：[`docs/schema.md`](docs/schema.md)  
- 标注：[`docs/annotation_guide_v0.md`](docs/annotation_guide_v0.md)
