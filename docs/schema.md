# VDR 任务 Schema 说明（冻结稿 v0）

本文对齐方案真源「Verified Dietary Reasoning」任务形式化：输出 \(y=(n, G, s, \tau)\)，每步留下 IR。

## 1. 终态输出 `vdr_output.schema.json`

| 字段 | 含义 |
|------|------|
| `dish_id` | 样本 ID（Nutrition5K `dish_id` 等） |
| `n` | 营养向量：必含 `energy_kcal / protein_g / carb_g / fat_g / mass_g`；可选 `micronutrients` |
| `G` | 证据图：`nodes` + `edges` |
| `s` | 核验状态枚举：`pass` \| `corrected` \| `abstain` |
| `tau` | 有序轨迹：`[{t, action, obs, state_summary}, ...]` |
| `abstain_reason` | `s=abstain` 时填写 |

### 1.1 证据图节点类型

`query` / `hyp` / `food_code` / `portion_g` / `nutrition` / `rule` / `dish`

### 1.2 证据边类型（至少）

| 边类型 | 操作定义 |
|--------|----------|
| `code<-query` | 检索查询文本绑定到选出的 FNDDS food_code |
| `code<-hyp` | 感知假设 hyp 绑定到 food_code |
| `portion<-code` | 该 code 对应的克重来源 |
| `n<-code×portion` | 营养分量 = FNDDS per-100g × (portion_g/100) |
| `n<-aggregate` | 盘级营养由各成分求和 |
| `rule_trigger` | 规则先验触发（verify_codes） |
| `corrects` | 纠错步：新状态修正旧状态 |
| `abstain_from` | 弃权所依据的失败证据 |

### 1.3 证据绑定（Evidence Binding）操作定义

**每个上报的营养分量必须能追溯到至少一条** `(food_code, portion_g, FNDDS_per_100g)` **计算路径**。

- 路径完整 → grounded，计入 Evidence Acc 分子。
- 缺失路径 → ungrounded，分子不计；人工子集上记 `evidence_ok=false`。
- `s=abstain` 时允许 `n` 字段为 `null`，但须给出 `abstain_reason` 与至少一条失败相关边。

## 2. 逐步 IR `ir_step.schema.json`

与方案 IR 表一致：

| 字段 | 含义 | 评测用途 |
|------|------|----------|
| `hyp_i` | 成分/区域假设 | 认错分析 |
| `code_i` / `cands_i` | FNDDS 码与 Top-k | Evidence Acc |
| `portion_g_i` / `portion_src_i` | 克重与来源 | 份量错 |
| `nutri_i` / `nutri_dish` | 查表缩放 / 盘级聚合 | MAE/MAPE |
| `verify_codes` | 触发规则码 | 控制器特征 |
| `evidence_edges` | 本步证据边 | G 完整性 |
| `action_t` / `obs_t` | 动作与工具返回 | 步骤监督 |
| `s` / `abstain_reason` | 终态（若本步决定） | 校准 |

## 3. 动作空间 `actions.json`

核心：`RETRIEVE` / `RE_PORTION` / `RE_DECOMPOSE` / `ABSTAIN` / `STOP`  
可选默认开：`CORRECT_CODE`  
**默认关**：`VOLUME_ESTIMATE`（Tanabe 警示：体积 Prompt 可灾难性失败）

规则核验不是动作，只提供 `verify_codes` 特征与硬安全护栏。

## 4. 文件位置

- 真源 JSON Schema：`schemas/*.json`
- Python 加载校验：`vdr/schemas/`
- 合法样例：`examples/{pass,corrected,abstain}.json`
