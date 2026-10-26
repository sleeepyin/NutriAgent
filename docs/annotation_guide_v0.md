# VDR 人工标注指南 v0（约 2 页）

面向 W3–W4 试标与金标准子集。只为**公开 Nutrition5K 图像**追加标签；**不新建私有采集**。营养硬标签沿用 N5K 官方，不要求标注员重估 kcal。

## 1. 目标

对每盘抽样图，判断系统（或老师轨迹）给出的证据是否成立、是否应当弃权、主冲突类型。服务指标：Evidence-Grounded Acc、Abstention AUROC、按 `conflict_type` 分层误差。

## 2. 每盘必填字段

| 字段 | 类型 | 说明 |
|------|------|------|
| `dish_id` | string | Nutrition5K ID |
| `evidence_ok` | bool | 上报营养是否可追溯到合理的 `(food_code, portion_g, FNDDS per-100g)` 路径，且 code/份量对可见食物合理 |
| `should_abstain` | bool | 以标注员视角：该图是否应拒绝给出可靠营养（遮挡/酱汁/库外/极度混杂等） |
| `conflict_type` | enum | 主冲突：`id` \| `portion` \| `db_gap` \| `sauce` \| `multi` \| `other` |
| `notes` | string | 可选短注（≤200 字） |

### conflict_type 定义

- **id**：认错食物/成分（code 错）
- **portion**：认对食物但克重明显不合理
- **db_gap**：FNDDS 难以覆盖的复合菜/地方菜
- **sauce**：酱汁/油分主导且视觉不可分
- **multi**：多成分严重遮挡或无法可靠分解
- **other**：其他（请在 notes 写一句）

若 `should_abstain=true`，仍选一个**最主要**的 `conflict_type`。

### evidence_ok 操作定义（与 schema 一致）

- 每个计入的营养分量须有完整计算路径；路径缺失 → `evidence_ok=false`。
- code 明显与可见食物不符，或份量离谱（例如米饭标成 20g/800g）→ `false`。
- 弃权样本：若系统正确弃权，`evidence_ok` 可标 `true`（表示「决策与证据一致」）或按项目配置单独计；**试标阶段约定**：系统弃权且标注员也认为应弃权 → `evidence_ok=true`，`should_abstain=true`。

## 3. 试标 30 盘流程（W3）

1. 培训 30–60 分钟：读本指南 + 看 5 个示范（pass/corrected/abstain 叙事）。
2. 两人独立标同一批 **30 盘**（分层：简单 / 混菜 / 酱汁各约 10）。
3. 计算 Cohen's **kappa**（至少对 `evidence_ok`、`should_abstain`、`conflict_type`）。
4. 目标 **kappa ≥ 0.6**；未达标 → 修订本指南歧义段 → 再试 10–20 盘。
5. 第三人仅裁决持续分歧项，不先参与独立轮。

## 4. 正式标注（W5–7，预告）

- 规模 **200–500** 盘；双人 + 第三人裁决。
- 只发布 `dish_id` + 标签；原图按 N5K 许可处理。
- 禁止使用测试集 GT 反推「标准答案步骤」污染弱监督（人工金标准是证据/弃权标签，不是把 GT kcal 写进轨迹老师）。

## 5. 伦理与禁止

- 非医疗器械；不作临床决策主张。
- 不私自爬取或拍摄未授权餐图作为主基准。
- 不沿用旧回归模型权重辅助「标准标注」。

## 6. 版本

- v0：W1 初稿，供试标前评审。
- 试标后根据 kappa 与歧义案例升为 v1。
