# Verified Dietary Reasoning（VDR）

> **2026-11-13（Asia/Shanghai）：现行方案已改为 ICDL。本页标题仍是历史项目名。** 方法真源是 Canvas「ICDL 官方方案（2026-11-13）：从增量组成变化学习食物营养估计」（https://cursor.com/canvas/store-745bdf38-f3d7-4313-8d66-08e4b6903e93/47e6af29-51f8-443c-954d-56b0c453b602）。当前只在分支 `icdl` 上做第一阶段数据可行性审计，且只解锁这一项。Nutrition5k 是当前唯一工作对象。第二阶段、第三阶段、完整网络、Agent、强化学习、OmniFood8K 外测、ACETADA 外测，以及不完整食材监督，全部锁定。停止规则不等于解锁第三阶段。OmniFood8K 不得提前当开发集，也不得在外部盲测前调参或泄漏标签。ACETADA 只作辅助外测，且须区分实际摄入量与餐前供应量。505 与 Cafe2 不再当独立验证集。不要改 `main`（`d43cb08`）。#73 已在备份提交 `28bd3c9c` 中，不要再合入。ICDL 尚未被证明。执行门禁以 [`docs/STATUS.md`](docs/STATUS.md) 文首为准。下文仍把双视角实验写成开工许可的句子，是 2026-11-12 的历史句，不要执行。


把膳食图像理解做成**可验证中间状态的多模态工具调用 Agent**测试床。

输出 \(y=(n, G, s, \tau)\)：营养向量 + 证据图 + 核验状态（`pass` / `corrected` / `abstain`）+ 纠错轨迹。  
kcal-proxy 弃权不作主创新。  
主方法仍叫 **VDR**。输出仍是 \(y=(n, G, s, \tau)\)。2026-11-12 晚方案真源是 Canvas「**VDR 官方方案（2026-11-12 晚）：面向物理约束营养估计的主动多模态证据获取（候选）**」（https://cursor.com/canvas/store-745bdf38-f3d7-4313-8d66-08e4b6903e93/d7be617d-d37e-4fe1-a4b5-9da2c86b27a2）。它取代同日早间把少样本质量校准当作候选创新点二的定位，也取代以前「Coverage-Constrained Active Specialist Selection + Error-Aware Q（Route A 能量+质量双赢）」的主张写法。候选工作标题尚未证实：Active Multimodal Evidence Acquisition for Physically Grounded Food Nutrition Estimation（面向物理约束营养估计的主动多模态证据获取）。**Mass Estimator** 是预训练几何-质量专家，冻结结果作历史证据；**Coverage Constraint** 是辅助机制。分层少样本质量校准是普通基线，不再是创新点二。执行门禁见 [`docs/STATUS.md`](docs/STATUS.md)（① #57、② #60、③ #63 squash `27d1dbfa` 已完成；Cafe2 能量审计 #65 squash `ecf859f5` 已完成；Mass-only 能量路径便宜诊断 #68 squash `f578e8f4` 已完成；#69 squash `ae0e73aa` 的空闲收口已降为历史；#70 squash `d622e6db` 的 Phase 1 已取消，仍是 0/8；**主声称：质量对 Direct 可外推（bootstrap 区间均为负）；对 Mass-only 基本打平（0.76 g 不算赢，不要当作方法优越性证据）；能量外测未稳**；**（历史，不是现行门禁。2026-11-12 晚曾把下一步写成 Nutrition5k 双视角实验。该实验已在备份提交 28bd3c9c 中做完并停止，不要重跑，不要再合入。现行下一步只在文首：第一阶段数据可行性审计。）** 不要把本文件合进 `main`，也不要把已停止的双视角实验当成开工许可；不是实验 B、实验 C、完整联合网络或其他编码的开工许可；Cafe2 只作已探索诊断集）。State-aware 闭环与开环 31 维 BC 为 **Ours-old / 消融对照**。暂不上 RL。  

Cafe2 能量审计已完成（#65），是 ③ 的诊断后续，与 #48 同口径，**不是门禁 ④**，也**不**解锁 **④–⑦**。配对 505 仍是开发 / 诊断集。冻结 Cafe2 n=228 不重画。#63 主表不变：能量相对 Direct **+11.23**、相对 Mass-only **+41.46**；质量相对 Direct **−22.80**、相对 Mass-only **−0.76**。#51 换码 **106** 盘已在该审计里拆开。**不**声称双赢。**不**从这些标签开工改方法。**Mass-only 能量路径便宜诊断已完成（#68，squash `f578e8f4`）**（2026-11-10 Asia/Shanghai）。只读冻结 Cafe2 n=228。能量是冻结质量估计器的盘级 `energy_kcal` 头（`vit_b_16`；`estimate_mass(...).nutrients.energy_kcal`），不读 food-id，也不把 FNDDS kcal/100 g 乘以克重。**222 / 228** 盘质量头与已提交克重相差在 0.001 g 以内（最大 **0.0002** g）；另外 **6** 盘停在 RETRIEVE，Mass-only 仍记整盘头。Oracle 标签密度 × 我方克数 **98.6201**，Mass-only **75.3870**，差距 **23.2332** kcal：更大的有符号抵消，不是更准的密度或食物码。不加权 kcal/g MAE：标签密度 **0**，食物表 **0.631**，head **0.753**。#65 通道抵消：oracle **0**，食物表 **58.2**，head **77.8**。真密度 × Mass-only 克重 **100.3152**（**+1.6951**）；换成能量头降到 **75.3870**（**−24.9283**）。食物表 kcal/g 更近 **139** 盘，头更近 **89** 盘。没有一盘的头密度误差低于 **0.0001** kcal/g。**58** 盘单组件贡献 **+23.23** 里的 **+19.85**（该片能量 MAE **38.06** vs 食物表 **154.51** / oracle **116.08**；kcal/g **1.023** vs **0.811**）。**6** 盘 RETRIEVE 贡献 **−1.56**。Bootstrap **10,000** 次、种子 **0**：**[+7.92, +39.06]** kcal，**99.83%** 为正。**（历史，不是现行门禁。2026-11-12 晚曾把下一步写成 Nutrition5k 双视角实验。该实验已在备份提交 28bd3c9c 中做完并停止，不要重跑，不要再合入。现行下一步只在文首：第一阶段数据可行性审计。）** Cafe2 只作已探索诊断集，不再用于新方法调参，也不再当下一轮完全盲测。禁止自动开实验 B、实验 C、完整联合网络、已取消的 Phase 1、④–⑦、P1 与重训。不要把本文件合进 main，也不要把已停止的双视角实验当成开工许可。不要重跑，不要改 `results/heldout_massonly_energy_path/`。

**双视角实验开工段（历史，已停止，不是现行许可）。** 本轮不补采、也不新造外部混合餐的配对第二视角；用收缩评价范围代替数据补救。实验 A 仅在 Nutrition5k 同时具备顶视 RGB 与侧视 RGB、可按 dish_id 对齐的子集上进行；先核可用规模，不编造配对条数；侧视不作度量双相机。ACETADA、VLM Food Benchmark 本轮不做双视角或主动获取主测（仍缺真实可请求的第二视图）。ECUSTFD 最多作辅助几何对照，不作混合餐主测试。Cafe2 仍是已探索诊断集。完整联合网络、实验 B、实验 C 仍锁。停止规则不变：若简单融合与几何融合不能稳定胜过单视图（主要是俯视）基线，停下来让 Triste 判断，不要强行第二视角，也不要自行换方案。无 RL，不开 fast Cloud Agent，RGB-D 不进主表。Q2 仍未完全通过。Q3 不得写成已解。当时的双视角实验现已做完并停止，不是再开工。主指标是 Mass MAE，以及质量预测跨视角的一致性。主单视图基线是俯视图像。若几何/语义融合不能稳定胜过单视图基线，不要强行第二视图。协议见 [`docs/STATUS.md`](docs/STATUS.md)。

**会刊：ACM MM 2027 或更晚**；备选 AAAI/IJCAI。MM 2026 Regular Paper 截稿为 2026-04-01 AoE，会议 2026-11-10 至 2026-11-14（里约热内卢）。MM 2027 在香港，日期与截稿尚未公布，不虚构 2027 截稿。算力假设 **2×4090D 24GB**。

> **方案真源（请先读）**：[PLAN.md](PLAN.md)

## 硬约束

- 不加载旧营养回归权重  
- 体积工具**默认关闭**  
- 公开数据：Nutrition5K + FNDDS；人工仅 200–500 盘；无私有大规模采集  
- 禁止退回「只报 MAE 的规则 Agent」作为投稿主叙事  

## 文档导航

| 文档 | 内容 |
|------|------|
| [PLAN.md](PLAN.md) | 完整实验方案（任务/基准/方法/基线/表/周程/MVP） |
| [docs/STATUS.md](docs/STATUS.md) | **当前执行状态**（已完成 / 待办） |
| [docs/VDR_P0.md](docs/VDR_P0.md) | **Ours-old / 历史脚手架**（Evidence-Aware Closed-Loop 对照），**不是**现行方法。31 维开环 = 消融 B5；ABSTAIN 0/24；无 RL |
| [docs/B1_B2_SCAFFOLD.md](docs/B1_B2_SCAFFOLD.md) | B1 DietAI24 / B2 FoodAgent 脚手架：dry-run only，未跑主表。历史 B1 能量 MAE 110.72 不是忠实 DietAI24 |
| [docs/COLLAB_GITHUB.md](docs/COLLAB_GITHUB.md) | **内网服务器 ↔ GitHub ↔ 助手**协作协议 |
| [docs/SERVER_SETUP.md](docs/SERVER_SETUP.md) | 服务器下载数据、冒烟、VLM 注意 |
| [docs/annotation_guide_v0.md](docs/annotation_guide_v0.md) | 人工标注指南 v0 |
| [docs/schema.md](docs/schema.md) | 输出/IR/证据绑定说明 |
| [docs/needs_human.md](docs/needs_human.md) | 需人工介入清单 |
| [docs/data_acquisition.md](docs/data_acquisition.md) | 数据状态与许可 |

## 快速开始（笔记本 / 服务器）

```bash
git clone <YOUR_REPO_URL> vdr-agent && cd vdr-agent
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

python scripts/validate_schema_examples.py
python scripts/fndds_retrieve_smoke.py   # 期望 n_foods=5624（若已含 foods.json）
python scripts/n5k_metadata_summary.py   # 期望 rgb_train=4059 / rgb_test=709
python scripts/rule_verify_toy_smoke.py
python scripts/b1_toy_smoke.py
python scripts/eval_toy_smoke.py
python scripts/weak_steps_smoke.py
python scripts/b1_b2_scaffold_smoke.py   # B1/B2 协议干跑，不写 results/
```

真实图像与 VLM：见 [docs/SERVER_SETUP.md](docs/SERVER_SETUP.md)。  
数据路径：

```bash
export VDR_N5K_ROOT=/path/to/nutrition5k
export VDR_FNDDS_PATH=/path/to/foods.json
```

## 仓库里有什么 / 没有什么

**有**：代码骨架、冻结 schema、toy 与真实 FNDDS `foods.json`（5624）、N5K 元数据与官方 split、方案与协作文档。  
**没有（勿期望 clone 即有）**：N5K 全量 imagery；原始 66MB FDC JSON（gitignore）。请在服务器下载 imagery。

## 与执行助手协作（内网）

1. 本仓库设为 GitHub **private**（推荐）。  
2. 服务器跑完 → 把指标/摘要写入 `results/`，结论写入 `docs/STATUS.md` → push。`notes/` 只留说明，历史周报已删除。  
3. 聊天里发**仓库 URL**或「看 main / 某 PR」。  
4. 助手**不能**进内网 SSH，只通过 GitHub 继续改代码与方案。  

细则：[docs/COLLAB_GITHUB.md](docs/COLLAB_GITHUB.md)。

## 目录（代码）

```
schemas/     冻结 JSON Schema
vdr/         包：tools / baselines / controller / eval / ir / weak
scripts/     校验、catalog 构建、冒烟
examples/    pass / corrected / abstain
data/toy/    冒烟假数据
results/     可同步小结果（初始为空）
notes/       历史周报已删除；见 notes/README.md 与 docs/STATUS.md 冻结节
```

## 许可与伦理

遵守 Nutrition5K / FNDDS 原许可。本系统为研究原型，**非医疗器械**，不作临床膳食决策主张。
