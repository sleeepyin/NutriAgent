# Verified Dietary Reasoning（VDR）

把膳食图像理解做成**可验证中间状态的多模态工具调用 Agent**测试床。

输出 \(y=(n, G, s, \tau)\)：营养向量 + 证据图 + 核验状态（`pass` / `corrected` / `abstain`）+ 纠错轨迹。  
主方法：可学习控制器（BC → DAgger → 可选小动作 RL）；规则仅先验/护栏。  
**主投 ACM Multimedia（CCF-A）**；备选 AAAI/IJCAI。算力假设 **2×4090D 24GB**。

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
| [docs/COLLAB_GITHUB.md](docs/COLLAB_GITHUB.md) | **内网服务器 ↔ GitHub ↔ 助手**协作协议 |
| [docs/SERVER_SETUP.md](docs/SERVER_SETUP.md) | 服务器下载数据、冒烟、VLM 注意 |
| [docs/annotation_guide_v0.md](docs/annotation_guide_v0.md) | 人工标注指南 v0 |
| [docs/schema.md](docs/schema.md) | 输出/IR/证据绑定说明 |
| [docs/w1_w4_checklist.md](docs/w1_w4_checklist.md) | W1–W4 看板 |
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
2. 服务器跑完 → 把指标/摘要写入 `results/`、`notes/` → push。  
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
notes/       周报（初始为空）
```

## 许可与伦理

遵守 Nutrition5K / FNDDS 原许可。本系统为研究原型，**非医疗器械**，不作临床膳食决策主张。
