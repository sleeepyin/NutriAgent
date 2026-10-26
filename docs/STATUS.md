# 执行状态（STATUS）

> 供 GitHub 同步：服务器跑完实验后，请在本文件或 `notes/` 周报中更新；执行助手据此继续改代码/方案。  
> **截止日期（文档写入时）**：2026-10-26（Asia/Shanghai）。

## 一句话

W1 基础设施在开发机（box）已基本打通：**schema + 骨架 + toy 冒烟 + 真实 FNDDS catalog（5624）+ N5K 元数据/官方 split**。  
**缺**：N5K 全量/评测 imagery、4090D 上 VLM、弱标签全量、试标 30、BC 训练。

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
| 标注指南 v0 | `docs/annotation_guide_v0.md` |
| 方案 MD | 根目录 `PLAN.md`（投稿/执行真源 MD 版） |

## 明确未完成

| 项 | 阻塞 |
|----|------|
| N5K **imagery**（overhead RGB 等） | 体积大；须在**内网服务器/大盘**下载（agent 进不了内网） |
| B0/B1 对接真实 VLM | 需 **2×4090D** 与模型权重 |
| 步骤弱标签全量（仅 train） | 需 imagery 或至少可跑感知；规则老师阈值 hold-out 调 |
| 试标 30 盘 + kappa≥0.6 | **人工** |
| 200–500 金标准 | **人工** |
| Ours-BC / RL | 依赖弱标签与特征管线 |
| 体积工具 | **保持默认关闭**（刻意不做） |
| 旧回归权重 | **禁止接入** |

## 建议服务器本周顺序

1. 按 [`SERVER_SETUP.md`](SERVER_SETUP.md) clone、装依赖、挂数据。  
2. 优先下 **overhead RGB**（或官方 split 子集），不必先下 181GB 全量 tar。  
3. 跑冒烟：`validate_schema` / `fndds_retrieve_smoke` / `n5k_metadata_summary`。  
4. 装 Qwen2.5-VL-7B（或同类），接 B0/B1 图像入口。  
5. 将指标 JSON/日志摘要 commit 到 `results/`，push；聊天告知执行助手「看 main」或贴 PR。

协作协议：[`COLLAB_GITHUB.md`](COLLAB_GITHUB.md)。人工清单：[`needs_human.md`](needs_human.md)。看板：[`w1_w4_checklist.md`](w1_w4_checklist.md)。
