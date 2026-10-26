# 执行状态（STATUS）

> 供 GitHub 同步：服务器跑完实验后，请在本文件或 `notes/` 周报中更新；执行助手据此继续改代码/方案。  
> **截止日期（文档写入时）**：2026-10-26（Asia/Shanghai）。

## 一句话

W1 基础设施在这台 2×4090D 上已打通：**schema + 骨架 + toy 冒烟 + 真实 FNDDS catalog（5624）+ N5K 元数据/官方 split + 俯拍 RGB + Qwen2.5-VL-7B 单盘冒烟**。  
B1 图像路径已做一版硬化（假设去重、portion grid、食材检索先验、`CODE_CONF_GAP` 时多一步再停）。**服务器拉下后请重跑同一 50 盘子集**（见文末命令）。  
**缺**：全量官方 test 的 B0/B1 主表、弱标签全量、试标 30、BC 训练。侧视视频未解压。50 盘子集数字不能当主表。

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
| N5K 俯拍 RGB | `data/nutrition5k/imagery/realsense_overhead/*/rgb.png`：**3490** 张，1.35GB。覆盖官方 rgb_train **2755/4059**、rgb_test **507/709**；另有 **228** 张不在官方 RGB split。未解 depth / 侧视视频。摘要：`results/n5k_overhead_rgb.json` |
| Qwen2.5-VL-7B | 权重在 `/var/tmp/vdr-models/Qwen2.5-VL-7B-Instruct`（不进 Git）。B0/B1 图像入口已接。单盘冒烟：`results/b0_b1_image_smoke.json`。50 盘官方 test 子集：`results/n5k_subset_b0_b1.json`。跑在空闲的 GPU 0；GPU 1 上原有 llama-server 未动 |
| B1 硬化（CPU，无 VLM） | 同名假设取中位数、不累加；`portion_grid` 默认克重网格接到查表；plain 果蔬查询压低 dip/soup/juice 等并偏向 raw；`CODE_CONF_GAP` 时先二次 `RETRIEVE`（仍是蘸酱/汤类）或确认性 `RE_PORTION` 再 `STOP`；终态 `n` 不含 `water_g`。冒烟：`scripts/b1_harden_smoke.py` |
| 标注指南 v0 | `docs/annotation_guide_v0.md` |
| 方案 MD | 根目录 `PLAN.md`（投稿/执行真源 MD 版） |

## 明确未完成

| 项 | 阻塞 |
|----|------|
| 全量官方 test 的 B0/B1 主表 | 50 盘子集已有能量 MAE（B0 **98.2**，B1 **112.1**）。还不是 709 盘全表 |
| 步骤弱标签全量（仅 train） | 感知已通。阈值只在 train hold-out 调，禁止用 test GT |
| 试标 30 盘 + kappa≥0.6 | **人工** |
| 200–500 金标准 | **人工** |
| Ours-BC / RL | 依赖弱标签与特征管线 |
| 体积工具 | **保持默认关闭**（刻意不做） |
| 旧回归权重 | **禁止接入** |

## 建议服务器本周顺序

1. 按 [`SERVER_SETUP.md`](SERVER_SETUP.md) clone、装依赖、挂数据。已完成（依赖装在 `.venv`，PyPI 主站不可达，用了清华镜像）。  
2. 优先下 **overhead RGB**。已从本机 tar 抽出 `rgb.png`，未解侧视视频。  
3. 冒烟已通过：`validate_schema` / `fndds_retrieve_smoke`（5624）/ `n5k_metadata_summary` / `rule_verify_toy_smoke` / `b1_toy_smoke`（5/5）/ `eval_toy_smoke`。  
4. Qwen2.5-VL-7B 已装上，B0/B1 图像入口和单盘冒烟已通过。  
5. 覆盖摘要在 `results/n5k_overhead_rgb.json`，单盘冒烟在 `results/b0_b1_image_smoke.json`。尚未 commit。  
6. 有图的官方 test 小子集已跑完：seed=42 的 50/507 盘，能量 MAE 为 B0 98.2 kcal、B1 112.1 kcal。不是全表。  
7. **B1 硬化已进代码**（针对上面 50 盘的三类失败：重复假设把克重加总、盲信 VLM `portion_g`、plain 菜名检索到 dip/soup/juice）。体积工具仍关，没有接入旧回归权重。全量 test 主表仍可之后再跑。弱标签仍只许用 train。

## 服务器请重跑 50 盘子集

拉下本分支后，在已有俯拍 RGB 和 Qwen2.5-VL 权重的机器上重跑**同一协议**（seed=42，官方 `rgb_test` 里有图的盘中抽 50）。会覆盖 `results/n5k_subset_b0_b1.jsonl` 和 `results/n5k_subset_b0_b1.json`：

```bash
python scripts/eval_n5k_subset.py --n 50 --seed 42 --split rgb_test
```

不需要 GPU 的回归：

```bash
python scripts/b1_harden_smoke.py
python scripts/validate_schema_examples.py
python scripts/fndds_retrieve_smoke.py
python scripts/rule_verify_toy_smoke.py
python scripts/b1_toy_smoke.py
python scripts/eval_toy_smoke.py
```

协作协议：[`COLLAB_GITHUB.md`](COLLAB_GITHUB.md)。人工清单：[`needs_human.md`](needs_human.md)。看板：[`w1_w4_checklist.md`](w1_w4_checklist.md)。
