# 执行状态（STATUS）

> 供 GitHub 同步：服务器跑完实验后，请在本文件或 `notes/` 周报中更新；执行助手据此继续改代码/方案。  
> **截止日期（文档写入时）**：2026-10-27（Asia/Shanghai）。  
> **可执行的下一步只在本文件。** 聊天不是 runbook。助手改下一步、跑法或方案时，先更新文档并推到 GitHub；服务器上的人或 AI 先 `git pull --ff-only`，再读「下一步」执行。  
> **服务器执行看「下一步（服务器立即执行）」。** 文末的 `--n 50` 和俯拍池 `--n 507` 都是已经跑完的记录，不要再执行。

## 下一步（服务器立即执行）

**没有新命令。** 有俯拍 RGB 的官方 `rgb_test` 池（507/709）已跑完，数字在下一节。不要重跑 `--n 507`，不要覆盖 `results/n5k_rgb_test_overhead_b0_b1.json` 与 `.jsonl`。默认的 `results/n5k_subset_b0_b1.json` 与 `.jsonl` 现在就是这一对，再跑脚本会截断覆盖。n=50 基线只在 `results/n5k_subset_b0_b1_n50_seed42.json` 与 `.jsonl`。

弱标签、改检索、改克重都还没有可执行命令。不要用这次 test 数字调阈值、检索或克重。体积工具仍关。旧回归权重仍不接入。下一轮先改本节并 push，再执行。

### 俯拍 RGB 池结果（2026-10-27）

- 代码：推理进程启动时日志为 `CODE 260c4b0`。`vdr/` 与 `scripts/` 相对 `1d4da37` 无差异；`9594f57` 只改了文档。祖先含 `1d4da37`。
- 时间：`2026-10-27T11:18:02+08:00` → `2026-10-27T12:14:52+08:00`（约 57 分钟），物理 GPU 0。GPU 1 上的 llama-server 未停。
- 解释器：仓库 `.venv` 没有 torch。按文首旧命令 `source .venv` 会每盘 `ModuleNotFoundError`，并立刻覆盖默认 json/jsonl。那次空跑没有拷到稳定文件名，默认文件已从 n=50 备份恢复后再跑。实际命令是 `CUDA_VISIBLE_DEVICES=0 /var/tmp/nutri-venv/bin/python -u scripts/eval_n5k_subset.py --n 507 --seed 42 --split rgb_test`。`CUDA_VISIBLE_DEVICES=0` 让加载器的 `cuda:0` 落在物理 GPU 0，没有绑到 GPU 1。
- 首行日志：`pool=507 subset=507 seed=42 split=rgb_test`
- n_b0_ok / n_b1_ok / n_b1_abstain：`507` / `507` / `0`
- 能量 MAE（summary 原值）：B0 **97.03**（中位 **63.51**，MAPE 65.17%，n=507），B1 **110.72**（中位 **63.57**，MAPE 71.61%，n=506）
- 同一次 summary 的其余 MAE：B0 蛋白质 8.79 g、碳水 10.39 g、脂肪 6.63 g、质量 76.46 g；B1 蛋白质 8.9 g、碳水 9.3 g、脂肪 8.84 g、质量 74.85 g。B1 这四项的 n 都是 506。
- 产物：`results/n5k_rgb_test_overhead_b0_b1.json` 与 `.jsonl`。成功跑完后，默认的 `results/n5k_subset_b0_b1.json` 与 `.jsonl` 与这一对相同。n=50 基线只在 `results/n5k_subset_b0_b1_n50_seed42.json` 与 `.jsonl`（能量 MAE 仍是 B0 98.29、B1 88.21）。
- 范围：官方 rgb_test 中有俯拍 RGB 且有标签的 507 盘。不是 709 盘全表。
- B1 能量 n=506：`dish_1557861216`（标签 0 kcal）VLM 返回空列表，`s=pass`，`n` 全是 null，MAE 不计这盘。弃权数仍是 0。
- 506 盘里 B1 能量绝对误差大于 B0 的有 269 盘。中位几乎相同，MAE 差距在尾部。例如 `dish_1562963289`（标签 350 kcal）B1 870.5 / B0 350.0；`dish_1565811139`（标签 902 kcal）B1 226.5 / B0 250.0。没有据此改检索或克重。
- 与 n=50 备份相比，这 50 个 dish_id 都在本池里：B1 的 `n` 与 VLM 原文都相同的 40/50，B0 的 `n` 相同的 37/50。其余按重跑抖动看。

```bash
# 已完成，不要执行
CUDA_VISIBLE_DEVICES=0 /var/tmp/nutri-venv/bin/python -u scripts/eval_n5k_subset.py --n 507 --seed 42 --split rgb_test
```

仍不要做：把这 507 盘写成 709 盘全表；打开体积工具；接入旧回归权重；用 test GT 调阈值、检索或克重；在没有新的文首命令时重跑或改 `vdr/`。弱标签仍只许用 train，本轮没有开始。

## 一句话

W1 基础设施在这台 2×4090D 上已打通：**schema + 骨架 + toy 冒烟 + 真实 FNDDS catalog（5624）+ N5K 元数据/官方 split + 俯拍 RGB + Qwen2.5-VL-7B 单盘冒烟**。  
B1 第二轮先验（菜品形态、未点名的仿肉、100 g 以上的高密度克重、Cheese+Crust 碎片合并）后，同一 50 盘子集已重跑：能量 MAE 为 B0 **98.3**、B1 **88.2**。上一轮是 97.4 / 107.9，硬化前是 98.2 / 112.1。  
有俯拍 RGB 的官方 `rgb_test` 池已跑完：能量 MAE 为 B0 **97.03**、B1 **110.72**（中位 63.51 / 63.57）。50 盘子集仍是 B0 **98.3**、B1 **88.2**，不能代替这 507 盘。  
**缺**：无俯拍 RGB 的官方 test 盘、弱标签全量、试标 30、BC 训练。侧视视频未解压。这 507 盘也不是 709 盘全表。

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
| Qwen2.5-VL-7B | 权重在 `/var/tmp/vdr-models/Qwen2.5-VL-7B-Instruct`（不进 Git）。B0/B1 图像入口已接。单盘冒烟：`results/b0_b1_image_smoke.json`。50 盘基线在 `results/n5k_subset_b0_b1_n50_seed42.json`（能量 MAE：B0 **98.3**，B1 **88.2**）。俯拍池 507 盘在 `results/n5k_rgb_test_overhead_b0_b1.json`（能量 MAE：B0 **97.03**，B1 **110.72**）。跑在 GPU 0；解释器是 `/var/tmp/nutri-venv`，不是没有 torch 的 `.venv`。GPU 1 上原有 llama-server 未动 |
| B1 硬化（CPU，无 VLM） | 同名假设取中位数、不累加；`portion_grid` 默认克重网格接到查表；plain 果蔬查询压低 dip/soup/juice 等并偏向 raw；`CODE_CONF_GAP` 时先二次 `RETRIEVE`（仍是蘸酱/汤类）或确认性 `RE_PORTION` 再 `STOP`；终态 `n` 不含 `water_g`。冒烟：`scripts/b1_harden_smoke.py` |
| B1 第二轮（CPU，无 VLM） | 查询里的 pudding/salad/pizza 等形态词不再落到短食材名；未写 meatless/vegetarian 时，培根这类肉类头不选仿肉。能量密度 ≥300 kcal/100g 且克重超过 100 g 才加连续惩罚，100 g 的一份坚果不动。`Cheese`+`Crust` 在检索前并成一次查询。体积仍关，不接旧回归权重。冒烟仍是 `scripts/b1_harden_smoke.py` |
| 标注指南 v0 | `docs/annotation_guide_v0.md` |
| 方案 MD | 根目录 `PLAN.md`（投稿/执行真源 MD 版） |

## 明确未完成

| 项 | 阻塞 |
|----|------|
| 有俯拍 RGB 的官方 rgb_test（507/709） | **已完成。** 能量 MAE：B0 **97.03**，B1 **110.72**。不要重跑。50 盘基线仍是 B0 **98.3**、B1 **88.2** |
| 无俯拍 RGB 的官方 test 盘 | 不进入这次命令，也不要写成 709 盘全表 |
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
5. 覆盖摘要在 `results/n5k_overhead_rgb.json`，单盘冒烟在 `results/b0_b1_image_smoke.json`。  
6. 有图的官方 test 小子集在 B1 硬化后重跑完毕（2026-10-27 10:02 Asia/Shanghai，GPU 0，约 6 分钟）：seed=42 的 50/507 盘，50/50 schema-valid，弃权 0。能量 MAE：B0 97.4 kcal（中位 60.9），B1 107.9 kcal（中位 48.8，MAPE 74%）。B1 质量 MAE 65.6 g，碳水 MAE 7.8 g。不是全表。  
7. 相对硬化前：B1 能量 MAE 112.1 → 107.9。22/50 盘的 B1 数字变了，其中 16 盘 VLM 原文相同。`dish_1557937335`（标签 47 kcal）五行重复红薯从 550 kcal / 500 g 收到 110 kcal / 100 g。`dish_1563898084`（标签 419 kcal）从 530 kcal 升到 845 kcal，质量仍是 380 g：`Chia Seed Pudding` 绑到了 Chia seeds。B0 有 16/50 盘与上次不同，能量 MAE 98.2 → 97.4，按重跑抖动看。体积工具仍关，没有接入旧回归权重。无俯拍 RGB 的官方 test 盘不在当时的命令里。弱标签仍只许用 train。  
8. 第二轮先验的同一协议已重跑（2026-10-27 10:40 Asia/Shanghai，GPU 0，约 6 分钟，代码 `1d4da37`）：50/50 schema-valid，弃权 0。能量 MAE：B0 98.3 kcal（中位 64.5），B1 88.2 kcal（中位 51.6，MAPE 72%）。B1 脂肪 MAE 5.9 g，蛋白质 MAE 7.4 g，质量 MAE 64.8 g，碳水 MAE 8.1 g。9/50 盘的 B1 数字变了，VLM 原文相同的 2 盘都变好。`dish_1563305513`（标签 606 kcal）Cheese+Crust 并成 stuffed-crust 芝士披萨，1100 kcal / 350 g → 548 kcal / 200 g。`dish_1559849497`（标签 521 kcal）高密度克重从 990 kcal / 300 g 收到 692 kcal / 250 g。`dish_1563898084` 在同一假设列下不再绑 Chia seeds，改为 Pudding, rice（845 → 522 kcal）；这次 VLM 把布丁写成 75 g，最终 495 kcal。B0 有 13/50 盘与上一轮不同，能量 MAE 97.4 → 98.3，按重跑抖动看。  
9. 俯拍 RGB 池已跑完（2026-10-27 11:18–12:15 Asia/Shanghai，GPU 0，约 57 分钟，代码 `1d4da37` / 启动日志 `260c4b0`）：507/507 schema-valid，弃权 0。能量 MAE：B0 97.03（中位 63.51），B1 110.72（中位 63.57，n=506）。无俯拍 RGB 的官方 test 盘没有补进这次。没有用这些数字改检索或克重。

## 50 盘子集重跑（2026-10-27）

协议已跑完。代码为 `9687a8c`，数字留在提交 `a14f6a3`。下面这条命令只作记录，**不要再跑**。

```bash
# 已完成，不要执行
python scripts/eval_n5k_subset.py --n 50 --seed 42 --split rgb_test
```

## 第二轮 50 盘（2026-10-27 10:40）

协议已跑完。代码为 `1d4da37`。能量 MAE：B0 **98.3**，B1 **88.2**。这一对现在只在 `results/n5k_subset_b0_b1_n50_seed42.json` 与 `.jsonl`。默认的 `results/n5k_subset_b0_b1.json` 与 `.jsonl` 已是俯拍池 507 盘。同一次拉代码后，CPU 回归已通过：`b1_harden_smoke`、`validate_schema_examples`、`fndds_retrieve_smoke`、`rule_verify_toy_smoke`、`b1_toy_smoke`（5/5）、`eval_toy_smoke`。

```bash
# 已完成，不要执行
python scripts/eval_n5k_subset.py --n 50 --seed 42 --split rgb_test
```

这一轮在该子集上 B1 能量 MAE 低于 B0。俯拍池 507 盘已经另跑完，能量 MAE 是 B0 97.03、B1 110.72，见文首。体积工具仍关。弱标签仍只许用 train。

协作协议：[`COLLAB_GITHUB.md`](COLLAB_GITHUB.md)。人工清单：[`needs_human.md`](needs_human.md)。看板：[`w1_w4_checklist.md`](w1_w4_checklist.md)。
