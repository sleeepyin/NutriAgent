# 执行状态（STATUS）

> 供 GitHub 同步：服务器跑完实验后，请在本文件更新。`notes/` 历史周报已删除，见 [`notes/README.md`](../notes/README.md)。执行助手据此继续改代码/方案。  
> **截止日期（文档写入时）**：2026-11-05（Asia/Shanghai）。**Cursor 额度已恢复**；闭环 State-aware BC 训练已落地（#24）；hold-out **FixRate** 主定义评测已落地（#26）。开环 31 维仍只作消融 **B5**。  
> **可执行的下一步只在本文件。** 聊天不是 runbook。助手改下一步、跑法或方案时，先更新文档并推到 GitHub；服务器上的人或 AI 先 `git pull --ff-only`，再读「下一步」执行。  
> **服务器执行看「下一步」。** 这一节没有新的弱标签 / 507 命令。历史 `--n 50`、俯拍池 `--n 507`、弱标签 jsonl、开环 BC 数字都是已完成记录，不要再执行。

## 下一步（立即执行）

**主表 MAE / AURC**（FixRate 已完成 #26）。

> 真源：根目录 [`PLAN.md`](../PLAN.md)、Canvas「VDR — Evidence-Aware Closed-Loop Tool Control」、[`docs/VDR_P0.md`](VDR_P0.md)、[`docs/STATE_FEATURE_DESIGN.md`](STATE_FEATURE_DESIGN.md)。闭环 `π(a_t|h_t,o_{t-1},b_t)` 已实现 + 训练 + hold-out 动作评测（#24）；**FixRate** hold-out 主定义已评测（#26）。`CLOSED_LOOP_READY=true`（见 `vdr/controller/state_features.py`）。Hold-out 动作侧 **ABSTAIN F1=0.5091**（mask none）；H2 mask-zero **ABSTAIN F1=0.1212**；overall Acc **次要**。开环 31 维 `build_features` 仍是消融 **B5**（`PolicyMLP` / `scripts/bc_mlp_smoke.py` 不变）。**不上 RL。不开 fast Cloud Agent。** 冻结结果不重写。

### 服务器

**没有新的 507 / 弱标签重跑命令。** 冻结文件勿覆盖，也不要重跑：

- `results/n5k_rgb_test_overhead_b0_b1.json` 与 `.jsonl`（sha256 前缀 `a4dfdd82e3a595f4…`）
- 默认的 `results/n5k_subset_b0_b1.json` 与 `.jsonl`（现在就是上面这一对）
- n=50 基线 `results/n5k_subset_b0_b1_n50_seed42.json` 与 `.jsonl`
- `results/weak_steps/rgb_train_fit_all.jsonl`（sha256 前缀 `ed47f9239bfecb3c…`）与 `results/weak_steps/rgb_train_holdout_all.jsonl`

不要用这 507 盘的 test GT 调阈值、检索、克重或碎片合并。体积工具仍关。旧回归权重仍不接入。弱标签不改 B1 先验。

历史 B0 能量 MAE **97.03**（n=507）可作 Direct-VLM 锚点。历史 B1 能量 MAE **110.72**（n=506）**不是忠实 DietAI24**，主表须按官方协议重做后再进。

### 助手进度

**闭环 State-aware BC 已落地**（2026-11-04 Asia/Shanghai，自托管 `gpu-12-container`，分支 `cursor/closed-loop-bc-20261104`）。

- 数据：`vdr/controller/closed_loop_data.py`（fit / hold-out 弱步骤；`o_prev` 为上一步 IR；`b_t.tools_used=step_i`）。
- 策略：`vdr/controller/policy_state_mlp.py`（79-d → 64 → 64 → 6，与 B5 容量对齐）。
- 训练/评测：`scripts/closed_loop_bc_train.py`；冒烟 `scripts/closed_loop_smoke.py` + `scripts/bc_mlp_smoke.py`（B5 路径仍 31-d）。
- 产物（**新路径**，未改冻结 weak_steps / n5k）：`results/closed_loop/metrics_holdout.json`、`report_holdout.txt`、`policy_state_mlp.npz`。
- **Full fit** 25864 行 / hold-out 2771 行，CPU ~14s fit。Hold-out（mask none）：overall Acc **0.9607**（次要）；**ABSTAIN F1=0.5091**（14/24）；per-class PR/F1 见 json。**H2 mask-zero**：Acc **0.7037**，**ABSTAIN F1=0.1212**（obs 块有关）。mask-shuffle ABSTAIN F1 **0.0870**。这不是 Full-set MAE / AURC 主表证明。
- `CLOSED_LOOP_READY=true`。B5 开环 MLP hold-out Acc **0.9700** / ABSTAIN F1=0 仍是消融对照，不是主方法。

**FixRate 主定义已评测**（2026-11-05 Asia/Shanghai，#26，squash `7e3084f`）。

- 主定义：`offline_policy_matched_terminal_counterfactual`（`scripts/closed_loop_fixrate.py`；冒烟 `scripts/closed_loop_fixrate_smoke.py`）。
- Hold-out：FixRate **0.7731**，WorsenRate **0.2177**，Δ **0.5554**，95% CI **[0.4818, 0.6280]**（ε=1 kcal，按 dish 聚类 bootstrap B=10000）。
- 工具成本 K=32：mean **6.29** / median **5**；**380/380** 在预算内。
- 产物：`results/closed_loop/fixrate_holdout.json`、`report_fixrate.txt`。
- 说明：不是 Full-set 能量 MAE / Coverage·Selective MAE / AURC 主表；consecutive-IR 诊断不是主指标。

**P0 冒烟 / B1-B2 脚手架**（#19 / #22）仍有效：`CLOSED_LOOP_READY` 以前为 false 的记录保留在历史段。B1 scaffold / B2 approx 仍等 live API。

### 助手 / 本地（优先）

1. **主表 MAE / AURC（立即）：** Full-set 能量 MAE + Coverage / Selective MAE / AURC，对比 Ours（closed-loop）与 B5（+ mask obs 消融）。用 `vdr/eval/selective.py` 与现有脚本；新产物只写在 `results/closed_loop/`。禁止用 test GT 调参。执行归属可记「营养论文代码编写」；本文件 STATUS 仍标 **方案执行**。
2. **稍后 API：** 忠实 DietAI24 / FoodAgent（Omni+MFP3D+NutritionX）。不要把 scaffold 数字写进冻结 `results/n5k_*`。
3. **P2 标注稍后：** kappa **0.7–0.8**；试标 30；指南仍 `docs/annotation_guide_v0.md` v0。
4. **禁止**：单独用 overall Acc 当方法证明；忽视 hold-out **ABSTAIN** per-class PR/F1；**单独用 FixRate 当完整方法证明（须配 MAE/AURC 主表）**；**把 consecutive-IR 诊断当主指标**；上 RL；开 fast Cloud Agent；用 test GT 调参；开体积工具；接旧回归权重；覆盖上面列出的冻结文件；把 **110.72** 当成忠实 DietAI24；把 B2 approx 写成 Fixed-MoE。

### 协作

docs-first：改下一步先改本文件并推仓。Cloud Agent **永远不开 fast**。

## 历史跑数（开环 v2 / 非主方法）

> 以下各段是 2026-10-27 已完成记录，命令都不要再执行。**开环 31 维 BC（线性 0.9206、MLP 0.9700、DAgger r1 0.9592）只是消融 B5**，不读 `o_{t-1}`，不是主方法，也不是方法证明。**旧 Ours-BC 已停。暂不上 RL。** 后文「不要把 `obs_t` 放进特征」是当时开环约束的原话，不是现行指令；现行主方法必须读 `o_{t-1}`。历史 B1 **110.72 不是忠实 DietAI24**。hold-out **ABSTAIN 0/24 是红警**。冻结文件引用保留，勿覆盖。

**n=8 冒烟已跑完**（2026-10-27 16:55–16:56 Asia/Shanghai，自托管 worker `gpu-12-container`，机器 `hostname` 为 `n012`）。CPU，进程里没有 `CUDA_VISIBLE_DEVICES`。解释器是仓库 `.venv` 的 Python 3.12（`jsonschema` 4.26）。`scripts/weak_steps_smoke.py` 通过（约 8 秒）。当时生成的是 fit 池官方顺序的前 8 盘，约 15 秒，61 步。**这次中间 jsonl 与 summary 已删除，不要再执行生成命令。** 弱步骤只保留全池 `results/weak_steps/rgb_train_fit_all` 与 `results/weak_steps/rgb_train_holdout_all`（各自含 `.summary.json`）。

当时 summary（teacher `0.1.0`）的数字仍有效：

- split / pool / n：`rgb_train` / `fit` / 8（无 seed）。有元数据的 fit 池共 **3679** 盘，hold-out **380** 盘。
- 动作：`RETRIEVE` 29、`RE_PORTION` 22、`STOP` 7、`CORRECT_CODE` 2、`ABSTAIN` 1。
- 终态：`corrected` 7、`abstain` 1。
- train 能量审计（pass/corrected 行对照 **train** 盘标签，不是 test 分）：n_scored **7**，`train_energy_mae` **35.0188**。`dish_1565799617` 为 abstain，不进这个 MAE。
- `uses_test_gt` false，`vlm` false，`volume_tool` false。当时 JSONL 61 行均过 `ir_step` schema，没有 `VOLUME_ESTIMATE`。
- 冻结文件未改：`results/n5k_rgb_test_overhead_b0_b1.jsonl` 的 sha256 仍是 `a4dfdd82e3a595f4…`。同目录 json 与 n=50 备份的 sha256 也与跑前一致。

**n=256 已跑完**（2026-10-27 17:00:09–17:07:05 Asia/Shanghai，同一台 `gpu-12-container` / `hostname` `n012`）。CPU，进程里没有 `CUDA_VISIBLE_DEVICES`。解释器仍是仓库 `.venv`。墙钟 **416** 秒（约 7 分钟），1809 步。含上面那 8 盘。**这次中间 jsonl 与 summary 已删除。** 不要按前缀盘数重跑。保留文件仍只有上面两份全池。

当时 summary（teacher `0.1.0`）的数字仍有效：

- split / pool / n：`rgb_train` / `fit` / 256（无 seed）。
- 动作：`RETRIEVE` 850、`RE_PORTION` 611、`STOP` 238、`CORRECT_CODE` 92、`ABSTAIN` 18。
- 终态：`corrected` 216、`pass` 22、`abstain` 18。
- train 能量审计（pass/corrected 对照 **train** 标签，不是 test 分）：n_scored **238**，`train_energy_mae` **25.3065**。18 盘 abstain 不进这个 MAE。
- `uses_test_gt` false，`vlm` false，`volume_tool` false。当时 JSONL 1809 行均过 `ir_step` schema，没有 `VOLUME_ESTIMATE`。
- 冻结文件未改：`results/n5k_rgb_test_overhead_b0_b1.jsonl` 的 sha256 仍是 `a4dfdd82e3a595f4…`。同目录 json 与 n=50 备份的 sha256 也与跑前一致。冻结能量 MAE 仍是 B0 **97.03** / B1 **110.72**（B0 可作 Direct-VLM 锚点；B1 **不是忠实 DietAI24**）。

**n=1024 已跑完**（2026-10-27 17:09:21–17:36:23 Asia/Shanghai，同一台 `gpu-12-container` / `hostname` `n012`）。CPU，进程里没有 `CUDA_VISIBLE_DEVICES`。解释器仍是仓库 `.venv`。墙钟 **1622** 秒（约 27 分钟），7128 步。前 256 个 dish_id 与 n=256 那次相同。**这次中间 jsonl 与 summary 已删除。** 不要按前缀盘数重跑。保留文件仍只有上面两份全池。

当时 summary（teacher `0.1.0`）的数字仍有效：

- split / pool / n：`rgb_train` / `fit` / 1024（无 seed）。
- 动作：`RETRIEVE` 3315、`RE_PORTION` 2396、`STOP` 955、`CORRECT_CODE` 393、`ABSTAIN` 69。
- 终态：`corrected` 859、`pass` 96、`abstain` 69。弃权比例 69/1024，与 n=256 的 18/256 接近。
- train 能量审计（pass/corrected 对照 **train** 标签，不是 test 分）：n_scored **955**，`train_energy_mae` **29.7984**。n=256 同一审计是 **25.3065**（n_scored 238）。前缀变长后 MAE 仍在三十上下，没有崩。69 盘 abstain 不进这个 MAE。
- `uses_test_gt` false，`vlm` false，`volume_tool` false。当时 JSONL 7128 行均过 `ir_step` schema，没有 `VOLUME_ESTIMATE`。
- 冻结文件未改。冻结能量 MAE 仍是 B0 **97.03** / B1 **110.72**（B0 可作 Direct-VLM 锚点；B1 **不是忠实 DietAI24**）。`results/n5k_rgb_test_overhead_b0_b1.jsonl` 的 sha256 仍是 `a4dfdd82e3a595f4…`。

**fit 全量已跑完**（2026-10-27 17:38:33–19:17:23 Asia/Shanghai，同一台 `gpu-12-container` / `hostname` `n012`）。CPU，进程里没有 `CUDA_VISIBLE_DEVICES`。解释器仍是仓库 `.venv`。墙钟 **5930** 秒（约 1 小时 39 分钟），25864 步。前 1024 个 dish_id 与 n=1024 那次相同（那次中间 jsonl 已删除）。命令已完成，不要再执行：

```bash
# 已完成，不要再执行
python scripts/gen_weak_steps_train.py --split rgb_train --pool fit --out results/weak_steps/rgb_train_fit_all.jsonl
```

summary（`results/weak_steps/rgb_train_fit_all.summary.json`，teacher `0.1.0`）：

- split / pool / n：`rgb_train` / `fit` / 全池 **3679**（`n_requested` 为空，无 seed）。
- 动作：`RETRIEVE` 12063、`RE_PORTION` 8654、`STOP` 3443、`CORRECT_CODE` 1468、`ABSTAIN` 236。
- 终态：`corrected` 3108、`pass` 335、`abstain` 236。弃权比例 236/3679，与 n=1024 的 69/1024 接近。
- train 能量审计（pass/corrected 对照 **train** 标签，不是 test 分）：n_scored **3443**，`train_energy_mae` **29.5844**。n=1024 同一审计是 **29.7984**（n_scored 955）。236 盘 abstain 不进这个 MAE。
- `uses_test_gt` false，`vlm` false，`volume_tool` false。JSONL 25864 行均过 `ir_step` schema，没有 `VOLUME_ESTIMATE`。
- 产物：`results/weak_steps/rgb_train_fit_all.jsonl`（25864 行，28,877,178 字节，约 27.5MB）与同名 `.summary.json`（约 1.6MB）。
- 速率约 **1.61 秒/盘**。冻结文件未改。冻结能量 MAE 仍是 B0 **97.03** / B1 **110.72**（B0 可作 Direct-VLM 锚点；B1 **不是忠实 DietAI24**）。`results/n5k_rgb_test_overhead_b0_b1.jsonl` 的 sha256 仍是 `a4dfdd82e3a595f4…`。

**hold-out 已跑完**（2026-10-27 19:20:14–19:30:45 Asia/Shanghai，同一台 `gpu-12-container` / `hostname` `n012`）。CPU，进程里没有 `CUDA_VISIBLE_DEVICES`。解释器仍是仓库 `.venv`。墙钟 **631** 秒，2771 步。380 个 dish_id 与 fit 全池不相交，并集是官方 `rgb_train` 的 **4059** 盘。命令已完成，不要再执行：

```bash
# 已完成，不要再执行
python scripts/gen_weak_steps_train.py --split rgb_train --pool holdout --out results/weak_steps/rgb_train_holdout_all.jsonl
```

summary（`results/weak_steps/rgb_train_holdout_all.summary.json`，teacher `0.1.0`）：

- split / pool / n：`rgb_train` / `holdout` / 全池 **380**（`n_requested` 为空，无 seed）。
- 动作：`RETRIEVE` 1285、`RE_PORTION` 949、`STOP` 356、`CORRECT_CODE` 157、`ABSTAIN` 24。
- 终态：`corrected` 328、`pass` 28、`abstain` 24。弃权比例 24/380，与 fit 全池的 236/3679 接近。
- train 能量审计（pass/corrected 对照 **train** 标签，不是 test 分）：n_scored **356**，`train_energy_mae` **30.7085**。fit 全池同一审计是 **29.5844**（n_scored 3443）。24 盘 abstain 不进这个 MAE。
- `uses_test_gt` false，`vlm` false，`volume_tool` false。JSONL 2771 行均过 `ir_step` schema，没有 `VOLUME_ESTIMATE`。
- 产物：`results/weak_steps/rgb_train_holdout_all.jsonl`（2771 行，3,105,355 字节，约 3.0MB）与同名 `.summary.json`（约 179KB）。
- 冻结文件未改。`results/weak_steps/rgb_train_fit_all.jsonl` 的 sha256 仍是 `ed47f9239bfecb3c…`，summary 仍是 `d11e6a3e547b38d0…`。冻结能量 MAE 仍是 B0 **97.03** / B1 **110.72**（B0 可作 Direct-VLM 锚点；B1 **不是忠实 DietAI24**）。`results/n5k_rgb_test_overhead_b0_b1.jsonl` 的 sha256 仍是 `a4dfdd82e3a595f4…`。

**train 弱步骤标签已经齐**：fit 3679 盘 + hold-out 380 盘。不要重跑这两份 jsonl，不要跑 rgb_test，不要开体积工具，不要用 507 盘的 test GT 改阈值、特征或这一节的准确率。

**BC 特征已落地（助手，CPU，无 GPU；开环 v2 / 消融 B5，非主方法）。** `vdr/controller/features.py` 的 `build_features` 只读动作前字段 `hyp_i`、`code_i`、`cands_i`、`portion_g_i`、`verify_codes`、`step_i`。标签是 `action_t`。`obs_t` 不进特征：`CORRECT_CODE` 和 `RE_PORTION` 的老师选定码或克重留在 `obs_t`。`vdr/controller/bc_data.py` 从 fit jsonl 组 `(h, a)`，动作准确率只在 hold-out jsonl 上算。`PolicyBC.fit` 不再是 `NotImplementedError`，但是 **numpy 多项逻辑回归**（80 步、学习率 2、L2 `1e-4`，零初始化，固定常数，没有在 hold-out 上搜）。这是线性基线，不是 Transformer，不回传 7B，不开体积工具。小 MLP 在下一段。动作头 6 个：`RETRIEVE`、`RE_PORTION`、`RE_DECOMPOSE`、`ABSTAIN`、`STOP`、`CORRECT_CODE`。`VOLUME_ESTIMATE` 不在头里。老师没有发出 `RE_DECOMPOSE`，这一列样本数是 0。

特征 **31** 维：假设是否出现、字数和词数；码是否出现、是不是 Top-1、在不在候选里、名次；候选条数、Top-1/Top-2 分数、分数差、均分；克重是否出现、克重/1500；9 个 `verify_codes` 一位有效码加一位未知码；规则触发数和 `1-Top1` 作为无 GT 的误差代理；`step_i` 相对固定步数上限 32 的比例、剩余比例、是否到顶。这些尺度没有在 hold-out 或 rgb_test 上拟合。

冒烟已在本机跑完（CPU，约 1 秒，不写结果文件，不读 rgb_test）。命令已完成，不要把它当成新的服务器任务：

```bash
# 已完成，不要再执行
python scripts/bc_features_smoke.py
```

- fit：3679 盘、25864 步。hold-out：380 盘、2771 步。两份 dish_id 不相交。
- hold-out 动作准确率 **0.9206**（2551/2771）。fit 集多数类 `RETRIEVE` 拿到 hold-out 上是 **0.4637**（1285/2771）。训练集上的拟合准确率是 0.9194（23780/25864），这不是要报的数。
- hold-out recall：`RETRIEVE` 1246/1285，`RE_PORTION` 949/949，`STOP` 356/356，`CORRECT_CODE` 0/157，`ABSTAIN` 0/24。后两类和二次 `RETRIEVE` 的差别在 `obs_t` 里，线性玩具把它们收成多数类。**ABSTAIN 0/24 是红警。** 当时写过「不要为了抬这两类把 `obs_t` 放进特征」；那是开环 v2 的约束，不是现行指令。31 维路径只作消融 B5。现行主方法必须读 `o_{t-1}`，见文首。
- 冻结文件未改。冻结能量 MAE 仍是 B0 **97.03** / B1 **110.72**（B0 可作 Direct-VLM 锚点；B1 **不是忠实 DietAI24**）。

**小 MLP BC 已落地（助手，CPU，无 GPU；开环 v2 / 消融 B5，非主方法）。** `PolicyMLP`（`vdr/controller/policy_mlp.py`）是 31→64→64→6，ReLU，dropout 0，L2 `1e-4`，Adam 学习率 `1e-2`（β1 0.9，β2 0.999），batch 512，80 epoch，种子 0。隐藏层 He 初始化，输出层用 `1/64` 的方差。类别权重是 fit 标签的逆平方根频数，再缩放到平均样本权重为 1：`RETRIEVE` 0.7471、`RE_PORTION` 0.8820、`STOP` 1.3984、`CORRECT_CODE` 2.1416、`ABSTAIN` 5.3412、`RE_DECOMPOSE` 0（fit 里没有这一类，预测时掩掉）。这些常数没有在 hold-out 或 rgb_test 上搜。`PolicyBC` 的超参没改。两边都只吃 `build_features`。不读 `obs_t`。没有 `VOLUME_ESTIMATE`。不回传 7B。不开体积工具。

冒烟已在本机跑完（CPU，数秒，只打 stdout，不写结果文件，不读 rgb_test）。命令已完成，不要把它当成新的服务器任务：

```bash
# 已完成，不要再执行
python scripts/bc_mlp_smoke.py
```

- fit：3679 盘、25864 步。hold-out：380 盘、2771 步。两份 dish_id 不相交。
- 线性 hold-out 动作准确率仍是 **0.9206**（2551/2771）。MLP hold-out 动作准确率 **0.9700**（2688/2771）。fit 集上 MLP 的拟合准确率是 0.9660（24984/25864），这不是要报的数。
- hold-out recall（命中/该类行数）：`RETRIEVE` 1273/1285（线性 1246/1285），`RE_PORTION` 917/949（线性 949/949），`STOP` 356/356（线性 356/356），`CORRECT_CODE` 142/157（线性 0/157），`ABSTAIN` 0/24（线性 0/24）。`RE_DECOMPOSE` 两边都是 0 行。
- 相对线性，MLP 把 142 个 `CORRECT_CODE` 从 `RE_PORTION` 里收回来，同时把 31 个 `RE_PORTION` 判成 `CORRECT_CODE`、1 个 `RE_PORTION` 判成 `RETRIEVE`，另有 27 个线性判成 `RE_PORTION` 的 `RETRIEVE` 被改对。净增 137 步。`RETRIEVE` / `RE_PORTION` / `STOP` 没有塌。
- `ABSTAIN` 仍然是 0。fit 上 236 行里 235 行和 `STOP` 的 31 维向量在四舍五入到 1e-6 后相同：老师写的都是空的 `hyp_i` / `code_i` / `cands_i` / `portion_g_i`，差别在 `obs_t` 和 `abstain_reason`，这两项不进特征。MLP 把 hold-out 的 24 个弃权步全部判成 `STOP`，和线性一样。**ABSTAIN 0/24 是红警**，overall Acc（含 **0.9700**）不能当方法证明。当时写过「不要为了抬 `ABSTAIN` 把 `obs_t` 放进特征」；那是开环 v2 的约束，不是现行指令。
- 弱步骤 summary 仍是 `uses_test_gt` false，`vlm` false，`volume_tool` false。冻结文件未改。冻结能量 MAE 仍是 B0 **97.03** / B1 **110.72**（B0 可作 Direct-VLM 锚点；B1 **不是忠实 DietAI24**）。

**DAgger 第 1 轮已落地（助手，CPU，无 GPU；开环 v2。旧 Ours-BC 已停，暂不上 RL）。** 专家是现有的 train-only 规则老师（`vdr/weak/teacher.py`，`0.1.0`）。学习者是 `PolicyMLP`，超参与上一节相同（31→64→64→6，dropout 0，L2 `1e-4`，Adam 学习率 `1e-2`，batch 512，80 epoch，种子 0），没有在 hold-out 或 rgb_test 上搜。先在 `results/weak_steps/rgb_train_fit_all.jsonl` 上拟合纯 BC MLP，再只在 fit 池上展开。环境走老师同一套工具：FNDDS 检索、盲 100 g 克重网格、`choose_code`、按老师规则的 `RE_PORTION` 和终态。学习者的动作驱动转移，标签始终是该状态上老师的 `action_t`。特征仍只来自 `build_features` 的动作前字段。`obs_t` 不进特征。没有 `VOLUME_ESTIMATE`。不开体积工具。不读 rgb_test / depth_test。不改 fit / hold-out jsonl，不改 507 文件。不需要 7B。

fit 行上预测已经和老师一致的盘不再展开：确定性的学习者和老师每步都一致时，会留在老师轨迹上，没有新状态。展开的盘里，访问到的状态在两种情况下追加进聚合：31 维特征键（四舍五入到 1e-6）不在 fit 矩阵里，或者学习者动作 ≠ 老师动作。已经在 BC 里、而且学习者判对的状态不重复追加。步数上限 48，连续 3 步除 `step_i` 外公开字段不变就停。这两个上限是固定护栏，不是 hold-out 搜出来的。老师轨迹最多 32 步。train 食材克重和 kcal 只留在老师查询里，用来决定 `a*` 和内部记账；下一步的 `portion_g_i` 不写入老师贴上去的克重。

冒烟不写结果文件。全池这一轮已在助手 CPU 上跑完，墙钟 **535** 秒（含前 8 盘专家轨迹与 `generate_dish_steps` 的对照，以及纯 BC 与聚合后各一次 MLP 拟合）。命令已完成，不要把它当成新的服务器任务，也不要设 `CUDA_VISIBLE_DEVICES`：

```bash
# 已完成，不要再执行
python scripts/dagger_smoke.py
python scripts/dagger_smoke.py --n 32
python scripts/dagger_smoke.py --all-fit
```

全池（这一轮要报的数）：

- 盘：fit **3679**。其中 **2913** 盘与老师一致，跳过。展开 **766** 盘，访问 **6061** 步，学习者动作 ≠ 老师 **1401** 步。
- 新行 **1490**（聚合 25864+1490=**27354**）。特征键是新的 **117** 行；其余 **1373** 行是学习者犯错的状态，特征键已经在 BC 里，再记一份老师标签。其中 **4** 行的旧标签集合里没有这次的老师动作。
- 新行的动作：`RETRIEVE` 145、`RE_PORTION` 945、`CORRECT_CODE` 201、`ABSTAIN` 196、`STOP` 3。没有 `RE_DECOMPOSE`，没有 `VOLUME_ESTIMATE`。
- hold-out 动作准确率：纯 BC MLP **0.9700**（2688/2771）→ DAgger **0.9592**（2658/2771）。线性基线仍是 **0.9206**（2551/2771）。少了 30 步。
- hold-out recall（命中/该类行数）：`RETRIEVE` 1285/1285（纯 BC 1273/1285），`RE_PORTION` 900/949（纯 BC 917/949），`STOP` 356/356，`CORRECT_CODE` 117/157（纯 BC 142/157），`ABSTAIN` 仍是 0/24。`RE_DECOMPOSE` 仍是 0 行。
- 边界往 `RETRIEVE` 和再记进去的 `RE_PORTION` 靠，`CORRECT_CODE` 少了 25 个命中。这不是改规则的依据。不要为了把准确率拉回去改特征、改 MLP 常数，或在 hold-out 上挑聚合方式。
- `ABSTAIN` 新行有 196 条，hold-out 仍然 **0/24（红警）**。动作前特征和 `STOP` 撞在一起，差别在 `obs_t` 和 `abstain_reason`。当时写过「不要为了抬 `ABSTAIN` 把 `obs_t` 放进特征」；那是开环 v2 的约束，不是现行指令。旧 Ours-BC 已停，不要为了把这轮准确率拉回去再开一轮。

同一条代码的前缀（不是另一套超参）：

- `--n 8`：展开 3 盘，新行 5（全是犯错副本，novel 0）。hold-out **2677/2771**。`CORRECT_CODE` 145/157，`RE_PORTION` 896/949，`RETRIEVE` 1280/1285，`ABSTAIN` 0/24。
- `--n 32`：展开 8 盘，新行 11（novel 1 + 副本 10）。hold-out **2671/2771**。`CORRECT_CODE` 148/157，`RE_PORTION` 889/949，`RETRIEVE` 1278/1285，`ABSTAIN` 0/24。

`uses_test_gt` false，`vlm` false，`volume_tool` false。冻结文件未改。`results/weak_steps/rgb_train_fit_all.jsonl` 的 sha256 仍是 `ed47f9239bfecb3c…`，`results/n5k_rgb_test_overhead_b0_b1.jsonl` 仍是 `a4dfdd82e3a595f4…`。冻结能量 MAE 仍是 B0 **97.03** / B1 **110.72**（B0 可作 Direct-VLM 锚点；B1 **不是忠实 DietAI24**）。

**DAgger 第 1 轮已落地（开环 v2，到此停止）。** 全池 hold-out 动作准确率 **0.9592**（2658/2771），纯 BC MLP 是 **0.9700**（2688/2771），线性基线 **0.9206**（2551/2771）。新行 **1490**。**没有新的服务器命令。** **旧 Ours-BC 已停。暂不上 RL。** 不要再开 DAgger 轮，也不要把 0.97 Acc 当方法证明；hold-out **ABSTAIN 0/24 是红警**。冻结能量 MAE 仍是 B0 **97.03** / B1 **110.72**（B0 可作 Direct-VLM 锚点；B1 **不是忠实 DietAI24**）。体积工具仍关。不要为了再跑一轮开 GPU，也不要重跑 `--n 507`。

### 俯拍 RGB 池结果（2026-10-27）

> 开环 v2 冻结跑数，文件引用保留，勿覆盖，勿重跑。B0 **97.03** 可作 Direct-VLM 锚点。B1 **110.72** **不是忠实 DietAI24**，主表须官方协议重做后再进。

- 代码：推理进程启动时日志为 `CODE 260c4b0`。`vdr/` 与 `scripts/` 相对 `1d4da37` 无差异；`9594f57` 只改了文档。祖先含 `1d4da37`。
- 时间：`2026-10-27T11:18:02+08:00` → `2026-10-27T12:14:52+08:00`（约 57 分钟），物理 GPU 0。GPU 1 上的 llama-server 未停。
- 解释器：仓库 `.venv` 没有 torch。按文首旧命令 `source .venv` 会每盘 `ModuleNotFoundError`，并立刻覆盖默认 json/jsonl。那次空跑没有拷到稳定文件名，默认文件已从 n=50 备份恢复后再跑。实际命令是 `CUDA_VISIBLE_DEVICES=0 /var/tmp/nutri-venv/bin/python -u scripts/eval_n5k_subset.py --n 507 --seed 42 --split rgb_test`。`CUDA_VISIBLE_DEVICES=0` 让加载器的 `cuda:0` 落在物理 GPU 0，没有绑到 GPU 1。
- 首行日志：`pool=507 subset=507 seed=42 split=rgb_test`
- n_b0_ok / n_b1_ok / n_b1_abstain：`507` / `507` / `0`
- 能量 MAE（summary 原值）：B0 **97.03**（中位 **63.51**，MAPE 65.17%，n=507），B1 **110.72**（中位 **63.57**，MAPE 71.61%，n=506；**不是忠实 DietAI24**）
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

仍不要做：把这 507 盘写成 709 盘全表；打开体积工具；接入旧回归权重；用 test GT 调阈值、检索、克重或碎片合并；重跑或覆盖上面的 507 文件。弱标签只许用 train。生成命令在「历史跑数」里，都已跑完，不要在这 507 盘上跑，也不要重跑。

### 507 分析（助手）

只读 `results/n5k_rgb_test_overhead_b0_b1.jsonl`（与默认 `results/n5k_subset_b0_b1.jsonl` 的 sha256 同为 `a4dfdd82e3a595f4…`，492261 字节）。没有改结果文件。配对明细就在本节。当时的周报已删除，不要再到 `notes/` 找那份列表。

`dish_1557861216`（标签 0 kcal）B1 返回空列表，`n` 全是 null。能量和宏量的胜负只在 **506** 盘上比。summary 里 B0 能量 MAE **97.03** 含这盘（绝对误差 0，n=507）。同一 506 盘上 B0 能量 MAE 是 **97.22**。下面的差都用这 506 盘。

**能量。** 配对 MAE：B0 **97.22**、B1 **110.72**，差 **+13.50** kcal。中位绝对误差 63.59 / 63.57。绝对误差之差（B1−B0）的中位数只有 **+3.5** kcal。B1 更好 / 更差 / 相同：**233 / 269 / 4**。平均有符号误差（预测−标签）B1 **+23.2** kcal，B0 **−3.9**。右尾更重：正差距之和 23242 kcal，负差距之和 −16412，净差 6830。差距最大的 10 盘占净差的 56%，20 盘占 94%；去掉差距最大的 25 盘后，平均差距从 +13.5 变为 **−1.6**。两边绝对误差都截到 150 kcal 时，MAE 差只剩 **+1.8**。两端各去掉 10% 的修剪均值仍是 B0 79.7、B1 88.6。中位持平；MAE 差距由右尾抬起，单独几盘解释不完。

**宏量与质量（同一 506 盘）。** 碳水 MAE B1 **9.30**、B0 **10.41**（更好 279 / 更差 224 / 平 3）。蛋白质 B1 **8.90**、B0 **8.81**（261 / 245）。脂肪 B1 **8.84**、B0 **6.65**，中位绝对误差 5.25 / 3.69（221 / 284 / 1）；脂肪的差距不是几盘尾部。质量 MAE B1 **74.85**、B0 **76.60**，中位绝对误差反而 52.5 / 48.5，有符号误差 B1 **+17.8 g**、B0 **−5.9 g**。能量变差不是质量 MAE 变差。

**尾部假设（不是改规则的依据）。** jsonl 只有 VLM 假设原文和最终 `n`，没有检索命中的 FNDDS 食物名。能量差距 >100 kcal 的有 **82** 盘（B1 高估超过 100 kcal 的 63 盘，低估超过 100 kcal 的 19 盘）。按互斥顺序：

1. **少计整盘（19）**。`dish_1563551220`：标签 562 kcal / 558 g，B1 148 / 210 g，B0 600。`dish_1563551194`：标签 472，B1 138，B0 500。
2. **披萨碎片，或旁边再加一整个披萨（18）**。`dish_1562963289`：标签 350，B1 870.5（Pasta Salad 150 g + Cheese Pizza 200 g），B0 350。`dish_1560203400`：标签 293，B1 756.5（Pepperoni 100 g + Cheese 50 g + Crust 25 g，三条都留下），B0 250。两条假设的 Cheese+Crust 合并仍然有效：`dish_1563305513` 标签 606，B1 548，B0 300。三条及以上现有代码不合并。这里只记录，不据此改合并。
3. **培根或香肠的克重（13）**。`dish_1564675380`：标签 87 kcal / 23 g，B1 494 / 100 g（假设只有 Bacon 100 g），B0 150。`dish_1559578579`：标签 143，B1 512（bacon 100 g + strawberry 50 g），B0 150。
4. **质量接近、能量密度偏高（10）**。`dish_1564588859`：标签 184 kcal / 257 g，B1 492 / 255 g（Feta + peach + chia pudding），B0 200。
5. **预测质量明显偏大（6）**。`dish_1566501575`：标签 252 kcal / 54 g，B1 764 / 420 g。
6. **其余 16**，形态类似，没跨过上面的切分。`dish_1559233445`：标签 24.5 kcal，B1 186（假设 Butter 25 g），B0 10。

`CODE_CONF_GAP` 在 506 盘里出现 447 次；B1 更差的 269 盘里 245 次，B1 更好的 233 盘里 199 次。这个码分不开胜负。

**和 n=50。** n=50 文件没有算错：能量 MAE B0 **98.29**、B1 **88.21**，B1 更好 29 盘、更差 21 盘。这 50 个 id 都在本池里。用 507 这次的预测重算这 50 盘，仍是 B1 **85.75**、B0 **98.22**（更好 27 / 更差 23）。B1 的营养向量和 VLM 原文与 n=50 文件相同的是 40/50，B0 营养向量相同的是 37/50。重跑抖动没有把子集结论翻过来。其余 **456** 盘：MAE B0 **97.11**、B1 **113.45**（更好 206 / 更差 246 / 平 4）。从 506 盘随机抽 50 盘、重复 20000 次（种子 0）：MAE 差（B1−B0）均值 **+13.5**、标准差 **15.3**；B1 的 MAE 更低的抽样占 **18.9%**。n=50 文件的差 **−10.1** 落在约 **5.9%** 分位。这 50 盘里只有 2 个落在全池差距最大的 50 盘，4 个落在 B1 更好最多的 50 盘。子集优势是真实的，也是偏有利的一抽。
## 一句话

W1 基础设施在这台 2×4090D 上已打通：**schema + 骨架 + toy 冒烟 + 真实 FNDDS catalog（5624）+ N5K 元数据/官方 split + 俯拍 RGB + Qwen2.5-VL-7B 单盘冒烟**。  
B1 第二轮先验（菜品形态、未点名的仿肉、100 g 以上的高密度克重、Cheese+Crust 碎片合并）后，同一 50 盘子集已重跑：能量 MAE 为 B0 **98.3**、B1 **88.2**。上一轮是 97.4 / 107.9，硬化前是 98.2 / 112.1。  
有俯拍 RGB 的官方 `rgb_test` 池已跑完：能量 MAE 为 B0 **97.03**（Direct-VLM 锚点）、B1 **110.72**（**不是忠实 DietAI24**；中位 63.51 / 63.57）。配对 506 盘上中位差距约 +3.5 kcal，MAE 差距在右尾。50 盘子集仍是 B0 **98.3**、B1 **88.2**，是偏有利的一抽，不能代替这 507 盘。见「507 分析（助手）」。  
**现行缺口**在文首：读 `o_{t-1}` 的 State-aware 闭环**训练**仍未做（P0 草图和离线脚本已进仓库，见 [`VDR_P0.md`](VDR_P0.md)；`CLOSED_LOOP_READY` 为假）。忠实 DietAI24 与 FoodAgent（或诚实 approx）、试标 30（kappa **0.7–0.8**）仍缺。开环 DAgger 第 1 轮的数字留作 B5：全池 hold-out **0.9592**，纯 BC MLP **0.9700**，新行 1490；`ABSTAIN` **0/24 是红警**。**旧 Ours-BC 已停。暂不上 RL。** 无俯拍 RGB 的官方 test 盘仍不进命令。侧视视频未解压。这 507 盘也不是 709 盘全表。

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
| 步骤弱标签老师（CPU，无 VLM） | `vdr/weak/teacher.py` + `scripts/gen_weak_steps_train.py`。只读 `rgb_train`。fit 全池在 `results/weak_steps/rgb_train_fit_all.jsonl`（3679 盘、25864 步、5930 秒；train 能量审计 MAE **29.5844**）。hold-out 在 `rgb_train_holdout_all.jsonl`（380 盘、2771 步、631 秒；train 能量审计 MAE **30.7085**）。两份 dish_id 不相交，并集 4059。n=8 / n=256 / n=1024 的中间 jsonl **已删除** |
| BC 特征 + 线性基线（CPU） | **开环 v2 / 消融 B5，非主方法。** `build_features` **31** 维，只读动作前字段，不读 `obs_t`。`load_bc_rows` 从 fit jsonl 组 `(h, a)`。`PolicyBC.fit` 是 numpy 多项逻辑回归。hold-out 动作准确率 **0.9206**（2551/2771）；多数类基线 0.4637。`CORRECT_CODE` 与 `ABSTAIN` 的 recall 是 0（**0/24 是红警**）。冒烟：`scripts/bc_features_smoke.py` |
| BC 小 MLP（CPU） | **开环 v2 / 消融 B5。** `PolicyMLP` 31→64→64→6，numpy，只在 fit 行上训练。hold-out 动作准确率 **0.9700**（2688/2771），不能当方法证明。`CORRECT_CODE` 142/157，`RE_PORTION` 917/949，`ABSTAIN` 仍是 **0/24（红警）**。不读 `obs_t`，没有 `VOLUME_ESTIMATE`。冒烟：`scripts/bc_mlp_smoke.py` |
| DAgger 第 1 轮（CPU） | **开环 v2。旧 Ours-BC 已停，暂不上 RL。** 规则老师当专家，`PolicyMLP` 当学习者。只在 fit 池展开。全池新行 **1490**（novel 117 + 犯错副本 1373）。hold-out 动作准确率 **0.9592**（2658/2771），低于纯 BC **0.9700**。`CORRECT_CODE` 117/157，`RETRIEVE` 1285/1285，`ABSTAIN` 仍是 **0/24（红警）**。不读 `obs_t`，没有 `VOLUME_ESTIMATE`。冒烟：`scripts/dagger_smoke.py` |
| 标注指南 v0 | `docs/annotation_guide_v0.md`。文首横幅：字段是 v0，相对 PLAN P2（Visibility / Code Match / …，kappa 0.7–0.8）部分过时；试标可用，不是最终协议 |
| 方案 MD | 根目录 `PLAN.md`（2026-11-04 起与 Canvas「VDR — Evidence-Aware Closed-Loop Tool Control」对齐） |
| P0 离线脚手架（未训练） | 设计 `docs/STATE_FEATURE_DESIGN.md`；混淆 / Provenance / Selective 脚本在 `scripts/`；mock `tests/fixtures/fndds_mock_foods.json`；草图 `vdr/controller/state_features.py`（`CLOSED_LOOP_READY` 假）。开环 31 维仍是 B5。**2026-11-04 离线冒烟已跑完**（数字在文首「助手进度」；不是方法证明，不是训练）。索引 [`docs/VDR_P0.md`](VDR_P0.md) |

## 明确未完成

| 项 | 阻塞 |
|----|------|
| 有俯拍 RGB 的官方 rgb_test（507/709） | **已完成（历史锚点）。** 能量 MAE：B0 **97.03**（Direct-VLM），B1 **110.72**（**非忠实 DietAI24**，主表须重做）。不要重跑。50 盘基线仍是 B0 **98.3**、B1 **88.2** |
| 无俯拍 RGB 的官方 test 盘 | 不进入这次命令，也不要写成 709 盘全表 |
| 步骤弱标签全量（仅 train） | **已完成。** fit 3679 + hold-out 380。没有新的生成命令。阈值仍只许在 hold-out 上看，禁止用 test GT |
| 试标 30 + 新标签 / kappa 0.7–0.8 | **人工，P2。** 指南仍是 v0，文首已标明相对 PLAN P2 部分过时。正式协议仍要扩到 Visibility / Code Match 等。不作 0.6 凑合 |
| 200–500 金标准 | **人工**，可分期；协议与 kappa 门禁不能省 |
| 旧开环 Ours-BC / RL | **已停 / 暂不上。** 31 维开环只作消融 B5：线性 **0.9206**，MLP **0.9700**，DAgger r1 **0.9592**（新行 1490）。`ABSTAIN` **0/24 是红警**。没有新的服务器训练命令。P0 脚本与未训练草图已进仓库（文首）；闭环训练仍要等额度 |
| 体积工具 | **保持默认关闭**（刻意不做） |
| 旧回归权重 | **禁止接入** |

## 建议服务器本周顺序

> 2026-10-27 的完成记录，不是现行任务单。下一步在文首。这里没有新命令。

1. 按 [`SERVER_SETUP.md`](SERVER_SETUP.md) clone、装依赖、挂数据。已完成（依赖装在 `.venv`，PyPI 主站不可达，用了清华镜像）。  
2. 优先下 **overhead RGB**。已从本机 tar 抽出 `rgb.png`，未解侧视视频。  
3. 冒烟已通过：`validate_schema` / `fndds_retrieve_smoke`（5624）/ `n5k_metadata_summary` / `rule_verify_toy_smoke` / `b1_toy_smoke`（5/5）/ `eval_toy_smoke`。  
4. Qwen2.5-VL-7B 已装上，B0/B1 图像入口和单盘冒烟已通过。  
5. 覆盖摘要在 `results/n5k_overhead_rgb.json`，单盘冒烟在 `results/b0_b1_image_smoke.json`。  
6. 有图的官方 test 小子集在 B1 硬化后重跑完毕（2026-10-27 10:02 Asia/Shanghai，GPU 0，约 6 分钟）：seed=42 的 50/507 盘，50/50 schema-valid，弃权 0。能量 MAE：B0 97.4 kcal（中位 60.9），B1 107.9 kcal（中位 48.8，MAPE 74%）。B1 质量 MAE 65.6 g，碳水 MAE 7.8 g。不是全表。  
7. 相对硬化前：B1 能量 MAE 112.1 → 107.9。22/50 盘的 B1 数字变了，其中 16 盘 VLM 原文相同。`dish_1557937335`（标签 47 kcal）五行重复红薯从 550 kcal / 500 g 收到 110 kcal / 100 g。`dish_1563898084`（标签 419 kcal）从 530 kcal 升到 845 kcal，质量仍是 380 g：`Chia Seed Pudding` 绑到了 Chia seeds。B0 有 16/50 盘与上次不同，能量 MAE 98.2 → 97.4，按重跑抖动看。体积工具仍关，没有接入旧回归权重。无俯拍 RGB 的官方 test 盘不在当时的命令里。弱标签仍只许用 train。  
8. 第二轮先验的同一协议已重跑（2026-10-27 10:40 Asia/Shanghai，GPU 0，约 6 分钟，代码 `1d4da37`）：50/50 schema-valid，弃权 0。能量 MAE：B0 98.3 kcal（中位 64.5），B1 88.2 kcal（中位 51.6，MAPE 72%）。B1 脂肪 MAE 5.9 g，蛋白质 MAE 7.4 g，质量 MAE 64.8 g，碳水 MAE 8.1 g。9/50 盘的 B1 数字变了，VLM 原文相同的 2 盘都变好。`dish_1563305513`（标签 606 kcal）Cheese+Crust 并成 stuffed-crust 芝士披萨，1100 kcal / 350 g → 548 kcal / 200 g。`dish_1559849497`（标签 521 kcal）高密度克重从 990 kcal / 300 g 收到 692 kcal / 250 g。`dish_1563898084` 在同一假设列下不再绑 Chia seeds，改为 Pudding, rice（845 → 522 kcal）；这次 VLM 把布丁写成 75 g，最终 495 kcal。B0 有 13/50 盘与上一轮不同，能量 MAE 97.4 → 98.3，按重跑抖动看。  
9. 俯拍 RGB 池已跑完（2026-10-27 11:18–12:15 Asia/Shanghai，GPU 0，约 57 分钟，代码 `1d4da37` / 启动日志 `260c4b0`）：507/507 schema-valid，弃权 0。能量 MAE：B0 97.03（中位 63.51），B1 110.72（中位 63.57，n=506；**不是忠实 DietAI24**）。无俯拍 RGB 的官方 test 盘没有补进这次。没有用这些数字改检索或克重。
10. 助手已把 507 盘的配对分析写入「507 分析（助手）」。当时的周报已删除，以该节为准。没有用这些数字改 B1。弱标签老师已落地。`gpu-12-container` 已跑完 fit 全池（2026-10-27 19:17）和 hold-out 全池（2026-10-27 19:30，631 秒，380 盘）。BC 特征与 numpy 逻辑回归基线已落地，hold-out 动作准确率 0.9206。小 MLP BC 已在助手侧用 CPU 跑完，hold-out 动作准确率 0.9700。DAgger 第 1 轮也已在助手 CPU 上跑完（全池 535 秒，新行 1490，hold-out 动作准确率 0.9592）。没有新的服务器命令。这几项是开环 v2：31 维只作消融 B5，`ABSTAIN` 0/24 是红警。**旧 Ours-BC 已停。暂不上 RL。** 下一步见文首，不是 507，也不是再开一轮 DAgger 或 RL。

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

这一轮在该子集上 B1 能量 MAE 低于 B0。俯拍池 507 盘已经另跑完，能量 MAE 是 B0 97.03、B1 110.72（B1 **不是忠实 DietAI24**）。体积工具仍关。弱标签仍只许用 train。现行下一步见文首。

协作协议：[`COLLAB_GITHUB.md`](COLLAB_GITHUB.md)。人工清单：[`needs_human.md`](needs_human.md)。W1–W4 看板已删除。进度以本文件、[`PLAN.md`](../PLAN.md) 与 [`VDR_P0.md`](VDR_P0.md) 为准。
