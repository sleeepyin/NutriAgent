# 执行状态（STATUS）

> 供 GitHub 同步：服务器跑完实验后，请在本文件或 `notes/` 周报中更新；执行助手据此继续改代码/方案。  
> **截止日期（文档写入时）**：2026-10-27（Asia/Shanghai）。  
> **可执行的下一步只在本文件。** 聊天不是 runbook。助手改下一步、跑法或方案时，先更新文档并推到 GitHub；服务器上的人或 AI 先 `git pull --ff-only`，再读「下一步」执行。  
> **服务器执行看「下一步（服务器立即执行）」。** 文末的 `--n 50` 命令是已经跑完的记录，不要再执行。

## 下一步（服务器立即执行）

**目标：** 用当前代码（含 `1d4da37` 的 B1 第二轮先验）对官方 `rgb_test` 里有俯拍 RGB、且有营养标签的全部盘子跑 B0 和 B1，备份现有 n=50 结果，再把 MAE、池大小、耗时和 commit 写回本文件并推送。

池子在已记录的覆盖下是 **507/709**。`--n 507` 在 `n` 大于等于池大小时使用整个池（按 `dish_id` 排序，不再抽样）。`scripts/eval_n5k_subset.py` 每次用截断写覆盖 `results/n5k_subset_b0_b1.json` 和 `results/n5k_subset_b0_b1.jsonl`。这两个文件现在是 n=50、seed=42 的基线（能量 MAE：B0 **98.3**，B1 **88.2**），必须先备份。

### 前提

- 仓库根目录。先 `git pull --ff-only`，确认本节能在 `docs/STATUS.md` 里看到。
- `source .venv/bin/activate`。依赖已装在 `.venv`（PyPI 主站不可达时用清华镜像，见 [`SERVER_SETUP.md`](SERVER_SETUP.md)）。
- 俯拍图在仓库内 `data/nutrition5k/imagery/realsense_overhead/<dish_id>/rgb.png`。脚本读的是这条路径，不读 `VDR_N5K_ROOT`。与已经跑通的 50 盘相同。FNDDS 默认用仓库内 `data/fndds/2019-2020/foods.json`；只有本机另有 catalog 时才需要已设好的 `VDR_FNDDS_PATH`。
- Qwen2.5-VL-7B 权重在 `/var/tmp/vdr-models/Qwen2.5-VL-7B-Instruct`（不进 Git）。路径不同时先 `export VDR_VLM_PATH=<目录>`。加载器固定 `device_map="cuda:0"`。
- 用空闲的物理 **GPU 0**。GPU 1 上原有 llama-server，不要停、不要抢。不要设置 `CUDA_VISIBLE_DEVICES` 把本任务绑到 GPU 1。开跑前看一眼 `nvidia-smi`。
- 体积工具保持代码里的默认关闭。不要加载旧回归权重。

### 命令

在仓库根目录整段执行。`set -e` 会在备份断言失败时停住，不会开跑。

```bash
set -euo pipefail
cd "$(git rev-parse --show-toplevel)"
git pull --ff-only
git merge-base --is-ancestor 1d4da37 HEAD
set +u
source .venv/bin/activate
set -u
# 权重不在默认路径时再取消下一行注释
# export VDR_VLM_PATH=/var/tmp/vdr-models/Qwen2.5-VL-7B-Instruct

# 1. 备份 n=50 / seed=42。备份文件已经存在时不覆盖。
test -f results/n5k_subset_b0_b1.json
test -f results/n5k_subset_b0_b1.jsonl
if [[ ! -e results/n5k_subset_b0_b1_n50_seed42.json ]]; then
  cp results/n5k_subset_b0_b1.json results/n5k_subset_b0_b1_n50_seed42.json
fi
if [[ ! -e results/n5k_subset_b0_b1_n50_seed42.jsonl ]]; then
  cp results/n5k_subset_b0_b1.jsonl results/n5k_subset_b0_b1_n50_seed42.jsonl
fi
python - <<'PY'
import json
from pathlib import Path
d = json.loads(Path("results/n5k_subset_b0_b1_n50_seed42.json").read_text())
b0 = d["b0"]["energy_kcal"]["mae"]
b1 = d["b1"]["energy_kcal"]["mae"]
assert d["split"] == "rgb_test" and d["seed"] == 42
assert d["n_requested"] == 50 and d["n_subset"] == 50
assert d["n_pool_with_rgb_and_label"] == 507
assert round(b0, 1) == 98.3 and round(b1, 1) == 88.2, (b0, b1)
n_lines = sum(1 for _ in open("results/n5k_subset_b0_b1_n50_seed42.jsonl"))
assert n_lines == 50, n_lines
print("n50 backup ok", b0, b1)
PY

# 2. 跑有俯拍 RGB 的官方 rgb_test 全池。
#    启动后几秒内确认日志含 pool=507 subset=507 seed=42 split=rgb_test。
#    否则立刻中断。50 盘约 6 分钟，507 盘约 1 小时；GPU 忙时按这个量级再留余量。
echo "CODE $(git rev-parse --short HEAD)"
date -Is
python scripts/eval_n5k_subset.py --n 507 --seed 42 --split rgb_test
date -Is

# 3. 先检查默认 summary，再拷到稳定文件名。不要覆盖第 1 步的 n50 备份。
python - <<'PY'
import json
from pathlib import Path
d = json.loads(Path("results/n5k_subset_b0_b1.json").read_text())
assert d["split"] == "rgb_test" and d["n_requested"] == 507
assert d["n_subset"] == d["n_pool_with_rgb_and_label"] == 507, (
    d["n_subset"], d["n_pool_with_rgb_and_label"]
)
n_lines = sum(1 for _ in open("results/n5k_subset_b0_b1.jsonl"))
assert n_lines == d["n_subset"] == 507, n_lines
base = json.loads(Path("results/n5k_subset_b0_b1_n50_seed42.json").read_text())
assert base["n_subset"] == 50 and round(base["b1"]["energy_kcal"]["mae"], 1) == 88.2
print("pool", d["n_pool_with_rgb_and_label"], "subset", d["n_subset"])
print("ok", d["n_b0_ok"], d["n_b1_ok"], "abstain_b1", d["n_b1_abstain"])
print("b0_energy", d["b0"]["energy_kcal"])
print("b1_energy", d["b1"]["energy_kcal"])
PY
cp results/n5k_subset_b0_b1.json results/n5k_rgb_test_overhead_b0_b1.json
cp results/n5k_subset_b0_b1.jsonl results/n5k_rgb_test_overhead_b0_b1.jsonl
```

看到 `n50 backup ok` 之后才会进入约 1 小时的推理。成功写回并推送之后，不要再贴整段。

日志里没有 `pool=507 subset=507`，或第 3 步断言失败时：停。不要把输出拷去稳定文件名，不要提交。默认文件若已不是 n=50，而备份断言也失败，从 git 恢复基线后再备份（`3d00c00` 与 `9023217` 上的这一对内容相同）：

```bash
git show 3d00c00:results/n5k_subset_b0_b1.json > results/n5k_subset_b0_b1_n50_seed42.json
git show 3d00c00:results/n5k_subset_b0_b1.jsonl > results/n5k_subset_b0_b1_n50_seed42.jsonl
```

然后重新跑上面的备份断言。通过之后再跑第 2 步。中途崩溃时默认 jsonl 是残缺的：保留 n50 备份，不要把它拷到 `results/n5k_rgb_test_overhead_b0_b1.*`。脚本不能断点续跑，修好后从第 2 步整段重跑。

### 跑完后写回并推送

把实测填进下面这段，放在本节末尾（括号换成日志和 JSON 里的原值，不要手算 MAE）：

```markdown
### 俯拍 RGB 池结果（服务器填写）

- 代码：`<git rev-parse --short HEAD>`（祖先含 `1d4da37`）
- 时间：`<date -Is 开始>` → `<date -Is 结束>`，GPU 0
- 首行日志：`pool=507 subset=507 seed=42 split=rgb_test`
- n_b0_ok / n_b1_ok / n_b1_abstain：`<summary>`
- 能量 MAE：B0 `<b0.energy_kcal.mae>`（中位 `<median_ae>`），B1 `<b1.energy_kcal.mae>`（中位 `<median_ae>`）
- 产物：`results/n5k_rgb_test_overhead_b0_b1.json` 与 `.jsonl`。成功跑完后，默认的 `results/n5k_subset_b0_b1.json` 与 `.jsonl` 与这一对相同。n=50 基线只在 `results/n5k_subset_b0_b1_n50_seed42.json` 与 `.jsonl`。
- 范围：官方 rgb_test 中有俯拍 RGB 且有标签的 507 盘。
```

下文已有的 50 盘数字（98.2 / 112.1、97.4 / 107.9、98.3 / 88.2）全部保留。然后：

```bash
git add \
  results/n5k_subset_b0_b1_n50_seed42.json \
  results/n5k_subset_b0_b1_n50_seed42.jsonl \
  results/n5k_subset_b0_b1.json \
  results/n5k_subset_b0_b1.jsonl \
  results/n5k_rgb_test_overhead_b0_b1.json \
  results/n5k_rgb_test_overhead_b0_b1.jsonl \
  docs/STATUS.md
git commit -m "[metric] B0/B1 on rgb_test overhead pool (507)"
git push
```

推送到当前分支。不要 force-push。不要把图像、权重或 `outputs/` 加进去。

### 不要做

- 未备份 `results/n5k_subset_b0_b1.json` 和 `.jsonl` 就运行。脚本总会覆盖这两个路径。
- 把这次结果写成 709 盘官方 test 全表。没有俯拍 RGB 的盘子不在池里，也不要为补齐 709 去改脚本或造图。
- 打开体积工具，或接入旧回归权重。
- 用 test GT 调阈值、检索或克重。数字异常只记在本文件。
- 本轮开始步骤弱标签。弱标签仍只许用 train，另开一轮。
- 为这次评测修改 `vdr/` 里的检索或 portion 逻辑。

## 一句话

W1 基础设施在这台 2×4090D 上已打通：**schema + 骨架 + toy 冒烟 + 真实 FNDDS catalog（5624）+ N5K 元数据/官方 split + 俯拍 RGB + Qwen2.5-VL-7B 单盘冒烟**。  
B1 第二轮先验（菜品形态、未点名的仿肉、100 g 以上的高密度克重、Cheese+Crust 碎片合并）后，同一 50 盘子集已重跑：能量 MAE 为 B0 **98.3**、B1 **88.2**。上一轮是 97.4 / 107.9，硬化前是 98.2 / 112.1。  
**当前要做的**是文首：有俯拍 RGB 的官方 `rgb_test` 池（507/709）。  
**缺**：无俯拍 RGB 的官方 test 盘、弱标签全量、试标 30、BC 训练。侧视视频未解压。50 盘子集数字不能当主表。

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
| Qwen2.5-VL-7B | 权重在 `/var/tmp/vdr-models/Qwen2.5-VL-7B-Instruct`（不进 Git）。B0/B1 图像入口已接。单盘冒烟：`results/b0_b1_image_smoke.json`。50 盘官方 test 子集（`1d4da37` 第二轮后重跑）：`results/n5k_subset_b0_b1.json`。能量 MAE：B0 **98.3**，B1 **88.2**。跑在空闲的 GPU 0；GPU 1 上原有 llama-server 未动 |
| B1 硬化（CPU，无 VLM） | 同名假设取中位数、不累加；`portion_grid` 默认克重网格接到查表；plain 果蔬查询压低 dip/soup/juice 等并偏向 raw；`CODE_CONF_GAP` 时先二次 `RETRIEVE`（仍是蘸酱/汤类）或确认性 `RE_PORTION` 再 `STOP`；终态 `n` 不含 `water_g`。冒烟：`scripts/b1_harden_smoke.py` |
| B1 第二轮（CPU，无 VLM） | 查询里的 pudding/salad/pizza 等形态词不再落到短食材名；未写 meatless/vegetarian 时，培根这类肉类头不选仿肉。能量密度 ≥300 kcal/100g 且克重超过 100 g 才加连续惩罚，100 g 的一份坚果不动。`Cheese`+`Crust` 在检索前并成一次查询。体积仍关，不接旧回归权重。冒烟仍是 `scripts/b1_harden_smoke.py` |
| 标注指南 v0 | `docs/annotation_guide_v0.md` |
| 方案 MD | 根目录 `PLAN.md`（投稿/执行真源 MD 版） |

## 明确未完成

| 项 | 阻塞 |
|----|------|
| 有俯拍 RGB 的官方 rgb_test（507/709） | **已授权，按文首执行。** 先备份 n=50。50 盘能量 MAE：B0 **98.3**，B1 **88.2**（B1 中位绝对误差 **51.6**） |
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
9. **现在执行文首。** 有俯拍 RGB 的官方 rgb_test 池（507）。无俯拍 RGB 的官方 test 盘不补进这次。

## 50 盘子集重跑（2026-10-27）

协议已跑完。代码为 `9687a8c`，数字留在提交 `a14f6a3`。下面这条命令只作记录，**不要再跑**；它会覆盖默认结果文件。当前要跑的是文首 `--n 507`。

```bash
# 已完成，不要执行
python scripts/eval_n5k_subset.py --n 50 --seed 42 --split rgb_test
```

## 第二轮 50 盘（2026-10-27 10:40）

协议已跑完。代码为 `1d4da37`。能量 MAE：B0 **98.3**，B1 **88.2**。结果目前在 `results/n5k_subset_b0_b1.json` 与 `.jsonl`；文首备份后，这一对只留在 `results/n5k_subset_b0_b1_n50_seed42.json` 与 `.jsonl`。同一次拉代码后，CPU 回归已通过：`b1_harden_smoke`、`validate_schema_examples`、`fndds_retrieve_smoke`、`rule_verify_toy_smoke`、`b1_toy_smoke`（5/5）、`eval_toy_smoke`。

```bash
# 已完成，不要执行
python scripts/eval_n5k_subset.py --n 50 --seed 42 --split rgb_test
```

这一轮在该子集上 B1 能量 MAE 低于 B0。下一步是文首的俯拍 RGB 池（507），不是无图的 709 盘表。体积工具仍关。弱标签仍只许用 train。

协作协议：[`COLLAB_GITHUB.md`](COLLAB_GITHUB.md)。人工清单：[`needs_human.md`](needs_human.md)。看板：[`w1_w4_checklist.md`](w1_w4_checklist.md)。
