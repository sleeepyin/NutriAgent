# GitHub 协作协议（内网服务器 ↔ 仓库 ↔ 执行助手）

服务器在内网，执行助手**不能 SSH**。约定用 **GitHub 同步代码、方案与小结果** 作为唯一协作面。

## 1. 仓库用途

| 可进库 | 禁止进库 |
|--------|----------|
| 代码、脚本、schema、文档（`PLAN.md` 等） | Nutrition5K **原始图像/视频**、深度图 |
| 小结果：指标 JSON、表 Markdown、日志**摘要** | 完整 FNDDS/FDC **大 zip / 66MB raw JSON** |
| 标注规范、dish_id 列表、人工标签 JSONL | 未脱敏隐私图；旧回归 `.pt` 权重 |
| `notes/` 周报 | `outputs/` 下大体积中间文件（默认 gitignore） |

建议仓库设为 **private**（论文未投前）。

## 2. 目录约定

```
PLAN.md                 # 方案真源（MD）
README.md               # 首页
docs/                   # STATUS / 协作 / 服务器 / 标注 / schema…
results/                # 可同步的小产物（指标、表、配置快照）
notes/                  # 周报、决策记录（YYYY-MM-DD-*.md）
schemas/ examples/ scripts/ vdr/
data/
  toy/                  # 可进库
  nutrition5k/
    dish_ids/ metadata/ # 可进库（小）
    imagery/            # 禁止 push（gitignore）
  fndds/
    toy/foods.json
    2019-2020/foods.json   # ~4MB，建议进库
    2019-2020/*FDC*66MB*   # 禁止 push
```

环境变量（服务器本机路径，**不要**把绝对路径写死进代码）：

```bash
export VDR_N5K_ROOT=/path/on/server/nutrition5k   # 含 imagery/ metadata/ dish_ids/
export VDR_FNDDS_PATH=/path/on/server/fndds/2019-2020/foods.json
```

## 3. Issue / PR 标题约定

| 前缀 | 含义 |
|------|------|
| `[W1]` … `[W16]` | 对应周程 |
| `[data]` | 数据就绪/路径变更 |
| `[smoke]` | 冒烟结果 |
| `[metric]` | 主表数字更新 |
| `[ann]` | 标注/kappa |
| `[ask]` | 需要助手改代码或改方案 |

示例：`[W3] weak labels ready`、`[metric] M1 draft energy MAE`、`[ask] B1 batch script OOM`。

## 4. 用户侧流程（服务器）

1. 在服务器跑实验 / 下数据（数据本身不进 Git）。  
2. 把**可共享**产物写入 `results/`（例如 `results/2026-10-27_b1_smoke.json`、`results/tables/m1_draft.md`）。  
3. 重要结论写一行到 `notes/YYYY-MM-DD.md`。  
4. 更新 `docs/STATUS.md` 中「服务器进度」小节（可自增）。  
5. `git add` → `commit` → `push`。  
6. 在聊天里：**贴 PR/commit 链接**，或说「看 `main` 最新 / 看某 PR」。

## 5. 助手侧流程

1. 通过 GitHub（或用户粘贴的 diff/文件）读取 `results/`、`notes/`、`STATUS.md`。  
2. 改代码、补脚本、修订 `PLAN.md` / 标注指南；**不**假设能进内网。  
3. 需要用户跑的命令写进 `docs/SERVER_SETUP.md` 或 `notes/` 的「请服务器执行」段落。当前评测命令在 [`STATUS.md`](STATUS.md) 文首「下一步（服务器立即执行）」：先备份 n=50 的 `results/n5k_subset_b0_b1.json` 与 `.jsonl`，再对官方 `rgb_test` 有俯拍 RGB 的池（507）跑 B0/B1。  
4. 不创建/代替用户开仓库；仓库 URL 由用户发来后继续协作。

## 6. 结果文件最小规范

`results/*.json` 建议字段：

```json
{
  "run_id": "2026-10-27_b1_n5k_smoke",
  "git_commit": "<hash>",
  "split": "rgb_test",
  "n_samples": 0,
  "metrics": {"energy_mae": null, "evidence_acc": null},
  "notes": "optional"
}
```

大日志只上传**尾部错误**或 `results/*.log.summary.txt`（建议 < 256KB）。

## 7. 冲突与安全

- 不要 force-push `main`（除非两人约定）。  
- 密钥、HuggingFace token、服务器地址不要进库；用环境变量或 GitHub Secrets（若用 CI）。  
- 遵守 N5K/FNDDS 许可；二次分发原图前再核协议。
