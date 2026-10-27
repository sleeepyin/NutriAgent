# W1–W4 执行看板（可勾选）

对齐方案「前 4 周立刻动手」。算力假设：**2×4090D 24GB**。体积工具默认关；不碰旧回归权重。

## W1 基础设施（本轮目标：本机 box 可完成部分）

- [x] 冻结任务 schema：`(n,G,s,tau)` + IR + actions
- [x] 仓库骨架 / README / 依赖清单
- [x] 样例 + `validate_schema_examples.py`
- [x] 数据获取文档 + `data/toy/`
- [x] 标注指南 v0 + `needs_human.md`
- [x] Toy FNDDS 片段 + `fndds_retrieve` / `nutrition_lookup` 可跑
- [x] `eval/metrics.py` 实现 + `scripts/eval_toy_smoke.py`
- [x] B1 RAG-once（无 VLM，给定 hyp）+ `scripts/b1_toy_smoke.py`
- [x] RuleVerify 骨架（≥3 规则）+ `scripts/rule_verify_toy_smoke.py`
- [x] N5K **元数据+官方 split** 已在 box（全量 imagery 仍缺，需大盘）
- [x] 俯拍 RGB 已抽出（3490 张 `rgb.png`，1.35GB；未解侧视视频和深度）。官方 RGB split 只覆盖到 train 2755/4059、test 507/709
- [x] FNDDS 2019–2020 已在 box；`foods.json` catalog 已构建（5624）
- [x] 默认 catalog 优先 `data/fndds/2019-2020/foods.json`（toy 回退保留）
- [x] FNDDS 检索索引冒烟（**真实 catalog** `data/fndds/2019-2020/foods.json`，5624；toy 仅回退）
- [x] B0/B1 图像入口已接 Qwen2.5-VL-7B；单盘官方 test 冒烟通过（`results/b0_b1_image_smoke.json`）

## W2 评测脚本与 B1 冒烟

- [x] N5K 官方 split 读入 + MAE/MAPE 脚手架接通真实标签（`vdr/data/n5k.py`，`scripts/eval_n5k_subset.py`）
- [x] `nutrition_lookup` + `fndds_retrieve` 最小可跑（toy）
- [x] B1 RAG-once 在 toy 冒烟（无 VLM）
- [x] Qwen2.5-VL-7B 在空闲的一张 4090D 上可推理（另一张卡上已有 llama-server，未占用）
- [x] B0/B1 在真实 N5K 小子集跑完（官方 rgb_test 有图 507 盘里 seed=42 抽 50）。第二轮先验后重跑：能量 MAE 为 B0 98.3、B1 88.2（上一轮 97.4 / 107.9，硬化前 98.2 / 112.1）。50 盘数字保留
- [x] 有俯拍 RGB 的官方 rgb_test 全池 B0/B1（507/709）已跑完。能量 MAE：B0 **97.03**、B1 **110.72**（中位 63.51 / 63.57）。稳定文件 `results/n5k_rgb_test_overhead_b0_b1.json`。n=50 基线仍在 `results/n5k_subset_b0_b1_n50_seed42.json`。不是 709 盘全表。体积工具仍关。本轮不做弱标签
- [x] portion_grid 接上查表打分（B1 默认克重网格 + 去重 + 食材检索先验；`scripts/b1_harden_smoke.py`）

## W3 规则老师与弱标签管道

- [x] 实现 `rule_verify` 初版（toy 阈值；**正式阈值仅在 train hold-out 调**）
- [x] 步骤弱标签生成器（仅 train；禁止 test GT）。`scripts/weak_steps_smoke.py` 可在 CPU 上跑。fit 3679 与 hold-out 380 已生成
- [ ] 标注指南定稿前试跑：`docs/annotation_guide_v0.md`
- [ ] **[人工]** 双人试标 **30 盘**（公开 N5K 图额外标签）
- [ ] **[人工]** 计算 Cohen's kappa；目标 **≥ 0.6**，否则修订指南

## W4 金标准启动 + BC 管线

- [ ] 分层抽样计划（简单/混菜/酱汁/高误差）锁定 200–500 盘池
- [ ] **[人工]** 启动双人正式标注（第三人裁决流程写清）
- [x] 控制器特征 `features.py`：Top-k gap + verify one-hot + 步数预算。`build_features` 31 维，不读 `obs_t`
- [x] BC 数据管线：弱标签轨迹 → `(h, a)`（`vdr/controller/bc_data.py` 在内存里组，不另写 jsonl）。`PolicyBC.fit` 目前是 numpy 逻辑回归玩具，不是 MLP
- [ ] 体积工具保持关闭；不做旧回归重训

## 图例

- **[人工]**：必须用户或标注员介入，本 agent 无法在 box 内独自完成。
