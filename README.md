# NutriAgent

`main` 已于 2026-11-13 清空，留给新方案。工作在 `digi-nutri`，不在 `main`。`main` 保持空初始提交 `d43cb084eba7693a45c114dfdf06b258611463f4`（短 `d43cb08`）。

执行门禁是 [`docs/STATUS.md`](docs/STATUS.md)。官方方案是[《特权深度与成分蒸馏：官方方案（v4）》](https://cursor.com/canvas/store-745bdf38-f3d7-4313-8d66-08e4b6903e93/d4e76222-c251-49c3-bf8e-10a7f8c2f9be)。[#95](https://github.com/sleeepyin/NutriAgent/pull/95) 的 v3 第一步已在官方 507 盘留出上未过 Q0，判定 `q0_fail_stop_wait_triste`，仍未合并。这不是通过。相对门禁 12/12 过了，当时的绝对门禁没过。对照是本轮同主干 ConvNeXt-Small B1，不是冻结的 #91 B1。[#93](https://github.com/sleeepyin/NutriAgent/pull/93)、[#91](https://github.com/sleeepyin/NutriAgent/pull/91)、[#88](https://github.com/sleeepyin/NutriAgent/pull/88)、[#86](https://github.com/sleeepyin/NutriAgent/pull/86)、[#84](https://github.com/sleeepyin/NutriAgent/pull/84) 也都未合并，都是失败记录，不是通过。本版 `docs/STATUS.md` squash 合并进 `digi-nutri` 之后，只解锁 DPF-Nutrition 与 Swin-Nutrition 的 2754/507 同划分重训复现。新教师训练、蒸馏、方案 B，以及把 #95、#93、#91、#88、#86、#84 合并进来，仍锁。允许做什么以门禁为准。本版门禁尚未 squash 合并进 `digi-nutri` 时，不写代码，也不从这份尚未合并的状态拉取请求开训。

此前方案与实验结果保存在分支：

- [`archive/main-before-new-scheme-2026-11-13`](https://github.com/sleeepyin/NutriAgent/tree/archive/main-before-new-scheme-2026-11-13)
- [`archive/vdr-old-plan-2026-11-13`](https://github.com/sleeepyin/NutriAgent/tree/archive/vdr-old-plan-2026-11-13)

两支都指向提交 `28bd3c9cae9795c822dc0af68f612ec1fd811e15`（含实验 A / PR #73）。不要从这些历史分支恢复或继续旧方案。
