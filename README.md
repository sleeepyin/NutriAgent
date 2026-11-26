# NutriAgent

`main` 已于 2026-11-13 清空，留给新方案。工作在 `digi-nutri`，不在 `main`。

执行门禁是 [`docs/STATUS.md`](docs/STATUS.md)。官方方案页是[《特权深度与成分蒸馏：官方方案（v3）》](https://cursor.com/canvas/store-745bdf38-f3d7-4313-8d66-08e4b6903e93/d4e76222-c251-49c3-bf8e-10a7f8c2f9be)。[#93](https://github.com/sleeepyin/NutriAgent/pull/93) 的统一教师 `T_multi` 已在官方 507 盘留出上未过 Q0，判定 `q0_fail_stop_wait_triste`，仍未合并。这不是通过。[#84](https://github.com/sleeepyin/NutriAgent/pull/84)、[#86](https://github.com/sleeepyin/NutriAgent/pull/86)、[#88](https://github.com/sleeepyin/NutriAgent/pull/88)、[#91](https://github.com/sleeepyin/NutriAgent/pull/91) 也都未合并：#84 是 v1 Q0 失败，#86 是 `train_gate_fail_stop`，#88 与 #91 是 `q0_fail_stop_wait_triste`。本版 `docs/STATUS.md` squash 合并进 `digi-nutri` 之后，只解锁 v3 第一步：新训同主干的 ConvNeXt-Small B1，以及带残差 Atwater 约束的 `T_multi`。蒸馏、方案 B，以及把 #84、#86、#88、#91、#93 合并进来，仍锁。允许做什么以门禁为准。本版门禁尚未 squash 合并进 `digi-nutri` 时，不写代码，也不从这份尚未合并的状态拉取请求开训。

此前方案与实验结果保存在分支：

- [`archive/main-before-new-scheme-2026-11-13`](https://github.com/sleeepyin/NutriAgent/tree/archive/main-before-new-scheme-2026-11-13)
- [`archive/vdr-old-plan-2026-11-13`](https://github.com/sleeepyin/NutriAgent/tree/archive/vdr-old-plan-2026-11-13)

两支都指向提交 `28bd3c9cae9795c822dc0af68f612ec1fd811e15`（含实验 A / PR #73）。不要从这些历史分支恢复或继续旧方案。
