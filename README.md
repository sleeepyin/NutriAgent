# NutriAgent

`main` 已于 2026-11-13 清空，留给新方案。工作在 `digi-nutri`，不在 `main`。

执行门禁是 [`docs/STATUS.md`](docs/STATUS.md)。官方方案页是[《特权深度与成分蒸馏：官方方案（v2）》](https://cursor.com/canvas/store-745bdf38-f3d7-4313-8d66-08e4b6903e93/d4e76222-c251-49c3-bf8e-10a7f8c2f9be)。v1 第一阶段已在官方 507 盘留出上停在 Q0。新版 Q0 第一次训练停在训练集门禁，判定为 `train_gate_fail_stop`，记在开放且未合并的 [#86](https://github.com/sleeepyin/NutriAgent/pull/86)。那不是 Q0 通过，也不合入 `digi-nutri` 或 `main`。当前只解锁 `D_table` 零值排查，以及按方案 A 重训几何教师 `T_geo`。新版 B1 与成分教师 `T_ing` 冻结，不重训。物理乘积蒸馏、方案 B，以及训练集门禁通过之前的 507 盘留出，仍锁。允许做什么以门禁为准。本版门禁尚未 squash 合并进 `digi-nutri` 时，不写代码。

此前方案与实验结果保存在分支：

- [`archive/main-before-new-scheme-2026-11-13`](https://github.com/sleeepyin/NutriAgent/tree/archive/main-before-new-scheme-2026-11-13)
- [`archive/vdr-old-plan-2026-11-13`](https://github.com/sleeepyin/NutriAgent/tree/archive/vdr-old-plan-2026-11-13)

两支都指向提交 `28bd3c9cae9795c822dc0af68f612ec1fd811e15`（含实验 A / PR #73）。不要从这些历史分支恢复或继续旧方案。
