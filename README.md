# NutriAgent

`main` 已于 2026-11-13 清空，留给新方案。工作在 `digi-nutri`，不在 `main`。

执行门禁是 [`docs/STATUS.md`](docs/STATUS.md)。官方方案页是[《特权深度与成分蒸馏：官方方案（v2）》](https://cursor.com/canvas/store-745bdf38-f3d7-4313-8d66-08e4b6903e93/d4e76222-c251-49c3-bf8e-10a7f8c2f9be)。v1 第一阶段已在官方 507 盘留出上停在 Q0。[#86](https://github.com/sleeepyin/NutriAgent/pull/86) 仍是未合并的第一次 v2 训练记录，判定 `train_gate_fail_stop`。[#88](https://github.com/sleeepyin/NutriAgent/pull/88) 仍是未合并的方案 A 官方留出记录，判定 `q0_fail_stop_wait_triste`。两份都不是通过，也不合入 `digi-nutri` 或 `main`。[#84](https://github.com/sleeepyin/NutriAgent/pull/84) 仍是未合并的 v1 Q0 失败记录。当前只解锁同一套新设置下重训 B1、`T_geo` 与 `T_ing`，并给 `T_geo` 加显式体积特征；判定只看 507 盘留出相对这一轮新的同设置 B1。蒸馏、方案 B、统一特权教师 `T_multi`，以及把 #84、#86、#88 合并进来，仍锁。允许做什么以门禁为准。本版门禁尚未 squash 合并进 `digi-nutri` 时，不写代码。

此前方案与实验结果保存在分支：

- [`archive/main-before-new-scheme-2026-11-13`](https://github.com/sleeepyin/NutriAgent/tree/archive/main-before-new-scheme-2026-11-13)
- [`archive/vdr-old-plan-2026-11-13`](https://github.com/sleeepyin/NutriAgent/tree/archive/vdr-old-plan-2026-11-13)

两支都指向提交 `28bd3c9cae9795c822dc0af68f612ec1fd811e15`（含实验 A / PR #73）。不要从这些历史分支恢复或继续旧方案。
