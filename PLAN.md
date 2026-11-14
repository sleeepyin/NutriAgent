# 计划

本文件只指向现行方案，不另立贡献主张，也不把尚未跑完的对照写成已经成立的方法。

现行小问题是 U2：视觉可观测的不完整食材标签。要检验的是：可见食材实例的营养求和系统性小于盘级总量时，一个把可观测食材和不可观测残差分开监督的协议，能否在盘级能量与三种宏量上，稳定同时优于整盘回归，并优于把食材标签当成完整的食材级监督。残差不当作零，也不摊回可见食材。推理是单张俯视 RGB。这不是「Nutrition5k 标签不全，因此训一个通用部分标签算法」。

方案真源是 2026-11-13（Asia/Shanghai）定稿的《不完整食材标签：官方可行性方案》：

https://cursor.com/canvas/store-745bdf38-f3d7-4313-8d66-08e4b6903e93/e21f31e5-d6b2-48a1-85e0-e806bbe3d0d6

执行以 [docs/STATUS.md](docs/STATUS.md) 为准。第一阶段已经在 `u2-partial-label` 的 `d330531bd266b553a5a78c3a3f379c668e594b2f`（短 `d330531b`，[#80](https://github.com/sleeepyin/NutriAgent/pull/80)）完成。当前只解锁第二阶段 B1 / B2 / B3 最小对照：三个同骨干、同预算的模型，不写新网络。新算法保持锁定。`main` 保持锁定。画布页首仍是定稿时只准备解锁第一阶段的句子；允许做什么以 STATUS 为准。

工作分支是 `u2-partial-label`，从空 `main`（`d43cb084eba7693a45c114dfdf06b258611463f4`，短 `d43cb08`）起。本分支不合入 `main`。不用 `icdl`，也不用 `u1-granularity` 做基线。U1 的可分离度与 ICDL 的组成差不是本计划的方法。

`archive/v4.0`、`archive/main-before-new-scheme-2026-11-13`、`archive/vdr-old-plan-2026-11-13`、`icdl` 和 `u1-granularity` 保存历史。已停止的路线不是本计划的起点。
