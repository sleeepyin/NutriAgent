# 计划

本文件只指向现行方案，不另立贡献主张。

现行小问题是 U1：可分离度感知的自适应营养估计粒度。测试时不看标签，按盘在整盘回归与食材级聚合之间选择，并稳定同时优于两种固定粒度。推理是单张俯视 RGB。

方案真源是 2026-11-13（Asia/Shanghai）定稿的《自适应营养估计粒度：官方可行性方案》：

https://cursor.com/canvas/store-745bdf38-f3d7-4313-8d66-08e4b6903e93/856389db-faaf-4e96-95b3-d9acd92af18d

执行以 [docs/STATUS.md](docs/STATUS.md) 为准。当前只解锁第一阶段数据审计。第二阶段最小对照、新算法和学习路由器保持锁定。

工作分支是 `u1-granularity`，从空 `main`（`d43cb084eba7693a45c114dfdf06b258611463f4`，短 `d43cb08`）起。本分支不合入 `main`。不用 `icdl` 做工作分支。

`archive/v4.0`、`archive/main-before-new-scheme-2026-11-13`、`archive/vdr-old-plan-2026-11-13` 和 `icdl` 保存历史。已停止的区间方案、VDR、Agent 和 ICDL 组成差学习不是本计划的方法。
