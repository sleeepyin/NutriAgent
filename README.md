# NutriAgent

`u1-granularity` 检验一个小问题：测试时不看标签，也不用食材真值个数或真值掩膜，按盘估计视觉可分离度，在整盘回归和食材级聚合之间选择，并稳定同时优于两种固定粒度。推理协议是单张俯视 RGB。平均意义上「分割更细是否更好」已由 Hand 等人在 1056 张 Nutrition5k 俯视 RGB 上回答：标准预处理下整盘平均 PMAE 19.91%；食材级减去整盘的配对差，95% 区间为 [1.51, 5.98] 个百分点，p=0.001。本分支要检验的是按盘选择粒度能否同时胜过两种固定粒度。

工作发生在 `u1-granularity`。`main` 保持不动。当前只解锁第一阶段数据审计：对齐公开标注、预先登记不泄漏真值的可分离度、写清与 `archive/v4.0` 的差别。不训练网络。第二阶段、新算法和学习路由器仍锁定。

执行清单见 [docs/STATUS.md](docs/STATUS.md)。方案索引见 [PLAN.md](PLAN.md)。官方可行性方案（2026-11-13，Asia/Shanghai 定稿）是[《自适应营养估计粒度：官方可行性方案》](https://cursor.com/canvas/store-745bdf38-f3d7-4313-8d66-08e4b6903e93/856389db-faaf-4e96-95b3-d9acd92af18d)。

## `main` 保持清空

`main` 已于 2026-11-13 清空，留给新方案。它仍是空初始提交 `d43cb084eba7693a45c114dfdf06b258611463f4`（短 `d43cb08`）。本分支的文档和工作不得合入 `main`。

## 历史分支

下列分支只保存已经停下的工作，不是现行方法，也不作为本分支的基线。

| 分支 | 提交 | 内容 |
|---|---|---|
| [`archive/main-before-new-scheme-2026-11-13`](https://github.com/sleeepyin/NutriAgent/tree/archive/main-before-new-scheme-2026-11-13) | `28bd3c9cae9795c822dc0af68f612ec1fd811e15` | 清空 `main` 前的树，含双视角实验 A / [#73](https://github.com/sleeepyin/NutriAgent/pull/73) |
| [`archive/vdr-old-plan-2026-11-13`](https://github.com/sleeepyin/NutriAgent/tree/archive/vdr-old-plan-2026-11-13) | `28bd3c9cae9795c822dc0af68f612ec1fd811e15` | 与上一支相同 |
| [`archive/v4.0`](https://github.com/sleeepyin/NutriAgent/tree/archive/v4.0) | `ce1dc0616002a2f76fcffcc8a3ef87addd8c8358` | 2026-10-26 的区间与尺度方案 |
| [`icdl`](https://github.com/sleeepyin/NutriAgent/tree/icdl) | `6ea6e849b88288b66d76fd063509e9b394daa7b4` | 已停止的 ICDL 线，含 [#76](https://github.com/sleeepyin/NutriAgent/pull/76) |

两支 `archive/main-before-new-scheme-2026-11-13` 与 `archive/vdr-old-plan-2026-11-13` 都指向 `28bd3c9`。不要从这些分支恢复或继续旧方案。`icdl` 上的组成差学习已经停止，不作为本分支的方法。
