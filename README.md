# NutriAgent

`u2-partial-label` 检验一个尚未证实的小问题：可见食材实例的营养求和若系统性小于盘级总量，一个把可观测食材和不可观测残差分开监督的协议，能否在盘级能量与三种宏量上，稳定同时优于整盘回归，并优于把食材标签当成完整监督。残差不当作零，也不摊回可见食材。推理协议是单张俯视 RGB。这不是已经证明的方法，也不是「Nutrition5k 标签不全，因此训一个通用部分标签算法」。

工作发生在 `u2-partial-label`。`main` 保持不动。当前只解锁第一阶段：冻结协议、刻画残差、预先登记按物理餐盘分组的训练 / 验证 / 留出，并写可复核审计报告。不训网络。第二阶段和新算法仍锁定。

执行清单见 [docs/STATUS.md](docs/STATUS.md)。方案索引见 [PLAN.md](PLAN.md)。官方可行性方案（2026-11-13，Asia/Shanghai 定稿）是[《不完整食材标签：官方可行性方案》](https://cursor.com/canvas/store-745bdf38-f3d7-4313-8d66-08e4b6903e93/e21f31e5-d6b2-48a1-85e0-e806bbe3d0d6)。

## `main` 保持清空

`main` 仍是空初始提交 `d43cb084eba7693a45c114dfdf06b258611463f4`（短 `d43cb08`）。本分支的文档和工作不得合入 `main`。

## 历史分支

下列分支只保存历史。它们不是现行方法，也不作为本分支的基线。

| 分支 | 提交 | 内容 |
|---|---|---|
| [`archive/main-before-new-scheme-2026-11-13`](https://github.com/sleeepyin/NutriAgent/tree/archive/main-before-new-scheme-2026-11-13) | `28bd3c9cae9795c822dc0af68f612ec1fd811e15` | 清空 `main` 前的树，含双视角实验 A / [#73](https://github.com/sleeepyin/NutriAgent/pull/73) |
| [`archive/vdr-old-plan-2026-11-13`](https://github.com/sleeepyin/NutriAgent/tree/archive/vdr-old-plan-2026-11-13) | `28bd3c9cae9795c822dc0af68f612ec1fd811e15` | 与上一支相同 |
| [`archive/v4.0`](https://github.com/sleeepyin/NutriAgent/tree/archive/v4.0) | `ce1dc0616002a2f76fcffcc8a3ef87addd8c8358` | 2026-10-26 的区间与尺度方案 |
| [`icdl`](https://github.com/sleeepyin/NutriAgent/tree/icdl) | `6ea6e849b88288b66d76fd063509e9b394daa7b4` | ICDL。第一阶段 1400 对、692 条序列的审计可保留；第二阶段 [#76](https://github.com/sleeepyin/NutriAgent/pull/76) 已停止 |
| [`u1-granularity`](https://github.com/sleeepyin/NutriAgent/tree/u1-granularity) | `a733641f16433bea6593abd35ee96031cd79e49a` | U1 可行性方案（[#77](https://github.com/sleeepyin/NutriAgent/pull/77)）。审计在 [#78](https://github.com/sleeepyin/NutriAgent/pull/78) / `9d7822f9`。可行性未过，未训练 |

不要从这些分支恢复或继续旧方案。U1 的可分离度与 ICDL 的组成差不是本分支的方法。`u1-granularity` 与 [#78](https://github.com/sleeepyin/NutriAgent/pull/78) 只作只读引用。
