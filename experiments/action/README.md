# 行动层

在仓库根目录运行：

```bash
python -m experiments.action.a1_group_routing
python -m experiments.action.a2_name_prior --skip-query
python -m experiments.action.a5_oracle_bound --skip-query
```

| 文件 | 状态 |
|---|---|
| `a1_group_routing.py` | A1，已完成。结果在 `results/action/a1_group_routing/` |
| `a2_name_prior.py` | A2a，已完成。A2b 用 VLM 名称替换标注类别名，脚本还没写 |
| `a5_oracle_bound.py` | A5 里的逐样本 oracle 上界，已完成 |
| `a3_auto_reference/` | A3，还没写 |
| `a4_ood_reject/` | A4，还没写 |
| `a5_mixed_flow/` | A5 端到端，还没写 |

A2b、A3、A4、A5 都只读 `cache/vlm/`，不各自再问一次模型。
