# Action confusion matrix / 动作混淆矩阵

Script: `scripts/action_confusion_matrix.py`.
Metrics: `vdr/eval/action_metrics.py`.

Per-class precision, recall, and F1, plus a confusion matrix with the ABSTAIN row marked. Overall accuracy is printed and is not the headline.

## Run

```bash
# Synthetic BC MLP hold-out check (STATUS recalls; ABSTAIN F1 = 0, Acc ≈ 0.9700)
python3 scripts/action_confusion_matrix.py --demo

# Optional text report. Do not point --out at frozen n5k_* or weak_steps jsonl.
python3 scripts/action_confusion_matrix.py --demo --out /tmp/bc_mlp_holdout_confusion_demo.txt

# Offline JSONL: each line needs gold + pred
#   {"action_t":"ABSTAIN","pred_action":"STOP"}
python3 scripts/action_confusion_matrix.py --jsonl path/to/preds.jsonl --out /tmp/cm.txt
```

No GPU. The script does not train and does not read `results/weak_steps/`.

## Expected demo numbers (BC MLP hold-out recalls)

| Class | Recall (hits/support) |
|-------|------------------------|
| RETRIEVE | 1273/1285 |
| RE_PORTION | 917/949 |
| STOP | 356/356 |
| CORRECT_CODE | 142/157 |
| **ABSTAIN** | **0/24** → F1 = **0** |
| RE_DECOMPOSE | 0/0 |

Overall Acc ≈ 0.9700 is **not** the headline. Report per-class PR/F1 and highlight ABSTAIN. This is the open-loop B5 MLP hold-out, not a closed-loop result. No RL.

Off-diagonal placement in `--demo` is a synthetic reconstruction consistent with the STATUS narrative (ABSTAIN→STOP; RE_PORTION↔CORRECT_CODE swaps). The full unpublished confusion matrix may differ slightly on non-ABSTAIN precisions. Column order is the P0 report order (`CORRECT_CODE` second), not `schemas/actions.json` enum order.
