# Food-id semantic audit

Read-only audit of the #51 same-head FNDDS swaps in `results/food_id_density/` (squash `6c8fda02`).

Regenerate:

```bash
python scripts/food_id_semantic_audit.py --self-check
python scripts/food_id_semantic_audit.py
```

Outputs in this folder:

- `report.md` — Exact / Near / Wrong, protein / carb / fat / energy MAE, and the calorie-versus-identity count.
- `human_review_components.csv`, `human_review_dishes.csv`, `human_review.md` — the 37-dish review pack.
- `summary.json` — the same numbers in machine-readable form.

Automatic labels are suggestions only. Every swapped component row is marked for human review. This folder does not unlock gates ③–⑦, P1, or a Mass / Q / τ retrain.
