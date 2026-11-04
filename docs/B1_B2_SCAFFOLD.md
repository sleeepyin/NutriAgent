# B1 / B2 scaffold

Landed: protocol interfaces, configs, and CPU dry-runs. Not landed: a faithful eval.

历史 B1 能量 MAE **110.72**（俯拍池 n=506）**不是忠实 DietAI24**。它来自 `vdr/baselines/b1_rag_once.py` 的硬化 RAG-once。主表在按 DietAI24 官方协议重做之前不能用这个数。

`docs/STATUS.md` is unchanged in this landing. Closed-loop training stays blocked. `CLOSED_LOOP_READY` stays false. Open-loop 31-d BC stays ablation B5. This smoke does not write `results/`.

---

## What is scaffold

| Path | Role |
|------|------|
| `vdr/baselines/dietai24/config.py` | Pinned DietAI24 call shape. Fidelity is `scaffold` only |
| `vdr/baselines/dietai24/interface.py` | Stage protocols and the published `v * grams / 100` sum |
| `vdr/baselines/dietai24/dry_run.py` | Fixture client. No GPT, no embeddings, no catalog read |
| `vdr/baselines/foodagent/config.py` | Fidelity is `approx` only. A fixed-mixture label is rejected |
| `vdr/baselines/foodagent/interface.py` | Per-element decision table and the published mass / unit formulas |
| `vdr/baselines/foodagent/approx.py` | Fixture attributes in, fixture arithmetic out. Experts are not started |
| `vdr/baselines/foodagent/dry_run.py` | Approx protocol entrypoint |
| `scripts/b1_b2_scaffold_smoke.py` | CPU smoke |

```bash
python3 scripts/b1_b2_scaffold_smoke.py
python3 -m vdr.baselines.dietai24.dry_run
python3 -m vdr.baselines.foodagent.dry_run
```

Toy energies printed by those commands are fixture arithmetic. They are not MAE.

---

## B1 — DietAI24 protocol stub

Published pipeline (Yan et al., Communications Medicine 2025; code `Runz96/DietAI24`):

1. Index FNDDS **2019–2020** (5624 descriptions) once with `text-embedding-3-large`.
2. GPT Vision describes the image; a GPT model expands that text into queries.
3. Embed the queries, search the vector store, drop duplicate codes.
4. GPT Vision keeps the codes that are actually present.
5. Choose an FNDDS portion descriptor (or an adjusted amount in FNDDS units) and map it to grams with the portions-and-weights table.
6. Scale each code's per-100 g vector by `grams/100` and sum. The paper reports **65** components. Temperature is 0. Reported MLLMs are GPT-4 Vision-Preview and GPT-4 1106-Preview.

The dry-run walks that order. It dedupes fixture codes, refuses a code that was not retrieved, refuses `portion_grid`, and applies the scaling formula to fixture per-100 g numbers. The public `n` has the five schema fields. The other 60 components are not run (`full_65_not_run: true`). Every live flag is false. `fidelity` is `scaffold`. `main_table_eligible` is false.

This is not an approx table row and not a faithful run. A config that asks for any other fidelity is rejected.

### What the hardened module still is

`vdr/baselines/b1_rag_once.py` is unchanged in behavior. It is Qwen hypotheses (on the image path), lexical FNDDS retrieval, duplicate-hypothesis merge, cheese+crust collapse, `portion_grid` with a density penalty, and one `CODE_CONF_GAP` follow-up. That is the system behind energy MAE **110.72**. Do not cite it as DietAI24.

The paper's ASA24 energy MAE (about 47.7 kcal per dish) is a literature anchor. This repository has not reproduced it.

---

## B2 — FoodAgent, approx only

Published router (Zhang et al., BSN 2025): Qwen2.5-Omni-7B decomposes the image, then **each element** goes to its own expert.

| Element | Expert | This repo |
|---------|--------|-----------|
| Important and visually clear | MFP3D volume (SAM + ZoeDepth), then `mass = volume × density` | Formula only, on fixture volume and density. Volume tool stays off. MFP3D is not called |
| Important but not clear | NutritionX retrieval, then unit count times the reference unit | Formula only. NutritionX is not called |
| Minor, or easy | The model's own numbers | Fixture vector. No LLM call |

The runnable fidelity is **`approx`**. Attributes are supplied by the fixture, not by Omni. A scaffold tie-break, not a rule from the paper's code: `minor` or `easy` wins over volume and retrieval.

There is no dish-level expert. An estimate whose expert disagrees with that element's attributes is rejected. A label that would present this baseline as a fixed mixture is rejected. Do not call this path Fixed-MoE.

---

## Not yet run

A later faithful eval still has to do all of the following. None of it is started here.

**DietAI24.** GPT Vision calls at temperature 0, a real `text-embedding-3-large` index of the 5624 FNDDS 2019–2020 descriptions, the FNDDS portions-and-weights file, all 65 components, and the paper's ASA24 / Nutrition5k subset protocol. Do not fit that run on the frozen 507-dish overhead pool. Do not write it into `results/n5k_*`.

**FoodAgent.** Qwen2.5-Omni-7B as the decomposer and router, MFP3D with SAM and ZoeDepth, and NutritionX. The volume tool stays off until an explicit ablation says otherwise. Until those three exist, keep `fidelity="approx"` and keep the run out of the main table.

Closed-loop training, RL, and `CLOSED_LOOP_READY=true` stay out of scope. B5 open-loop code stays.
