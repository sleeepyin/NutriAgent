# CAIA-MM agent extension

Unattended implementation of `Agent创新点_详细完整实施方案.md` (v1.2) on **Level A public data** (Protocol A).

The myopic value-of-information policy did not beat the best fixed action. Group-conditional freeze (`experiments/e13_group_freeze.py`) only switches the action between datasets. `experiments/e14_llm_prior.py` asks the local Qwen3.8-27B server for a name-only mass prior and shrinks it toward the leave-one-out category median. On MetaFood3D the frozen rule lowers held-out WAPE and MAE and raises MAPE. On ECUSTFD the coin still wins. Write-ups: `docs/实验结果/Agent_千问类别先验实验.md` and section 16 of `docs/论文重构方案_完整版.md`. The paper text is unchanged. The API key is not stored in this repository.

## What the autonomous runner does

1. Phase 0: literature matrix (no first-claim), preregistration + SHA256, unit tests
2. E0: asset/checksum audit, group manifests (N5K cache has **no dish id** — recorded)
3. Background: clamp-free clean 3ch/4ch retrain (`e01`, optional, does not block)
4. E2–E10: action table, tool-value gates, myopic VOI, terminal + all-state conformal, locked test, cross-domain recovery, LLM-off ablation, figures
5. Phase 3/4: write Interact SOP (physical collection is **not** executed)
6. Phase 8: paper integration notes filled only with real artifacts

If E1 gates fail, the runner **degrades** to conformal triage or a single-action trigger and continues.

## Start / monitor / stop

```bash
export MM_CFA_ROOT=/home/mm-cfa-repro
bash /home/mm-cfa-repro/run_caia_mm_autonome.sh
bash /home/mm-cfa-repro/run_caia_mm_autonome.sh --status
bash /home/mm-cfa-repro/run_caia_mm_autonome.sh --stop
```

Logs: `agent_extension/logs/`  
Results: `agent_extension/results/`  
Final: `agent_extension/results/FINAL_REPORT.md`

## Environment

Uses conda env `nutri` (`/opt/miniconda3/envs/nutri/bin/python`) when present.

Optional:

- `CAIA_SKIP_E01=1` — do not launch clean retrain
- `CAIA_E01_EPOCHS=100` — retrain length
- `CAIA_FRESH=1` — ignore previous COMPLETE/state
- `CAIA_HANG_SEC=1200` — watchdog stale-heartbeat threshold
