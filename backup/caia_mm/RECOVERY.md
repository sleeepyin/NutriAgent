# Automatic recovery playbook (wired into the runner)

| Failure | Automatic response |
|---|---|
| SSH disconnect | `setsid` + watchdog; process stays up |
| Orchestrator crash | Watchdog restarts from `pipeline_state.json` (completed steps skipped) |
| Heartbeat stale > hang_sec | Kill hung process, restart |
| Missing critical asset | E0 records gate; later steps fail soft and are logged |
| Checksum mismatch | Warn and continue; do not hide |
| 3ch/4ch inference fail | Retry 3×; if still fail, E2 marked fail, later steps soft-fail |
| Clean retrain fail / OOM | Keep existing 3ch-clean + 4ch historical; depth-zero ablation remains the fair fallback |
| E1 all gains <5% | Degrade A: conformal triage (auto / acquire / abstain) |
| One action wins >90% | Degrade B: single-action trigger |
| Gain-model R² ≤ 0 | Keep heuristic / triage controller; do not claim learned VOI |
| No LLM API | Parser + heuristic planner + validator; core numbers unchanged |
| No MM-CFA-Interact data | Write SOP, skip Phase 4, stay Level A |
| NV zero-target coverage collapse | Report empirical coverage + recovery curve; **no 90% claim** |
| Locked hash mismatch | E7 `--locked` refuses confirmatory test |
| Too many watchdog restarts | Write `logs/FATAL` and stop |
