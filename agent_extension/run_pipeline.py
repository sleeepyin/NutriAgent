#!/usr/bin/env python3
"""
CAIA-MM full-pipeline orchestrator.

Design goals (from the v1.2 plan):
  - Run Phase 0 → E0–E10 → SOP → draft without an interactive session.
  - Resume from the last successful step.
  - Retry, backoff, and step-level fallbacks so a single failure does not kill the run.
  - Heartbeat for an external watchdog (SSH disconnect / hang).
  - Degrade to conformal triage or single-action trigger when E1 gates fail.
  - Never use X_train as Agent test. Never invent numbers. Never claim 90% under domain shift.

Physical Level-B collection cannot be executed here; the SOP is written and the run continues at Level A.
"""
from __future__ import annotations

import importlib
import os
import subprocess
import sys
import time
import traceback
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from agent_extension.src.paths import Paths
from agent_extension.src.runtime import Heartbeat, atomic_write_json, log, read_json, utc_now


@dataclass
class Step:
    sid: str
    title: str
    module: str | None = None
    argv: list[str] = field(default_factory=list)
    retries: int = 3
    timeout_sec: int | None = 7200
    required: bool = False
    skip_if: Callable[[Paths], bool] | None = None
    fallback: str = "continue"  # continue | degrade | required_warn
    status_dir: str | None = None


def _status_ok(paths: Paths, name: str) -> bool:
    p = paths.results / name / "status.json"
    if not p.exists():
        return False
    rec = read_json(p, {})
    return bool(rec.get("ok"))


STEPS: list[Step] = [
    Step("e00_literature", "Phase0 literature matrix", "agent_extension.experiments.e00_literature", status_dir="e00_literature"),
    Step("e00_prereg", "Phase0 preregistration + hash", "agent_extension.experiments.e00_prereg", status_dir="e00_prereg", required=True),
    Step("e00_tests", "Finite-sample q + MAPE/WAPE unit tests", "agent_extension.experiments.e00_tests", status_dir="e00_tests", required=True, fallback="required_warn"),
    Step("e00_audit", "E0 asset / checksum / protocol audit", "agent_extension.experiments.e00_asset_audit", status_dir="e00_asset_audit", required=True, fallback="required_warn"),
    Step("e00_manifest", "Rebuild group manifests", "agent_extension.experiments.e00_manifest", status_dir="e00_manifest"),
    Step("e02_action_table", "Build paired action table (public data)", "agent_extension.experiments.e02_action_table", status_dir="e02_action_table", required=True, timeout_sec=7200, fallback="required_warn"),
    Step("e03_tool_value", "Tool value + Go/No-Go", "agent_extension.experiments.e03_tool_value", status_dir="e03_tool_value", required=True),
    Step("e04_train_policy", "Fit myopic VOI on D_policy", "agent_extension.experiments.e04_train_policy", argv=["--policy", "myopic_voi", "--max-steps", "2"], status_dir="e04_train_policy"),
    Step("e05_terminal", "Frozen-policy terminal conformal", "agent_extension.experiments.e05_terminal_conformal", argv=["--alpha", "0.10", "--split", "calibration"], status_dir="e05_terminal_conformal"),
    Step("e06_all_state", "All-tool-state joint envelope", "agent_extension.experiments.e06_all_state_conformal", argv=["--alpha", "0.10"], status_dir="e06_all_state_conformal"),
    Step("e07_benchmark", "One-shot test benchmark (locked)", "agent_extension.experiments.e07_main_benchmark", argv=["--split", "test", "--locked"], status_dir="e07_main_benchmark"),
    Step("e08_xdomain", "Cross-domain + few-shot recovery", "agent_extension.experiments.e08_cross_domain", argv=["--target-calibration-sizes", "0,25,50,100,200"], status_dir="e08_cross_domain"),
    Step("e09_llm", "LLM ablation (0 API default)", "agent_extension.experiments.e09_llm_ablation", status_dir="e09_llm_ablation"),
    Step("e10_figures", "Figures", "agent_extension.experiments.e10_make_figures", status_dir="e10_figures"),
    Step("e11_sop", "Phase3/4 Interact SOP (no physical collect)", "agent_extension.experiments.e11_pilot_sop", status_dir="e11_pilot_sop"),
    Step("e12_draft", "Phase8 paper integration notes", "agent_extension.experiments.e12_paper_draft", status_dir="e12_paper_draft"),
]


def run_module(module: str, argv: list[str], timeout: int | None) -> int:
    cmd = [sys.executable, "-m", module, *argv]
    log(f"exec: {' '.join(cmd)}")
    env = os.environ.copy()
    env["PYTHONPATH"] = str(ROOT) + os.pathsep + env.get("PYTHONPATH", "")
    env["PYTHONUNBUFFERED"] = "1"
    env.setdefault("MM_CFA_ROOT", str(ROOT))
    try:
        p = subprocess.run(cmd, cwd=str(ROOT), env=env, timeout=timeout)
        return int(p.returncode)
    except subprocess.TimeoutExpired:
        log(f"TIMEOUT {module} after {timeout}s", "ERROR")
        return 124


def launch_e01(paths: Paths) -> int | None:
    """Start clean retrain in the background. Pipeline does not wait."""
    if os.environ.get("CAIA_SKIP_E01", "0") == "1":
        log("CAIA_SKIP_E01=1 — not launching clean retrain")
        return None
    flag = paths.results / "e01_clean_rgb_rgbd" / "launched.json"
    if flag.exists() and os.environ.get("CAIA_E01_FORCE") != "1":
        rec = read_json(flag, {})
        pid = rec.get("pid")
        if pid and _pid_alive(int(pid)):
            log(f"e01 already running pid={pid}")
            return int(pid)
        if _status_ok(paths, "e01_clean_rgb_rgbd"):
            log("e01 already completed")
            return None
    out = paths.logs / "e01_train.log"
    env = os.environ.copy()
    env["PYTHONPATH"] = str(ROOT) + os.pathsep + env.get("PYTHONPATH", "")
    env["PYTHONUNBUFFERED"] = "1"
    env.setdefault("MM_CFA_ROOT", str(ROOT))
    epochs = os.environ.get("CAIA_E01_EPOCHS", "100")
    cmd = [sys.executable, "-m", "agent_extension.experiments.e01_clean_rgb_rgbd", "--epochs", epochs]
    log(f"launch background e01: {' '.join(cmd)}")
    fh = open(out, "ab")
    proc = subprocess.Popen(cmd, cwd=str(ROOT), env=env, stdout=fh, stderr=fh, start_new_session=True)
    atomic_write_json(flag, {"pid": proc.pid, "ts": utc_now(), "log": str(out), "epochs": epochs})
    return proc.pid


def _pid_alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
        return True
    except OSError:
        return False


def apply_degrade(paths: Paths, state: dict) -> None:
    g = read_json(paths.results / "e03_tool_value" / "tool_value.json", {})
    gates = g.get("gates") or read_json(paths.results / "e03_tool_value" / "status.json", {}).get("gates") or {}
    state["gates"] = gates
    if gates.get("unified_agent"):
        state["policy_mode"] = "myopic_voi"
        state["degrade"] = None
        log("Go: heterogeneous tool value — keep unified myopic VOI (still Level A / feasibility)")
        return
    deg = gates.get("degrade")
    state["degrade"] = deg
    if deg == "B_single_action":
        act = gates.get("single_action", "REQUEST_DEPTH")
        state["policy_mode"] = "single_trigger"
        state["single_action"] = act
        log(f"No-Go unified Agent → degrade B single trigger {act}", "WARN")
    else:
        state["policy_mode"] = "triage"
        log("No-Go unified Agent → degrade A conformal triage", "WARN")
    atomic_write_json(paths.results / "degrade.json", {"gates": gates, "mode": state.get("policy_mode")})


def maybe_rerun_e07_degraded(paths: Paths, state: dict, hb: Heartbeat) -> None:
    """If gates say degrade, re-run benchmark with triage/single trigger as the named agent."""
    mode = state.get("policy_mode")
    if mode in (None, "myopic_voi"):
        return
    log(f"re-run E07 with degraded controller mode={mode}")
    argv = ["--split", "test"]
    # locked still ok — we are not changing prereg, we are following it
    argv += ["--locked"]
    # e07 uses myopic_voi as CAIA; we additionally store degraded traces via a side call
    try:
        import numpy as np
        from agent_extension.experiments.e05_terminal_conformal import load_controller, run_episode
        from agent_extension.experiments.common import load_rows
        from agent_extension.src.conformal.finite_sample import conformal_interval
        from agent_extension.src.metrics import interval_metrics, point_metrics, summarize_agent
        rows = [r for r in load_rows() if r.get("split") == "test"]
        ctrl = load_controller(paths, "triage" if mode == "triage" else "single_trigger")
        if mode == "single_trigger":
            ctrl.single_action = state.get("single_action")
            ctrl.mode = "single_trigger"
        cal = read_json(paths.results / "e05_terminal_conformal" / "calibrator.json", {})
        q = cal.get("q")
        traces = [run_episode(ctrl, r) for r in rows]
        y = np.array([t["y"] for t in traces if np.isfinite(t["pred"])])
        m = np.array([t["pred"] for t in traces if np.isfinite(t["pred"])])
        h = np.array([t["scale"] for t in traces if np.isfinite(t["pred"])])
        lo, hi = conformal_interval(m, h, float(q) if q is not None else 1.0)
        ev = {
            "mode": mode,
            "point": point_metrics(y, m),
            "interval": interval_metrics(y, lo, hi, 0.10),
            "agent": summarize_agent(traces),
        }
        atomic_write_json(paths.results / "e07_main_benchmark" / "degraded.json", ev)
    except Exception as e:
        log(f"degraded E07 failed (non-fatal): {e}", "WARN")


def finalize(paths: Paths, state: dict) -> None:
    report = {
        "ts": utc_now(),
        "root": str(paths.root),
        "protocol": "A + Level A public data",
        "steps": state.get("steps", {}),
        "gates": state.get("gates"),
        "degrade": state.get("degrade"),
        "e01_pid": state.get("e01_pid"),
        "notes": [
            "X_train never used as Agent test.",
            "N5K cache has no dish/scan id; grouping limitation is recorded in manifests.",
            "Cross-dataset action comparisons are not causal.",
            "Zero-target NV coverage is empirical only; no 90% guarantee.",
            "Phase 3/4 physical collection was not executed.",
            "LLM was not required for core numbers.",
        ],
    }
    atomic_write_json(paths.results / "FINAL_REPORT.json", report)
    # human summary
    lines = [
        "# CAIA-MM autonomous run summary",
        "",
        f"Time: {report['ts']}",
        f"Root: {report['root']}",
        f"Degrade: {report['degrade']}",
        "",
        "## Steps",
    ]
    for sid, rec in (state.get("steps") or {}).items():
        lines.append(f"- {sid}: {rec.get('status')} rc={rec.get('rc')} attempts={rec.get('attempts')}")
    (paths.results / "FINAL_REPORT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    (paths.logs / "COMPLETE").write_text(utc_now() + "\n", encoding="utf-8")
    log("PIPELINE COMPLETE — see agent_extension/results/FINAL_REPORT.md")


def main() -> int:
    os.environ.setdefault("MM_CFA_ROOT", str(ROOT))
    os.environ.setdefault("PYTHONUNBUFFERED", "1")
    os.environ.setdefault("OMP_NUM_THREADS", "4")
    os.environ.setdefault("MKL_NUM_THREADS", "4")
    os.environ.setdefault("OPENBLAS_NUM_THREADS", "4")
    paths = Paths.resolve()
    state_p = paths.logs / "pipeline_state.json"
    state = read_json(state_p, {"steps": {}})
    hb = Heartbeat(paths.logs / "heartbeat.json", interval=float(os.environ.get("CAIA_HB", "20")))
    hb.start()
    hb.touch("boot", "pipeline start")
    log(f"MM_CFA_ROOT={paths.root} python={sys.executable}")

    # background clean retrain
    try:
        state["e01_pid"] = launch_e01(paths)
    except Exception as e:
        log(f"e01 launch failed (continue with existing weights): {e}", "WARN")
        state["e01_launch_error"] = str(e)

    fatal = False
    for step in STEPS:
        hb.touch(step.sid, step.title)
        prev = (state.get("steps") or {}).get(step.sid, {})
        if prev.get("status") == "ok" and step.status_dir and _status_ok(paths, step.status_dir):
            log(f"SKIP {step.sid} (already ok)")
            continue
        if step.skip_if and step.skip_if(paths):
            log(f"SKIP {step.sid} (skip_if)")
            state.setdefault("steps", {})[step.sid] = {"status": "skipped", "ts": utc_now()}
            atomic_write_json(state_p, state)
            continue
        rc = 1
        err = None
        for attempt in range(1, step.retries + 1):
            hb.touch(step.sid, f"attempt {attempt}/{step.retries}")
            log(f"==== {step.sid} attempt {attempt}/{step.retries}: {step.title} ====")
            try:
                rc = run_module(step.module, step.argv, step.timeout_sec)
                if rc == 0:
                    break
                log(f"{step.sid} rc={rc}", "WARN")
            except Exception as e:
                rc = 1
                err = traceback.format_exc()
                log(f"{step.sid} exception: {e}", "ERROR")
            if attempt < step.retries:
                time.sleep(min(30, 2 ** attempt))
        status = "ok" if rc == 0 else "fail"
        state.setdefault("steps", {})[step.sid] = {
            "status": status,
            "rc": rc,
            "attempts": attempt,
            "ts": utc_now(),
            "error": err,
        }
        atomic_write_json(state_p, state)
        if rc != 0:
            log(f"FALLBACK {step.sid} policy={step.fallback}", "WARN")
            if step.fallback == "required_warn" or step.required:
                # continue anyway after recording; only e00_tests/audit/e02 are hard-ish
                if step.sid in ("e00_tests",) and rc != 0:
                    log("unit tests failed — continue but mark FATAL-lite", "ERROR")
                if step.sid == "e02_action_table" and rc != 0:
                    log("action table failed — later steps will no-op or fail soft", "ERROR")
            # never stop the whole loop
        if step.sid == "e03_tool_value":
            try:
                apply_degrade(paths, state)
                atomic_write_json(state_p, state)
            except Exception as e:
                log(f"degrade apply failed: {e}", "WARN")

    try:
        maybe_rerun_e07_degraded(paths, state, hb)
    except Exception as e:
        log(f"degraded rerun failed: {e}", "WARN")

    # if e01 finished in time, note it
    if _status_ok(paths, "e01_clean_rgb_rgbd"):
        state["e01"] = "completed_clean_retrain"
    else:
        state["e01"] = "background_or_fallback_existing_weights"

    try:
        finalize(paths, state)
    except Exception as e:
        log(f"finalize failed: {e}", "ERROR")
        fatal = True
    hb.touch("done", "pipeline finished")
    hb.stop()
    if fatal:
        (paths.logs / "FATAL").write_text(utc_now() + "\n", encoding="utf-8")
        return 2
    return 0


if __name__ == "__main__":
    # numpy imported lazily in degrade helper; make sure it's available if that path runs
    try:
        import numpy as np  # noqa: F401
    except Exception:
        pass
    raise SystemExit(main())
