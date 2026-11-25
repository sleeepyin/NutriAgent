"""Train B1, T_geo, and T_ing, then open the holdout at most once."""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PYTHON = sys.executable


def _env(updates: dict | None = None) -> dict:
    env = os.environ.copy()
    previous = env.get("PYTHONPATH", "")
    env["PYTHONPATH"] = str(ROOT) if not previous else str(ROOT) + os.pathsep + previous
    if updates:
        env.update(updates)
    return env


def _run(args: list[str], log_name: str, env: dict | None = None) -> int:
    log_path = ROOT / "outputs" / "logs" / log_name
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("a") as handle:
        handle.write("\n" + " ".join(args) + "\n")
        handle.flush()
        process = subprocess.run(
            args,
            cwd=ROOT,
            env=_env() if env is None else env,
            stdout=handle,
            stderr=subprocess.STDOUT,
        )
    return process.returncode


def _popen(model: str, gpu: str) -> subprocess.Popen:
    env = _env({"CUDA_VISIBLE_DEVICES": gpu})
    log_path = ROOT / "outputs" / "logs" / f"{model}.log"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    handle = log_path.open("a")
    process = subprocess.Popen(
        [PYTHON, "-m", "digi_nutri.train", "--model", model, "--device", "cuda:0"],
        cwd=ROOT,
        env=env,
        stdout=handle,
        stderr=subprocess.STDOUT,
    )
    process._log_handle = handle  # type: ignore[attr-defined]
    return process


def _close(process: subprocess.Popen) -> None:
    handle = getattr(process, "_log_handle", None)
    if handle is not None:
        handle.close()


def main() -> None:
    os.chdir(ROOT)
    holdout = ROOT / "results" / "v2_q0" / "holdout_metrics.json"
    if holdout.exists():
        raise SystemExit("holdout_metrics.json already exists; refusing to train or evaluate again")
    cache_code = _run([PYTHON, "scripts/build_cache.py"], "cache.log")
    if cache_code != 0:
        raise SystemExit(f"cache build failed with {cache_code}")
    geo = _popen("t_geo", "1")
    base = _popen("b1", "0")
    base_code = base.wait()
    _close(base)
    if base_code != 0:
        geo.wait()
        _close(geo)
        raise SystemExit(f"b1 failed with {base_code}")
    ingredient = _popen("t_ing", "0")
    geo_code = geo.wait()
    ing_code = ingredient.wait()
    _close(geo)
    _close(ingredient)
    if geo_code != 0 or ing_code != 0:
        raise SystemExit(f"teacher failed geo={geo_code} ing={ing_code}")
    train_code = _run(
        [PYTHON, "-m", "digi_nutri.evaluate", "--split", "train", "--device", "cuda:0"],
        "eval_train.log",
        env=_env({"CUDA_VISIBLE_DEVICES": "0"}),
    )
    if train_code != 0:
        raise SystemExit(f"train eval failed with {train_code}")
    train_metrics = json.loads((ROOT / "results" / "v2_q0" / "train_metrics.json").read_text())
    if not train_metrics["train_gate_pass"]:
        (ROOT / "results" / "v2_q0" / "verdict.json").write_text(
            json.dumps(
                {
                    "verdict": "train_gate_fail_stop",
                    "holdout_evaluated": False,
                    "failures": train_metrics["train_gate_failures"],
                },
                indent=2,
            )
            + "\n"
        )
        report_code = _run([PYTHON, "-m", "digi_nutri.report"], "report.log")
        if report_code != 0:
            raise SystemExit(f"report failed with {report_code}")
        print("train gate failed; holdout not opened", flush=True)
        return
    holdout_code = _run(
        [PYTHON, "-m", "digi_nutri.evaluate", "--split", "holdout", "--device", "cuda:0"],
        "eval_holdout.log",
        env=_env({"CUDA_VISIBLE_DEVICES": "0"}),
    )
    if holdout_code != 0:
        raise SystemExit(f"holdout eval failed with {holdout_code}")
    report_code = _run([PYTHON, "-m", "digi_nutri.report"], "report.log")
    if report_code != 0:
        raise SystemExit(f"report failed with {report_code}")
    verdict = json.loads((ROOT / "results" / "v2_q0" / "verdict.json").read_text())
    print(verdict["verdict"], flush=True)


if __name__ == "__main__":
    main()
