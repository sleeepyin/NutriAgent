"""Commit only the official result files, then push the existing branch."""

from __future__ import annotations

import json
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PATHS = [
    "results/tmulti/train_metrics.json",
    "results/tmulti/holdout_metrics.json",
    "results/tmulti/predictions_train.csv",
    "results/tmulti/predictions_holdout.csv",
    "results/tmulti/REPORT.md",
]


def _run(args: list[str], check: bool = True) -> subprocess.CompletedProcess:
    return subprocess.run(args, cwd=ROOT, text=True, capture_output=True, check=check)


def main() -> None:
    missing = [path for path in PATHS if not (ROOT / path).is_file()]
    if missing:
        raise SystemExit(f"result files are missing: {missing}")
    verdict = json.loads((ROOT / "results/tmulti/holdout_metrics.json").read_text())["verdict"]
    _run(["git", "add", "--", *PATHS])
    staged = _run(["git", "diff", "--cached", "--name-only"]).stdout.split()
    if set(staged) - set(PATHS):
        _run(["git", "reset"], check=False)
        raise SystemExit(f"refusing to commit unexpected paths: {staged}")
    if not staged:
        print("result files are already committed", flush=True)
        return
    message = f"Record the single official T_multi holdout.\n\nVerdict: {verdict}\n"
    _run(["git", "commit", "-m", message])
    branch = _run(["git", "rev-parse", "--abbrev-ref", "HEAD"]).stdout.strip()
    delay = 4
    for attempt in range(4):
        pushed = _run(["git", "push", "-u", "origin", branch], check=False)
        if pushed.returncode == 0:
            print(pushed.stdout or pushed.stderr, flush=True)
            return
        if attempt == 3:
            sys.stderr.write(pushed.stderr)
            raise SystemExit(pushed.returncode)
        time.sleep(delay)
        delay *= 2


if __name__ == "__main__":
    main()
