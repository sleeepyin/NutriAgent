"""Train once, then commit only the result files on this branch."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
BRANCH = "cursor/digi-nutri-v3-step1-de25"
RESULT_FILES = (
    "results/v3_step1/train_metrics.json",
    "results/v3_step1/holdout_metrics.json",
    "results/v3_step1/predictions_train.csv",
    "results/v3_step1/predictions_holdout.csv",
    "results/v3_step1/REPORT.md",
)


def main() -> None:
    completed = subprocess.run(
        [sys.executable, str(ROOT / "scripts" / "run_v3_step1.py")],
        cwd=ROOT,
        check=False,
    )
    if completed.returncode != 0:
        raise SystemExit(completed.returncode)
    holdout = ROOT / "results" / "v3_step1" / "holdout_metrics.json"
    if not holdout.is_file():
        raise SystemExit("training finished without the holdout metrics file")
    subprocess.check_call(["git", "add", "--", *RESULT_FILES], cwd=ROOT)
    staged = subprocess.check_output(
        ["git", "diff", "--cached", "--name-only"],
        cwd=ROOT,
        text=True,
    )
    if not staged.strip():
        print("result files were already committed", flush=True)
        return
    subprocess.check_call(
        [
            "git",
            "commit",
            "-m",
            "Record the single v3 step 1 train metrics and official holdout.",
        ],
        cwd=ROOT,
    )
    subprocess.check_call(["git", "push", "-u", "origin", BRANCH], cwd=ROOT)
    print("result files pushed", flush=True)


if __name__ == "__main__":
    main()
