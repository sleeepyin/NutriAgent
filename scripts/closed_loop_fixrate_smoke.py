#!/usr/bin/env python3
"""Synthetic FixRate contract. Does not read weak steps and does not write results/."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from vdr.eval.fixrate import BUDGET_K, FIXRATE_EPSILON_KCAL, run_self_check  # noqa: E402


def main() -> int:
    run_self_check()
    print(f"closed_loop_fixrate_smoke ok epsilon={FIXRATE_EPSILON_KCAL} K={BUDGET_K}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
