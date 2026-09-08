#!/usr/bin/env python3
"""Phase 3/4: write MM-CFA-Interact SOP. Physical collection cannot be executed here."""
from __future__ import annotations

import argparse
from pathlib import Path

from .common import add_common_args, out_dir, save_status
from agent_extension.src.paths import Paths

SOP = """# MM-CFA-Interact pilot SOP (60 instances)

This automated run cannot collect food, weigh plates, or recruit users.
If a directory `MM-CFA-Interact/` with per-instance paired actions appears, re-run the pipeline to ingest it.

## Per instance (weight is the primary endpoint)

1. Initial natural RGB
2. Same scene + standard card / ArUco
3. Top view
4. ~45° side view
5. RGB-D if the device supports it
6. Structured semantic labels (morphology, container, preparation, hidden_liquid, added_fat, food_code)
7. Scale weight, twice
8. Time, retries, 1–5 burden
9. Strip EXIF / faces / location

## Rules

- Do not change the food between actions
- Reference object coplanar with food
- Collect ALL actions first; offline Agent only sees chosen subsets
- Nutrient GT only if recipe weighing or package label is auditable
- Ethics / consent before any participant work

## Gate to continue to 400

- Action success >=90%
- Question comprehension >=90%
- Each main subgroup has preliminary samples
"""


def main(argv=None) -> int:
    ap = add_common_args(argparse.ArgumentParser())
    args = ap.parse_args(argv)
    paths = Paths.resolve()
    d = out_dir("e11_pilot_sop", args)
    (d / "MM_CFA_Interact_pilot_SOP.md").write_text(SOP, encoding="utf-8")
    interact = paths.root / "MM-CFA-Interact"
    found = interact.exists() and any(interact.iterdir())
    save_status(d, True, physical_data_found=found, next="skip Phase 4 unless data appears")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
