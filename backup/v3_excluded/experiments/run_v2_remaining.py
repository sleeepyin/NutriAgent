"""Run the unfinished v2.0 experiments that can run on this machine.

D3 needs SAM 3D weights that are not in the tree, so it is not started.
The vision pass uses the local Qwen3.8-27B server. This process itself
does not take a GPU.
"""
from __future__ import annotations

import os

os.environ["CUDA_VISIBLE_DEVICES"] = ""

import sys
import traceback
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
for entry in (str(ROOT), str(ROOT / "src")):
    if entry not in sys.path:
        sys.path.insert(0, entry)

from experiments.vlm.names import self_check as names_check
from experiments.vlm.parse import self_check as parse_check
from mmcfa.runtime import log


def main() -> int:
    names_check()
    parse_check()
    from experiments.vlm.infer_holdout import main as infer

    code = infer()
    if code != 0:
        log(f"infer exit {code}")
        return code
    from experiments.vlm.points import main as points

    if points() != 0:
        return 1
    from experiments.action.a2b_vlm_name import main as a2b
    from experiments.action.a3_auto_reference.run import main as a3
    from experiments.action.a4_ood_reject.run import main as a4
    from experiments.action.a5_mixed_flow.run import main as a5
    from experiments.diagnosis.d6_capacity.vlm_weight import main as d6

    failed = []
    for name, fn in (("d6_vlm", d6), ("a2b", a2b), ("a3", a3), ("a4", a4), ("a5", a5)):
        try:
            rc = fn()
            log(f"{name} exit {rc}")
            if rc not in (0, None):
                failed.append(name)
        except Exception:
            traceback.print_exc()
            failed.append(name)
    if failed:
        log(f"downstream failed: {failed}")
        return 1
    log("v2 remaining pipeline finished")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
