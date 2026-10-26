#!/usr/bin/env python3
"""Validate example VDR outputs (and optional IR steps) against frozen schemas."""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from jsonschema import ValidationError  # noqa: E402

from vdr.schemas import (  # noqa: E402
    IR_STEP_SCHEMA_PATH,
    VDR_OUTPUT_SCHEMA_PATH,
    load_schema,
    validate_instance,
)

EXAMPLES = [
    ROOT / "examples" / "pass.json",
    ROOT / "examples" / "corrected.json",
    ROOT / "examples" / "abstain.json",
]

IR_SMOKE = {
    "dish_id": "toy_0001",
    "step_i": 0,
    "component_i": 0,
    "hyp_i": "white rice",
    "code_i": "26100100",
    "cands_i": [{"code": "26100100", "score": 0.9, "name": "rice"}],
    "portion_g_i": 150.0,
    "portion_src_i": "vlm_guess",
    "nutri_i": {"energy_kcal": 195.0},
    "nutri_dish": {"energy_kcal": 220.0},
    "verify_codes": [],
    "evidence_edges": [{"src": "c1", "dst": "h1", "type": "code<-hyp"}],
    "action_t": "STOP",
    "obs_t": {},
    "s": "pass",
    "abstain_reason": None,
}


def main() -> int:
    out_schema = load_schema(VDR_OUTPUT_SCHEMA_PATH)
    ir_schema = load_schema(IR_STEP_SCHEMA_PATH)
    failed = 0

    print(f"Validating VDR outputs against {VDR_OUTPUT_SCHEMA_PATH.name}")
    for path in EXAMPLES:
        with path.open("r", encoding="utf-8") as f:
            inst = json.load(f)
        try:
            validate_instance(inst, out_schema)
            print(f"  OK  {path.relative_to(ROOT)}  s={inst.get('s')}")
        except ValidationError as e:
            failed += 1
            print(f"  FAIL {path.relative_to(ROOT)}: {e.message}")

    print(f"Validating IR smoke against {IR_STEP_SCHEMA_PATH.name}")
    try:
        validate_instance(IR_SMOKE, ir_schema)
        print("  OK  inline IR smoke")
    except ValidationError as e:
        failed += 1
        print(f"  FAIL inline IR smoke: {e.message}")

    if failed:
        print(f"FAILED: {failed} instance(s)")
        return 1
    print("All schema validations passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
