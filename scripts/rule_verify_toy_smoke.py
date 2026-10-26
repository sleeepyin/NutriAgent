#!/usr/bin/env python3
"""Unit-style smoke for rule_verify on constructed states."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from vdr.tools.rule_verify import verify  # noqa: E402


def expect(name: str, state: dict, must_contain: set[str], must_not: set[str] | None = None) -> bool:
    got = set(verify(state))
    must_not = must_not or set()
    ok = must_contain.issubset(got) and got.isdisjoint(must_not)
    status = "OK" if ok else "FAIL"
    print(f"  {status} {name}: got={sorted(got)} expect_supset={sorted(must_contain)}")
    return ok


def main() -> int:
    cases_ok = True

    # Clean grounded state: Atwater-consistent rice-like macros
    # E=130, P=2.7, C=28, F=0.3 -> 4*2.7+4*28+9*0.3 = 10.8+112+2.7=125.5 ≈ 130
    cases_ok &= expect(
        "clean_grounded",
        {
            "n": {"energy_kcal": 130.0, "protein_g": 2.7, "carb_g": 28.0, "fat_g": 0.3},
            "evidence_edges": [
                {"src": "n1", "dst": "c1", "type": "n<-code×portion"},
            ],
            "portions": [150.0],
            "cands": [{"code": "A", "score": 0.9}, {"code": "B", "score": 0.5}],
        },
        must_contain=set(),
        must_not={"UNGROUNDED", "ENERGY_ATWATER_MISMATCH", "ENERGY_OUTLIER", "CODE_CONF_GAP"},
    )

    cases_ok &= expect(
        "ungrounded_numeric",
        {
            "n": {"energy_kcal": 200.0, "protein_g": 10.0, "carb_g": 20.0, "fat_g": 5.0},
            "evidence_edges": [],
        },
        must_contain={"UNGROUNDED"},
    )

    cases_ok &= expect(
        "atwater_mismatch",
        {
            "n": {"energy_kcal": 800.0, "protein_g": 2.0, "carb_g": 2.0, "fat_g": 2.0},
            "evidence_edges": [
                {"src": "n1", "dst": "c1", "type": "n<-code×portion"},
            ],
        },
        must_contain={"ENERGY_ATWATER_MISMATCH"},
    )

    cases_ok &= expect(
        "energy_outlier",
        {
            "n": {"energy_kcal": 9000.0, "protein_g": 10.0, "carb_g": 10.0, "fat_g": 10.0},
            "evidence_edges": [
                {"src": "n1", "dst": "c1", "type": "n<-code×portion"},
            ],
        },
        must_contain={"ENERGY_OUTLIER"},
    )

    cases_ok &= expect(
        "code_conf_gap",
        {
            "n": {"energy_kcal": 100.0, "protein_g": 5.0, "carb_g": 10.0, "fat_g": 3.0},
            "evidence_edges": [
                {"src": "n1", "dst": "c1", "type": "n<-code×portion"},
            ],
            "cands": [{"code": "A", "score": 0.55}, {"code": "B", "score": 0.52}],
        },
        must_contain={"CODE_CONF_GAP"},
    )

    cases_ok &= expect(
        "portion_extreme",
        {
            "n": {"energy_kcal": 100.0, "protein_g": 5.0, "carb_g": 10.0, "fat_g": 3.0},
            "evidence_edges": [
                {"src": "n1", "dst": "c1", "type": "n<-code×portion"},
            ],
            "portions": [0.1],
        },
        must_contain={"PORTION_EXTREME"},
    )

    print(
        "rule_verify_toy_smoke: "
        + ("All rule cases passed." if cases_ok else "FAILED some cases.")
    )
    return 0 if cases_ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
