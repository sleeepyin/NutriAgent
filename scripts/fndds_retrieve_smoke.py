#!/usr/bin/env python3
"""Smoke Top-k retrieve against the active FNDDS catalog (real if built)."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from vdr.tools.fndds_catalog import catalog_stats, clear_catalog_cache  # noqa: E402
from vdr.tools.fndds_retrieve import retrieve  # noqa: E402
from vdr.tools.nutrition_lookup import lookup_and_scale  # noqa: E402

QUERIES = [
    "white rice",
    "grilled chicken breast",
    "broccoli",
    "banana",
    "milk",
]


def main() -> int:
    clear_catalog_cache()
    stats = catalog_stats()
    print(f"catalog: path={stats['path']} n_foods={stats['n_foods']} version={stats['version']}")
    if stats["n_foods"] <= 20:
        print("WARN: catalog looks like toy (<=20). Did you run build_fndds_catalog.py?")
    for q in QUERIES:
        rr = retrieve(q, top_k=3)
        print(f"\nquery={q!r} selected={rr.selected_code}")
        for c in rr.cands:
            print(f"  {c['score']:.3f}  {c['code']}  {c.get('name')}")
        if rr.selected_code:
            scaled = lookup_and_scale(rr.selected_code, 100.0)
            print(f"  per100g: {scaled}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
