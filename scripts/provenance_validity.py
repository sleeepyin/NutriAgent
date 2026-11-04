#!/usr/bin/env python3
"""Provenance Validity CLI: n == code × portion/100 × per100g.

Demo uses tests/fixtures/fndds_mock_foods.json (offline, not the 5624-code
release). A real catalog must identify FNDDS 2019–2020 or the loader raises.
Does not train and does not read test GT.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Sequence

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from vdr.eval.provenance import (  # noqa: E402
    DEFAULT_MOCK_CATALOG,
    evaluate,
    format_summary,
    load_catalog,
)


def demo_records() -> list[dict[str, Any]]:
    """Tiny synthetic cases against the mock catalog. Two should pass."""
    return [
        {
            "id": "ok_milk_200g",
            "food_code": "11111000",
            "portion_g": 200.0,
            "nutrients": {
                "energy_kcal": 122.0,
                "protein_g": 6.3,
                "carb_g": 9.6,
                "fat_g": 6.5,
                "mass_g": 200.0,
            },
        },
        {
            "id": "bad_chicken_energy",
            "food_code": "24122120",
            "portion_g": 100.0,
            "nutrients": {
                "energy_kcal": 999.0,
                "protein_g": 31.0,
                "carb_g": 0.0,
                "fat_g": 3.6,
                "mass_g": 100.0,
            },
        },
        {
            "id": "ok_apple_150g",
            "food_code": "75113000",
            "portion_g": 150.0,
            "nutrients": {
                "energy_kcal": 78.0,
                "protein_g": 0.39,
                "carb_g": 20.715,
                "fat_g": 0.255,
                "mass_g": 150.0,
            },
        },
        {
            "id": "missing_code",
            "portion_g": 50.0,
            "nutrients": {"energy_kcal": 10.0, "protein_g": 0.0, "carb_g": 0.0, "fat_g": 0.0},
        },
    ]


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--catalog", type=Path, default=None, help="FNDDS foods.json or mock")
    parser.add_argument("--jsonl", type=Path, help="Records to check")
    parser.add_argument("--demo", action="store_true", help="Run mock unit cases")
    parser.add_argument("--atol", type=float, default=1e-2)
    parser.add_argument("--out", type=Path, help="Write summary text")
    args = parser.parse_args(list(argv) if argv is not None else None)

    catalog_path = args.catalog or DEFAULT_MOCK_CATALOG
    try:
        catalog, info = load_catalog(catalog_path)
    except (FileNotFoundError, ValueError) as exc:
        sys.stderr.write(f"{exc}\n")
        return 2

    if args.demo:
        records: list[dict[str, Any]] = demo_records()
    elif args.jsonl:
        records = []
        with args.jsonl.open(encoding="utf-8") as handle:
            for line in handle:
                line = line.strip()
                if line:
                    records.append(json.loads(line))
    else:
        parser.error("provide --jsonl or --demo")
        return 2

    summary = evaluate(records, catalog, atol=args.atol)
    summary.update(
        {
            "catalog_kind": info.kind,
            "pinned_release": info.pinned_release,
            "n_catalog_foods": info.n_foods,
            "full_release": info.full_release,
        }
    )
    text = format_summary(summary)
    slim = {key: value for key, value in summary.items() if key != "rows"}
    slim["rows_brief"] = [
        {"id": row["id"], "ok": row["ok"], "broken_edges": row["broken_edges"]}
        for row in summary["rows"]
    ]
    slim["catalog"] = info.as_dict()
    sys.stdout.write(text)

    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(text, encoding="utf-8")
        args.out.with_suffix(".json").write_text(json.dumps(slim, indent=2), encoding="utf-8")

    if args.demo and info.kind == "mock" and summary["n_pass"] != 2:
        sys.stderr.write(f"demo expected 2 passes, got {summary['n_pass']}\n")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
