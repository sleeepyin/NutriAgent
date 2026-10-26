#!/usr/bin/env python3
"""B1 RAG-once smoke (no VLM): toy dishes -> schema-valid VDR JSON."""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from vdr.baselines.b1_rag_once import predict_from_hypotheses  # noqa: E402
from vdr.schemas import VDR_OUTPUT_SCHEMA_PATH, load_schema, validate_instance  # noqa: E402

TOY_DIR = ROOT / "data" / "toy" / "dishes"
OUT_DIR = ROOT / "outputs" / "toy" / "b1"


def main() -> int:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    schema = load_schema(VDR_OUTPUT_SCHEMA_PATH)
    dishes = sorted(TOY_DIR.glob("toy_*.json"))
    ok = 0
    energies = []
    for path in dishes:
        with path.open("r", encoding="utf-8") as f:
            d = json.load(f)
        pred = predict_from_hypotheses(d["dish_id"], d.get("components") or [])
        validate_instance(pred, schema)
        out_path = OUT_DIR / f"{d['dish_id']}_b1.json"
        with out_path.open("w", encoding="utf-8") as f:
            json.dump(pred, f, ensure_ascii=False, indent=2)
            f.write("\n")
        ok += 1
        e = pred["n"].get("energy_kcal")
        energies.append(e)
        print(
            f"  {d['dish_id']}: s={pred['s']} energy={e} "
            f"verify={pred['meta'].get('verify_codes')} -> {out_path.relative_to(ROOT)}"
        )

    matched = [e for e in energies if e is not None]
    print(
        f"b1_toy_smoke: OK {ok}/{len(dishes)} schema-valid; "
        f"mean_energy={sum(matched)/len(matched):.1f}" if matched else
        f"b1_toy_smoke: OK {ok}/{len(dishes)} schema-valid; mean_energy=n/a"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
