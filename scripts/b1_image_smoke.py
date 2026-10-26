#!/usr/bin/env python3
"""One official rgb_test overhead photo through B1 (VLM + FNDDS) and B0."""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from vdr.baselines.b0_direct_vlm import predict as predict_b0  # noqa: E402
from vdr.baselines.b1_rag_once import predict as predict_b1  # noqa: E402
from vdr.schemas import VDR_OUTPUT_SCHEMA_PATH, load_schema, validate_instance  # noqa: E402

N5K = ROOT / "data" / "nutrition5k"
OUT = ROOT / "results" / "b0_b1_image_smoke.json"


def label_for(dish_id: str) -> dict[str, float] | None:
    for name in ("dish_metadata_cafe1.csv", "dish_metadata_cafe2.csv"):
        path = N5K / "metadata" / name
        with path.open(encoding="utf-8") as handle:
            for line in handle:
                parts = line.strip().split(",")
                if not parts or parts[0] != dish_id or len(parts) < 6:
                    continue
                return {
                    "energy_kcal": float(parts[1]),
                    "mass_g": float(parts[2]),
                    "fat_g": float(parts[3]),
                    "carb_g": float(parts[4]),
                    "protein_g": float(parts[5]),
                }
    return None


def first_test_image() -> tuple[str, Path]:
    ids = [
        line.strip()
        for line in (N5K / "dish_ids" / "splits" / "rgb_test_ids.txt").read_text().splitlines()
        if line.strip()
    ]
    for dish_id in ids:
        image = N5K / "imagery" / "realsense_overhead" / dish_id / "rgb.png"
        if image.is_file():
            return dish_id, image
    raise SystemExit("no official rgb_test dish has an overhead rgb.png")


def main() -> int:
    dish_id, image = first_test_image()
    schema = load_schema(VDR_OUTPUT_SCHEMA_PATH)
    print(f"image {image}", flush=True)
    b1 = predict_b1(str(image), dish_id=dish_id)
    validate_instance(b1, schema)
    print(
        f"B1 s={b1['s']} energy={b1['n'].get('energy_kcal')} "
        f"hyps={b1.get('meta', {}).get('vlm', {}).get('n_hypotheses')}",
        flush=True,
    )
    b0 = predict_b0(str(image), dish_id=dish_id)
    validate_instance(b0, schema)
    print(f"B0 s={b0['s']} energy={b0['n'].get('energy_kcal')}", flush=True)
    label = label_for(dish_id)
    payload = {
        "dish_id": dish_id,
        "image": str(image.relative_to(ROOT)),
        "label": label,
        "b1": b1,
        "b0": b0,
    }
    OUT.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"wrote {OUT.relative_to(ROOT)}", flush=True)
    hyps = (b1.get("meta") or {}).get("vlm", {}).get("n_hypotheses", 0)
    if hyps < 1:
        print("FAIL: VLM returned no component hypotheses")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
