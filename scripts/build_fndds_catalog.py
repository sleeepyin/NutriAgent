#!/usr/bin/env python3
"""Build VDR-compatible foods.json from FDC SurveyFoods JSON and/or FNDDS xlsx.

Primary source: FoodData_Central_survey_food_json_*.json (SurveyFoods[]).
Fallback / cross-check: FNDDS_Nutrient_Values.xlsx + Foods_and_Beverages.xlsx.

Output schema matches data/fndds/toy/foods.json:
  {version, note, foods: [{code, name, aliases, per_100g: {energy_kcal, protein_g, carb_g, fat_g}}]}
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DIR = ROOT / "data" / "fndds" / "2019-2020"
DEFAULT_JSON = DEFAULT_DIR / "FoodData_Central_survey_food_json_2022-10-28.json"
DEFAULT_OUT = DEFAULT_DIR / "foods.json"

# FDC nutrient ids (FoodData Central)
NUTRIENT_MAP = {
    1008: "energy_kcal",  # Energy, kcal
    1003: "protein_g",  # Protein
    1005: "carb_g",  # Carbohydrate, by difference
    1004: "fat_g",  # Total lipid (fat)
}
# Also match by nutrient number string as backup
NUTRIENT_NUMBER_MAP = {
    "208": "energy_kcal",
    "203": "protein_g",
    "205": "carb_g",
    "204": "fat_g",
}


def _extract_macros(food_nutrients: list[dict[str, Any]]) -> dict[str, float]:
    out: dict[str, float] = {}
    for fn in food_nutrients or []:
        nut = fn.get("nutrient") or {}
        key = None
        nid = nut.get("id")
        if nid in NUTRIENT_MAP:
            key = NUTRIENT_MAP[nid]
        else:
            num = str(nut.get("number", ""))
            key = NUTRIENT_NUMBER_MAP.get(num)
        if key is None:
            continue
        amt = fn.get("amount")
        if amt is None:
            continue
        out[key] = float(amt)
    return out


def _aliases_from_description(desc: str) -> list[str]:
    d = (desc or "").strip()
    if not d:
        return []
    aliases = {d.lower()}
    # first clause before comma often is the head term
    head = d.split(",")[0].strip().lower()
    if head:
        aliases.add(head)
    return sorted(aliases)


def build_from_fdc_json(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    foods_in = data.get("SurveyFoods") or data.get("surveyFoods") or []
    foods: list[dict[str, Any]] = []
    skipped = 0
    for item in foods_in:
        code = item.get("foodCode")
        if code is None:
            skipped += 1
            continue
        code_s = str(code)
        name = (item.get("description") or "").strip() or code_s
        macros = _extract_macros(item.get("foodNutrients") or [])
        if "energy_kcal" not in macros:
            skipped += 1
            continue
        # ensure all four macros present (default 0 if missing)
        per_100g = {
            "energy_kcal": float(macros.get("energy_kcal", 0.0)),
            "protein_g": float(macros.get("protein_g", 0.0)),
            "carb_g": float(macros.get("carb_g", 0.0)),
            "fat_g": float(macros.get("fat_g", 0.0)),
        }
        entry: dict[str, Any] = {
            "code": code_s,
            "name": name,
            "aliases": _aliases_from_description(name),
            "per_100g": per_100g,
        }
        # optional mass-related: water g / 100g if present
        for fn in item.get("foodNutrients") or []:
            nut = fn.get("nutrient") or {}
            if nut.get("id") == 1051 or str(nut.get("number")) == "255":
                if fn.get("amount") is not None:
                    entry["per_100g"]["water_g"] = float(fn["amount"])
                break
        # keep a couple default portions (gram weights) if available
        portions = []
        for p in (item.get("foodPortions") or [])[:5]:
            gw = p.get("gramWeight")
            if gw is None:
                continue
            portions.append(
                {
                    "gram_weight": float(gw),
                    "description": p.get("portionDescription") or "",
                }
            )
        if portions:
            entry["portions"] = portions
        if item.get("fdcId") is not None:
            entry["fdc_id"] = item["fdcId"]
        foods.append(entry)
    print(f"FDC JSON: kept={len(foods)} skipped={skipped} from {path.name}")
    return foods


def build_from_xlsx(nutrient_xlsx: Path, foods_xlsx: Path | None = None) -> list[dict[str, Any]]:
    """Optional fallback using At-a-Glance style xlsx."""
    try:
        import openpyxl
    except ImportError as e:
        raise SystemExit("openpyxl required for xlsx path: pip install openpyxl") from e

    wb = openpyxl.load_workbook(nutrient_xlsx, read_only=True, data_only=True)
    ws = wb.active
    rows = ws.iter_rows(values_only=True)
    header = [str(c).strip() if c is not None else "" for c in next(rows)]
    # Heuristic column names used in FNDDS At A Glance exports
    def find_col(*cands: str) -> int | None:
        lower = [h.lower() for h in header]
        for c in cands:
            for i, h in enumerate(lower):
                if c in h:
                    return i
        return None

    i_code = find_col("food code", "foodcode")
    i_desc = find_col("main food description", "food description", "description")
    i_energy = find_col("energy (kcal)", "energy\n(kcal)", "calorie")
    i_protein = find_col("protein (g)", "protein\n(g)")
    i_carb = find_col("carbohydrate (g)", "carb")
    i_fat = find_col("total fat (g)", "total fat", "fat (g)")
    # Sometimes energy column is just "Energy"
    if i_energy is None:
        i_energy = find_col("energy")
    print("xlsx header map:", {
        "code": i_code, "desc": i_desc, "E": i_energy, "P": i_protein, "C": i_carb, "F": i_fat
    })
    if i_code is None or i_energy is None:
        wb.close()
        raise ValueError(f"Cannot map columns in {nutrient_xlsx}; header={header[:20]}")

    foods: list[dict[str, Any]] = []
    for row in rows:
        if not row or row[i_code] is None:
            continue
        code = str(int(row[i_code])) if isinstance(row[i_code], float) else str(row[i_code])
        name = str(row[i_desc]).strip() if i_desc is not None and row[i_desc] else code

        def cell(idx: int | None) -> float:
            if idx is None:
                return 0.0
            v = row[idx]
            if v is None or v == "":
                return 0.0
            return float(v)

        per_100g = {
            "energy_kcal": cell(i_energy),
            "protein_g": cell(i_protein),
            "carb_g": cell(i_carb),
            "fat_g": cell(i_fat),
        }
        foods.append(
            {
                "code": code,
                "name": name,
                "aliases": _aliases_from_description(name),
                "per_100g": per_100g,
            }
        )
    wb.close()
    print(f"XLSX: kept={len(foods)} from {nutrient_xlsx.name}")
    return foods


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--fdc-json", type=Path, default=DEFAULT_JSON)
    ap.add_argument("--nutrient-xlsx", type=Path, default=DEFAULT_DIR / "FNDDS_Nutrient_Values.xlsx")
    ap.add_argument("--out", type=Path, default=DEFAULT_OUT)
    ap.add_argument(
        "--source",
        choices=["json", "xlsx", "auto"],
        default="auto",
        help="auto: prefer FDC JSON if present",
    )
    args = ap.parse_args()

    foods: list[dict[str, Any]] = []
    source_note = ""
    if args.source in ("json", "auto") and args.fdc_json.is_file():
        foods = build_from_fdc_json(args.fdc_json)
        source_note = f"Built from {args.fdc_json.name} (FDC SurveyFoods)."
    elif args.source in ("xlsx", "auto") and args.nutrient_xlsx.is_file():
        foods = build_from_xlsx(args.nutrient_xlsx)
        source_note = f"Built from {args.nutrient_xlsx.name}."
    else:
        print("ERROR: no usable FNDDS source found", file=sys.stderr)
        return 1

    # Dedup by code (keep first)
    seen: set[str] = set()
    uniq: list[dict[str, Any]] = []
    for f in foods:
        if f["code"] in seen:
            continue
        seen.add(f["code"])
        uniq.append(f)

    payload = {
        "version": "fndds-2019-2020-vdr-0.1",
        "note": (
            source_note
            + " Per-100g macros from USDA FNDDS/FDC survey foods. "
            "Replace path via VDR_FNDDS_PATH if needed."
        ),
        "foods": uniq,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
        f.write("\n")
    size_mb = args.out.stat().st_size / (1024 * 1024)
    print(f"Wrote {len(uniq)} foods -> {args.out} ({size_mb:.2f} MB)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
