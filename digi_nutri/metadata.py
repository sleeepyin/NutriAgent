"""Nutrition5k dish metadata. Ingredient calories and macros are discarded."""

from __future__ import annotations

from pathlib import Path


def parse_dish_line(line: str) -> dict:
    """Parse one cafe metadata row.

    Layout: dish_id, energy, mass, fat, carb, protein, then repeating
    (ingredient id, name, grams, calories, fat, carb, protein).

    Ingredient grams, calories, and macros are read only to keep the columns
    aligned. The returned record contains the dish id, the five dish-level
    supervision targets, and the ingredient names.
    """

    parts = [item.strip() for item in line.strip().split(",")]
    if parts and parts[-1] == "":
        parts = parts[:-1]
    if len(parts) < 6:
        raise ValueError("metadata row has fewer than 6 fields")
    rest = parts[6:]
    if len(rest) % 7 != 0:
        raise ValueError(f"ingredient columns are not groups of 7 ({len(rest)})")
    names: list[str] = []
    for offset in range(0, len(rest), 7):
        ing_id, name, _grams, _cal, _fat, _carb, _pro = rest[offset : offset + 7]
        if not ing_id.startswith("ingr_"):
            raise ValueError(f"expected an ingredient id, got {ing_id!r}")
        if name:
            names.append(name)
    try:
        targets = [float(parts[i]) for i in range(1, 6)]
    except ValueError as exc:
        raise ValueError("dish targets are not numeric") from exc
    return {"dish_id": parts[0], "y": targets, "names": names}


def load_metadata(paths: list[Path] | tuple[Path, ...]) -> dict[str, dict]:
    dishes: dict[str, dict] = {}
    for path in paths:
        with Path(path).open(encoding="utf-8-sig") as handle:
            for line_no, line in enumerate(handle, 1):
                if not line.strip():
                    continue
                record = parse_dish_line(line)
                dish_id = record["dish_id"]
                if dish_id in dishes:
                    raise ValueError(f"duplicate dish_id {dish_id} at {path}:{line_no}")
                dishes[dish_id] = record
    return dishes
