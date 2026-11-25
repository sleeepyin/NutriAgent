"""Nutrition5k dish metadata. Plate-level labels only."""

from __future__ import annotations

from pathlib import Path

from digi_nutri.protocol import NUTRIENTS


class MetadataError(ValueError):
    pass


def parse_dish_line(line: str) -> dict:
    """Parse one cafe metadata row.

    The repeating ingredient tail is (id, name, grams, calories, fat, carb,
    protein). Grams and per-ingredient nutrition are discarded here so they
    cannot be passed to a model.
    """
    parts = line.strip().split(",")
    if len(parts) < 6:
        raise MetadataError("dish row has fewer than 6 fields")
    dish_id = parts[0].strip()
    try:
        calories, mass, fat, carb, protein = (float(parts[i]) for i in range(1, 6))
    except ValueError as exc:
        raise MetadataError(f"{dish_id} has non-numeric plate labels") from exc
    rest = parts[6:]
    ingredients = []
    for index in range(0, len(rest) - 6, 7):
        ing_id = rest[index].strip()
        name = rest[index + 1].strip()
        if not ing_id and not name:
            continue
        ingredients.append({"id": ing_id, "name": name})
    nutrition = {
        "energy": calories,
        "fat": fat,
        "carb": carb,
        "protein": protein,
    }
    return {
        "dish_id": dish_id,
        "mass": mass,
        "nutrition": nutrition,
        "ingredients": ingredients,
    }


def _same_labels(left: dict, right: dict) -> bool:
    if abs(left["mass"] - right["mass"]) > 1e-6:
        return False
    return all(abs(left["nutrition"][key] - right["nutrition"][key]) <= 1e-6 for key in NUTRIENTS)


def load_dish_table(cafe1: str | Path, cafe2: str | Path) -> dict[str, dict]:
    dishes: dict[str, dict] = {}
    for path in (cafe1, cafe2):
        for line in Path(path).read_text().splitlines():
            if not line.strip():
                continue
            dish = parse_dish_line(line)
            previous = dishes.get(dish["dish_id"])
            if previous is not None and not _same_labels(previous, dish):
                raise MetadataError(f"conflicting labels for {dish['dish_id']}")
            dishes[dish["dish_id"]] = dish
    return dishes


def nutrition_vector(dish: dict) -> list[float]:
    return [float(dish["nutrition"][key]) for key in NUTRIENTS]
