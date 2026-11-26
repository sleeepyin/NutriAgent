"""Plate-level Nutrition5k labels. Ingredient grams are discarded."""

from __future__ import annotations

from pathlib import Path


class MetadataError(ValueError):
    pass


def parse_dish_line(line: str) -> dict:
    parts = line.strip().split(",")
    if len(parts) < 6:
        raise MetadataError("dish row has fewer than 6 fields")
    dish_id = parts[0].strip()
    try:
        calories, mass, fat, carb, protein = (float(parts[i]) for i in range(1, 6))
    except ValueError as exc:
        raise MetadataError(f"{dish_id} has non-numeric plate labels") from exc
    return {
        "dish_id": dish_id,
        "mass": mass,
        "nutrition": {
            "energy": calories,
            "fat": fat,
            "carb": carb,
            "protein": protein,
        },
    }


def _same(left: dict, right: dict) -> bool:
    if abs(left["mass"] - right["mass"]) > 1e-6:
        return False
    keys = ("energy", "fat", "carb", "protein")
    return all(abs(left["nutrition"][key] - right["nutrition"][key]) <= 1e-6 for key in keys)


def load_dishes(cafe1: str | Path, cafe2: str | Path) -> dict[str, dict]:
    dishes: dict[str, dict] = {}
    for path in (cafe1, cafe2):
        for line in Path(path).read_text().splitlines():
            if not line.strip():
                continue
            dish = parse_dish_line(line)
            previous = dishes.get(dish["dish_id"])
            if previous is not None and not _same(previous, dish):
                raise MetadataError(f"conflicting labels for {dish['dish_id']}")
            dishes[dish["dish_id"]] = dish
    return dishes


def label_vector(dish: dict, order: tuple[str, ...]) -> list[float]:
    values = {"mass": float(dish["mass"]), **{key: float(dish["nutrition"][key]) for key in dish["nutrition"]}}
    return [values[name] for name in order]
