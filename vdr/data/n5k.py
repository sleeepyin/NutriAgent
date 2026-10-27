"""Official Nutrition5K splits, dish labels, and overhead RGB paths.

Dish nutrition and per-ingredient grams are for scoring, and for the
train-only weak-step teacher (`vdr.weak`). Do not pass them into perception,
retrieval query text, portion-grid hints, or student-policy features.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
N5K = ROOT / "data" / "nutrition5k"
LABEL_FIELDS = ("energy_kcal", "mass_g", "fat_g", "carb_g", "protein_g")
METADATA_FILES = ("dish_metadata_cafe1.csv", "dish_metadata_cafe2.csv")


def read_ids(split_name: str) -> list[str]:
    path = N5K / "dish_ids" / "splits" / f"{split_name}_ids.txt"
    return [line.strip() for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def load_labels() -> dict[str, dict[str, float]]:
    labels: dict[str, dict[str, float]] = {}
    for name in ("dish_metadata_cafe1.csv", "dish_metadata_cafe2.csv"):
        path = N5K / "metadata" / name
        with path.open(encoding="utf-8") as handle:
            for line in handle:
                parts = line.strip().split(",")
                if len(parts) < 6 or not parts[0].startswith("dish_"):
                    continue
                try:
                    mass, fat, carb, protein = (float(parts[i]) for i in range(2, 6))
                    energy = float(parts[1])
                except ValueError:
                    continue
                labels[parts[0]] = {
                    "energy_kcal": energy,
                    "mass_g": mass,
                    "fat_g": fat,
                    "carb_g": carb,
                    "protein_g": protein,
                }
    return labels


@dataclass(frozen=True)
class Ingredient:
    """One Nutrition5K ingredient block. Grams and kcal are privileged."""

    ingr_id: str
    name: str
    mass_g: float
    energy_kcal: float
    fat_g: float
    carb_g: float
    protein_g: float


@dataclass(frozen=True)
class DishMeta:
    """Dish-level nutrition plus the ingredient list from cafe metadata."""

    dish_id: str
    energy_kcal: float
    mass_g: float
    fat_g: float
    carb_g: float
    protein_g: float
    ingredients: tuple[Ingredient, ...]


def test_split_ids() -> set[str]:
    """Official ids the weak teacher must never emit (rgb_test and depth_test)."""
    ids: set[str] = set()
    for name in ("rgb_test", "depth_test"):
        ids.update(read_ids(name))
    return ids


def parse_dish_metadata_line(line: str) -> DishMeta | None:
    """Parse one cafe metadata line.

    Nutrition5K writes comma-separated blocks with no header:
    dish fields, then repeating
    ``ingr_id, name, grams, kcal, fat, carb, protein``.
    Names may themselves contain commas. The five numbers are the fields
    immediately before the next ``ingr_`` token (or the end of the line).
    """
    parts = [part.strip() for part in line.strip().split(",")]
    if len(parts) < 6 or not parts[0].startswith("dish_"):
        return None
    try:
        energy, mass, fat, carb, protein = (float(parts[i]) for i in range(1, 6))
    except ValueError:
        return None
    ingredients: list[Ingredient] = []
    index = 6
    while index < len(parts):
        if not parts[index].startswith("ingr_"):
            break
        end = index + 1
        while end < len(parts) and not parts[end].startswith("ingr_"):
            end += 1
        block = parts[index:end]
        # id + name + five numbers. A comma inside the name makes this longer.
        if len(block) < 7:
            break
        try:
            grams, kcal, fat_i, carb_i, pro_i = (float(block[i]) for i in range(-5, 0))
        except ValueError:
            break
        name = ",".join(block[1:-5]).strip()
        ingredients.append(
            Ingredient(
                ingr_id=block[0],
                name=name,
                mass_g=grams,
                energy_kcal=kcal,
                fat_g=fat_i,
                carb_g=carb_i,
                protein_g=pro_i,
            )
        )
        index = end
    return DishMeta(
        dish_id=parts[0],
        energy_kcal=energy,
        mass_g=mass,
        fat_g=fat,
        carb_g=carb,
        protein_g=protein,
        ingredients=tuple(ingredients),
    )


def load_dish_metadata() -> dict[str, DishMeta]:
    """Cafe1 then cafe2. A repeated dish_id keeps the cafe2 row."""
    dishes: dict[str, DishMeta] = {}
    for name in METADATA_FILES:
        path = N5K / "metadata" / name
        with path.open(encoding="utf-8") as handle:
            for line in handle:
                dish = parse_dish_metadata_line(line)
                if dish is not None:
                    dishes[dish.dish_id] = dish
    return dishes


def overhead_rgb(dish_id: str) -> Path:
    return N5K / "imagery" / "realsense_overhead" / dish_id / "rgb.png"


def dishes_with_overhead(split_name: str = "rgb_test") -> list[dict]:
    """Official split rows that have both a label and an overhead rgb.png."""
    labels = load_labels()
    rows = []
    for dish_id in read_ids(split_name):
        image = overhead_rgb(dish_id)
        label = labels.get(dish_id)
        if label is None or not image.is_file():
            continue
        rows.append({"dish_id": dish_id, "image": image, "label": label})
    return rows
