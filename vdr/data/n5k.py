"""Official Nutrition5K splits, dish labels, and overhead RGB paths.

Labels are for scoring. Callers must not feed them back into perception or
into train-only weak labels.
"""

from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
N5K = ROOT / "data" / "nutrition5k"
LABEL_FIELDS = ("energy_kcal", "mass_g", "fat_g", "carb_g", "protein_g")


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
