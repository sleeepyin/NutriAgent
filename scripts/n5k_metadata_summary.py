#!/usr/bin/env python3
"""Summarize Nutrition5K metadata + official RGB/depth splits (no imagery required)."""

from __future__ import annotations

import csv
import statistics
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
N5K = ROOT / "data" / "nutrition5k"


def count_ids(path: Path) -> list[str]:
    if not path.is_file():
        return []
    return [ln.strip() for ln in path.read_text(encoding="utf-8").splitlines() if ln.strip()]


def parse_dish_metadata(path: Path) -> list[dict]:
    """N5K cafe metadata CSV has no header.
    Format: dish_id, total_calories, total_mass, total_fat, total_carb, total_protein, then ingredient blocks...
    """
    rows = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            parts = line.strip().split(",")
            if len(parts) < 6:
                continue
            try:
                rows.append(
                    {
                        "dish_id": parts[0],
                        "energy_kcal": float(parts[1]),
                        "mass_g": float(parts[2]),
                        "fat_g": float(parts[3]),
                        "carb_g": float(parts[4]),
                        "protein_g": float(parts[5]),
                    }
                )
            except ValueError:
                continue
    return rows


def _stat(vals: list[float]) -> str:
    if not vals:
        return "n=0"
    return (
        f"n={len(vals)} mean={statistics.mean(vals):.1f} "
        f"median={statistics.median(vals):.1f} "
        f"min={min(vals):.1f} max={max(vals):.1f}"
    )


def main() -> int:
    splits = N5K / "dish_ids" / "splits"
    print("=== Nutrition5K split counts ===")
    for name in (
        "rgb_train_ids.txt",
        "rgb_test_ids.txt",
        "depth_train_ids.txt",
        "depth_test_ids.txt",
    ):
        ids = count_ids(splits / name)
        print(f"  {name}: {len(ids)}")

    all_ids = count_ids(N5K / "dish_ids" / "dish_ids_all.txt")
    print(f"  dish_ids_all.txt: {len(all_ids)}")

    meta_rows: list[dict] = []
    for name in ("dish_metadata_cafe1.csv", "dish_metadata_cafe2.csv"):
        p = N5K / "metadata" / name
        if p.is_file():
            part = parse_dish_metadata(p)
            print(f"=== metadata {name}: {len(part)} dishes ===")
            meta_rows.extend(part)
    if meta_rows:
        # dedup by dish_id
        by_id = {r["dish_id"]: r for r in meta_rows}
        meta_rows = list(by_id.values())
        print(f"=== combined unique dishes: {len(meta_rows)} ===")
        for field in ("energy_kcal", "mass_g", "protein_g", "carb_g", "fat_g"):
            vals = [r[field] for r in meta_rows]
            print(f"  {field}: {_stat(vals)}")

    ing = N5K / "metadata" / "ingredients_metadata.csv"
    if ing.is_file():
        with ing.open("r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            n_ing = sum(1 for _ in reader)
        print(f"=== ingredients_metadata.csv rows: {n_ing} ===")

    imagery = N5K / "imagery" / "realsense_overhead"
    rgb_ids = {
        p.parent.name
        for p in imagery.glob("*/rgb.png")
        if p.is_file()
    }
    rgb_train = set(count_ids(splits / "rgb_train_ids.txt"))
    rgb_test = set(count_ids(splits / "rgb_test_ids.txt"))
    print(f"=== overhead rgb.png: {len(rgb_ids)} ===")
    print(f"  in official rgb_train: {len(rgb_ids & rgb_train)} / {len(rgb_train)}")
    print(f"  in official rgb_test: {len(rgb_ids & rgb_test)} / {len(rgb_test)}")
    print(f"  not in official rgb split: {len(rgb_ids - rgb_train - rgb_test)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
