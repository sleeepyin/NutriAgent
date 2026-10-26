"""FNDDS per-100g lookup and portion scaling."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from .fndds_catalog import foods_by_code

_MACRO_KEYS = ("energy_kcal", "protein_g", "carb_g", "fat_g")

# Frozen public nutrition vector. water_g and other FNDDS fields are not part of y.n.
FROZEN_N_KEYS = ("energy_kcal", "protein_g", "carb_g", "fat_g", "mass_g")


def freeze_nutrition_vector(nutri: dict[str, Any] | None) -> dict[str, float | None]:
    """Keep only the schema fields of ``n``. Extra keys such as water_g are dropped."""
    src = nutri or {}
    out: dict[str, float | None] = {}
    for key in FROZEN_N_KEYS:
        value = src.get(key)
        if value is None:
            out[key] = None
        else:
            out[key] = float(value)
    return out


def lookup_per_100g(
    code: str,
    catalog_path: str | Path | None = None,
    **kwargs: Any,
) -> dict[str, float]:
    """Return energy/protein/carb/fat (and optional micros) per 100g for a food code."""
    table = foods_by_code(catalog_path)
    if code not in table:
        raise KeyError(f"Unknown food code: {code}")
    per = table[code].get("per_100g") or {}
    out: dict[str, float] = {}
    for k in _MACRO_KEYS:
        if k in per:
            out[k] = float(per[k])
    for k, v in per.items():
        if k not in out:
            out[k] = float(v)
    return out


def scale_by_portion(per_100g: dict[str, float], portion_g: float) -> dict[str, float]:
    """Scale a per-100g vector by portion_g / 100."""
    factor = float(portion_g) / 100.0
    return {k: float(v) * factor for k, v in per_100g.items()}


def aggregate_dish(components: list[dict[str, float]]) -> dict[str, float]:
    """Sum component nutrition dicts key-wise."""
    out: dict[str, float] = {}
    for comp in components:
        for k, v in comp.items():
            out[k] = out.get(k, 0.0) + float(v)
    return out


def lookup_and_scale(
    code: str,
    portion_g: float,
    catalog_path: str | Path | None = None,
) -> dict[str, float]:
    """Convenience: lookup per-100g then scale by portion."""
    return scale_by_portion(lookup_per_100g(code, catalog_path=catalog_path), portion_g)
