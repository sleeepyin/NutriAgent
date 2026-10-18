"""Structured semantic answers -> versioned density / nutrient prior. No LLM guessing grams."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

ONTOLOGY = {
    "morphology": ("whole-solid", "cut-solid", "loose-pile", "compact-mixture", "liquid", "soup"),
    "container": ("none", "plate", "bowl", "cup", "package"),
    "preparation": ("raw", "boiled", "steamed", "fried", "baked", "mixed"),
    "hidden_liquid": ("none", "low", "high"),
    "added_fat": ("none", "low", "high", "unknown"),
}

FORBIDDEN_QUESTIONS = ("how many grams", "how many calories", "weight", "kcal")


@dataclass
class SemanticResult:
    success: bool
    food_code: str | None
    density_prior: float
    nutrients_per_g: dict[str, float]
    reason: str = ""


class SemanticDensityTool:
    def __init__(self, table: pd.DataFrame | None = None):
        self.table = table
        self.cat_stats: dict[str, dict] = {}
        if table is not None:
            self.fit(table)

    def fit(self, table: pd.DataFrame) -> None:
        g = table.copy()
        if "density" not in g.columns and {"weight_g", "volume_cm3"} <= set(g.columns):
            g["density"] = g["weight_g"] / g["volume_cm3"].replace(0, np.nan)
        self.cat_stats = {}
        if "cat" in g.columns:
            for cat, sub in g.groupby("cat"):
                self.cat_stats[str(cat)] = {
                    "density": float(sub["density"].median()) if "density" in sub else float("nan"),
                    "energy": float(sub["energy"].median()) if "energy" in sub.columns else float("nan"),
                    "fat": float(sub["fat"].median()) if "fat" in sub.columns else float("nan"),
                    "carb": float(sub["carb"].median()) if "carb" in sub.columns else float("nan"),
                    "prot": float(sub["prot"].median()) if "prot" in sub.columns else float("nan"),
                    "n": int(len(sub)),
                }

    def lookup(self, cat: str | None, global_density: float = 0.7) -> SemanticResult:
        if not cat or cat not in self.cat_stats:
            return SemanticResult(False, None, global_density, {}, "no_match")
        s = self.cat_stats[cat]
        dens = s["density"] if np.isfinite(s["density"]) else global_density
        nutr = {
            "calories": s["energy"],
            "fat": s["fat"],
            "carbs": s["carb"],
            "protein": s["prot"],
        }
        return SemanticResult(True, cat, float(dens), nutr, "category_prior")
