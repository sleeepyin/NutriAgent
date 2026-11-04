"""FoodAgent dynamic-routing interfaces.

Published assignment (Zhang et al., BSN 2025), one expert per element:

- important and visually clear → monocular volume (MFP3D, with SAM and ZoeDepth)
- important but not clear → retrieval over NutritionX
- minor, or easy to estimate → the reasoning model's own numbers

``route_element`` is that decision table applied to attributes the caller
already filled in. It is not Qwen2.5-Omni. A run that uses it is ``approx``.

Tie-break owned by this scaffold, not by the paper's code: ``minor`` or
``easy`` wins over the volume and retrieval branches. The paper's model
emits one category and does not publish a collision rule.

Mass for the volume branch is the published ``volume × density``. Nutrition
for that branch scales a fixture per-100 g vector by the resulting mass.
The retrieval branch scales a fixture per-unit vector by the unit count.
The direct branch copies the fixture vector. None of those tables are NutritionX.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Mapping, Protocol, Sequence, runtime_checkable

from vdr.baselines.foodagent.config import FoodAgentConfig, FoodAgentScaffoldError

MACRO_KEYS = ("energy_kcal", "protein_g", "carb_g", "fat_g")
PUBLIC_N_KEYS = ("energy_kcal", "protein_g", "carb_g", "fat_g", "mass_g")


class ExpertId(str, Enum):
    VOLUME_MFP3D = "volume_mfp3d"
    RAG_NUTRITIONX = "rag_nutritionx"
    LLM_DIRECT = "llm_direct"


@dataclass(frozen=True)
class ImageRef:
    dish_id: str
    uri: str


@dataclass(frozen=True)
class FoodElement:
    """One decomposed element. Numeric fields are fixture fuel, not router inputs.

    The router reads ``importance``, ``clarity``, and ``easy`` only.
    """

    name: str
    importance: str
    clarity: str
    easy: bool = False
    volume_cm3: float | None = None
    density_g_per_cm3: float | None = None
    per_100g: tuple[tuple[str, float], ...] | None = None
    unit_name: str | None = None
    unit_count: float | None = None
    grams_per_unit: float | None = None
    per_unit: tuple[tuple[str, float], ...] | None = None
    direct_nutrition: tuple[tuple[str, float], ...] | None = None


@dataclass(frozen=True)
class RouteDecision:
    name: str
    expert: ExpertId
    source: str


@dataclass(frozen=True)
class ComponentEstimate:
    name: str
    expert: ExpertId
    live: bool
    nutrition: tuple[tuple[str, float], ...]
    detail: tuple[tuple[str, float | str | bool], ...]


def _require_macros(values: Mapping[str, float]) -> None:
    missing = [key for key in MACRO_KEYS if key not in values]
    if missing:
        raise FoodAgentScaffoldError(f"nutrient fixture missing {missing}")


def route_element(element: FoodElement) -> ExpertId:
    """Pick one expert from this element's attributes. Not a dish-level choice."""

    if element.importance not in {"important", "minor"}:
        raise FoodAgentScaffoldError(f"importance must be important or minor, got {element.importance!r}")
    if element.clarity not in {"clear", "not_clear"}:
        raise FoodAgentScaffoldError(f"clarity must be clear or not_clear, got {element.clarity!r}")
    if element.importance == "minor" or element.easy:
        return ExpertId.LLM_DIRECT
    if element.clarity == "clear":
        return ExpertId.VOLUME_MFP3D
    return ExpertId.RAG_NUTRITIONX


def mass_from_volume(volume_cm3: float, density_g_per_cm3: float) -> float:
    """Published category-1 mass: ``volume × density``."""

    if volume_cm3 < 0 or density_g_per_cm3 < 0:
        raise FoodAgentScaffoldError("volume and density must be >= 0")
    return float(volume_cm3) * float(density_g_per_cm3)


def nutrition_from_per_100g(per_100g: Mapping[str, float], grams: float) -> dict[str, float]:
    _require_macros(per_100g)
    if grams < 0:
        raise FoodAgentScaffoldError("grams must be >= 0")
    factor = float(grams) / 100.0
    scaled = {key: float(per_100g[key]) * factor for key in MACRO_KEYS}
    scaled["mass_g"] = float(grams)
    return scaled


def nutrition_from_units(
    per_unit: Mapping[str, float],
    unit_count: float,
    grams_per_unit: float,
) -> dict[str, float]:
    """Category-2 total: unit count times the reference unit, including grams."""

    _require_macros(per_unit)
    if unit_count < 0 or grams_per_unit < 0:
        raise FoodAgentScaffoldError("unit count and grams per unit must be >= 0")
    scaled = {key: float(per_unit[key]) * float(unit_count) for key in MACRO_KEYS}
    scaled["mass_g"] = float(grams_per_unit) * float(unit_count)
    return scaled


def nutrition_from_direct(direct: Mapping[str, float]) -> dict[str, float]:
    """Category-3: accept the fixture vector, which stands in for an LLM guess."""

    missing = [key for key in PUBLIC_N_KEYS if key not in direct]
    if missing:
        raise FoodAgentScaffoldError(f"direct nutrition missing {missing}")
    return {key: float(direct[key]) for key in PUBLIC_N_KEYS}


def sum_nutrition(rows: Sequence[Mapping[str, float]]) -> dict[str, float]:
    total = {key: 0.0 for key in PUBLIC_N_KEYS}
    if not rows:
        raise FoodAgentScaffoldError("nothing to sum")
    for row in rows:
        for key in PUBLIC_N_KEYS:
            total[key] += float(row[key])
    return total


def freeze_pairs(values: Mapping[str, float]) -> tuple[tuple[str, float], ...]:
    return tuple((key, float(values[key])) for key in PUBLIC_N_KEYS)


@runtime_checkable
class ElementDecomposer(Protocol):
    def decompose(self, image: ImageRef, config: FoodAgentConfig) -> tuple[FoodElement, ...]:
        """List elements. A faithful client is the Omni reasoning model."""


@runtime_checkable
class ExpertRouter(Protocol):
    def route(
        self,
        elements: Sequence[FoodElement],
        config: FoodAgentConfig,
    ) -> tuple[RouteDecision, ...]:
        """One decision per element. Decisions may name different experts."""


@runtime_checkable
class ExpertEstimator(Protocol):
    def estimate(
        self,
        element: FoodElement,
        decision: RouteDecision,
        config: FoodAgentConfig,
    ) -> ComponentEstimate:
        """Run the expert named by ``decision``. The approx client does not call it."""
