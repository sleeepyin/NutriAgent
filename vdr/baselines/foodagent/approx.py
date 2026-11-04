"""Approx FoodAgent experts.

Routing uses ``route_element`` on fixture attributes. Estimates apply the
published mass and unit formulas to fixture numbers. No expert process starts.
"""

from __future__ import annotations

from typing import Sequence

from vdr.baselines.foodagent.config import FoodAgentConfig, FoodAgentScaffoldError
from vdr.baselines.foodagent.interface import (
    ComponentEstimate,
    ExpertId,
    FoodElement,
    RouteDecision,
    freeze_pairs,
    mass_from_volume,
    nutrition_from_direct,
    nutrition_from_per_100g,
    nutrition_from_units,
    route_element,
)


def route_elements(
    elements: Sequence[FoodElement],
    config: FoodAgentConfig,
) -> tuple[RouteDecision, ...]:
    """One decision per element, in order. There is no dish-level expert field."""

    del config
    return tuple(
        RouteDecision(name=element.name, expert=route_element(element), source="fixture_attributes")
        for element in elements
    )


def estimate_element(
    element: FoodElement,
    decision: RouteDecision,
    config: FoodAgentConfig,
) -> ComponentEstimate:
    """Dispatch on the per-element decision. A mismatched expert is rejected."""

    del config
    expected = route_element(element)
    if decision.expert != expected or decision.name != element.name:
        raise FoodAgentScaffoldError(
            f"{element.name} was assigned {decision.expert.value}, "
            f"but its attributes route to {expected.value}"
        )
    if decision.expert is ExpertId.VOLUME_MFP3D:
        return _volume(element)
    if decision.expert is ExpertId.RAG_NUTRITIONX:
        return _retrieval(element)
    if decision.expert is ExpertId.LLM_DIRECT:
        return _direct(element)
    raise FoodAgentScaffoldError(f"unknown expert {decision.expert!r}")


def _volume(element: FoodElement) -> ComponentEstimate:
    if element.volume_cm3 is None or element.density_g_per_cm3 is None or element.per_100g is None:
        raise FoodAgentScaffoldError(f"{element.name} is missing volume fixture fields")
    mass = mass_from_volume(element.volume_cm3, element.density_g_per_cm3)
    per_100g = {key: value for key, value in element.per_100g}
    nutrition = nutrition_from_per_100g(per_100g, mass)
    return ComponentEstimate(
        name=element.name,
        expert=ExpertId.VOLUME_MFP3D,
        live=False,
        nutrition=freeze_pairs(nutrition),
        detail=(
            ("volume_cm3", element.volume_cm3),
            ("density_g_per_cm3", element.density_g_per_cm3),
            ("mass_formula", "volume_cm3*density_g_per_cm3"),
            ("mfp3d_called", False),
            ("volume_tool_called", False),
        ),
    )


def _retrieval(element: FoodElement) -> ComponentEstimate:
    if (
        element.unit_name is None
        or element.unit_count is None
        or element.grams_per_unit is None
        or element.per_unit is None
    ):
        raise FoodAgentScaffoldError(f"{element.name} is missing NutritionX fixture fields")
    per_unit = {key: value for key, value in element.per_unit}
    nutrition = nutrition_from_units(per_unit, element.unit_count, element.grams_per_unit)
    return ComponentEstimate(
        name=element.name,
        expert=ExpertId.RAG_NUTRITIONX,
        live=False,
        nutrition=freeze_pairs(nutrition),
        detail=(
            ("unit_name", element.unit_name),
            ("unit_count", element.unit_count),
            ("grams_per_unit", element.grams_per_unit),
            ("database", "NutritionX"),
            ("nutritionx_called", False),
        ),
    )


def _direct(element: FoodElement) -> ComponentEstimate:
    if element.direct_nutrition is None:
        raise FoodAgentScaffoldError(f"{element.name} is missing a direct nutrition fixture")
    direct = {key: value for key, value in element.direct_nutrition}
    nutrition = nutrition_from_direct(direct)
    return ComponentEstimate(
        name=element.name,
        expert=ExpertId.LLM_DIRECT,
        live=False,
        nutrition=freeze_pairs(nutrition),
        detail=(
            ("source", "fixture_stand_in_for_llm"),
            ("llm_called", False),
        ),
    )
