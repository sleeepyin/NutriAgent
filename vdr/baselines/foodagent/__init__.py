"""FoodAgent dynamic-routing scaffold.

The runnable path is fidelity ``approx``: fixture attributes, the published
per-element decision table, and fixture arithmetic. It does not call
Qwen2.5-Omni, MFP3D, or NutritionX. The CPU entrypoint is
``python -m vdr.baselines.foodagent.dry_run``.
"""

from vdr.baselines.foodagent.config import (
    MISSING_FOR_FAITHFUL_RUN,
    DisallowedRoutingLabel,
    FaithfulFoodAgentNotImplemented,
    FoodAgentConfig,
    FoodAgentScaffoldError,
)
from vdr.baselines.foodagent.interface import ExpertId, ExpertRouter, mass_from_volume, route_element

__all__ = [
    "DisallowedRoutingLabel",
    "ExpertId",
    "ExpertRouter",
    "FaithfulFoodAgentNotImplemented",
    "FoodAgentConfig",
    "FoodAgentScaffoldError",
    "MISSING_FOR_FAITHFUL_RUN",
    "mass_from_volume",
    "route_element",
]
