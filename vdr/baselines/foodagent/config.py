"""FoodAgent scaffold config.

The published system routes each food element to its own expert. This
landing cannot run that system: Qwen2.5-Omni, MFP3D, SAM, ZoeDepth, and
NutritionX are not in the repo, and the volume tool stays off. The only
legal fidelity flag is ``approx``.

A dish-level single expert is not a supported mode. Labels that collapse
the baseline into a fixed mixture are rejected.
"""

from __future__ import annotations

from dataclasses import dataclass


PAPER = "Zhang et al., BSN 2025, FoodAgent"
ROUTER_MODEL = "Qwen2.5-Omni-7B"
MISSING_FOR_FAITHFUL_RUN = (
    "Qwen2.5-Omni-7B",
    "MFP3D",
    "SAM",
    "ZoeDepth",
    "NutritionX",
)


class FoodAgentScaffoldError(RuntimeError):
    """The approx fixture or config broke the per-element routing contract."""


class FaithfulFoodAgentNotImplemented(FoodAgentScaffoldError):
    """A live FoodAgent eval is not in this landing."""


class DisallowedRoutingLabel(FoodAgentScaffoldError):
    """Raised when a label would present this baseline as a fixed mixture."""


def assert_routing_label_allowed(label: str) -> None:
    folded = label.lower().replace("_", "").replace("-", "").replace(" ", "")
    if "fixedmoe" in folded:
        raise DisallowedRoutingLabel(
            f"refusing routing label {label!r}; FoodAgent here is per-element "
            "dynamic routing with fidelity 'approx'"
        )


@dataclass(frozen=True)
class FoodAgentConfig:
    """Approx call shape. Live experts, the volume tool, and table claims stay off."""

    baseline_id: str = "b2_foodagent"
    fidelity: str = "approx"
    routing: str = "dynamic_per_element"
    allow_live: bool = False
    volume_tool_enabled: bool = False
    main_table_eligible: bool = False

    def __post_init__(self) -> None:
        assert_routing_label_allowed(self.baseline_id)
        assert_routing_label_allowed(self.fidelity)
        assert_routing_label_allowed(self.routing)
        if self.baseline_id != "b2_foodagent":
            raise FoodAgentScaffoldError(f"unexpected baseline id {self.baseline_id!r}")
        if self.fidelity != "approx":
            raise FaithfulFoodAgentNotImplemented(
                "FoodAgent fidelity must stay 'approx' until Omni, MFP3D, and "
                f"NutritionX actually run. Refusing {self.fidelity!r}."
            )
        if self.routing != "dynamic_per_element":
            raise FoodAgentScaffoldError(
                "routing is one expert per element, chosen from that element's attributes"
            )
        if self.allow_live or self.volume_tool_enabled or self.main_table_eligible:
            raise FaithfulFoodAgentNotImplemented(
                "live FoodAgent experts are not implemented; volume stays off"
            )
