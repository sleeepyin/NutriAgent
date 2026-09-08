"""Episode / action / tool schemas. JSON-serializable dataclasses."""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Literal

ACTIONS = (
    "STOP",
    "REQUEST_REFERENCE",
    "REQUEST_VIEW",
    "REQUEST_DEPTH",
    "ASK_SEMANTIC",
    "ABSTAIN",
)
COARSE_ACTIONS = ("PHYSICAL", "SEMANTIC", "STOP", "ABSTAIN")
PHYSICAL_SUB = ("REQUEST_REFERENCE", "REQUEST_VIEW", "REQUEST_DEPTH")
OUTPUTS = ("weight", "calories", "fat", "carbs", "protein")

PRIOR_COST = {
    "STOP": 0.0,
    "ASK_SEMANTIC": 1.0,
    "REQUEST_VIEW": 2.0,
    "REQUEST_REFERENCE": 3.0,
    "REQUEST_DEPTH": 3.0,
    "ABSTAIN": 1.0,
}

SplitName = Literal["base", "policy", "calibration", "test", "unused"]


@dataclass
class UncertaintyFactors:
    u_scale: float = 0.0
    u_geometry: float = 0.0
    u_density: float = 0.0
    u_domain: float = 0.0
    u_quality: float = 0.0

    def as_vec(self) -> list[float]:
        return [self.u_scale, self.u_geometry, self.u_density, self.u_domain, self.u_quality]


@dataclass
class ToolResult:
    name: str
    success: bool
    point: dict[str, float] = field(default_factory=dict)
    interval: dict[str, list[float]] = field(default_factory=dict)
    features: dict[str, float] = field(default_factory=dict)
    error: str | None = None
    oracle: bool = False
    version: str = "v1"

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class ActionDecision:
    action: str
    reason_code: str
    expected_gain: float
    estimated_cost: float
    remaining_budget: float
    allowed: bool
    values: dict[str, float] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class Episode:
    episode_id: str
    dataset: str
    group_id: str
    split: SplitName
    y_weight: float
    y_nutrients: list[float] | None = None
    morphology: str = "unknown"
    domain: str = "n5k"
    available: dict[str, bool] = field(default_factory=dict)
    states: dict[str, dict[str, Any]] = field(default_factory=dict)
    meta: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def default_available(dataset: str) -> dict[str, bool]:
    d = {a: False for a in ACTIONS}
    d["STOP"] = True
    d["ABSTAIN"] = True
    if dataset == "n5k":
        d["REQUEST_DEPTH"] = True
        d["ASK_SEMANTIC"] = True
    elif dataset == "nvreal":
        d["REQUEST_VIEW"] = True
        d["ASK_SEMANTIC"] = True
    elif dataset == "ecustfd":
        d["REQUEST_REFERENCE"] = True
        d["REQUEST_VIEW"] = True
    elif dataset == "mf3d":
        d["ASK_SEMANTIC"] = True
        d["REQUEST_VIEW"] = True
    return d
