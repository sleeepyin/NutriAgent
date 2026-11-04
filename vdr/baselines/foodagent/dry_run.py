"""CPU dry-run of the FoodAgent approx path.

Decomposes a fixture, routes each element on its own, applies the published
conversion formulas, and emits schema-valid VDR JSON. Nothing is written
under ``results/``. The volume tool is not imported.
"""

from __future__ import annotations

import sys
from dataclasses import dataclass
from typing import Any, Sequence

from vdr.baselines.foodagent.approx import estimate_element, route_elements
from vdr.baselines.foodagent.config import (
    MISSING_FOR_FAITHFUL_RUN,
    PAPER,
    ROUTER_MODEL,
    FoodAgentConfig,
    FoodAgentScaffoldError,
)
from vdr.baselines.foodagent.interface import (
    PUBLIC_N_KEYS,
    ComponentEstimate,
    ElementDecomposer,
    ExpertEstimator,
    ExpertId,
    ExpertRouter,
    FoodElement,
    ImageRef,
    RouteDecision,
    sum_nutrition,
)
from vdr.schemas import VDR_OUTPUT_SCHEMA_PATH, load_schema, validate_instance

BUILTIN_IMAGE = ImageRef(dish_id="scaffold_foodagent_toy", uri="fixture://no-pixels")

BUILTIN_ELEMENTS = (
    FoodElement(
        name="bacon",
        importance="important",
        clarity="clear",
        volume_cm3=10.0,
        density_g_per_cm3=2.0,
        per_100g=(
            ("energy_kcal", 250.0),
            ("protein_g", 20.0),
            ("carb_g", 0.0),
            ("fat_g", 10.0),
        ),
    ),
    FoodElement(
        name="scrambled eggs",
        importance="important",
        clarity="not_clear",
        unit_name="1 cup",
        unit_count=2.0,
        grams_per_unit=40.0,
        per_unit=(
            ("energy_kcal", 80.0),
            ("protein_g", 6.0),
            ("carb_g", 1.0),
            ("fat_g", 5.0),
        ),
    ),
    FoodElement(
        name="blueberries",
        importance="minor",
        clarity="clear",
        direct_nutrition=(
            ("energy_kcal", 15.0),
            ("protein_g", 1.0),
            ("carb_g", 3.0),
            ("fat_g", 0.0),
            ("mass_g", 25.0),
        ),
    ),
)


class FixtureFoodAgent:
    """Decomposer that returns the supplied elements. The router and estimator are shared."""

    def __init__(self, elements: tuple[FoodElement, ...]) -> None:
        self.elements = elements

    def decompose(self, image: ImageRef, config: FoodAgentConfig) -> tuple[FoodElement, ...]:
        del image, config
        return self.elements

    def route(
        self,
        elements: Sequence[FoodElement],
        config: FoodAgentConfig,
    ) -> tuple[RouteDecision, ...]:
        return route_elements(elements, config)

    def estimate(
        self,
        element: FoodElement,
        decision: RouteDecision,
        config: FoodAgentConfig,
    ) -> ComponentEstimate:
        return estimate_element(element, decision, config)


@dataclass
class DryRunResult:
    output: dict[str, Any]
    trace: dict[str, Any]


def _null_n() -> dict[str, None]:
    return {key: None for key in PUBLIC_N_KEYS}


def _nutrition_map(estimate: ComponentEstimate) -> dict[str, float]:
    return {key: float(value) for key, value in estimate.nutrition}


def run_foodagent_approx(
    client: ElementDecomposer | ExpertRouter | ExpertEstimator,
    image: ImageRef,
    config: FoodAgentConfig | None = None,
) -> DryRunResult:
    """Route each element, estimate from fixtures, sum, and validate."""

    config = config or FoodAgentConfig()
    if not isinstance(client, (ElementDecomposer, ExpertRouter, ExpertEstimator)):
        raise FoodAgentScaffoldError("client does not implement the FoodAgent protocols")

    elements = client.decompose(image, config)
    decisions = client.route(elements, config)
    if len(decisions) != len(elements):
        raise FoodAgentScaffoldError("router must return one decision per element")

    estimates: list[ComponentEstimate] = []
    for element, decision in zip(elements, decisions):
        estimate = client.estimate(element, decision, config)
        if estimate.live or estimate.expert != decision.expert:
            raise FoodAgentScaffoldError("approx estimates stay offline and keep their route")
        estimates.append(estimate)

    routes = [
        {"name": decision.name, "expert": decision.expert.value, "source": decision.source}
        for decision in decisions
    ]
    experts_used = sorted({decision.expert.value for decision in decisions})
    trace = {
        "routing": config.routing,
        "fidelity": config.fidelity,
        "live_models_called": False,
        "routes": routes,
        "n_experts_used": len(experts_used),
        "experts_used": experts_used,
        "volume_tool_called": False,
        "nutritionx_called": False,
    }
    if not estimates:
        output = _abstain_output(image, config, trace)
    else:
        total = sum_nutrition([_nutrition_map(estimate) for estimate in estimates])
        output = _pass_output(image, config, trace, estimates, total)
    validate_instance(output, load_schema(VDR_OUTPUT_SCHEMA_PATH))
    return DryRunResult(output=output, trace=trace)


def _meta(config: FoodAgentConfig, trace: dict[str, Any]) -> dict[str, Any]:
    return {
        "baseline": config.baseline_id,
        "fidelity": "approx",
        "routing": "dynamic_per_element",
        "main_table_eligible": False,
        "live_models_called": False,
        "paper": PAPER,
        "router_model": ROUTER_MODEL,
        "router_actually_used": "fixture_attributes_not_qwen2.5_omni",
        "volume_tool_called": False,
        "nutritionx_called": False,
        "mfp3d_called": False,
        "missing_for_faithful_run": list(MISSING_FOR_FAITHFUL_RUN),
        "routes": trace["routes"],
        "n_experts_used": trace["n_experts_used"],
        "n_source": "toy_fixture_formula",
        "fndds_codes_bound": False,
        "wrote_results": False,
    }


def _pass_output(
    image: ImageRef,
    config: FoodAgentConfig,
    trace: dict[str, Any],
    estimates: Sequence[ComponentEstimate],
    total: dict[str, float],
) -> dict[str, Any]:
    nodes: list[dict[str, Any]] = [{"id": "dish", "type": "dish", "label": image.dish_id}]
    edges: list[dict[str, Any]] = []
    tau: list[dict[str, Any]] = [
        {
            "t": 0,
            "action": "RE_DECOMPOSE",
            "obs": {
                "stage": "decompose",
                "live": False,
                "routes": trace["routes"],
            },
            "state_summary": "foodagent approx decompose; Omni was not called",
        }
    ]
    step = 1
    direct_details: list[dict[str, Any]] = []
    for index, estimate in enumerate(estimates):
        nutrition = _nutrition_map(estimate)
        hyp_id = f"h{index}"
        query_id = f"q{index}"
        code_id = f"c{index}"
        portion_id = f"p{index}"
        nutri_id = f"n{index}"
        nodes.extend(
            [
                {"id": hyp_id, "type": "hyp", "label": estimate.name},
                {"id": query_id, "type": "query", "label": estimate.name},
                {
                    "id": code_id,
                    "type": "food_code",
                    "label": estimate.expert.value,
                    "attrs": {
                        "code": None,
                        "catalog_hit": False,
                        "expert": estimate.expert.value,
                        "binding": "not_fndds",
                    },
                },
                {
                    "id": portion_id,
                    "type": "portion_g",
                    "label": f"{nutrition['mass_g']}g",
                    "attrs": {"portion_g": nutrition["mass_g"], "src": estimate.expert.value},
                },
                {
                    "id": nutri_id,
                    "type": "nutrition",
                    "label": estimate.name,
                    "attrs": nutrition,
                },
            ]
        )
        edges.extend(
            [
                {"src": code_id, "dst": query_id, "type": "code<-query"},
                {"src": code_id, "dst": hyp_id, "type": "code<-hyp"},
                {"src": portion_id, "dst": code_id, "type": "portion<-code"},
                {
                    "src": nutri_id,
                    "dst": code_id,
                    "type": "n<-code×portion",
                    "attrs": {"via_portion": portion_id, "portion_g": nutrition["mass_g"]},
                },
                {"src": "dish", "dst": nutri_id, "type": "n<-aggregate"},
            ]
        )
        if estimate.expert is ExpertId.LLM_DIRECT:
            direct_details.append(
                {
                    "name": estimate.name,
                    "expert": estimate.expert.value,
                    "live": False,
                    "detail": {key: value for key, value in estimate.detail},
                }
            )
            continue
        tau.append(
            {
                "t": step,
                "action": _action_for(estimate.expert),
                "obs": {
                    "stage": "estimate",
                    "name": estimate.name,
                    "expert": estimate.expert.value,
                    "live": False,
                    "detail": {key: value for key, value in estimate.detail},
                },
                "state_summary": (
                    f"approx {estimate.expert.value} for {estimate.name}; expert was not executed"
                ),
            }
        )
        step += 1
    tau.append(
        {
            "t": step,
            "action": "STOP",
            "obs": {
                "s": "pass",
                "n_source": "toy_fixture_formula",
                "routes": trace["routes"],
                "llm_direct": direct_details,
            },
            "state_summary": "approx dry-run emit; not a live FoodAgent estimate",
        }
    )
    return {
        "dish_id": image.dish_id,
        "n": {key: total[key] for key in PUBLIC_N_KEYS},
        "G": {"nodes": nodes, "edges": edges},
        "s": "pass",
        "tau": tau,
        "abstain_reason": None,
        "meta": _meta(config, trace),
    }


def _action_for(expert: ExpertId) -> str:
    """Map a live-tool expert onto the frozen action set.

    The direct branch has no tool call. Its fixture numbers ride on the
    final STOP observation instead of a second terminating action.
    """

    if expert is ExpertId.VOLUME_MFP3D:
        return "RE_PORTION"
    if expert is ExpertId.RAG_NUTRITIONX:
        return "RETRIEVE"
    raise FoodAgentScaffoldError(f"{expert.value} has no tool action")


def _abstain_output(image: ImageRef, config: FoodAgentConfig, trace: dict[str, Any]) -> dict[str, Any]:
    return {
        "dish_id": image.dish_id,
        "n": _null_n(),
        "G": {
            "nodes": [
                {"id": "dish", "type": "dish", "label": image.dish_id},
                {"id": "h0", "type": "hyp", "label": "no elements in approx fixture"},
            ],
            "edges": [{"src": "dish", "dst": "h0", "type": "abstain_from"}],
        },
        "s": "abstain",
        "tau": [
            {
                "t": 0,
                "action": "RE_DECOMPOSE",
                "obs": {"stage": "decompose", "live": False, "n_elements": 0},
                "state_summary": "approx decompose found no elements",
            },
            {
                "t": 1,
                "action": "ABSTAIN",
                "obs": {"s": "abstain", "live": False},
                "state_summary": "scaffold abstain because the fixture listed no elements",
            },
        ],
        "abstain_reason": "approx fixture listed no elements; experts were not called",
        "meta": _meta(config, trace),
    }


def run_builtin_foodagent_dry_run() -> DryRunResult:
    return run_foodagent_approx(FixtureFoodAgent(BUILTIN_ELEMENTS), BUILTIN_IMAGE)


def main(argv: Sequence[str] | None = None) -> int:
    args = list(sys.argv[1:] if argv is None else argv)
    if args:
        print("usage: python -m vdr.baselines.foodagent.dry_run", file=sys.stderr)
        return 2
    result = run_builtin_foodagent_dry_run()
    experts = ",".join(route["expert"] for route in result.trace["routes"])
    print(
        "foodagent approx dry-run "
        f"fidelity={result.output['meta']['fidelity']} "
        f"routing={result.output['meta']['routing']} "
        f"experts={experts} "
        f"toy_fixture_energy_kcal={result.output['n']['energy_kcal']} "
        "wrote_results=false"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
