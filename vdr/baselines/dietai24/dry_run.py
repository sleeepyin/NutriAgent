"""CPU dry-run of the DietAI24 stage order.

The fixture client returns canned stage payloads and sets every live flag
false. The runner still dedupes codes, checks that inferred codes were
retrieved, rejects a portion grid, and applies ``v * grams / 100``.

No GPT call, no embedding index, no FNDDS file read, no write under ``results/``.
"""

from __future__ import annotations

import sys
from dataclasses import dataclass
from typing import Any, Sequence

from vdr.baselines.dietai24.config import (
    HARDENED_B1_IS_FAITHFUL_DIETAI24,
    HARDENED_B1_MODULE,
    NUTRIENT_COUNT,
    OFFICIAL_CODE,
    OFFICIAL_ENTRYPOINTS,
    PAPER,
    PUBLIC_N_KEYS,
    DietAI24Config,
    DietAI24ScaffoldError,
    FaithfulDietAI24NotImplemented,
)
from vdr.baselines.dietai24.interface import (
    DIETAI24_STAGES,
    CandidateCode,
    DietAI24DryCase,
    DietAI24Estimator,
    DietAI24Indexer,
    DietAI24Retriever,
    ImageRef,
    IndexManifest,
    NutrientSpec,
    PortionSpec,
    QueryExpansion,
    dedupe_candidates,
    nutrient_map,
    scale_per_100g,
    sum_contributions,
)
from vdr.schemas import VDR_OUTPUT_SCHEMA_PATH, load_schema, validate_instance

BUILTIN_IMAGE = ImageRef(dish_id="scaffold_dietai24_toy", uri="fixture://no-pixels")

BUILTIN_CASE = DietAI24DryCase(
    description="toy rice and cucumber, no pixels",
    queries=("cooked white rice", "raw cucumber", "cooked white rice"),
    candidates=(
        CandidateCode("90000001", "fixture rice, not an FNDDS hit", 0.91),
        CandidateCode("90000001", "duplicate rice code, dropped", 0.10),
        CandidateCode("90000002", "fixture cucumber, not an FNDDS hit", 0.80),
    ),
    selected_codes=("90000001", "90000002"),
    portions=(
        PortionSpec("90000001", "1 cup", 100.0),
        PortionSpec("90000002", "10 slices", 100.0),
    ),
    nutrients=(
        NutrientSpec(
            "90000001",
            (
                ("energy_kcal", 130.0),
                ("protein_g", 3.0),
                ("carb_g", 28.0),
                ("fat_g", 0.0),
            ),
        ),
        NutrientSpec(
            "90000002",
            (
                ("energy_kcal", 16.0),
                ("protein_g", 1.0),
                ("carb_g", 4.0),
                ("fat_g", 0.0),
            ),
        ),
    ),
)

EMPTY_CASE = DietAI24DryCase(
    description="fixture with no recognized foods",
    queries=(),
    candidates=(),
    selected_codes=(),
    portions=(),
    nutrients=(),
)


class FixtureDietAI24:
    """Protocol client backed by a ``DietAI24DryCase``. Models are not called."""

    def __init__(self, case: DietAI24DryCase) -> None:
        self.case = case

    def index(self, config: DietAI24Config) -> IndexManifest:
        del config
        return IndexManifest(
            release="2019-2020",
            expected_food_codes=5624,
            embedding_model="text-embedding-3-large",
            executed=False,
            reason="dry-run does not embed the 5624 FNDDS descriptions",
        )

    def expand_queries(self, image: ImageRef, config: DietAI24Config) -> QueryExpansion:
        del image, config
        return QueryExpansion(
            description=self.case.description,
            queries=self.case.queries,
            live=False,
        )

    def retrieve(
        self,
        expansion: QueryExpansion,
        index: IndexManifest,
        config: DietAI24Config,
    ) -> tuple[CandidateCode, ...]:
        del expansion, index, config
        return self.case.candidates

    def infer_codes(
        self,
        image: ImageRef,
        candidates: Sequence[CandidateCode],
        config: DietAI24Config,
    ) -> tuple[str, ...]:
        del image, candidates, config
        return self.case.selected_codes

    def choose_portions(
        self,
        image: ImageRef,
        codes: Sequence[str],
        config: DietAI24Config,
    ) -> tuple[PortionSpec, ...]:
        del image, codes, config
        return self.case.portions

    def per_100g_rows(
        self,
        codes: Sequence[str],
        config: DietAI24Config,
    ) -> tuple[NutrientSpec, ...]:
        del codes, config
        return self.case.nutrients


@dataclass
class DryRunResult:
    output: dict[str, Any]
    trace: dict[str, Any]


def _null_n() -> dict[str, None]:
    return {key: None for key in PUBLIC_N_KEYS}


def _require_offline(manifest: IndexManifest, expansion: QueryExpansion) -> None:
    if manifest.executed or expansion.live:
        raise FaithfulDietAI24NotImplemented(
            "dry-run refuses a client that claims the index or the MLLM ran"
        )


def run_dietai24_protocol(
    client: DietAI24Indexer | DietAI24Retriever | DietAI24Estimator,
    image: ImageRef,
    config: DietAI24Config | None = None,
) -> DryRunResult:
    """Walk the six published stages and emit schema-valid VDR JSON.

    Aggregation always uses the published scaling formula. Empty recognition
    abstains. A client that claims a live model ran is rejected.
    """

    config = config or DietAI24Config()
    if not isinstance(client, (DietAI24Indexer, DietAI24Retriever, DietAI24Estimator)):
        raise DietAI24ScaffoldError("client does not implement the DietAI24 stage protocols")

    manifest = client.index(config)
    expansion = client.expand_queries(image, config)
    _require_offline(manifest, expansion)
    if manifest.release != config.fndds_release:
        raise DietAI24ScaffoldError("index release drifted from the pinned protocol")
    if manifest.expected_food_codes != config.expected_food_codes:
        raise DietAI24ScaffoldError("index code count drifted from 5624")

    raw_candidates = client.retrieve(expansion, manifest, config)
    candidates = dedupe_candidates(raw_candidates)
    selected = client.infer_codes(image, candidates, config)
    known = {candidate.code: candidate for candidate in candidates}
    for code in selected:
        if code not in known:
            raise DietAI24ScaffoldError(
                f"inferred code {code} was not in the retrieved set"
            )

    trace: dict[str, Any] = {
        "stages": list(DIETAI24_STAGES),
        "live_models_called": False,
        "index_executed": manifest.executed,
        "n_queries": len(expansion.queries),
        "n_candidates_raw": len(raw_candidates),
        "n_candidates": len(candidates),
        "n_selected": len(selected),
        "portion_grid_used": False,
        "nutrients_scaled_in_scaffold": len(PUBLIC_N_KEYS) - 1,
        "nutrient_count_protocol": NUTRIENT_COUNT,
    }

    if not selected:
        output = _abstain_output(image, config, trace, expansion, candidates)
        _validate(output)
        return DryRunResult(output=output, trace=trace)

    portions = {spec.code: spec for spec in client.choose_portions(image, selected, config)}
    rows = {spec.code: spec for spec in client.per_100g_rows(selected, config)}
    contributions: list[dict[str, float]] = []
    bound: list[dict[str, Any]] = []
    for code in selected:
        portion = portions.get(code)
        nutrient = rows.get(code)
        if portion is None or nutrient is None:
            raise DietAI24ScaffoldError(f"fixture missing portion or per-100 g row for {code}")
        if portion.from_portion_grid:
            raise DietAI24ScaffoldError("DietAI24 portions are descriptors, not portion_grid")
        if not portion.descriptor:
            raise DietAI24ScaffoldError("portion descriptor is required")
        scaled = scale_per_100g(nutrient_map(nutrient), portion.grams)
        contributions.append(scaled)
        bound.append(
            {
                "code": code,
                "description": known[code].description,
                "descriptor": portion.descriptor,
                "grams": portion.grams,
                "catalog_hit": False,
                "contribution": scaled,
            }
        )

    total = sum_contributions(contributions)
    trace["foods"] = bound
    output = _pass_output(image, config, trace, expansion, candidates, bound, total)
    _validate(output)
    return DryRunResult(output=output, trace=trace)


def _meta(config: DietAI24Config, trace: dict[str, Any]) -> dict[str, Any]:
    return {
        "baseline": config.baseline_id,
        "fidelity": config.fidelity,
        "main_table_eligible": False,
        "live_models_called": False,
        "paper": PAPER,
        "official_code": OFFICIAL_CODE,
        "official_entrypoints": list(OFFICIAL_ENTRYPOINTS),
        "protocol_stages": list(DIETAI24_STAGES),
        "fndds_release": config.fndds_release,
        "expected_food_codes": config.expected_food_codes,
        "nutrient_count_protocol": config.nutrient_count,
        "nutrients_scaled_in_scaffold": len(PUBLIC_N_KEYS) - 1,
        "full_65_not_run": True,
        "embedding_model": config.embedding_model,
        "mllm_family": config.mllm_family,
        "mllm_called": False,
        "temperature": config.temperature,
        "portion_source": "fixture_stand_in_for_fndds_portions_and_weights",
        "portion_grid_used": False,
        "hardened_b1_module": HARDENED_B1_MODULE,
        "hardened_b1_energy_mae_110_72_is_faithful_dietai24": HARDENED_B1_IS_FAITHFUL_DIETAI24,
        "n_source": "toy_fixture_formula",
        "index_executed": trace["index_executed"],
        "wrote_results": False,
    }


def _pass_output(
    image: ImageRef,
    config: DietAI24Config,
    trace: dict[str, Any],
    expansion: QueryExpansion,
    candidates: Sequence[CandidateCode],
    bound: list[dict[str, Any]],
    total: dict[str, float],
) -> dict[str, Any]:
    nodes: list[dict[str, Any]] = [{"id": "dish", "type": "dish", "label": image.dish_id}]
    edges: list[dict[str, Any]] = []
    for index, food in enumerate(bound):
        query_id = f"q{index}"
        hyp_id = f"h{index}"
        code_id = f"c{index}"
        portion_id = f"p{index}"
        nutri_id = f"n{index}"
        nodes.extend(
            [
                {"id": query_id, "type": "query", "label": food["description"]},
                {"id": hyp_id, "type": "hyp", "label": food["description"]},
                {
                    "id": code_id,
                    "type": "food_code",
                    "label": f"fixture:{food['code']}",
                    "attrs": {"code": food["code"], "catalog_hit": False},
                },
                {
                    "id": portion_id,
                    "type": "portion_g",
                    "label": f"{food['grams']}g",
                    "attrs": {
                        "portion_g": food["grams"],
                        "descriptor": food["descriptor"],
                        "src": "fndds_descriptor_fixture",
                    },
                },
                {
                    "id": nutri_id,
                    "type": "nutrition",
                    "label": f"{food['code']} scaled",
                    "attrs": food["contribution"],
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
                    "attrs": {"via_portion": portion_id, "portion_g": food["grams"]},
                },
                {"src": "dish", "dst": nutri_id, "type": "n<-aggregate"},
            ]
        )
    tau = [
        {
            "t": 0,
            "action": "RETRIEVE",
            "obs": {
                "stage": "retrieve_candidates",
                "live": False,
                "description": expansion.description,
                "n_candidates": len(candidates),
                "codes": [candidate.code for candidate in candidates],
            },
            "state_summary": "dietai24 dry-run retrieve; embeddings were not built",
        },
        {
            "t": 1,
            "action": "RE_PORTION",
            "obs": {
                "stage": "estimate_portions",
                "live": False,
                "portion_grid_used": False,
                "portions": [
                    {"code": food["code"], "descriptor": food["descriptor"], "grams": food["grams"]}
                    for food in bound
                ],
            },
            "state_summary": "dietai24 dry-run portion descriptors; portion_grid was not used",
        },
        {
            "t": 2,
            "action": "STOP",
            "obs": {"stage": "aggregate_nutrients", "s": "pass", "n_source": "toy_fixture_formula"},
            "state_summary": "scaffold dry-run emit; not a live DietAI24 estimate",
        },
    ]
    return {
        "dish_id": image.dish_id,
        "n": {key: total[key] for key in PUBLIC_N_KEYS},
        "G": {"nodes": nodes, "edges": edges},
        "s": "pass",
        "tau": tau,
        "abstain_reason": None,
        "meta": _meta(config, trace),
    }


def _abstain_output(
    image: ImageRef,
    config: DietAI24Config,
    trace: dict[str, Any],
    expansion: QueryExpansion,
    candidates: Sequence[CandidateCode],
) -> dict[str, Any]:
    del expansion
    return {
        "dish_id": image.dish_id,
        "n": _null_n(),
        "G": {
            "nodes": [
                {"id": "dish", "type": "dish", "label": image.dish_id},
                {"id": "h0", "type": "hyp", "label": "no food codes in dry-run fixture"},
            ],
            "edges": [{"src": "dish", "dst": "h0", "type": "abstain_from"}],
        },
        "s": "abstain",
        "tau": [
            {
                "t": 0,
                "action": "RETRIEVE",
                "obs": {
                    "stage": "retrieve_candidates",
                    "live": False,
                    "n_candidates": len(candidates),
                },
                "state_summary": "dietai24 dry-run retrieve returned no codes to keep",
            },
            {
                "t": 1,
                "action": "ABSTAIN",
                "obs": {"stage": "infer_food_codes", "s": "abstain", "live": False},
                "state_summary": "scaffold abstain because the fixture named no foods",
            },
        ],
        "abstain_reason": "dry-run fixture recognized no foods; models were not called",
        "meta": _meta(config, trace),
    }


def _validate(output: dict[str, Any]) -> None:
    validate_instance(output, load_schema(VDR_OUTPUT_SCHEMA_PATH))


def run_builtin_dietai24_dry_run() -> DryRunResult:
    return run_dietai24_protocol(FixtureDietAI24(BUILTIN_CASE), BUILTIN_IMAGE)


def run_empty_dietai24_dry_run() -> DryRunResult:
    image = ImageRef(dish_id="scaffold_dietai24_empty", uri="fixture://no-pixels")
    return run_dietai24_protocol(FixtureDietAI24(EMPTY_CASE), image)


def main(argv: Sequence[str] | None = None) -> int:
    args = list(sys.argv[1:] if argv is None else argv)
    if args not in ([], ["--empty"]):
        print("usage: python -m vdr.baselines.dietai24.dry_run [--empty]", file=sys.stderr)
        return 2
    result = run_empty_dietai24_dry_run() if args == ["--empty"] else run_builtin_dietai24_dry_run()
    energy = result.output["n"]["energy_kcal"]
    print(
        "dietai24 dry-run "
        f"fidelity={result.output['meta']['fidelity']} "
        f"main_table_eligible={result.output['meta']['main_table_eligible']} "
        f"s={result.output['s']} "
        f"toy_fixture_energy_kcal={energy} "
        "wrote_results=false"
    )
    if "--json" in args:
        json.dump(result.output, sys.stdout, indent=2)
        sys.stdout.write("\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
