#!/usr/bin/env python3
"""CPU smoke for the DietAI24 and FoodAgent scaffolds.

Imports the new interfaces and runs the toy dry-runs. No GPU, no VLM
weights, no writes under results/.
"""

from __future__ import annotations

import hashlib
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from vdr.baselines.b1_rag_once import predict_from_hypotheses  # noqa: E402
from vdr.baselines.dietai24.config import (  # noqa: E402
    HARDENED_B1_ENERGY_MAE,
    HARDENED_B1_IS_FAITHFUL_DIETAI24,
    DietAI24Config,
    FaithfulDietAI24NotImplemented,
)
from vdr.baselines.dietai24.dry_run import (  # noqa: E402
    FixtureDietAI24,
    run_builtin_dietai24_dry_run,
    run_dietai24_protocol,
    run_empty_dietai24_dry_run,
)
from vdr.baselines.dietai24.interface import (  # noqa: E402
    DIETAI24_STAGES,
    CandidateCode,
    DietAI24DryCase,
    DietAI24Estimator,
    DietAI24Indexer,
    DietAI24Retriever,
    ImageRef as DietImageRef,
    NutrientSpec,
    PortionSpec,
    scale_per_100g,
)
from vdr.baselines.foodagent.config import (  # noqa: E402
    DisallowedRoutingLabel,
    FaithfulFoodAgentNotImplemented,
    FoodAgentConfig,
)
from vdr.baselines.foodagent.dry_run import (  # noqa: E402
    BUILTIN_ELEMENTS,
    FixtureFoodAgent,
    run_builtin_foodagent_dry_run,
    run_foodagent_approx,
)
from vdr.baselines.foodagent.interface import (  # noqa: E402
    ExpertId,
    ExpertRouter,
    FoodElement,
    ImageRef as FoodImageRef,
    RouteDecision,
    mass_from_volume,
    route_element,
)
from vdr.baselines.foodagent.approx import estimate_element  # noqa: E402
from vdr.controller.features import FEATURE_DIM, build_features  # noqa: E402
from vdr.controller.state_features import CLOSED_LOOP_READY  # noqa: E402
from vdr.schemas import VDR_OUTPUT_SCHEMA_PATH, load_schema, validate_instance  # noqa: E402
from vdr.tools.volume_estimate import ENABLED_BY_DEFAULT  # noqa: E402

FROZEN = (
    "results/n5k_rgb_test_overhead_b0_b1.json",
    "results/n5k_rgb_test_overhead_b0_b1.jsonl",
    "results/n5k_subset_b0_b1.json",
    "results/n5k_subset_b0_b1.jsonl",
    "results/n5k_subset_b0_b1_n50_seed42.json",
    "results/n5k_subset_b0_b1_n50_seed42.jsonl",
    "results/weak_steps/rgb_train_fit_all.jsonl",
    "results/weak_steps/rgb_train_holdout_all.jsonl",
)
FROZEN_PREFIXES = {
    "results/n5k_rgb_test_overhead_b0_b1.jsonl": "a4dfdd82e3a595f4",
    "results/weak_steps/rgb_train_fit_all.jsonl": "ed47f9239bfecb3c",
}
FORBIDDEN_IMPORT_TOKENS = (
    "portion_grid",
    "volume_estimate",
    "b1_rag_once",
    "qwen25",
    "torch",
)


def check(name: str, ok: bool, detail: str = "") -> bool:
    print(f"  {'OK' if ok else 'FAIL'} {name}" + (f" — {detail}" if detail else ""))
    return ok


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def frozen_hashes() -> dict[str, str]:
    out: dict[str, str] = {}
    for rel in FROZEN:
        path = ROOT / rel
        if path.is_file():
            out[rel] = _sha256(path)
    return out


def result_names() -> tuple[str, ...]:
    base = ROOT / "results"
    if not base.is_dir():
        return ()
    names: list[str] = []
    for path in base.rglob("*"):
        if not path.is_file():
            continue
        rel = path.relative_to(ROOT).as_posix()
        if rel.startswith("results/n5k_") or rel.startswith("results/weak_steps/"):
            names.append(rel)
    return tuple(sorted(names))


def _close(left: float, right: float) -> bool:
    return abs(left - right) < 1e-9


def test_dietai24() -> bool:
    ok = True
    fixture = FixtureDietAI24  # class, protocols checked on an instance
    client = fixture(
        DietAI24DryCase("x", (), (), (), (), ())
    )
    ok &= check("indexer protocol", isinstance(client, DietAI24Indexer))
    ok &= check("retriever protocol", isinstance(client, DietAI24Retriever))
    ok &= check("estimator protocol", isinstance(client, DietAI24Estimator))
    ok &= check("six published stages", DIETAI24_STAGES[0] == "index" and DIETAI24_STAGES[-1] == "aggregate_nutrients")
    ok &= check("hardened flag stays false", HARDENED_B1_IS_FAITHFUL_DIETAI24 is False)
    ok &= check("hardened mae is the historical 110.72", HARDENED_B1_ENERGY_MAE == 110.72)

    scaled = scale_per_100g(
        {"energy_kcal": 200.0, "protein_g": 10.0, "carb_g": 0.0, "fat_g": 5.0},
        50.0,
    )
    ok &= check("scale uses grams/100", scaled["energy_kcal"] == 100.0 and scaled["mass_g"] == 50.0)

    result = run_builtin_dietai24_dry_run()
    output = result.output
    validate_instance(output, load_schema(VDR_OUTPUT_SCHEMA_PATH))
    ok &= check("fidelity scaffold", output["meta"]["fidelity"] == "scaffold")
    ok &= check("not main table", output["meta"]["main_table_eligible"] is False)
    ok &= check("models not called", output["meta"]["live_models_called"] is False and output["meta"]["mllm_called"] is False)
    ok &= check("65 nutrients not run", output["meta"]["full_65_not_run"] is True)
    ok &= check("portion grid not used", output["meta"]["portion_grid_used"] is False)
    ok &= check("historical b1 is not this protocol", output["meta"]["hardened_b1_energy_mae_110_72_is_faithful_dietai24"] is False)
    ok &= check("duplicate codes dropped", result.trace["n_candidates_raw"] == 3 and result.trace["n_candidates"] == 2)
    energy = output["n"]["energy_kcal"]
    ok &= check("toy energy is fixture sum 146", _close(energy, 146.0), str(energy))
    ok &= check("toy mass is 200", _close(output["n"]["mass_g"], 200.0))
    ok &= check("toy energy is not 110.72", not _close(energy, 110.72))
    ok &= check("toy energy is not the literature anchor", not _close(energy, 47.7))
    ok &= check("s pass on a complete fixture", output["s"] == "pass")
    descriptors = output["tau"][1]["obs"]["portions"]
    ok &= check("portion step keeps descriptors", descriptors[0]["descriptor"] == "1 cup")

    empty = run_empty_dietai24_dry_run()
    ok &= check("empty fixture abstains", empty.output["s"] == "abstain")
    ok &= check("empty n is null", all(value is None for value in empty.output["n"].values()))

    raised = False
    try:
        DietAI24Config(fidelity="faithful")
    except FaithfulDietAI24NotImplemented:
        raised = True
    ok &= check("faithful DietAI24 config is rejected", raised)

    grid_case = DietAI24DryCase(
        description="grid",
        queries=("a",),
        candidates=(CandidateCode("90000003", "fixture", 1.0),),
        selected_codes=("90000003",),
        portions=(PortionSpec("90000003", "1 cup", 10.0, True),),
        nutrients=(
            NutrientSpec(
                "90000003",
                (("energy_kcal", 1.0), ("protein_g", 0.0), ("carb_g", 0.0), ("fat_g", 0.0)),
            ),
        ),
    )
    grid_rejected = False
    try:
        run_dietai24_protocol(
            FixtureDietAI24(grid_case),
            DietImageRef("scaffold_dietai24_grid", "fixture://no-pixels"),
        )
    except Exception as exc:
        grid_rejected = "portion_grid" in str(exc)
    ok &= check("portion_grid portion is rejected", grid_rejected)

    missing = DietAI24DryCase(
        description="missing",
        queries=("a",),
        candidates=(CandidateCode("90000009", "fixture", 1.0),),
        selected_codes=("90000008",),
        portions=(),
        nutrients=(),
    )
    missing_rejected = False
    try:
        run_dietai24_protocol(
            FixtureDietAI24(missing),
            DietImageRef("scaffold_dietai24_missing", "fixture://no-pixels"),
        )
    except Exception as exc:
        missing_rejected = "not in the retrieved set" in str(exc)
    ok &= check("code outside the retrieved set is rejected", missing_rejected)
    return ok


def test_foodagent() -> bool:
    ok = True
    client = FixtureFoodAgent(BUILTIN_ELEMENTS)
    ok &= check("router protocol", isinstance(client, ExpertRouter))
    bacon, eggs, berries = BUILTIN_ELEMENTS
    ok &= check("clear important routes to volume", route_element(bacon) is ExpertId.VOLUME_MFP3D)
    ok &= check("unclear important routes to retrieval", route_element(eggs) is ExpertId.RAG_NUTRITIONX)
    ok &= check("minor clear routes to direct", route_element(berries) is ExpertId.LLM_DIRECT)
    easy = FoodElement("garnish", "important", "clear", easy=True)
    ok &= check("easy wins over volume", route_element(easy) is ExpertId.LLM_DIRECT)
    heavier = FoodElement(
        bacon.name,
        bacon.importance,
        bacon.clarity,
        volume_cm3=999.0,
        density_g_per_cm3=1.0,
    )
    ok &= check("volume number does not change the route", route_element(heavier) is route_element(bacon))
    ok &= check("mass is volume times density", mass_from_volume(10.0, 2.0) == 20.0)

    tampered = False
    try:
        estimate_element(
            bacon,
            RouteDecision(bacon.name, ExpertId.RAG_NUTRITIONX, "tamper"),
            FoodAgentConfig(),
        )
    except Exception as exc:
        tampered = "attributes route to" in str(exc)
    ok &= check("a mismatched expert is rejected", tampered)

    result = run_builtin_foodagent_dry_run()
    output = result.output
    validate_instance(output, load_schema(VDR_OUTPUT_SCHEMA_PATH))
    routes = [row["expert"] for row in result.trace["routes"]]
    ok &= check(
        "three elements, three experts",
        routes == ["volume_mfp3d", "rag_nutritionx", "llm_direct"],
        ",".join(routes),
    )
    ok &= check("fidelity approx", output["meta"]["fidelity"] == "approx")
    ok &= check("routing is per element", output["meta"]["routing"] == "dynamic_per_element")
    ok &= check("foodagent not main table", output["meta"]["main_table_eligible"] is False)
    ok &= check("experts not called", output["meta"]["live_models_called"] is False)
    ok &= check("volume tool not called", output["meta"]["volume_tool_called"] is False)
    ok &= check("nutritionx not called", output["meta"]["nutritionx_called"] is False)
    ok &= check("no fndds codes bound", output["meta"]["fndds_codes_bound"] is False)
    energy = output["n"]["energy_kcal"]
    ok &= check("toy energy is fixture sum 225", _close(energy, 225.0), str(energy))
    ok &= check("toy protein is 17", _close(output["n"]["protein_g"], 17.0))
    ok &= check("toy mass is 125", _close(output["n"]["mass_g"], 125.0))
    actions = [step["action"] for step in output["tau"]]
    ok &= check(
        "tool steps then one STOP",
        actions == ["RE_DECOMPOSE", "RE_PORTION", "RETRIEVE", "STOP"],
        ",".join(actions),
    )
    ok &= check("direct branch is on STOP, not a fake tool", len(output["tau"][-1]["obs"]["llm_direct"]) == 1)

    empty = run_foodagent_approx(
        FixtureFoodAgent(()),
        FoodImageRef("scaffold_foodagent_empty", "fixture://no-pixels"),
    )
    ok &= check("empty foodagent abstains", empty.output["s"] == "abstain")
    ok &= check("empty foodagent n is null", all(value is None for value in empty.output["n"].values()))

    for label in ("Fixed-MoE", "fixed_moe", "fixed moe"):
        rejected = False
        try:
            FoodAgentConfig(fidelity=label)
        except DisallowedRoutingLabel:
            rejected = True
        ok &= check(f"label {label!r} is rejected", rejected)

    faithful = False
    try:
        FoodAgentConfig(fidelity="faithful")
    except FaithfulFoodAgentNotImplemented:
        faithful = True
    ok &= check("faithful FoodAgent config is rejected", faithful)

    volume_on = False
    try:
        FoodAgentConfig(volume_tool_enabled=True)
    except FaithfulFoodAgentNotImplemented:
        volume_on = True
    ok &= check("volume tool cannot be enabled here", volume_on)
    return ok


def test_neighbors_untouched() -> bool:
    ok = True
    ok &= check("closed loop stays off", CLOSED_LOOP_READY is False)
    ok &= check("volume default stays off", ENABLED_BY_DEFAULT is False)
    ok &= check("open-loop features still exist", FEATURE_DIM == 31 and callable(build_features))
    ok &= check("hardened predict still exists", callable(predict_from_hypotheses))
    for rel in (
        "vdr/controller/features.py",
        "vdr/controller/policy_bc.py",
        "vdr/controller/policy_mlp.py",
        "vdr/controller/dagger.py",
    ):
        text = (ROOT / rel).read_text(encoding="utf-8")
        ok &= check(f"{rel} still uses build_features", "build_features" in text)
    packages = (ROOT / "vdr" / "baselines" / "dietai24", ROOT / "vdr" / "baselines" / "foodagent")
    for package in packages:
        hits: list[str] = []
        for path in sorted(package.glob("*.py")):
            for line in path.read_text(encoding="utf-8").splitlines():
                stripped = line.strip()
                if not (stripped.startswith("import ") or stripped.startswith("from ")):
                    continue
                found = [token for token in FORBIDDEN_IMPORT_TOKENS if token in stripped]
                if found:
                    hits.append(f"{path.relative_to(ROOT)}: {stripped}")
        ok &= check(f"{package.name} does not import the hardened path or live models", not hits, "; ".join(hits))
    return ok


def test_cli() -> bool:
    ok = True
    runs = (
        ("dietai24", [sys.executable, "-m", "vdr.baselines.dietai24.dry_run"], "fidelity=scaffold"),
        ("dietai24 empty", [sys.executable, "-m", "vdr.baselines.dietai24.dry_run", "--empty"], "s=abstain"),
        ("foodagent", [sys.executable, "-m", "vdr.baselines.foodagent.dry_run"], "fidelity=approx"),
    )
    for name, cmd, needle in runs:
        proc = subprocess.run(cmd, cwd=ROOT, check=False, capture_output=True, text=True)
        text = proc.stdout
        ok &= check(
            f"cli {name} exit 0",
            proc.returncode == 0 and proc.stderr.strip() == "",
            proc.stderr.strip(),
        )
        ok &= check(f"cli {name} marker", needle in text and "wrote_results=false" in text, text.strip())
    return ok


def main() -> int:
    print("b1/b2 scaffold smoke (no GPU, no live models)")
    if "torch" in sys.modules:
        print("  FAIL torch was already imported")
        return 1
    before = frozen_hashes()
    names_before = result_names()
    for rel, prefix in FROZEN_PREFIXES.items():
        digest = before.get(rel, "")
        if not digest.startswith(prefix):
            print(f"  FAIL frozen prefix {rel}: {digest[:16]}")
            return 1
        print(f"  OK frozen prefix {rel}")
    ok = True
    ok &= test_dietai24()
    ok &= test_foodagent()
    ok &= test_neighbors_untouched()
    ok &= test_cli()
    after = frozen_hashes()
    ok &= check("frozen result hashes unchanged", before == after)
    ok &= check("n5k and weak_steps file list unchanged", names_before == result_names())
    ok &= check("torch was not imported", "torch" not in sys.modules)
    if not ok:
        print("FAILED")
        return 1
    print("OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
