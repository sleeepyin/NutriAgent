#!/usr/bin/env python3
"""Score the frozen #53+#51 protocol on the external held-out list.

The selector, Q weights, terminal margin, mass checkpoint, and same-head
density swap are the ones already in the rollout. This script does not train,
does not change tau, and does not open P1. Cafe labels are read only in the
score phase, after the predictions exist.

    python scripts/heldout_eval.py --self-check
    python scripts/heldout_eval.py --phase mass
    python scripts/heldout_eval.py --phase rollout
    python scripts/heldout_eval.py --phase direct
    python scripts/heldout_eval.py --phase score
"""

from __future__ import annotations

import hashlib
import json
import os
import statistics
import sys
import time
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

# The process-wide PYTHONPATH on this machine shadows torch. Drop it before imports.
if os.environ.get("PYTHONPATH"):
    os.environ.pop("PYTHONPATH", None)
    os.execv(sys.executable, [sys.executable, *sys.argv])

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "1")

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import vdr.controller.features  # noqa: E402,F401
from vdr.data.n5k import overhead_rgb  # noqa: E402
from vdr.eval.fixrate import BUDGET_K, TOOL_ACTIONS, summarize_tool_counts  # noqa: E402

PLAN = ROOT / "results" / "heldout_plan"
OUT = ROOT / "results" / "heldout_eval"
Q_WEIGHTS = ROOT / "results" / "qphi" / "q_phi.npz"
Q_CALIBRATION = ROOT / "results" / "qphi" / "calibration.json"
FROZEN = (
    Q_WEIGHTS,
    Q_CALIBRATION,
    ROOT / "results" / "mass_estimator" / "train_meta.json",
    ROOT / "results" / "mass_estimator" / "weights-part-00.npz",
    ROOT / "results" / "mass_estimator" / "weights-part-01.npz",
    ROOT / "results" / "mass_estimator" / "weights-part-02.npz",
    ROOT / "results" / "p0d" / "same_budget_dishes.jsonl",
    ROOT / "results" / "coverage_early_stop" / "paired_scores.jsonl",
    ROOT / "results" / "coverage_early_stop" / "summary.json",
    ROOT / "results" / "food_id_density" / "summary.json",
    ROOT / "results" / "qphi_counterfactual" / "summary.json",
    ROOT / "results" / "food_id_semantic_audit" / "summary.json",
    ROOT / "results" / "weak_steps" / "rgb_train_fit_all.jsonl",
    ROOT / "results" / "n5k_rgb_test_overhead_b0_b1.jsonl",
    ROOT / "docs" / "STATUS.md",
    ROOT / "PLAN.md",
    PLAN / "heldout_dish_ids.txt",
    PLAN / "dev505_dish_ids.txt",
    PLAN / "fit_dish_ids.txt",
    PLAN / "proof.json",
)
PROTECTED_PARTS = frozenset(
    {
        "food_id_density",
        "coverage_early_stop",
        "qphi_counterfactual",
        "food_id_semantic_audit",
        "p0d",
        "audit",
        "qphi",
        "mass_estimator",
        "weak_steps",
        "closed_loop",
        "heldout_plan",
    }
)


class HeldoutEvalError(RuntimeError):
    """The held-out run cannot be scored without breaking a freeze."""


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def frozen_hashes() -> dict[str, str]:
    missing = [str(path.relative_to(ROOT)) for path in FROZEN if not path.is_file()]
    if missing:
        raise HeldoutEvalError("missing frozen file: " + ", ".join(missing))
    return {str(path.relative_to(ROOT)): sha256_file(path) for path in FROZEN}


def assert_unchanged(before: Mapping[str, str]) -> None:
    after = frozen_hashes()
    changed = sorted(name for name, digest in before.items() if after.get(name) != digest)
    if changed:
        raise HeldoutEvalError("frozen files changed: " + ", ".join(changed))


def assert_output(path: Path) -> None:
    resolved = path.resolve()
    allowed = OUT.resolve()
    if resolved != allowed and allowed not in resolved.parents:
        raise HeldoutEvalError(f"refusing to write outside {OUT}")
    if PROTECTED_PARTS.intersection(resolved.parts):
        raise HeldoutEvalError(f"refusing to write into a frozen directory: {path}")


def read_ids(path: Path) -> list[str]:
    return [line.strip() for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def script_module(name: str) -> Any:
    import importlib.util

    path = ROOT / "scripts" / f"{name}.py"
    spec = importlib.util.spec_from_file_location(f"heldout_{name}", path)
    if spec is None or spec.loader is None:
        raise HeldoutEvalError(f"cannot load {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def planned_ids(limit: int | None) -> list[str]:
    ids = script_module("heldout_plan").load_heldout_ids(PLAN)
    dev = set(read_ids(PLAN / "dev505_dish_ids.txt"))
    fit = set(read_ids(PLAN / "fit_dish_ids.txt"))
    if set(ids) & dev or set(ids) & fit:
        raise HeldoutEvalError("held-out list overlaps the 505 or the fit set")
    if limit is None:
        return ids
    if limit < 1:
        raise HeldoutEvalError("--limit must be >= 1")
    return ids[:limit]


def out_dir(limit: int | None) -> Path:
    path = OUT / "smoke" if limit is not None else OUT
    assert_output(path / "placeholder")
    path.mkdir(parents=True, exist_ok=True)
    return path


def load_jsonl_rows(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        return []
    rows = []
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def done_ids(path: Path) -> set[str]:
    return {str(row["dish_id"]) for row in load_jsonl_rows(path) if row.get("dish_id")}


def failed_ids(path: Path, phase: str) -> set[str]:
    return {
        str(row["dish_id"])
        for row in load_jsonl_rows(path)
        if row.get("phase") == phase and row.get("dish_id")
    }


def append_json(path: Path, rows: Iterable[Mapping[str, Any]]) -> None:
    assert_output(path)
    with path.open("a", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
        handle.flush()


def finite(value: Any) -> float | None:
    if isinstance(value, bool) or value is None:
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    if number != number or number in (float("inf"), float("-inf")):
        return None
    return number


def group_steps(path: Path) -> dict[str, list[dict[str, Any]]]:
    grouped: dict[str, list[dict[str, Any]]] = {}
    for row in load_jsonl_rows(path):
        grouped.setdefault(str(row["dish_id"]), []).append(row)
    for steps in grouped.values():
        steps.sort(key=lambda step: int(step["step_i"]))
    return grouped


def step_summary(values: Sequence[int]) -> dict[str, Any]:
    if not values:
        return {"n": 0, "mean": None, "median": None, "min": None, "max": None}
    ordered = sorted(int(value) for value in values)
    return {
        "n": len(ordered),
        "mean": sum(ordered) / len(ordered),
        "median": float(statistics.median(ordered)),
        "min": ordered[0],
        "max": ordered[-1],
    }


def mae(errors: Sequence[float]) -> float:
    if not errors:
        raise HeldoutEvalError("MAE asked for an empty set")
    return sum(abs(value) for value in errors) / len(errors)


def f4(value: float | None) -> str:
    if value is None:
        return ""
    return f"{value:.4f}"


def run_phase_loop(
    phase: str,
    ids: list[str],
    directory: Path,
    success_name: str,
    run_one: Any,
) -> None:
    success_path = directory / success_name
    failure_path = directory / "failures.jsonl"
    assert_output(success_path)
    finished = done_ids(success_path)
    skipped_fail = failed_ids(failure_path, phase)
    pending = [dish_id for dish_id in ids if dish_id not in finished and dish_id not in skipped_fail]
    print(
        f"{phase}: pending={len(pending)} done={len(finished)} "
        f"previous_failures={len(skipped_fail & set(ids))}",
        flush=True,
    )
    failures: list[str] = []
    for index, dish_id in enumerate(pending, start=1):
        started = time.perf_counter()
        try:
            rows = run_one(dish_id)
        except HeldoutEvalError:
            raise
        except (OSError, ValueError, RuntimeError, TimeoutError) as exc:
            if "out of memory" in str(exc).lower():
                raise
            failures.append(dish_id)
            print(f"[{phase} {index}/{len(pending)}] {dish_id} FAIL {exc}", flush=True)
            continue
        append_json(success_path, rows)
        print(
            f"[{phase} {index}/{len(pending)}] {dish_id} rows={len(rows)} "
            f"{time.perf_counter() - started:.1f}s",
            flush=True,
        )
    if failures:
        print(f"{phase}: retrying {len(failures)} once", flush=True)
    still: list[tuple[str, str]] = []
    for dish_id in failures:
        if dish_id in done_ids(success_path):
            continue
        try:
            rows = run_one(dish_id)
        except HeldoutEvalError:
            raise
        except (OSError, ValueError, RuntimeError, TimeoutError) as exc:
            if "out of memory" in str(exc).lower():
                raise
            still.append((dish_id, f"{type(exc).__name__}: {exc}"))
            print(f"[{phase} retry] {dish_id} FAIL {exc}", flush=True)
            continue
        append_json(success_path, rows)
        print(f"[{phase} retry] {dish_id} rows={len(rows)}", flush=True)
    for dish_id, error in still:
        append_json(failure_path, [{"phase": phase, "dish_id": dish_id, "error": error}])


def mass_phase(ids: list[str], directory: Path) -> None:
    from vdr.tools.mass_estimator import estimate_mass

    def run_one(dish_id: str) -> list[dict[str, Any]]:
        estimate = estimate_mass(overhead_rgb(dish_id))
        if estimate.get("rgb_only") is not True or estimate.get("depth_used") is not False:
            raise HeldoutEvalError(f"{dish_id}: mass estimate is not RGB-only")
        energy = finite((estimate.get("nutrients") or {}).get("energy_kcal"))
        mass = finite(estimate.get("mass_g"))
        if energy is None or mass is None:
            raise HeldoutEvalError(f"{dish_id}: mass-only prediction is not finite")
        return [
            {
                "dish_id": dish_id,
                "energy_kcal": energy,
                "mass_g": mass,
                "backbone": estimate.get("backbone"),
                "tool": "estimate_mass",
                "rgb_only": True,
                "depth_used": False,
            }
        ]

    run_phase_loop("mass", ids, directory, "mass_only.jsonl", run_one)


def rollout_phase(ids: list[str], directory: Path) -> None:
    from vdr.controller.on_policy import rollout_dish
    from vdr.controller.q_calibration import load_calibrated_policy
    from vdr.perception.qwen38 import server_ready
    from vdr.tools.mass_estimator import load_mass_model

    policy = load_calibrated_policy(Q_WEIGHTS, Q_CALIBRATION)
    calibration = json.loads(Q_CALIBRATION.read_text(encoding="utf-8"))
    tau = float(calibration["tau"])
    if float(policy.terminal_margin) != tau:
        raise HeldoutEvalError("loaded fence does not match calibration.json")
    weight_hash = sha256_file(Q_WEIGHTS)
    if not weight_hash.startswith(str(calibration["weights_sha256_prefix"])):
        raise HeldoutEvalError("Q weight hash does not match the calibration record")
    load_mass_model()
    vlm = server_ready()
    dev = set(read_ids(PLAN / "dev505_dish_ids.txt"))
    fit = set(read_ids(PLAN / "fit_dish_ids.txt"))

    def run_one(dish_id: str) -> list[dict[str, Any]]:
        if dish_id in dev or dish_id in fit:
            raise HeldoutEvalError(f"{dish_id} is in the development or fit set")
        result = rollout_dish(
            dish_id,
            policy,
            image_path=overhead_rgb(dish_id),
            allow_outside_rgb_test=True,
        )
        if not result.steps:
            raise HeldoutEvalError(f"{dish_id}: empty trajectory")
        return result.steps

    print(f"rollout vlm={vlm.get('model_alias')} tau={tau}", flush=True)
    run_phase_loop("rollout", ids, directory, "rollout.jsonl", run_one)
    if sha256_file(Q_WEIGHTS) != weight_hash:
        raise HeldoutEvalError("Q weights changed during rollout")


def direct_phase(ids: list[str], directory: Path) -> None:
    import torch

    free, _total = torch.cuda.mem_get_info()
    if "VDR_VLM_MAX_MEMORY" not in os.environ and free < 15 * (1024**3):
        cap = max(4, int(free / (1024**3)) - 2)
        os.environ["VDR_VLM_MAX_MEMORY"] = f"{cap}GiB"
        print(
            f"direct-vlm placement cap {os.environ['VDR_VLM_MAX_MEMORY']} "
            f"(free {free / (1024**3):.1f} GiB). Same bf16 checkpoint and prompt.",
            flush=True,
        )
    from vdr.baselines.b0_direct_vlm import predict

    def run_one(dish_id: str) -> list[dict[str, Any]]:
        image = overhead_rgb(dish_id)
        prediction = predict(str(image), dish_id=dish_id)
        nutri = prediction.get("n") if isinstance(prediction, Mapping) else None
        if not isinstance(nutri, Mapping):
            raise HeldoutEvalError(f"{dish_id}: direct prediction has no nutrition object")
        energy = finite(nutri.get("energy_kcal"))
        mass = finite(nutri.get("mass_g"))
        if energy is None or mass is None:
            raise HeldoutEvalError(f"{dish_id}: direct prediction is not finite")
        meta = prediction.get("meta") if isinstance(prediction.get("meta"), Mapping) else {}
        vlm_meta = meta.get("vlm") if isinstance(meta.get("vlm"), Mapping) else {}
        return [
            {
                "dish_id": dish_id,
                "energy_kcal": energy,
                "mass_g": mass,
                "protein_g": finite(nutri.get("protein_g")),
                "carb_g": finite(nutri.get("carb_g")),
                "fat_g": finite(nutri.get("fat_g")),
                "model_path": vlm_meta.get("model_path"),
                "placement_max_memory": os.environ.get("VDR_VLM_MAX_MEMORY"),
                "raw": vlm_meta.get("raw"),
            }
        ]

    run_phase_loop("direct", ids, directory, "direct_vlm.jsonl", run_one)


def _coverage_counts(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    n = len(rows)
    complete = sum(1 for row in rows if row["plate_complete"])
    return {
        "n": n,
        "complete": complete,
        "coverage": None if n == 0 else complete / n,
        "stop_unwritten": sum(1 for row in rows if row["stop_unwritten"]),
        "mass_writes": sum(int(row["n_mass_writes"]) for row in rows),
    }


def _method_block(
    name: str,
    rows: Sequence[Mapping[str, Any]],
    energy_key: str,
    mass_key: str,
) -> dict[str, Any]:
    energy = [float(row[energy_key]) - float(row["label_energy_kcal"]) for row in rows]
    mass = [float(row[mass_key]) - float(row["label_mass_g"]) for row in rows]
    return {
        "method": name,
        "n": len(rows),
        "energy_mae_kcal": mae(energy),
        "mass_mae_g": mae(mass),
    }


def score_phase(ids: list[str], directory: Path, *, allow_partial: bool) -> None:
    from vdr.eval.same_budget import SameBudgetRefused, audit_mass_portions

    plan = script_module("heldout_plan")
    commit_from_steps = script_module("coverage_early_stop").commit_from_steps
    labels = plan.cafe2_score_labels(ids)

    rollout_path = directory / "rollout.jsonl"
    direct_path = directory / "direct_vlm.jsonl"
    mass_path = directory / "mass_only.jsonl"
    failure_path = directory / "failures.jsonl"
    traces = group_steps(rollout_path)
    direct = {str(row["dish_id"]): row for row in load_jsonl_rows(direct_path)}
    mass_rows = {str(row["dish_id"]): row for row in load_jsonl_rows(mass_path)}
    failures = load_jsonl_rows(failure_path)
    failed = {phase: failed_ids(failure_path, phase) for phase in ("rollout", "direct", "mass")}
    attempted = {
        "rollout": set(traces) | failed["rollout"],
        "direct": set(direct) | failed["direct"],
        "mass": set(mass_rows) | failed["mass"],
    }
    missing = {phase: [dish_id for dish_id in ids if dish_id not in done] for phase, done in attempted.items()}
    if any(missing.values()) and not allow_partial:
        detail = {phase: values[:5] for phase, values in missing.items() if values}
        raise HeldoutEvalError(f"score refused; dishes not attempted: {detail}")

    per_dish: list[dict[str, Any]] = []
    for dish_id, steps in traces.items():
        if dish_id not in set(ids):
            raise HeldoutEvalError(f"{dish_id} is in the trajectory but not in the planned list")
        facts = commit_from_steps(steps)
        label = labels.get(dish_id)
        if label is None:
            raise HeldoutEvalError(f"{dish_id} has no cafe label")
        direct_row = direct.get(dish_id)
        mass_row = mass_rows.get(dish_id)
        aligned = 0
        for step in steps:
            obs = step.get("obs_t") if isinstance(step.get("obs_t"), Mapping) else {}
            info = obs.get("density_align") if isinstance(obs.get("density_align"), Mapping) else {}
            if info.get("applied") is True:
                aligned += 1
        per_dish.append(
            {
                "dish_id": dish_id,
                "label_energy_kcal": float(label["energy_kcal"]),
                "label_mass_g": float(label["mass_g"]),
                "ours_energy_kcal": facts["energy_kcal"],
                "ours_mass_g": facts["mass_g"],
                "direct_energy_kcal": None if direct_row is None else direct_row.get("energy_kcal"),
                "direct_mass_g": None if direct_row is None else direct_row.get("mass_g"),
                "mass_only_energy_kcal": None if mass_row is None else mass_row.get("energy_kcal"),
                "mass_only_mass_g": None if mass_row is None else mass_row.get("mass_g"),
                "decision_action": facts["decision_action"],
                "tool_count": facts["tool_count"],
                "n_steps": facts["n_steps"],
                "n_hypotheses": facts["n_hypotheses"],
                "n_mass_writes": facts["n_mass_writes"],
                "plate_complete": facts["plate_complete"],
                "stop_unwritten": facts["stop_unwritten"],
                "density_align_applied": aligned,
            }
        )
    paired = [
        row
        for row in per_dish
        if finite(row["ours_energy_kcal"]) is not None
        and finite(row["ours_mass_g"]) is not None
        and finite(row["direct_energy_kcal"]) is not None
        and finite(row["direct_mass_g"]) is not None
        and finite(row["mass_only_energy_kcal"]) is not None
        and finite(row["mass_only_mass_g"]) is not None
    ]
    if not paired and not allow_partial:
        raise HeldoutEvalError("no dish has a finite prediction from all three methods")
    hollow: dict[str, Any]
    try:
        hollow = audit_mass_portions(traces)
    except SameBudgetRefused as exc:
        hollow = {"hollow_refused": str(exc)}
    tools = summarize_tool_counts([int(row["tool_count"]) for row in paired], budget_k=int(BUDGET_K))
    steps = step_summary([int(row["n_steps"]) for row in paired])
    all_tools = summarize_tool_counts([int(row["tool_count"]) for row in per_dish], budget_k=int(BUDGET_K))
    all_steps = step_summary([int(row["n_steps"]) for row in per_dish])
    methods = {}
    if paired:
        methods = {
            "coverage_constrained": _method_block("coverage_constrained", paired, "ours_energy_kcal", "ours_mass_g"),
            "direct_vlm": _method_block("direct_vlm", paired, "direct_energy_kcal", "direct_mass_g"),
            "mass_only": _method_block("mass_only", paired, "mass_only_energy_kcal", "mass_only_mass_g"),
        }
        ours = methods["coverage_constrained"]
        direct_block = methods["direct_vlm"]
        mass_block = methods["mass_only"]
        methods["coverage_constrained"]["energy_minus_direct_kcal"] = (
            ours["energy_mae_kcal"] - direct_block["energy_mae_kcal"]
        )
        methods["coverage_constrained"]["mass_minus_direct_g"] = ours["mass_mae_g"] - direct_block["mass_mae_g"]
        methods["coverage_constrained"]["energy_minus_mass_only_kcal"] = (
            ours["energy_mae_kcal"] - mass_block["energy_mae_kcal"]
        )
        methods["coverage_constrained"]["mass_minus_mass_only_g"] = ours["mass_mae_g"] - mass_block["mass_mae_g"]
    payload = {
        "protocol": "heldout_cafe2_frozen_53_plus_51",
        "main_table": limit_is_main(directory),
        "definition": (
            "External cafe2 held-out. The main method is the frozen calibrated "
            "Q/Cost selector with the coverage guard and the same-head density "
            "swap already inside the rollout. Direct-VLM is one Qwen2.5-VL "
            "nutrition guess. Mass-only is one frozen estimate_mass RGB forward. "
            "Cafe2 dish-level energy is 0 on every row, so the energy target is the "
            "sum of ingredient calories. Mass is the dish-level gram total, which "
            "matches the ingredient grams. Labels are read only here, to score. "
            "No threshold, code, or portion was chosen from these labels."
        ),
        "n_planned": len(ids),
        "n_rollout": len(traces),
        "n_direct": len(direct),
        "n_mass_only": len(mass_rows),
        "n_paired": len(paired),
        "missing_attempts": {phase: len(values) for phase, values in missing.items()},
        "failures": failures,
        "methods": methods,
        "coverage_paired": _coverage_counts(paired),
        "coverage_all_rollouts": _coverage_counts(per_dish),
        "steps_paired": steps,
        "tools_paired": tools,
        "steps_all_rollouts": all_steps,
        "tools_all_rollouts": all_tools,
        "density_align_applied_dishes": sum(1 for row in per_dish if int(row["density_align_applied"]) > 0),
        "unwritten_residuals": [
            {
                "dish_id": row["dish_id"],
                "n_mass_writes": int(row["n_mass_writes"]),
                "n_hypotheses": int(row["n_hypotheses"]),
                "decision_action": row["decision_action"],
            }
            for row in per_dish
            if not row["plate_complete"]
        ],
        "hollow_re_portion": hollow,
        "q_weights_retrained": False,
        "mass_retrained": False,
        "tau_retuned": False,
        "p1_unlocked": False,
        "rl": False,
        "heldout_used_for_selection": False,
        "real_tool_cost": "not measured; that is gate ④",
        "framing": {
            "development_505_is_not_this_table": True,
            "status_53_does_not_prove_the_method": True,
            "status_51_is_density_kcal_heuristic_auto_exact_0": True,
            "q_approx_rule_much_worse_than_oracle": True,
        },
        "tool_actions": sorted(TOOL_ACTIONS),
    }
    scores_path = directory / "scores.jsonl"
    summary_path = directory / "summary.json"
    report_path = directory / "report.md"
    assert_output(scores_path)
    scores_path.write_text("", encoding="utf-8")
    append_json(scores_path, per_dish)
    summary_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    report_path.write_text(render_report(payload), encoding="utf-8")
    print(
        f"wrote {report_path.relative_to(ROOT)} paired={len(paired)} planned={len(ids)}",
        flush=True,
    )


def limit_is_main(directory: Path) -> bool:
    return directory.resolve() == OUT.resolve()


def render_report(payload: Mapping[str, Any]) -> str:
    methods = payload["methods"]
    coverage = payload["coverage_paired"]
    tools = payload["tools_paired"]
    steps = payload["steps_paired"]
    title = "External held-out evaluation" if payload["main_table"] else "SMOKE prefix, not the held-out table"
    lines = [
        f"# {title}",
        "",
        "The paired 505 rgb_test dishes remain a development / diagnostic set. "
        "This table is the cafe2 held-out named in `results/heldout_plan/`. "
        "#53 does not prove the method. #51 is a density / kcal heuristic "
        "(automatic Exact on the development food-code audit was 0). "
        "On the separate one-step audit, Q ≈ rule ≪ oracle. "
        "Nothing here retrained Mass or Q, refit τ, or opened P1. "
        "Held-out labels were not used to choose a method, stop early, move τ, or edit a code or a portion. "
        "Cafe2 energy targets are ingredient-calorie sums because the dish-level energy column is 0 on every cafe2 row. "
        "Mass targets are the dish-level gram totals.",
        "",
        payload["definition"],
        "",
        f"Planned dishes: {payload['n_planned']}. Trajectories: {payload['n_rollout']}. "
        f"Direct-VLM finite: {payload['n_direct']}. Mass-only finite: {payload['n_mass_only']}. "
        f"Paired (all three finite): {payload['n_paired']}.",
        "",
    ]
    if not methods:
        lines.append("No paired dishes yet, so there is no MAE row.")
        lines.append("")
        return "\n".join(lines)
    ours = methods["coverage_constrained"]
    direct = methods["direct_vlm"]
    mass_only = methods["mass_only"]
    lines.extend(
        [
            "## Main table",
            "",
            "MAE is the mean absolute error on the paired dishes. "
            "Coverage is the fraction of those dishes whose every named component has a mass-estimator write. "
            "Tool calls count RETRIEVE, CORRECT_CODE, RE_PORTION, and RE_DECOMPOSE up to the committing step. "
            "Direct-VLM and Mass-only do not take those controller actions. "
            "Real latency, token, and dollar cost are gate ④ and are not in this table.",
            "",
            "| Method | n | Energy MAE (kcal) | Mass MAE (g) | Coverage | Steps mean | Tool calls mean |",
            "|---|---:|---:|---:|---:|---:|---:|",
            f"| Coverage-Constrained + density/same-head | {ours['n']} | {f4(ours['energy_mae_kcal'])} | "
            f"{f4(ours['mass_mae_g'])} | {f4(coverage['coverage'])} | {f4(steps['mean'])} | {f4(tools['mean'])} |",
            f"| Direct-VLM | {direct['n']} | {f4(direct['energy_mae_kcal'])} | {f4(direct['mass_mae_g'])} |  | 1 | 0 |",
            f"| Mass-only (frozen Mass) | {mass_only['n']} | {f4(mass_only['energy_mae_kcal'])} | "
            f"{f4(mass_only['mass_mae_g'])} |  | 1 | 0 |",
            "",
            f"Energy MAE minus Direct-VLM: {ours['energy_minus_direct_kcal']:+.4f} kcal. "
            f"Mass MAE minus Direct-VLM: {ours['mass_minus_direct_g']:+.4f} g. "
            f"Energy MAE minus Mass-only: {ours['energy_minus_mass_only_kcal']:+.4f} kcal. "
            f"Mass MAE minus Mass-only: {ours['mass_minus_mass_only_g']:+.4f} g.",
            "",
            "## Coverage and budget",
            "",
            f"Paired dishes with every named component mass-written: {coverage['complete']} / {coverage['n']} "
            f"({coverage['coverage']:.4f}). STOP while a named component is still unwritten: {coverage['stop_unwritten']}. "
            f"Mass-estimator writes: {coverage['mass_writes']}.",
            "",
        ]
    )
    residuals = payload.get("unwritten_residuals") or []
    if residuals:
        actions = sorted({str(residual["decision_action"]) for residual in residuals})
        ended = " and ".join(actions)
        lines.append(
            f"{len(residuals)} dishes still leave a named component without a mass write. "
            f"They ended on {ended}, not on STOP. The rollout does not invent a portion when retrieval has not produced a code."
        )
        lines.append("")
        for residual in residuals:
            lines.append(
                f"- `{residual['dish_id']}`: {residual['n_mass_writes']} of {residual['n_hypotheses']} "
                f"components mass-written, ended on {residual['decision_action']}."
            )
        lines.append("")
    lines.extend(
        [
            f"Steps on the paired dishes: mean {steps['mean']:.4f}, median {steps['median']}, "
            f"min {steps['min']}, max {steps['max']}.",
            "",
            f"Tool calls on the paired dishes: mean {tools['mean']:.4f}, median {tools['median']}, "
            f"min {tools['min']}, max {tools['max']}. "
            f"Within K={tools['budget_K']}: {tools['n_within_budget']} / {tools['n_dishes']}.",
            "",
            f"Same figures on every trajectory, including dishes a baseline did not score: "
            f"coverage {payload['coverage_all_rollouts']['complete']} / {payload['coverage_all_rollouts']['n']}, "
            f"steps mean {f4(payload['steps_all_rollouts']['mean'])}, "
            f"tool mean {f4(payload['tools_all_rollouts']['mean'])}.",
            "",
            f"Dishes where the frozen same-head density swap applied: {payload['density_align_applied_dishes']}. "
            "That swap is the #51 kcal heuristic. It is not an identity correction.",
            "",
            "## What this run did not do",
            "",
            "Mass, Q, and τ were not retrained. P1 stayed closed. There is no RL. "
            "Gate ④ real tool cost, gate ⑤ faithful B1/B2, and gate ⑥ provenance were not started. "
            "The development 505 energy MAE 64.2792 and mass MAE 32.9908 are not rows of this table.",
            "",
        ]
    )
    if payload["failures"]:
        lines.append("## Failures")
        lines.append("")
        for row in payload["failures"]:
            lines.append(f"- {row.get('phase')} `{row.get('dish_id')}`: {row.get('error')}")
        lines.append("")
    if payload["hollow_re_portion"].get("hollow_refused"):
        lines.append("Mass-portion audit refused a trajectory:")
        lines.append("")
        lines.append(payload["hollow_re_portion"]["hollow_refused"])
        lines.append("")
    return "\n".join(lines)


def run_self_check() -> None:
    from vdr.controller.on_policy import OnPolicyRefused, rollout_dish

    try:
        rollout_dish("dish_not_in_rgb_test", object())
    except OnPolicyRefused as exc:
        if "rgb_test" not in str(exc):
            raise HeldoutEvalError("the default guard did not name rgb_test") from exc
    else:
        raise HeldoutEvalError("a non-rgb_test id was accepted without the held-out flag")

    class StopPolicy:
        feature_kind = "z"

        def predict_actions(self, _features: Any) -> list[str]:
            return ["STOP"]

    result = rollout_dish(
        "dish_heldout_guard",
        StopPolicy(),
        perceive_fn=lambda _path: ([], {"model_alias": "stub", "serving": "stub"}),
        retrieve_fn=lambda *_args, **_kwargs: None,
        lookup_fn=lambda *_args, **_kwargs: {},
        portion_fn=lambda *_args, **_kwargs: None,
        allow_outside_rgb_test=True,
        check_replay=False,
    )
    if result.steps[-1]["action_t"] != "STOP":
        raise HeldoutEvalError("held-out flag did not roll the stub dish")
    if abs(mae([1.0, -3.0]) - 2.0) > 1e-12:
        raise HeldoutEvalError("MAE self-check failed")
    try:
        assert_output(ROOT / "results" / "coverage_early_stop" / "report.md")
    except HeldoutEvalError:
        pass
    else:
        raise HeldoutEvalError("a frozen directory was accepted as an output")
    print("heldout eval self-check ok", flush=True)


def main() -> int:
    import argparse

    parser = argparse.ArgumentParser(description="Evaluate the frozen protocol on the external held-out")
    parser.add_argument("--self-check", action="store_true")
    parser.add_argument(
        "--phase",
        choices=("mass", "rollout", "direct", "score", "all"),
        default="all",
    )
    parser.add_argument("--limit", type=int, default=None, help="Sorted-list prefix for a smoke. Not the main table.")
    args = parser.parse_args()
    if args.self_check:
        run_self_check()
        return 0
    before = frozen_hashes()
    ids = planned_ids(args.limit)
    directory = out_dir(args.limit)
    phases = ("mass", "rollout", "direct", "score") if args.phase == "all" else (args.phase,)
    for phase in phases:
        if phase == "mass":
            mass_phase(ids, directory)
        elif phase == "rollout":
            rollout_phase(ids, directory)
        elif phase == "direct":
            direct_phase(ids, directory)
        elif phase == "score":
            score_phase(ids, directory, allow_partial=args.limit is not None)
        assert_unchanged(before)
    if args.limit is not None:
        print("smoke prefix is not the held-out main table", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
