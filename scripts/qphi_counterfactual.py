#!/usr/bin/env python3
"""Gate ①: dish-level leakage audit and a one-step Q counterfactual.

Reads frozen weak steps, frozen ``Q`` weights, and the frozen τ. Writes only
``results/qphi_counterfactual/``. Does not refit Mass, Q, or τ, and does not
edit #46 / #48 / #51 / #53 artifacts.

    python scripts/qphi_counterfactual.py
    python scripts/qphi_counterfactual.py --max-dishes 8
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from pathlib import Path
from typing import Any, Mapping, Sequence

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np  # noqa: E402

from vdr.controller.mass_sigma import collect_dish_sigma, estimate_from_sigma_record  # noqa: E402
from vdr.controller.nutrition_loss import energy_abs_error, nutrition_loss  # noqa: E402
from vdr.controller.policy_bc import BC_ACTIONS  # noqa: E402
from vdr.controller.q_data import HOLDOUT_JSONL, load_q_rows  # noqa: E402
from vdr.controller.q_state import Z_DIM, build_z  # noqa: E402
from vdr.controller.selector import QCostPolicy  # noqa: E402
from vdr.data.n5k import load_labels, test_split_ids  # noqa: E402
from vdr.controller.qphi_counterfactual import (  # noqa: E402
    DECISIVE_ENERGY_KCAL,
    FIT_JSONL,
    PROTOCOL,
    PROTOCOL_ID,
    SIGMA_CACHE,
    TAU_PATH,
    BranchExecutor,
    audit_weak_step_split,
    belief_at,
    build_decision_record,
    choose_fixed,
    choose_q,
    choose_rule,
    component_count,
    coverage_legal,
    development_dish_ids,
    json_ready,
    load_frozen_q,
    load_frozen_tau,
    prepare_budget,
    run_self_check,
    score_branch,
    summarize_policy,
    value_calibration,
)
from vdr.ir.io import read_ir_jsonl  # noqa: E402
from vdr.tools.fndds_retrieve import retrieve  # noqa: E402
from vdr.tools.mass_estimator import assert_safe_result_dir  # noqa: E402
from vdr.tools.nutrition_lookup import lookup_per_100g  # noqa: E402
from vdr.tools.portion_grid import search_portion  # noqa: E402
from vdr.weak.teacher import pool_of_dish  # noqa: E402

OUT_DIR = ROOT / "results" / "qphi_counterfactual"
Q_WEIGHTS = ROOT / "results" / "qphi" / "q_phi.npz"
POLICIES = ("oracle", "q_phi", "rule_utility", "random", "fixed_pipeline")
FROZEN = (
    ROOT / "results" / "weak_steps" / "rgb_train_fit_all.jsonl",
    ROOT / "results" / "weak_steps" / "rgb_train_holdout_all.jsonl",
    ROOT / "results" / "weak_steps" / "rgb_train_fit_all.summary.json",
    ROOT / "results" / "weak_steps" / "rgb_train_holdout_all.summary.json",
    ROOT / "results" / "n5k_rgb_test_overhead_b0_b1.jsonl",
    ROOT / "results" / "n5k_rgb_test_overhead_b0_b1.json",
    ROOT / "results" / "n5k_subset_b0_b1.jsonl",
    ROOT / "results" / "n5k_subset_b0_b1.json",
    ROOT / "results" / "qphi" / "q_phi.npz",
    ROOT / "results" / "qphi" / "calibration.json",
    ROOT / "results" / "qphi" / "calibration_holdout.json",
    ROOT / "results" / "qphi" / "mass_sigma_rgb_train.json",
    ROOT / "results" / "mass_estimator" / "train_meta.json",
    ROOT / "results" / "coverage_early_stop" / "summary.json",
    ROOT / "results" / "coverage_early_stop" / "paired_scores.jsonl",
    ROOT / "results" / "food_id_density" / "summary.json",
    ROOT / "results" / "audit" / "error_audit.json",
    ROOT / "results" / "p0d" / "same_budget.json",
)


def _sha256(path: Path) -> str | None:
    if not path.is_file():
        return None
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _frozen_hashes() -> dict[str, str | None]:
    return {str(path.relative_to(ROOT)): _sha256(path) for path in FROZEN}


def _load_sigma(path: Path) -> dict[str, dict[str, Any]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("rgb_only") is not True or payload.get("depth_used") is not False:
        raise RuntimeError("mass sigma cache is not the RGB-only forward")
    if payload.get("uses_rgb_test_gt") is not False:
        raise RuntimeError("mass sigma cache says it used rgb_test labels")
    dishes = payload.get("dishes")
    if not isinstance(dishes, dict):
        raise RuntimeError("mass sigma cache has no dishes")
    return dishes


def _group_holdout(path: Path) -> dict[str, list[dict[str, Any]]]:
    banned = test_split_ids()
    grouped: dict[str, list[dict[str, Any]]] = {}
    for record in read_ir_jsonl(path):
        dish_id = record.get("dish_id")
        if not isinstance(dish_id, str):
            raise RuntimeError("hold-out row missing dish_id")
        if dish_id in banned:
            raise RuntimeError(f"hold-out file contains test dish {dish_id}")
        if pool_of_dish(dish_id) != "holdout":
            raise RuntimeError(f"{dish_id} is in the hold-out file but hashes to fit")
        grouped.setdefault(dish_id, []).append(record)
    for dish_id, steps in grouped.items():
        steps.sort(key=lambda row: int(row["step_i"]))
    return grouped


def _labels_without_test() -> dict[str, dict[str, float]]:
    banned = test_split_ids()
    labels = {dish_id: row for dish_id, row in load_labels().items() if dish_id not in banned}
    if banned.intersection(labels):
        raise RuntimeError("test ids remained in the scoring table")
    return labels


def _slice_name(record: Mapping[str, Any]) -> str:
    if "STOP" in record["legal"]:
        return "closed_component"
    if int(record["n_legal"]) >= 2:
        return "coverage_choice"
    return "coverage_single"


def _mean_abs_gap(records: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """Teacher's logged transition versus the live tool on that same action name."""
    gaps: dict[str, list[float]] = {}
    n_logged_illegal = 0
    n_compared = 0
    for record in records:
        action = record.get("logged_action")
        teacher = record.get("teacher_energy_delta")
        live = record.get("live_logged_energy_delta")
        if action is None:
            continue
        if not record.get("logged_in_legal"):
            n_logged_illegal += 1
            continue
        if teacher is None or live is None:
            continue
        gaps.setdefault(str(action), []).append(abs(float(teacher) - float(live)))
        n_compared += 1
    by_action = {
        action: {"n": len(values), "mean_abs_kcal": round(float(sum(values) / len(values)), 4)}
        for action, values in sorted(gaps.items())
    }
    flat = [value for values in gaps.values() for value in values]
    return {
        "n_compared": n_compared,
        "n_logged_action_not_in_A_legal": n_logged_illegal,
        "mean_abs_kcal": None if not flat else round(float(sum(flat) / len(flat)), 4),
        "by_action": by_action,
        "note": (
            "The weak-step teacher picks CORRECT_CODE from ingredient kcal/g and "
            "writes RE_PORTION from ingredient grams snapped to the grid. "
            "The branch uses the alternate retrieve code and estimate_mass. "
            "A large gap means the counterfactual did not reuse the privileged transition."
        ),
    }


def _policy_block(records: Sequence[Mapping[str, Any]], vectors: Sequence[np.ndarray], budget: float | None) -> dict[str, Any]:
    views = [prepare_budget(record, budget, vector) for record, vector in zip(records, vectors)]
    return {policy: summarize_policy(views, policy) for policy in POLICIES}


def _md_table(block: Mapping[str, Mapping[str, Any]]) -> list[str]:
    if not block or "oracle" not in block or "mean_energy_regret_kcal" not in block.get("oracle", {}):
        return ["No states in this slice."]
    header = (
        "| Policy | Identification | Exact tie-break | Energy improvement (kcal) | "
        "Energy regret (kcal) | Mass improvement (g) | Mass regret (g) | Mean cost |"
    )
    rule = "|---|---:|---:|---:|---:|---:|---:|---:|"
    lines = [header, rule]
    labels = {
        "oracle": "Oracle",
        "q_phi": "Q_φ (frozen + fence)",
        "rule_utility": "Rule Utility",
        "random": "Random (expectation)",
        "fixed_pipeline": "Fixed Pipeline",
    }
    for policy in POLICIES:
        row = block[policy]
        lines.append(
            "| {name} | {ident} | {exact} | {energy} | {regret} | {mass} | {mass_regret} | {cost} |".format(
                name=labels[policy],
                ident=_cell(row["best_action_identification_rate"]),
                exact=_cell(row["exact_tiebreak_rate"]),
                energy=_cell(row["mean_energy_improvement_kcal"]),
                regret=_cell(row["mean_energy_regret_kcal"]),
                mass=_cell(row["mean_mass_improvement_g"]),
                mass_regret=_cell(row["mean_mass_regret_g"]),
                cost=_cell(row["mean_cost"]),
            )
        )
    return lines


def _cell(value: Any) -> str:
    if value is None:
        return "—"
    return str(value)


def _without_noop(records: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    """Drop the unimplemented ``RE_DECOMPOSE`` no-op and rescore.

    The live rollout does not call a second perception model for that action.
    The branch leaves nutrition unchanged. Frozen ``Q`` masks it because the
    fit set never logged it. Rule and fixed selections in this run do not
    pick it, so their stored actions stay valid.
    """
    filtered: list[dict[str, Any]] = []
    for record in records:
        view = dict(record)
        view["legal"] = [action for action in record["legal"] if action != "RE_DECOMPOSE"]
        view["branches"] = {action: row for action, row in record["branches"].items() if action != "RE_DECOMPOSE"}
        view["n_legal"] = len(view["legal"])
        if not view["legal"]:
            continue
        filtered.append(prepare_budget(view, None, None))
    return filtered


def _reading(primary: Mapping[str, Mapping[str, Any]], leakage: Mapping[str, Any], extra: Mapping[str, Any]) -> list[str]:
    q_row = primary["q_phi"]
    rule = primary["rule_utility"]
    random_row = primary["random"]
    fixed = primary["fixed_pipeline"]
    q_regret = q_row["mean_energy_regret_kcal"]
    others = [rule["mean_energy_regret_kcal"], random_row["mean_energy_regret_kcal"], fixed["mean_energy_regret_kcal"]]
    lines = [
        "Reading. Route A scores energy. Mass is the parallel log of the energy-chosen action, including its regret against the mass oracle.",
        "This table is not the #53 development MAE. #53 is coverage-constrained selection on the rgb_test 505, which is a development and diagnostic set. Those dishes are not in this sample. The #53 energy drop is the coverage constraint, not evidence that Q got smarter.",
    ]
    if leakage["leakage_exists"]:
        lines.append(
            "The weak-step split leaks dishes across fit and hold-out. Q was not retrained. "
            "The numbers below are still the frozen checkpoint on the recorded hold-out file, and they are not a clean dish-held-out test."
        )
    elif q_regret is None or any(value is None for value in others):
        lines.append("Energy regret is missing on this slice, so the comparison does not support a claim about learned action values.")
    elif q_regret < min(others):
        lines.append(
            f"On this one-step hold-out, frozen Q_φ energy regret is {q_regret} kcal, "
            f"rule utility {rule['mean_energy_regret_kcal']}, fixed pipeline {fixed['mean_energy_regret_kcal']}, "
            f"random {random_row['mean_energy_regret_kcal']}. "
            f"Q is ahead of those three. The margin over the rule is "
            f"{round(rule['mean_energy_regret_kcal'] - q_regret, 4)} kcal, "
            f"while both remain about {q_regret} kcal behind the oracle. "
            f"Identification is {q_row['best_action_identification_rate']} for Q and "
            f"{rule['best_action_identification_rate']} for the rule."
        )
        lines.append(
            f"States whose energy-best legal action is the RE_DECOMPOSE no-op: {extra['n_noop_oracle']} "
            f"of {extra['n_headline']}. On those states the no-op keeps the current error and every real tool increases it. "
            f"Q regret there is {extra['q_regret_on_noop_oracle']} kcal and the rule is {extra['rule_regret_on_noop_oracle']} kcal. "
            "Frozen Q cannot emit RE_DECOMPOSE: that head was masked because the fit weak steps never logged it. "
            f"On the other states, Q regret is {extra['q_regret_elsewhere']} kcal and the rule is {extra['rule_regret_elsewhere']} kcal."
        )
        lines.append(
            "Removing that no-op from A_legal and rescoring the same states gives the sensitivity table below. "
            "This run does not treat the no-op as a reason to edit Q. #53 remains coverage-constrained selection on the development 505, not a claim that Q got smarter."
        )
    else:
        lines.append(
            f"On this one-step hold-out, frozen Q_φ regret ({q_regret} kcal) does not beat every non-oracle baseline "
            f"(rule {rule['mean_energy_regret_kcal']}, random {random_row['mean_energy_regret_kcal']}, fixed {fixed['mean_energy_regret_kcal']}). "
            "The learned Q is not, on this test, a stronger action-value rule than the baselines it fails to beat. "
            "The development-set energy drop at #53 stays a coverage-constraint result."
        )
    spaced: list[str] = []
    for line in lines:
        spaced.append(line)
        spaced.append("")
    return spaced


def _write_report(path: Path, payload: Mapping[str, Any]) -> None:
    leakage = payload["leakage_audit"]
    primary = payload["slices"]["coverage_choice_mass"]["policies"]
    lines = [
        "# Gate ① — leakage audit and counterfactual Q",
        "",
        PROTOCOL,
        "",
        "## Leakage audit",
        "",
        f"Conclusion: `{leakage['conclusion']}`. `leakage_exists={str(leakage['leakage_exists']).lower()}`.",
        "",
        f"Fit is {leakage['fit_dishes']} dishes / {leakage['fit_rows']} steps "
        f"({leakage['fit_dishes_with_multiple_steps']} dishes have more than one step). "
        f"Hold-out is {leakage['holdout_dishes']} dishes / {leakage['holdout_rows']} steps "
        f"({leakage['holdout_dishes_with_multiple_steps']} dishes have more than one step).",
        "",
        f"Dishes present in both files: {leakage['dishes_in_both_splits']}. "
        f"Rows on those dishes: {leakage['rows_belonging_to_dishes_in_both_splits']}. "
        f"Dishes whose file disagrees with `pool_of_dish`: {leakage['misfiled_dishes']}.",
        "",
        "The cut is `sha256(dish_id)` modulo 10, not a shuffle of decision steps. "
        "Q training consumes the fit file and scores the hold-out file. "
        "No weights were updated after this audit.",
        "",
        "## Sample",
        "",
        f"Population id: `{payload['protocol_id']}`. Smoke run: {payload['smoke']}.",
        "",
        f"Hold-out dishes scored: {payload['n_holdout_dishes']}. "
        f"Decision states branched: {payload['n_states']}. "
        f"Coverage-choice states (|A_legal|≥2 and STOP/ABSTAIN illegal): {payload['slices']['coverage_choice']['n']}. "
        f"Headline keeps the {payload['slices']['coverage_choice_mass']['n']} of those whose dish has a frozen RGB mass forward, "
        f"so RE_PORTION is a real specialist call. "
        f"Of the headline states, {payload['slices']['decisive_energy']['n']} move post energy error by more than {DECISIVE_ENERGY_KCAL:g} kcal.",
        "",
        f"Development dishes in `#53` paired file: {payload['n_development_dishes']}. "
        f"Overlap with this sample: {payload['n_overlap_development']}. "
        f"rgb_test ids in the sample: {payload['n_rgb_test_in_sample']}.",
        "",
        f"Frozen τ = {payload['tau']}. Q sha256 prefix `{payload['q_sha256'][:16]}`. "
        f"Matches calibration prefix: {payload['q_sha_matches_calibration']}. "
        "Mass weights, Q weights, and τ were not edited. "
        f"Fresh mass forwards (cache misses only): {payload['mass_forwards_fresh']}. "
        f"RE_PORTION cache hits / misses during branches: {payload['mass_branch_hits']} / {payload['mass_branch_misses']}.",
        "",
        "Portions from `RE_PORTION` use the frozen dish-level mass mean split equally across components, "
        "because these states have no vision gram hints. Ingredient grams are not hints. "
        "The #51 same-head swap is not applied inside a one-step branch: it runs on a finished multi-component plate, "
        "and the earlier components in these logs were written by the privileged teacher.",
        "",
        "## Coverage-choice with a frozen mass forward (headline)",
        "",
        f"n = {payload['slices']['coverage_choice_mass']['n']}. "
        "Identification is membership in the energy-optimal set. "
        "Regret is post energy error minus the oracle post energy error. "
        "Improvement is pre error minus post error, on states where both energies exist.",
        "",
        *_md_table(primary),
        "",
        *_reading(primary, leakage, payload["noop_split"]),
        "",
        "### Chosen-action counts (headline, no-op included)",
        "",
    ]
    for policy in POLICIES:
        lines.append(f"- {policy}: `{json.dumps(primary[policy]['actions'], sort_keys=True)}`")
    lines.extend(
        [
            "",
            "## Sensitivity: A_legal without the RE_DECOMPOSE no-op",
            "",
            "Same headline states. RE_DECOMPOSE is removed before the oracle, the rule, random, and the fixed pipeline are scored. "
            "Q is not re-argmaxed. Its frozen action is kept when it is still legal, which it is, because the masked head does not select RE_DECOMPOSE.",
            "",
            *_md_table(payload["no_decompose"]["policies"]),
            "",
            "### Chosen-action counts (no-op removed)",
            "",
        ]
    )
    for policy in POLICIES:
        lines.append(
            f"- {policy}: `{json.dumps(payload['no_decompose']['policies'][policy]['actions'], sort_keys=True)}`"
        )
    lines.extend(
        [
            "",
            "## All coverage-choice states, including dishes with no overhead RGB",
            "",
            "RE_PORTION on a dish with no frozen forward fails and does not invent grams. "
            "Those states stay out of the headline so a missing image is not scored as a specialist call.",
            "",
            *_md_table(payload["slices"]["coverage_choice"]["policies"]),
            "",
            "## Decisive energy slice",
            "",
            f"Headline states whose legal actions disagree by more than {DECISIVE_ENERGY_KCAL:g} kcal "
            "in post energy error. The 1 kcal cutoff is the existing FixRate ε. It was not refit.",
            "",
            *_md_table(payload["slices"]["decisive_energy"]["policies"]),
            "",
            "## Closed-component states",
            "",
            "STOP is legal because the component is not open. This is not the coverage-constrained choice. "
            "Shown so the terminal states are not dropped silently.",
            "",
            *_md_table(payload["slices"]["closed_component"]["policies"]),
            "",
            "## Budget curves",
            "",
            "A cap keeps actions with cost ≤ B. Oracle, rule utility, random, and the fixed pipeline re-select inside that set. "
            "Q_φ does not re-argmax and τ does not change. If the frozen Q action costs more than B, that state is idle "
            "(the pre-action error is kept). Caps are the three action costs already in the controller (1, 2, 3), not a search.",
            "",
            "The curve is not monotone. At cost ≤ 1, a state whose only legal tools are RETRIEVE (cost 2) and RE_DECOMPOSE (cost 3) "
            "cannot act, so the error stays put. At cost ≤ 2 that state must retrieve, and retrieval can increase energy error. "
            "The no-op that would have kept the error costs 3, so it enters only at the last cap.",
            "",
        ]
    )
    for cap, block in payload["budget_curves"].items():
        lines.append(f"### Cost ≤ {cap}")
        lines.append("")
        policies = block.get("policies") or {}
        if "q_phi" not in policies:
            lines.append("No headline states.")
            lines.append("")
            continue
        lines.append(f"n = {block['n']}. Idle Q states: {policies['q_phi']['n_idle']}.")
        lines.append("")
        lines.extend(_md_table(policies))
        lines.append("")
    cal = payload["value_calibration"]
    lines.extend(
        [
            "## Value calibration",
            "",
            f"n action-values = {cal['n']}. MAE(Q, ΔL) = {cal['mae']}. "
            f"Pearson(Q, ΔL) = {cal['pearson_delta_l']}. Pearson(Q, kcal improvement) = {cal['pearson_energy_kcal']}. "
            f"Slope of realized ΔL on predicted Q = {cal['slope_realized_on_predicted']} "
            f"(intercept {cal['intercept_realized_on_predicted']}).",
            "",
            "The slope is a description. It was not written back into the fence or the weights.",
            "",
            "| Q bin | n | Mean predicted Q | Mean realized ΔL | Mean realized kcal |",
            "|---|---:|---:|---:|---:|",
        ]
    )
    for row in cal["bins"]:
        lines.append(
            f"| {row['q_lo']}–{row['q_hi']} | {row['n']} | {row['mean_predicted_q']} | "
            f"{row['mean_realized_delta_l']} | {row['mean_realized_energy_delta_kcal']} |"
        )
    lines.extend(["", "### By action", ""])
    for action in BC_ACTIONS:
        row = cal["per_action"][action]
        lines.append(
            f"- {action}: n={row['n']}, MAE={row.get('mae')}, "
            f"mean Q={row.get('mean_predicted_q')}, mean ΔL={row.get('mean_realized_delta_l')}, "
            f"Pearson={row.get('pearson')}"
        )
    gap = payload["teacher_vs_live_tool"]
    lines.extend(
        [
            "",
            "## Logged teacher transition vs live tool",
            "",
            f"Mean absolute energy-delta gap on the logged action name: {gap['mean_abs_kcal']} kcal "
            f"(n={gap['n_compared']}). Logged actions outside A_legal: {gap['n_logged_action_not_in_A_legal']}.",
            "",
        ]
    )
    for action, row in gap["by_action"].items():
        lines.append(f"- {action}: n={row['n']}, mean |Δkcal teacher − live| = {row['mean_abs_kcal']}")
    lines.extend(
        [
            "",
            "## Freeze and bans",
            "",
            f"Frozen hashes unchanged: {payload['frozen_unchanged']}.",
            "No Mass retrain. No Q retrain. No τ retune. No rgb_test label was a scoring target or a threshold. "
            "No reinforcement learning. No P1. Gates ②–⑦ were not started. "
            "RE_PORTION does not fall back to the gram grid or to ingredient grams when the mass forward is missing.",
            "",
            "## Blockers",
            "",
        ]
    )
    if payload["blockers"]:
        for item in payload["blockers"]:
            lines.append(f"- {item}")
    else:
        lines.append("- None that stop this one-step report. The limitations above still bound what the numbers mean.")
    lines.append("")
    path.write_text("\n".join(lines), encoding="utf-8")


def _write_leakage_md(path: Path, leakage: Mapping[str, Any]) -> None:
    lines = [
        "# Weak-step fit / hold-out leakage audit",
        "",
        f"Conclusion: `{leakage['conclusion']}`.",
        "",
        f"`leakage_exists` = {str(leakage['leakage_exists']).lower()}.",
        "",
        leakage["question"],
        "",
        f"Rule: `{leakage['split_rule']}`.",
        "",
        f"Fit dishes {leakage['fit_dishes']}, rows {leakage['fit_rows']}, "
        f"multi-step dishes {leakage['fit_dishes_with_multiple_steps']}.",
        f"Hold-out dishes {leakage['holdout_dishes']}, rows {leakage['holdout_rows']}, "
        f"multi-step dishes {leakage['holdout_dishes_with_multiple_steps']}.",
        f"Dishes in both splits: {leakage['dishes_in_both_splits']}.",
        f"Misfiled against the hash: {leakage['misfiled_dishes']}.",
        "",
        leakage["note"],
        "",
        "Q, Mass, and τ were not retrained.",
        "",
    ]
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description="Gate ① leakage audit and counterfactual Q")
    parser.add_argument("--out-dir", type=Path, default=OUT_DIR)
    parser.add_argument("--max-dishes", type=int, default=None, help="Prefix of sorted hold-out dishes. Omit for the full hold-out.")
    args = parser.parse_args()
    out_dir = args.out_dir if args.out_dir.is_absolute() else ROOT / args.out_dir
    if out_dir.name != "qphi_counterfactual":
        raise RuntimeError(f"refusing to write outside results/qphi_counterfactual: {out_dir}")
    assert_safe_result_dir(out_dir)
    if args.max_dishes is not None and args.max_dishes < 1:
        raise RuntimeError("--max-dishes must be >= 1")

    print("self-check", flush=True)
    run_self_check()
    before = _frozen_hashes()
    started = time.perf_counter()

    leakage = audit_weak_step_split(ROOT / FIT_JSONL, ROOT / HOLDOUT_JSONL)
    print(
        f"leakage conclusion={leakage['conclusion']} both={leakage['dishes_in_both_splits']} "
        f"fit_dishes={leakage['fit_dishes']} holdout_dishes={leakage['holdout_dishes']}",
        flush=True,
    )

    tau = load_frozen_tau(ROOT / TAU_PATH)
    calibration = json.loads((ROOT / TAU_PATH).read_text(encoding="utf-8"))
    model = load_frozen_q(Q_WEIGHTS)
    q_sha = _sha256(Q_WEIGHTS) or ""
    prefix = str(calibration.get("weights_sha256_prefix") or "")
    if not q_sha.startswith(prefix):
        raise RuntimeError("q_phi.npz does not match the calibration weight prefix")
    policy = QCostPolicy(model, terminal_margin=tau)

    sigma_for_z = _load_sigma(ROOT / SIGMA_CACHE)
    grouped = _group_holdout(ROOT / HOLDOUT_JSONL)
    dish_ids = sorted(grouped)
    if args.max_dishes is not None:
        dish_ids = dish_ids[: args.max_dishes]
        grouped = {dish_id: grouped[dish_id] for dish_id in dish_ids}
    labels = _labels_without_test()
    missing_labels = [dish_id for dish_id in dish_ids if dish_id not in labels]
    if missing_labels:
        raise RuntimeError(f"{len(missing_labels)} hold-out dishes have no train label")
    development = development_dish_ids(ROOT / "results" / "coverage_early_stop" / "paired_scores.jsonl")
    banned = test_split_ids()
    overlap = sorted(set(dish_ids) & development)
    test_hits = sorted(set(dish_ids) & banned)
    if overlap:
        dish_ids = [dish_id for dish_id in dish_ids if dish_id not in set(overlap)]
        grouped = {dish_id: grouped[dish_id] for dish_id in dish_ids}
    if test_hits:
        raise RuntimeError(f"sample contains rgb_test ids: {test_hits[:5]}")

    # z uses the cache Q already saw. A missing dish gets a forward only for the
    # RE_PORTION tool effect, and that forward is not written back into results/qphi.
    sigma_for_tools = dict(sigma_for_z)
    missing_mass = [dish_id for dish_id in dish_ids if dish_id not in sigma_for_z]
    fresh_n = 0
    if missing_mass:
        print(f"frozen mass forward for {len(missing_mass)} hold-out dishes missing from the cache", flush=True)
        fresh, _backbone = collect_dish_sigma(missing_mass, batch_size=16)
        sigma_for_tools.update(fresh)
        fresh_n = len(fresh)
        print(f"fresh mass forwards {fresh_n}", flush=True)

    if args.max_dishes is None:
        print("checking z against the frozen Q row builder", flush=True)
        reference = load_q_rows(ROOT / HOLDOUT_JSONL, expect_pool="holdout", sigma_by_dish=sigma_for_z)
        ref_index = {(reference.dish_ids[i], int(reference.step_indices[i])): i for i in range(reference.n_rows)}
    else:
        reference = None
        ref_index = {}

    executor = BranchExecutor(
        retrieve_fn=retrieve,
        lookup_fn=lookup_per_100g,
        portion_fn=search_portion,
        mass_by_dish=sigma_for_tools,
    )
    records: list[dict[str, Any]] = []
    vectors: list[np.ndarray] = []
    n_states = 0
    for done, dish_id in enumerate(dish_ids, start=1):
        steps = grouped[dish_id]
        n_components = component_count(steps)
        label = labels[dish_id]
        prev: dict[str, Any] | None = None
        seen_portion = False
        cached = sigma_for_z.get(dish_id)
        for index, row in enumerate(steps):
            injected = estimate_from_sigma_record(cached) if seen_portion and cached is not None else None
            vector = build_z(row, prev, belief_at(int(row["step_i"])), mass_estimate=injected)
            if vector.shape != (Z_DIM,):
                raise RuntimeError(f"z width {vector.shape}")
            if reference is not None:
                slot = ref_index.get((dish_id, int(row["step_i"])))
                if slot is None:
                    raise RuntimeError(f"Q rows missing {dish_id} step {row['step_i']}")
                if not np.allclose(vector, reference.Z[slot]):
                    raise RuntimeError(f"z drifted from q_data on {dish_id} step {row['step_i']}")
            legal = coverage_legal(vector)
            q_action, q_source, q_values = choose_q(model, vector, tau)
            batched = policy.predict_actions(vector.reshape(1, -1))[0]
            if batched != q_action:
                raise RuntimeError(f"Q choice drifted from QCostPolicy on {dish_id} step {row['step_i']}")
            rule_action = choose_rule(vector, legal)
            fixed_action = choose_fixed(vector, legal)
            branches: dict[str, dict[str, Any]] = {}
            for action in BC_ACTIONS:
                if not legal[action]:
                    continue
                outcome = executor.run(row, action, n_components=n_components)
                branches[action] = score_branch(
                    outcome,
                    label=label,
                    pre_dish=row.get("nutri_dish"),
                    q_value=q_values[action],
                )
            pre_dish = row.get("nutri_dish") if isinstance(row.get("nutri_dish"), dict) else None
            pre_energy = energy_abs_error(pre_dish, label)
            pre_mass = None
            if pre_dish is not None and pre_dish.get("mass_g") is not None:
                pre_mass = abs(float(pre_dish["mass_g"]) - float(label["mass_g"]))
            pre_loss = nutrition_loss(pre_dish, label)
            record = build_decision_record(
                dish_id=dish_id,
                step_i=int(row["step_i"]),
                component_i=row.get("component_i") if isinstance(row.get("component_i"), int) else None,
                n_components=n_components,
                legal=legal,
                pre_energy=pre_energy,
                pre_mass=pre_mass,
                pre_loss=pre_loss,
                branches=branches,
                q_action=q_action,
                q_source=q_source,
                rule_action=rule_action,
                fixed_action=fixed_action,
            )
            logged = str(row.get("action_t"))
            record["mass_available"] = dish_id in sigma_for_tools
            record["logged_action"] = logged
            record["logged_in_legal"] = bool(legal.get(logged, False))
            teacher_delta = None
            if index + 1 < len(steps):
                nxt = steps[index + 1].get("nutri_dish")
                if pre_energy is not None and isinstance(nxt, dict) and nxt.get("energy_kcal") is not None:
                    teacher_delta = pre_energy - abs(float(nxt["energy_kcal"]) - float(label["energy_kcal"]))
            record["teacher_energy_delta"] = teacher_delta
            live_logged = branches.get(logged, {}).get("energy_delta") if logged in branches else None
            record["live_logged_energy_delta"] = live_logged
            record["q_source"] = q_source
            records.append(record)
            vectors.append(vector)
            n_states += 1
            prev = row
            if str(row.get("action_t")) == "RE_PORTION":
                seen_portion = True
        if done % 40 == 0 or done == len(dish_ids):
            print(f"dishes {done}/{len(dish_ids)} states {n_states}", flush=True)
    if reference is not None and reference.n_rows != n_states:
        raise RuntimeError(f"Q rows {reference.n_rows} != branched states {n_states}")

    buckets: dict[str, list[tuple[dict[str, Any], np.ndarray]]] = {
        "coverage_choice": [],
        "coverage_choice_mass": [],
        "coverage_single": [],
        "closed_component": [],
        "decisive_energy": [],
    }
    for record, vector in zip(records, vectors):
        name = _slice_name(record)
        buckets[name].append((record, vector))
        if name == "coverage_choice" and record.get("mass_available"):
            buckets["coverage_choice_mass"].append((record, vector))
            if record["decisive_energy"]:
                buckets["decisive_energy"].append((record, vector))

    slices: dict[str, Any] = {}
    for name, pairs in buckets.items():
        recs = [item[0] for item in pairs]
        vecs = [item[1] for item in pairs]
        slices[name] = {"n": len(recs), "policies": _policy_block(recs, vecs, None) if recs else {policy: {"n": 0} for policy in POLICIES}}

    headline_records = [item[0] for item in buckets["coverage_choice_mass"]]
    headline_vectors = [item[1] for item in buckets["coverage_choice_mass"]]
    budget_curves = {}
    for cap in (1.0, 2.0, 3.0):
        budget_curves[str(int(cap))] = {
            "cap": cap,
            "n": len(headline_records),
            "policies": _policy_block(headline_records, headline_vectors, cap) if headline_records else {},
        }
    calibration = value_calibration(headline_records)
    gap = _mean_abs_gap(headline_records)
    noop_rows = [record for record in headline_records if record["oracle_action"] == "RE_DECOMPOSE"]
    other_rows = [record for record in headline_records if record["oracle_action"] != "RE_DECOMPOSE"]

    def _mean_regret(subset: Sequence[Mapping[str, Any]], action_key: str) -> float | None:
        values: list[float] = []
        for record in subset:
            branch = record["branches"].get(str(record[action_key]))
            oracle = record["branches"].get(str(record["oracle_action"]))
            if not branch or not oracle:
                continue
            if branch.get("post_energy_ae") is None or oracle.get("post_energy_ae") is None:
                continue
            values.append(float(branch["post_energy_ae"]) - float(oracle["post_energy_ae"]))
        if not values:
            return None
        return round(sum(values) / len(values), 4)

    noop_split = {
        "n_headline": len(headline_records),
        "n_noop_oracle": len(noop_rows),
        "q_regret_on_noop_oracle": _mean_regret(noop_rows, "q_action"),
        "rule_regret_on_noop_oracle": _mean_regret(noop_rows, "rule_action"),
        "q_regret_elsewhere": _mean_regret(other_rows, "q_action"),
        "rule_regret_elsewhere": _mean_regret(other_rows, "rule_action"),
        "note": (
            "RE_DECOMPOSE does not change nutrition. It is legal while a component is open, "
            "and it is the energy-best action when every real tool increases absolute energy error. "
            "The frozen Q head masks it."
        ),
    }
    no_decompose_views = _without_noop(headline_records)
    no_decompose = {
        "n": len(no_decompose_views),
        "policies": {policy: summarize_policy(no_decompose_views, policy) for policy in POLICIES},
    }

    blockers: list[str] = []
    if leakage["leakage_exists"]:
        blockers.append(
            "Dish leakage: the same dish_id has steps in fit and hold-out. Documented. Q was not retrained."
        )
    if leakage["misfiled_dishes"]:
        blockers.append("Some dishes sit in a file that disagrees with pool_of_dish.")
    if overlap:
        blockers.append(f"Excluded {len(overlap)} dishes that also appear in the #53 development file.")
    if fresh_n:
        blockers.append(
            f"{fresh_n} hold-out dishes were missing from the frozen sigma cache. "
            "A forward of the frozen mass checkpoint filled them in memory only. The cache file was not rewritten."
        )
    q_illegal = slices["coverage_choice_mass"]["policies"].get("q_phi", {}).get("n_q_illegal_on_full_set", 0)
    if q_illegal:
        blockers.append(f"Frozen Q selected an action outside A_legal on {q_illegal} coverage-choice states.")
    if executor.mass_misses:
        blockers.append(
            f"RE_PORTION had no frozen mass forward on {executor.mass_misses} branches, all outside the mass-available headline. "
            "Grams were not invented."
        )
    blockers.append(
        "One-step branches, not a full-horizon rollout. Budget curves are one-step cost caps."
    )
    blockers.append(
        "Weak-step states have no vision gram hints, so RE_PORTION uses the frozen equal split of the dish mass."
    )
    blockers.append(
        "The #51 plate-level code swap is not inside the one-step branch."
    )

    after = _frozen_hashes()
    if before != after:
        changed = [key for key in before if before[key] != after.get(key)]
        raise RuntimeError(f"frozen files changed: {changed}")

    payload: dict[str, Any] = {
        "protocol_id": PROTOCOL_ID,
        "protocol": PROTOCOL,
        "smoke": args.max_dishes is not None,
        "max_dishes": args.max_dishes,
        "tau": tau,
        "tau_source": "results/qphi/calibration.json",
        "q_weights": "results/qphi/q_phi.npz",
        "q_sha256": q_sha,
        "q_sha_matches_calibration": True,
        "retrained_mass": False,
        "retrained_q": False,
        "retuned_tau": False,
        "rl": False,
        "uses_rgb_test_gt": False,
        "p1": False,
        "gates_2_through_7_started": False,
        "leakage_audit": leakage,
        "n_holdout_dishes": len(dish_ids),
        "n_states": n_states,
        "n_development_dishes": len(development),
        "n_overlap_development": len(overlap),
        "n_rgb_test_in_sample": len(test_hits),
        "mass_forwards_fresh": fresh_n,
        "mass_branch_hits": executor.mass_hits,
        "mass_branch_misses": executor.mass_misses,
        "retrieve_cache": len(executor.retrieve_cache),
        "slices": slices,
        "budget_curves": budget_curves,
        "value_calibration": calibration,
        "teacher_vs_live_tool": gap,
        "noop_split": noop_split,
        "no_decompose": no_decompose,
        "q_sources": {
            source: sum(1 for record in headline_records if record["q_source"] == source)
            for source in sorted({record["q_source"] for record in headline_records})
        },
        "blockers": blockers,
        "frozen_unchanged": True,
        "seconds": round(time.perf_counter() - started, 1),
    }
    out_dir.mkdir(parents=True, exist_ok=True)
    ready = json_ready(payload)
    (out_dir / "summary.json").write_text(json.dumps(ready, indent=2) + "\n", encoding="utf-8")
    (out_dir / "leakage_audit.json").write_text(json.dumps(json_ready(leakage), indent=2) + "\n", encoding="utf-8")
    _write_leakage_md(out_dir / "leakage_audit.md", leakage)
    _write_report(out_dir / "report.md", ready)
    # Plain-text twin for the same numbers. The markdown is the readable copy.
    text_lines = [
        f"protocol {PROTOCOL_ID}",
        f"leakage {leakage['conclusion']} leakage_exists {leakage['leakage_exists']}",
        f"coverage_choice n {slices['coverage_choice']['n']}",
        f"coverage_choice_mass n {slices['coverage_choice_mass']['n']}",
    ]
    if slices["coverage_choice_mass"]["n"]:
        for policy in POLICIES:
            row = slices["coverage_choice_mass"]["policies"][policy]
            text_lines.append(
                f"{policy} id {row['best_action_identification_rate']} "
                f"dE {row['mean_energy_improvement_kcal']} regret {row['mean_energy_regret_kcal']} "
                f"dM {row['mean_mass_improvement_g']} mass_regret {row['mean_mass_regret_g']} "
                f"cost {row['mean_cost']}"
            )
    text_lines.append(
        f"calibration n {calibration['n']} mae {calibration['mae']} pearson {calibration['pearson_delta_l']}"
    )
    text_lines.append("no_decompose")
    for policy in POLICIES:
        row = no_decompose["policies"][policy]
        text_lines.append(
            f"nodecomp {policy} id {row['best_action_identification_rate']} "
            f"dE {row['mean_energy_improvement_kcal']} regret {row['mean_energy_regret_kcal']}"
        )
    text_lines.append(f"seconds {payload['seconds']}")
    (out_dir / "report.txt").write_text("\n".join(text_lines) + "\n", encoding="utf-8")
    with (out_dir / "decisions.jsonl").open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(json_ready(record), ensure_ascii=False) + "\n")
    # Hashes one more time after the writes, which must not have touched frozen paths.
    if _frozen_hashes() != before:
        raise RuntimeError("frozen files changed while writing the new result directory")
    print(f"wrote {out_dir.relative_to(ROOT)} seconds {payload['seconds']}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
