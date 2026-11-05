#!/usr/bin/env python3
"""Calibrate Q/Cost decisions on the fit set, then score the hold-out once.

The learned Q weights stay as they are. The only change is the decision
rule: a stop or abstain that beats the best legal specialist by less than
the fit-set median-plus-three-MAD fence defers to the rule utility. The
fence is written before the hold-out is scored. No reinforcement learning.
rgb_test labels are not read. Frozen n5k and weak_steps files are hashed
before and after and must not change. This is not the full-set same-budget
table.

    python scripts/qphi_calibrate.py
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from collections import Counter
from pathlib import Path
from typing import Any, Mapping, Sequence

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from vdr.controller.on_policy import rgb_test_ids, rollout_dish  # noqa: E402
from vdr.controller.policy_bc import BC_ACTIONS  # noqa: E402
from vdr.controller.q_calibration import (  # noqa: E402
    MARGIN_MAD_MULTIPLIER,
    RULE_ID,
    SPECIALIST_ACTIONS,
    calibrated_decision,
    load_calibrated_policy,
    robust_fence,
    terminal_win_excesses,
)
from vdr.controller.q_data import FIT_JSONL, HOLDOUT_JSONL, load_q_rows  # noqa: E402
from vdr.controller.q_phi import QPhi  # noqa: E402
from vdr.controller.selector import HeuristicUtilityPolicy, QCostPolicy  # noqa: E402
from vdr.controller.utility import ACTION_COST  # noqa: E402
from vdr.tools.mass_estimator import assert_safe_result_dir  # noqa: E402

OUT_DIR = ROOT / "results" / "qphi"
WEIGHTS = OUT_DIR / "q_phi.npz"
CALIBRATION_PATH = OUT_DIR / "calibration.json"
HOLDOUT_METRICS_PATH = OUT_DIR / "calibration_holdout.json"
REPORT_PATH = OUT_DIR / "report_calibration.txt"
# P0c published this comparison. Reproduced here; not used to choose the fence.
P0C_AGREEMENT = 0.4489354023818116
FROZEN = (
    ROOT / "results" / "weak_steps" / "rgb_train_fit_all.jsonl",
    ROOT / "results" / "weak_steps" / "rgb_train_holdout_all.jsonl",
    ROOT / "results" / "weak_steps" / "rgb_train_fit_all.summary.json",
    ROOT / "results" / "weak_steps" / "rgb_train_holdout_all.summary.json",
    ROOT / "results" / "n5k_rgb_test_overhead_b0_b1.jsonl",
    ROOT / "results" / "n5k_rgb_test_overhead_b0_b1.json",
    ROOT / "results" / "n5k_subset_b0_b1.jsonl",
    ROOT / "results" / "n5k_subset_b0_b1.json",
)
NOT_P0D = (
    "Q/Cost decision calibration on logged rgb_train weak steps. "
    "Not the same-budget full-set energy table. "
    "No reinforcement learning. rgb_test labels were not read."
)
FORMULA = (
    "On fit weak steps where uncalibrated Q/Cost picks stop or abstain and a "
    "specialist is legal, excess = best terminal Q/Cost - best legal specialist "
    "Q/Cost. tau = median(excess) + 3 * MAD(excess). MAD is the median absolute "
    "deviation from the median. The multiplier 3 is the usual robust fence and "
    "was not scored on the official hold-out or on rgb_test. At decision time, "
    "if a terminal action wins by less than tau, the rule utility selects. "
    "Otherwise the action stays argmax Q/Cost. Learned Q weights are unchanged."
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


def _hist(actions: Sequence[str]) -> dict[str, int]:
    counts = Counter(actions)
    return {name: int(counts.get(name, 0)) for name in BC_ACTIONS}


def _agree(left: Sequence[str], right: Sequence[str]) -> float:
    if len(left) != len(right) or not left:
        raise ValueError("agreement needs two equal non-empty sequences")
    return sum(a == b for a, b in zip(left, right)) / len(left)


def _selection_summary(actions: Sequence[str]) -> dict[str, Any]:
    labels = [str(action) for action in actions]
    nonstop = [action for action in labels if action != "STOP"]
    specialists = [action for action in nonstop if action in SPECIALIST_ACTIONS]
    mass_or_retrieval = [action for action in nonstop if action in ("RETRIEVE", "RE_PORTION")]
    n_nonstop = len(nonstop)
    return {
        "n": len(labels),
        "histogram": _hist(labels),
        "n_stop": int(sum(action == "STOP" for action in labels)),
        "n_abstain": int(sum(action == "ABSTAIN" for action in labels)),
        "n_nonstop": n_nonstop,
        "n_specialist_on_nonstop": len(specialists),
        "specialist_call_rate_on_nonstop": None if n_nonstop == 0 else len(specialists) / n_nonstop,
        "n_mass_or_retrieval_on_nonstop": len(mass_or_retrieval),
        "mass_or_retrieval_call_rate_on_nonstop": (
            None if n_nonstop == 0 else len(mass_or_retrieval) / n_nonstop
        ),
        "n_retrieve": int(sum(action == "RETRIEVE" for action in labels)),
        "n_re_portion": int(sum(action == "RE_PORTION" for action in labels)),
        "n_correct_code": int(sum(action == "CORRECT_CODE" for action in labels)),
    }


def _controller_stub(policy: Any, *, label: str) -> dict[str, Any]:
    """One rgb_test id, stub tools, no cafe label."""
    dish_id = sorted(rgb_test_ids())[0]
    calls: list[int] = []

    def mass_fn(image, candidates, component_index=None, component_hints_g=None, catalog_path=None, **kwargs):
        del image, component_index, component_hints_g, catalog_path, kwargs
        calls.append(len(list(candidates or [])))
        return {
            "portion_g": 210.0,
            "mass_g": 210.0,
            "sigma": {
                "mass_g": 12.0,
                "energy_kcal": 40.0,
                "protein_g": 2.0,
                "carb_g": 4.0,
                "fat_g": 2.0,
            },
            "nutrients": {"energy_kcal": 180.0, "protein_g": 2.0, "carb_g": 20.0, "fat_g": 1.0},
            "rgb_only": True,
            "depth_used": False,
            "portion_src": "mass_estimator",
        }

    class Result:
        cands = [
            {"code": "111", "score": 0.91, "name": "Apple raw"},
            {"code": "222", "score": 0.80, "name": "Apple juice"},
        ]

    def retrieve_fn(query, top_k=5, catalog_path=None):
        del top_k, catalog_path
        found = Result()
        found.query = query  # type: ignore[attr-defined]
        return found

    def lookup_fn(code, catalog_path=None):
        del code, catalog_path
        return {"energy_kcal": 100.0, "protein_g": 1.0, "carb_g": 10.0, "fat_g": 1.0}

    class Portion:
        portion_g = 100.0

    rolled = rollout_dish(
        dish_id,
        policy,
        image_path="overhead/rgb.png",
        perceive_fn=lambda _path: (
            [{"name": "apple", "portion_g": 180.0}],
            {"model_alias": "qwen3.8-27b", "serving": "self-check"},
        ),
        retrieve_fn=retrieve_fn,
        lookup_fn=lookup_fn,
        portion_fn=lambda code, hint_g=100.0, catalog_path=None: Portion(),
        mass_fn=mass_fn,
        check_replay=True,
    )
    actions = [step["action_t"] for step in rolled.steps]
    portion_src = None
    for step in rolled.steps:
        if step["action_t"] == "RE_PORTION":
            portion_src = step["obs_t"].get("portion_src")
    return {
        "selector": label,
        "dish_id": dish_id,
        "actions": actions,
        "ended_reason": rolled.ended_reason,
        "n_steps": len(rolled.steps),
        "mass_calls": len(calls),
        "retrieval_calls": int(sum(action == "RETRIEVE" for action in actions)),
        "portion_src": portion_src,
        "first_action": None if not actions else actions[0],
        "replay_ok": True,
        "read_cafe_label": False,
        "vlm": False,
        "note": "Stub perception and stub estimate_mass. Not an rgb_test score.",
    }


def _report(metrics: Mapping[str, Any]) -> str:
    raw = metrics["holdout_uncalibrated"]
    cal = metrics["holdout_calibrated"]
    stub_raw = metrics["controller_stub"]["uncalibrated_q_over_cost"]
    stub_cal = metrics["controller_stub"]["calibrated_q_over_cost"]
    fence = metrics["fence"]
    lines = [
        "Q/Cost calibration — unresolved terminal margin defers to the rule utility",
        (
            f"fit_rows={metrics['fit_rows']} holdout_rows={metrics['holdout_rows']} "
            f"tau={fence['tau']:.6f} median_excess={fence['median']:.6f} mad={fence['mad']:.6f}"
        ),
        (
            "hold-out agreement with the rule utility "
            f"uncalibrated {raw['agreement_with_rule_utility']:.4f} "
            f"calibrated {cal['agreement_with_rule_utility']:.4f} "
            f"(P0c baseline {P0C_AGREEMENT:.4f}, n={cal['n']})"
        ),
        "uncalibrated actions " + " ".join(f"{name}={raw['histogram'][name]}" for name in BC_ACTIONS),
        "calibrated actions " + " ".join(f"{name}={cal['histogram'][name]}" for name in BC_ACTIONS),
        "rule utility actions " + " ".join(
            f"{name}={metrics['holdout_rule_utility']['histogram'][name]}" for name in BC_ACTIONS
        ),
        (
            "specialist call rate on non-stop steps "
            f"uncalibrated {raw['specialist_call_rate_on_nonstop']:.4f} "
            f"({raw['n_specialist_on_nonstop']}/{raw['n_nonstop']}) "
            f"calibrated {cal['specialist_call_rate_on_nonstop']:.4f} "
            f"({cal['n_specialist_on_nonstop']}/{cal['n_nonstop']})"
        ),
        (
            "mass or retrieval among non-stop steps "
            f"uncalibrated {raw['mass_or_retrieval_call_rate_on_nonstop']:.4f} "
            f"calibrated {cal['mass_or_retrieval_call_rate_on_nonstop']:.4f}"
        ),
        (
            f"calibrated counts retrieve={cal['n_retrieve']} mass={cal['n_re_portion']} "
            f"code_fix={cal['n_correct_code']} stop={cal['n_stop']} abstain={cal['n_abstain']}"
        ),
        (
            "decisions kept on Q/Cost "
            f"{metrics['n_kept_q_over_cost']} / {metrics['holdout_rows']}; "
            f"deferred to the rule utility {metrics['n_deferred_to_rule_utility']}"
        ),
        f"stub uncalibrated {stub_raw['actions']} mass_calls={stub_raw['mass_calls']}",
        (
            f"stub calibrated {stub_cal['actions']} mass_calls={stub_cal['mass_calls']} "
            f"portion_src={stub_cal['portion_src']}"
        ),
        NOT_P0D,
    ]
    return "\n".join(lines) + "\n"


def main() -> int:
    parser = argparse.ArgumentParser(description="Calibrate unresolved Q/Cost terminal decisions")
    parser.add_argument("--fit", type=Path, default=FIT_JSONL)
    parser.add_argument("--holdout", type=Path, default=HOLDOUT_JSONL)
    parser.add_argument("--out-dir", type=Path, default=OUT_DIR)
    parser.add_argument("--weights", type=Path, default=WEIGHTS)
    args = parser.parse_args()
    assert_safe_result_dir(args.out_dir)
    if not args.weights.is_file():
        raise SystemExit(f"missing Q weights {args.weights}")

    before = _frozen_hashes()
    started = time.perf_counter()
    sigma_path = args.out_dir / "mass_sigma_rgb_train.json"
    if not sigma_path.is_file():
        raise SystemExit(f"missing mass sigma cache {sigma_path}")
    sigma_payload = json.loads(sigma_path.read_text(encoding="utf-8"))
    if sigma_payload.get("uses_rgb_test_gt") is not False or sigma_payload.get("rgb_only") is not True:
        raise SystemExit("mass sigma cache is not the RGB-only fit cache")
    sigma = sigma_payload["dishes"]

    # Fence from the fit pool only. The hold-out file is not read yet.
    fit = load_q_rows(args.fit, expect_pool="fit", sigma_by_dish=sigma)
    model = QPhi.from_npz(args.weights)
    fit_q = model.predict_q(fit.Z)
    excesses = terminal_win_excesses(fit.Z, fit_q)
    fence = robust_fence(excesses)
    tau = float(fence["tau"])
    if tau <= 0.0:
        raise SystemExit(f"terminal margin fence is not positive: {tau}")
    print(
        f"fit rows={fit.n_rows} terminal wins with a specialist={int(fence['n'])} "
        f"median={float(fence['median']):.6f} mad={float(fence['mad']):.6f} tau={tau:.6f}",
        flush=True,
    )

    args.out_dir.mkdir(parents=True, exist_ok=True)
    calibration_path = args.out_dir / CALIBRATION_PATH.name
    calibration = {
        "rule": RULE_ID,
        "tau": tau,
        "multiplier": MARGIN_MAD_MULTIPLIER,
        "formula": FORMULA,
        "population": (
            "rgb_train fit weak steps whose uncalibrated Q/Cost action is stop or "
            "abstain and a specialist is legal. The official hold-out is not an input."
        ),
        "fit_rows": fit.n_rows,
        "fit_dishes": fit.n_dishes,
        "n_terminal_wins_with_legal_specialist": int(fence["n"]),
        "median_excess": float(fence["median"]),
        "mad_excess": float(fence["mad"]),
        "mean_excess": float(fence["mean"]),
        "fraction_terminal_wins_below_tau": float(fence["fraction_below_tau"]),
        "costs": ACTION_COST,
        "weights": str(args.weights.relative_to(ROOT)) if args.weights.is_relative_to(ROOT) else str(args.weights),
        "weights_sha256_prefix": _sha256(args.weights)[:16] if _sha256(args.weights) else None,
        "holdout_used_for_selection": False,
        "uses_rgb_test_gt": False,
        "rl": False,
        "not_p0d": NOT_P0D,
    }
    calibration_path.write_text(json.dumps(calibration, indent=2) + "\n", encoding="utf-8")

    policy = load_calibrated_policy(args.weights, calibration_path)
    if abs(float(policy.terminal_margin) - tau) > 1e-15:
        raise SystemExit("reloaded terminal margin does not match the fence")
    uncalibrated = QCostPolicy(model)
    heuristic = HeuristicUtilityPolicy()

    hold = load_q_rows(args.holdout, expect_pool="holdout", sigma_by_dish=sigma)
    rule_actions = heuristic.predict_actions(hold.Z)
    raw_actions = uncalibrated.predict_actions(hold.Z)
    cal_actions = policy.predict_actions(hold.Z)
    raw_summary = _selection_summary(raw_actions)
    cal_summary = _selection_summary(cal_actions)
    rule_summary = _selection_summary(rule_actions)
    raw_agreement = _agree(raw_actions, rule_actions)
    cal_agreement = _agree(cal_actions, rule_actions)
    n_changed = int(sum(a != b for a, b in zip(raw_actions, cal_actions)))
    hold_q = model.predict_q(hold.Z)
    sources: list[str] = []
    replay: list[str] = []
    for row_z, row_q in zip(hold.Z, hold_q):
        payload = {action: float(row_q[index]) for index, action in enumerate(BC_ACTIONS)}
        action, source = calibrated_decision(payload, row_z, tau)
        replay.append(action)
        sources.append(source)
    if replay != cal_actions:
        raise SystemExit("QCostPolicy disagreed with calibrated_decision")
    n_rule = int(sum(source == "rule_utility" for source in sources))
    n_q = int(sum(source == "q_over_cost" for source in sources))

    stub_raw = _controller_stub(uncalibrated, label="uncalibrated_q_over_cost")
    stub_cal = _controller_stub(policy, label="calibrated_q_over_cost")
    stub_rule = _controller_stub(heuristic, label="rule_utility")

    metrics: dict[str, Any] = {
        "task": "Q/Cost terminal-margin calibration",
        "rule": RULE_ID,
        "rl": False,
        "uses_rgb_test_gt": False,
        "holdout_used_for_selection": False,
        "not_p0d": NOT_P0D,
        "formula": FORMULA,
        "p0c_agreement_q_vs_rule_utility": P0C_AGREEMENT,
        "fence": {
            "tau": tau,
            "median": float(fence["median"]),
            "mad": float(fence["mad"]),
            "multiplier": MARGIN_MAD_MULTIPLIER,
            "mean": float(fence["mean"]),
            "n_terminal_wins_with_legal_specialist": int(fence["n"]),
            "fraction_terminal_wins_below_tau": float(fence["fraction_below_tau"]),
        },
        "fit_rows": fit.n_rows,
        "fit_dishes": fit.n_dishes,
        "holdout_rows": hold.n_rows,
        "holdout_dishes": hold.n_dishes,
        "seconds": round(time.perf_counter() - started, 2),
        "holdout_uncalibrated": {
            "agreement_with_rule_utility": raw_agreement,
            **raw_summary,
        },
        "holdout_calibrated": {
            "agreement_with_rule_utility": cal_agreement,
            **cal_summary,
        },
        "holdout_rule_utility": rule_summary,
        "n_deferred_to_rule_utility": n_rule,
        "n_kept_q_over_cost": n_q,
        "n_actions_changed_from_uncalibrated": n_changed,
        "controller_stub": {
            "uncalibrated_q_over_cost": stub_raw,
            "calibrated_q_over_cost": stub_cal,
            "rule_utility": stub_rule,
        },
        "weights": calibration["weights"],
        "calibration": str(calibration_path.relative_to(ROOT)),
    }

    if abs(raw_agreement - P0C_AGREEMENT) > 1e-12:
        raise SystemExit(
            f"uncalibrated agreement {raw_agreement} did not reproduce the P0c baseline {P0C_AGREEMENT}"
        )
    if not (cal_agreement > P0C_AGREEMENT and cal_agreement > raw_agreement):
        raise SystemExit(f"calibrated agreement {cal_agreement} is not above {P0C_AGREEMENT}")
    if stub_rule["actions"] != ["RETRIEVE", "RE_PORTION", "STOP"] or stub_rule["mass_calls"] != 1:
        raise SystemExit(f"rule-utility stub did not call mass: {stub_rule}")
    if stub_raw["actions"] != ["STOP"] or stub_raw["mass_calls"] != 0:
        raise SystemExit(f"uncalibrated stub changed from the P0c stop-on-step-1 trace: {stub_raw}")
    if stub_cal["first_action"] == "STOP" or stub_cal["mass_calls"] < 1:
        raise SystemExit(f"calibrated stub did not call the mass estimator: {stub_cal}")
    if "RETRIEVE" not in stub_cal["actions"] or "RE_PORTION" not in stub_cal["actions"]:
        raise SystemExit(f"calibrated stub did not retrieve and portion: {stub_cal}")
    if stub_cal["portion_src"] != "mass_estimator":
        raise SystemExit(f"mass step did not record portion_src=mass_estimator: {stub_cal}")
    if cal_summary["n_retrieve"] <= raw_summary["n_retrieve"]:
        raise SystemExit("calibration did not raise retrieval calls on the hold-out")
    if cal_summary["n_re_portion"] <= raw_summary["n_re_portion"]:
        raise SystemExit("calibration did not raise mass calls on the hold-out")
    specialist_rate = cal_summary["specialist_call_rate_on_nonstop"]
    if specialist_rate is None or specialist_rate < 0.9:
        raise SystemExit(f"specialist call rate on non-stop steps is {specialist_rate}")
    if n_rule <= 0 or n_q <= n_rule:
        raise SystemExit(f"Q/Cost no longer makes the majority of decisions (q={n_q}, rule={n_rule})")

    after = _frozen_hashes()
    if after != before:
        changed = [key for key in before if before[key] != after.get(key)]
        raise SystemExit(f"frozen files changed: {changed}")
    metrics["frozen_sha256"] = {key: None if value is None else value[:16] for key, value in before.items()}

    holdout_path = args.out_dir / HOLDOUT_METRICS_PATH.name
    report_path = args.out_dir / REPORT_PATH.name
    holdout_path.write_text(json.dumps(metrics, indent=2) + "\n", encoding="utf-8")
    report_path.write_text(_report(metrics), encoding="utf-8")
    print(_report(metrics), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
