#!/usr/bin/env python3
"""Lock the abstain score on the fit pool. Does not read the official hold-out.

    python scripts/closed_loop_abstain_calib.py --self-check
    python scripts/closed_loop_abstain_calib.py

Writes ``results/closed_loop/abstain_calibration.json`` only. Does not update
``policy_state_mlp.npz`` and does not write the hold-out energy table.
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np  # noqa: E402

from vdr.controller.closed_loop_data import FIT_JSONL  # noqa: E402
from vdr.controller.state_features import STATE_FEATURE_NAMES  # noqa: E402
from vdr.data.n5k import test_split_ids  # noqa: E402
from vdr.eval.abstain_calibration import (  # noqa: E402
    SCORE_RULE_ID,
    AbstainCalibrationError,
    candidate_scores,
    run_self_check,
    select_rule,
)
from vdr.eval.energy_selective import EnergyEvalRefused, dish_readings  # noqa: E402
from vdr.eval.selective import aurc  # noqa: E402
from scripts.closed_loop_mae_aurc import (  # noqa: E402
    _cafe_labels,
    _frozen_paths,
    _group_steps,
    _load_policy,
    _maps,
    _predict,
    _sha256,
    _stack_state,
)

OUT = ROOT / "results" / "closed_loop" / "abstain_calibration.json"
WEIGHTS = ROOT / "results" / "closed_loop" / "policy_state_mlp.npz"
TRAIN_META = ROOT / "results" / "closed_loop" / "train_meta.json"


def _rows(fit_path: Path) -> tuple[list[dict], dict]:
    forbidden = test_split_ids()
    steps = _group_steps(fit_path, expect_pool="fit", allow_test=False)
    leaked = sorted(dish_id for dish_id in steps if dish_id in forbidden)
    if leaked:
        raise AbstainCalibrationError(f"fit trajectories include test dishes: {leaked[:3]}")
    state, keys, _truth = _stack_state(steps)
    policy = _load_policy(WEIGHTS, TRAIN_META)
    actions, _softmax = _predict(policy, state)
    pred_map, score_map = _maps(keys, actions, _softmax)
    labels = _cafe_labels(set(steps))
    readings = dish_readings(steps, pred_map, score_map, labels)
    probs = np.asarray(policy.predict_proba(state), dtype=np.float64)
    names = list(policy.actions)
    i_stop = names.index("STOP")
    i_abs = names.index("ABSTAIN")
    energy_index = STATE_FEATURE_NAMES.index("obs_nutri_energy_norm")
    key_to_row = {key: index for index, key in enumerate(keys)}
    built: list[dict] = []
    n_early = 0
    n_grounded = 0
    for reading in readings:
        if not reading.grounded_stop or reading.energy_kcal is None or reading.error_kcal is None:
            continue
        n_grounded += 1
        if not reading.decision_is_last:
            n_early += 1
        index = key_to_row[(reading.dish_id, reading.decision_step_i)]
        built.append(
            {
                "dish_id": reading.dish_id,
                "energy_kcal": float(reading.energy_kcal),
                "error_kcal": float(reading.error_kcal),
                "p_abstain": float(probs[index, i_abs]),
                "p_stop": float(probs[index, i_stop]),
                "obs_energy_norm": float(state[index, energy_index]),
            }
        )
    meta = {
        "n_dishes": len(steps),
        "n_grounded_stop_scored": n_grounded,
        "n_early_decision": n_early,
        "n_rows_for_selection": len(built),
    }
    return built, meta


def _all_fit_aurc(rows: list[dict]) -> dict[str, float]:
    errors = [float(row["error_kcal"]) for row in rows]
    table: dict[str, float] = {}
    for rule in candidate_scores(rows[0]):
        scores = [candidate_scores(row)[rule] for row in rows]
        table[rule] = float(aurc(errors, scores)["aurc"])
    return table


def main() -> int:
    if "--self-check" in sys.argv[1:]:
        run_self_check()
        print(f"closed_loop_abstain_calib self-check ok rule={SCORE_RULE_ID}")
        return 0
    if len(sys.argv) != 1:
        raise AbstainCalibrationError("only --self-check is accepted; there is no hold-out flag")

    run_self_check()
    started = time.perf_counter()
    fit_path = ROOT / FIT_JSONL
    for path in (fit_path, WEIGHTS, TRAIN_META):
        if not path.is_file():
            raise AbstainCalibrationError(f"missing {path}")
    frozen = _frozen_paths()
    before = {path.name: _sha256(path) for path in frozen}
    rows, meta = _rows(fit_path)
    selection = select_rule(rows)
    if selection["winner"] != SCORE_RULE_ID:
        detail = json.dumps(selection["fit_internal_val"]["candidates"], indent=2)
        raise AbstainCalibrationError(
            "fit-internal AURC did not lock "
            f"{SCORE_RULE_ID}; winner={selection['winner']}\n{detail}"
        )
    descriptive = _all_fit_aurc(rows)
    payload = {
        **selection,
        "population": (
            "rgb_train fit pool, logged STOP with status pass or corrected and a "
            "finite force-answer energy. Not the official hold-out. Not rgb_test."
        ),
        "fit_rows": meta,
        "descriptive_all_fit_aurc_not_used_for_selection": descriptive,
        "discarded": {
            "softmax_p_abstain": (
                "The saved head's P(ABSTAIN) is a candidate and loses on the "
                "fit-internal slice, so the action head is not retrained and no "
                "temperature or class-weight is applied."
            ),
            "early_stop": (
                f"{meta['n_early_decision']} of {meta['n_grounded_stop_scored']} "
                "fit grounded-STOP decisions are before the last logged step. "
                "Early STOP is not the AURC failure on this checkpoint."
            ),
            "obs_nutri_energy_norm": (
                "The previous-step observation energy is not the committed total. "
                "It is a candidate and is not the winner."
            ),
        },
        "action_head": "results/closed_loop/policy_state_mlp.npz was not modified",
        "inputs": {
            "fit_jsonl": "results/weak_steps/rgb_train_fit_all.jsonl",
            "fit_sha256_prefix": before.get("rgb_train_fit_all.jsonl", "")[:16],
            "weights": "results/closed_loop/policy_state_mlp.npz",
            "weights_sha256_prefix": before.get("policy_state_mlp.npz", "")[:16],
            "official_holdout_read": False,
            "results_n5k_read": False,
        },
        "elapsed_seconds": round(time.perf_counter() - started, 2),
    }
    if OUT.parent.resolve() != (ROOT / "results" / "closed_loop").resolve():
        raise AbstainCalibrationError(f"refusing to write {OUT}")
    OUT.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    after = {path.name: _sha256(path) for path in frozen}
    changed = sorted(name for name, digest in before.items() if after.get(name) != digest)
    if changed:
        OUT.unlink(missing_ok=True)
        raise AbstainCalibrationError("frozen files changed: " + ", ".join(changed))
    val = payload["fit_internal_val"]["candidates"]
    print(
        f"locked {SCORE_RULE_ID} on fit-internal val "
        f"n={payload['fit_internal_val']['n_val']} "
        f"aurc={val[SCORE_RULE_ID]['aurc']:.4f} "
        f"(softmax {val['softmax_p_abstain']['aurc']:.4f}); "
        f"early={meta['n_early_decision']}; wrote {OUT.relative_to(ROOT)}"
    )
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (AbstainCalibrationError, EnergyEvalRefused) as exc:
        print(f"ABSTAIN CALIBRATION REFUSED: {exc}", file=sys.stderr)
        raise SystemExit(1) from exc
