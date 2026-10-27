#!/usr/bin/env python3
"""CPU smoke for BC features and the numpy logistic toy.

Checks that ``build_features`` ignores ``obs_t``, that the loader refuses
test-split names, then fits on ``rgb_train_fit_all.jsonl`` and reports
action accuracy only on ``rgb_train_holdout_all.jsonl``.

No VLM, no volume tool, no rgb_test labels. Hyperparameters stay at the
module defaults (not selected on the hold-out).

    python scripts/bc_features_smoke.py
"""

from __future__ import annotations

import sys
import time
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np  # noqa: E402

from vdr.controller.bc_data import (  # noqa: E402
    BCDataRefused,
    action_counts,
    assert_bc_jsonl_path,
    load_bc_rows,
)
from vdr.controller.features import (  # noqa: E402
    FEATURE_DIM,
    FEATURE_NAMES,
    IGNORED_STATE_FIELDS,
    PRE_ACTION_FIELDS,
    STEP_BUDGET,
    VERIFY_CODE_VOCAB,
    build_features,
)
from vdr.controller.policy_bc import BC_ACTIONS, PolicyBC  # noqa: E402
from vdr.ir.io import read_ir_jsonl  # noqa: E402
from vdr.weak.teacher import PROTECTED_RESULT_NAMES  # noqa: E402

FIT_PATH = ROOT / "results/weak_steps/rgb_train_fit_all.jsonl"
HOLDOUT_PATH = ROOT / "results/weak_steps/rgb_train_holdout_all.jsonl"
PREFIX_ROWS = 8


def check(name: str, ok: bool, detail: str = "") -> bool:
    print(f"  {'OK' if ok else 'FAIL'} {name}" + (f" — {detail}" if detail else ""))
    return ok


def _index(name: str) -> int:
    return FEATURE_NAMES.index(name)


def test_feature_contract() -> bool:
    ok = True
    ok &= check("feature names are unique", len(FEATURE_NAMES) == len(set(FEATURE_NAMES)) == FEATURE_DIM, str(FEATURE_DIM))
    ok &= check(
        "pre-action fields are the six IR inputs",
        PRE_ACTION_FIELDS
        == ("hyp_i", "code_i", "cands_i", "portion_g_i", "verify_codes", "step_i"),
    )
    ok &= check("obs_t is not a feature field", "obs_t" not in PRE_ACTION_FIELDS and "obs_t" in IGNORED_STATE_FIELDS)
    ok &= check("volume action is not in the BC set", "VOLUME_ESTIMATE" not in BC_ACTIONS, str(BC_ACTIONS))
    refused_narrow = False
    try:
        PolicyBC(n_actions=5)
    except ValueError:
        refused_narrow = True
    ok &= check("n_actions=5 does not drop CORRECT_CODE", refused_narrow)
    ok &= check("default action head has six names", PolicyBC().n_actions == len(BC_ACTIONS) == 6)
    unfitted = False
    try:
        PolicyBC().predict_action(np.zeros(FEATURE_DIM))
    except RuntimeError:
        unfitted = True
    ok &= check("predict before fit raises", unfitted)

    base = {
        "hyp_i": "brown rice",
        "code_i": "AAA",
        "cands_i": [
            {"code": "AAA", "score": 0.80, "name": "rice"},
            {"code": "BBB", "score": 0.70, "name": "other"},
        ],
        "portion_g_i": 100.0,
        "verify_codes": ["CODE_CONF_GAP"],
        "step_i": 4,
        "portion_src_i": "grid_search",
        "nutri_i": {"energy_kcal": 1.0},
        "nutri_dish": {"energy_kcal": 10.0},
        "evidence_edges": [{"src": "a", "dst": "b", "type": "n<-aggregate"}],
        "s": None,
        "abstain_reason": None,
        "component_i": 1,
        "dish_id": "dish_toy",
        "action_t": "RE_PORTION",
        "obs_t": {
            "portion_g": 150.0,
            "from_portion_g": 100.0,
            "to": "SECRET",
            "from": "AAA",
            "query": "brown rice cooked",
            "privileged": True,
            "label_quality": "weak",
        },
        "train_gt_energy_kcal": 999.0,
    }
    first = build_features(base)
    leaked = dict(base)
    leaked["obs_t"] = {"portion_g": 1.0, "to": "OTHER", "query": "not the hyp", "privileged": True}
    leaked["nutri_dish"] = {"energy_kcal": 12345.0}
    leaked["action_t"] = "VOLUME_ESTIMATE"
    leaked["train_gt_energy_kcal"] = 0.0
    leaked["s"] = "abstain"
    second = build_features(leaked)
    ok &= check(
        "obs_t and other post-action fields do not change h",
        first.shape == (FEATURE_DIM,) and np.array_equal(first, second),
        f"dim={first.shape[0]}",
    )
    via_kwargs = build_features(base, obs_t={"portion_g": 5.0}, action_t="STOP")
    ok &= check("kwargs cannot inject obs_t", np.array_equal(first, via_kwargs))

    ok &= check("dtype is float64 and finite", first.dtype == np.float64 and bool(np.isfinite(first).all()))
    ok &= check("hyp present", first[_index("hyp_present")] == 1.0)
    ok &= check("code is top1", first[_index("code_is_top1")] == 1.0 and first[_index("code_in_cands")] == 1.0)
    ok &= check(
        "score gap is top1-top2",
        abs(first[_index("score_gap")] - 0.10) < 1e-12
        and abs(first[_index("top1_score")] - 0.80) < 1e-12
        and abs(first[_index("top2_score")] - 0.70) < 1e-12,
        f"gap={first[_index('score_gap')]} top1={first[_index('top1_score')]} top2={first[_index('top2_score')]}",
    )
    gap_at = _index("verify_CODE_CONF_GAP")
    other_verify = [
        first[_index(f"verify_{name}")] for name in VERIFY_CODE_VOCAB if name != "CODE_CONF_GAP"
    ]
    ok &= check(
        "CODE_CONF_GAP one-hot",
        first[gap_at] == 1.0 and all(value == 0.0 for value in other_verify) and first[_index("verify_other")] == 0.0,
    )
    ok &= check("portion scaled by 1500", abs(first[_index("portion_g_norm")] - (100.0 / 1500.0)) < 1e-12)
    ok &= check(
        "step budget uses the fixed cap",
        abs(first[_index("step_frac")] - (4.0 / STEP_BUDGET)) < 1e-12 and first[_index("step_over_budget")] == 0.0,
    )

    over = build_features({"step_i": STEP_BUDGET + 5, "hyp_i": None, "code_i": None, "cands_i": [], "portion_g_i": None, "verify_codes": []})
    ok &= check(
        "step at or past the cap saturates",
        over[_index("step_frac")] == 1.0
        and over[_index("step_remaining_frac")] == 0.0
        and over[_index("step_over_budget")] == 1.0
        and over[_index("hyp_present")] == 0.0,
    )

    unknown = build_features({"verify_codes": ["NOT_A_RULE", "CODE_CONF_GAP"], "cands_i": [], "step_i": 0})
    ok &= check(
        "unknown verify code sets the other bit",
        unknown[_index("verify_other")] == 1.0 and unknown[_index("verify_CODE_CONF_GAP")] == 1.0,
    )

    ranked = build_features(
        {
            "code_i": "CCC",
            "cands_i": [
                {"code": "AAA", "score": 0.9},
                {"code": "BBB", "score": 0.5},
                {"code": "CCC", "score": 0.4},
            ],
            "step_i": 1,
        }
    )
    ok &= check(
        "code rank is 2 of 5 slots",
        ranked[_index("code_is_top1")] == 0.0
        and ranked[_index("code_in_cands")] == 1.0
        and abs(ranked[_index("code_rank_norm")] - (2.0 / 4.0)) < 1e-12,
    )
    missing = build_features({"code_i": "ZZZ", "cands_i": [{"code": "AAA", "score": 0.2}], "step_i": 0})
    ok &= check(
        "code absent from cands is rank 1",
        missing[_index("code_present")] == 1.0
        and missing[_index("code_in_cands")] == 0.0
        and missing[_index("code_rank_norm")] == 1.0,
    )
    return ok


def test_path_guards() -> bool:
    ok = True
    samples = [
        ROOT / "results" / "n5k_rgb_test_overhead_b0_b1.jsonl",
        ROOT / "results" / "n5k_subset_b0_b1.jsonl",
        ROOT / "results" / "n5k_subset_b0_b1_n50_seed42.jsonl",
        ROOT / "results" / "weak_steps" / "rgb_test_steps.jsonl",
        ROOT / "results" / "weak_steps" / "depth_test_all.jsonl",
        ROOT / "results" / "weak_steps" / "rgb_train_fit_all.summary.json",
    ]
    for path in samples:
        blocked = False
        try:
            assert_bc_jsonl_path(path)
        except BCDataRefused:
            blocked = True
        ok &= check(f"protect {path.name}", blocked)
    for name in PROTECTED_RESULT_NAMES:
        if not name.endswith(".jsonl"):
            continue
        blocked = False
        try:
            assert_bc_jsonl_path(ROOT / "results" / name)
        except BCDataRefused:
            blocked = True
        ok &= check(f"protect frozen {name}", blocked)
    return ok


def test_prefix_rows() -> bool:
    if not FIT_PATH.is_file():
        return check("fit jsonl present", False, str(FIT_PATH))
    rows = load_bc_rows(FIT_PATH, expect_pool="fit", limit=PREFIX_ROWS)
    ok = check(
        f"prefix {PREFIX_ROWS} fit rows",
        rows.n_rows == PREFIX_ROWS and rows.feature_dim == FEATURE_DIM and rows.X.dtype == np.float64,
        f"rows={rows.n_rows} dim={rows.feature_dim} dishes={rows.n_dishes}",
    )
    ok &= check("prefix labels are BC actions", set(rows.y.tolist()).issubset(set(BC_ACTIONS)), str(sorted(set(rows.y.tolist()))))
    matched = True
    for index, record in enumerate(read_ir_jsonl(FIT_PATH)):
        if index >= PREFIX_ROWS:
            break
        mutated = dict(record)
        mutated["obs_t"] = {"portion_g": 999.0, "to": "LEAK", "from": "LEAK", "query": "leak"}
        mutated["nutri_dish"] = {"energy_kcal": 1.0}
        if not np.array_equal(build_features(record), build_features(mutated)):
            matched = False
            break
        if not np.array_equal(rows.X[index], build_features(record)):
            matched = False
            break
    ok &= check("prefix rows match build_features and ignore obs_t", matched)
    return ok


def _recall(y_true: np.ndarray, y_pred: list[str]) -> dict[str, tuple[int, float]]:
    truth = [str(item) for item in y_true.tolist()]
    out: dict[str, tuple[int, float]] = {}
    for action in BC_ACTIONS:
        idx = [i for i, label in enumerate(truth) if label == action]
        if not idx:
            out[action] = (0, float("nan"))
            continue
        hit = sum(1 for i in idx if y_pred[i] == action)
        out[action] = (len(idx), hit / len(idx))
    return out


def test_holdout_accuracy() -> bool:
    if not FIT_PATH.is_file() or not HOLDOUT_PATH.is_file():
        return check("fit and hold-out jsonl present", False, f"fit={FIT_PATH.is_file()} holdout={HOLDOUT_PATH.is_file()}")
    started = time.perf_counter()
    fit_rows = load_bc_rows(FIT_PATH, expect_pool="fit")
    hold_rows = load_bc_rows(HOLDOUT_PATH, expect_pool="holdout")
    fit_ids = set(fit_rows.dish_ids)
    hold_ids = set(hold_rows.dish_ids)
    disjoint = fit_ids.isdisjoint(hold_ids)
    policy = PolicyBC()
    policy.fit(fit_rows.X, fit_rows.y)
    hold_pred = policy.predict_actions(hold_rows.X)
    train_pred = policy.predict_actions(fit_rows.X)
    hold_hits = sum(1 for left, right in zip(hold_rows.y.tolist(), hold_pred) if str(left) == right)
    train_hits = sum(1 for left, right in zip(fit_rows.y.tolist(), train_pred) if str(left) == right)
    hold_acc = hold_hits / hold_rows.n_rows
    train_acc = train_hits / fit_rows.n_rows
    majority = Counter(str(item) for item in fit_rows.y.tolist()).most_common(1)[0][0]
    majority_acc = sum(1 for label in hold_rows.y.tolist() if str(label) == majority) / hold_rows.n_rows
    elapsed = time.perf_counter() - started
    recalls = _recall(hold_rows.y, hold_pred)
    recall_bits = []
    for action, (count, value) in recalls.items():
        if count == 0:
            recall_bits.append(f"{action}:n=0")
        else:
            recall_bits.append(f"{action}:n={count} recall={value:.4f}")
    detail = (
        f"feature_dim={FEATURE_DIM} actions={list(BC_ACTIONS)} "
        f"fit_rows={fit_rows.n_rows} fit_dishes={fit_rows.n_dishes} "
        f"holdout_rows={hold_rows.n_rows} holdout_dishes={hold_rows.n_dishes} "
        f"holdout_acc={hold_acc:.4f} ({hold_hits}/{hold_rows.n_rows}) "
        f"train_acc={train_acc:.4f} ({train_hits}/{fit_rows.n_rows}) "
        f"majority={majority} majority_holdout_acc={majority_acc:.4f} "
        f"fit_actions={action_counts(fit_rows.y)} "
        f"holdout_recall={' '.join(recall_bits)} "
        f"seconds={elapsed:.2f}"
    )
    print(detail)
    ok = True
    ok &= check("fit and hold-out dishes are disjoint", disjoint, f"fit={len(fit_ids)} holdout={len(hold_ids)}")
    ok &= check("feature dim on both pools", fit_rows.feature_dim == hold_rows.feature_dim == FEATURE_DIM)
    ok &= check(
        "hold-out accuracy is a probability above the fit-set majority baseline",
        0.0 <= hold_acc <= 1.0 and hold_acc > majority_acc,
        f"holdout_acc={hold_acc:.4f} majority_holdout_acc={majority_acc:.4f}",
    )
    # The linear toy should recover the coarse pre-action rule (empty -> STOP,
    # hypothesis and no candidates -> RETRIEVE, else mostly RE_PORTION).
    ok &= check(
        "hold-out accuracy clears 0.85",
        hold_acc >= 0.85,
        f"{hold_acc:.4f}",
    )
    single = policy.predict_action(hold_rows.X[0])
    ok &= check("predict_action matches the first hold-out row", single == hold_pred[0], single)
    return ok


def main() -> int:
    print("bc features smoke")
    ok = True
    ok &= test_feature_contract()
    ok &= test_path_guards()
    ok &= test_prefix_rows()
    ok &= test_holdout_accuracy()
    if not ok:
        print("FAILED")
        return 1
    print("All BC feature smoke checks passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
