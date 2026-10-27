#!/usr/bin/env python3
"""CPU smoke for the small MLP behavior-cloning policy.

Fits ``PolicyBC`` (linear baseline) and ``PolicyMLP`` on
``results/weak_steps/rgb_train_fit_all.jsonl`` only, then prints action
accuracy and per-class recall on ``results/weak_steps/rgb_train_holdout_all.jsonl``.

Stdout only. Does not write a result file and does not read rgb_test.
Hyperparameters stay at the module defaults (not selected on the hold-out).

    python scripts/bc_mlp_smoke.py
"""

from __future__ import annotations

import json
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
from vdr.controller.features import FEATURE_DIM, build_features  # noqa: E402
from vdr.controller.policy_bc import BC_ACTIONS, PolicyBC  # noqa: E402
from vdr.controller.policy_mlp import (  # noqa: E402
    MLP_BATCH_SIZE,
    MLP_DROPOUT,
    MLP_EPOCHS,
    MLP_HIDDEN_SIZES,
    MLP_L2,
    MLP_LR,
    MLP_SEED,
    PolicyMLP,
    inverse_sqrt_class_weights,
)
from vdr.weak.teacher import PROTECTED_RESULT_NAMES  # noqa: E402

FIT_PATH = ROOT / "results/weak_steps/rgb_train_fit_all.jsonl"
HOLDOUT_PATH = ROOT / "results/weak_steps/rgb_train_holdout_all.jsonl"
FIT_SUMMARY = ROOT / "results/weak_steps/rgb_train_fit_all.summary.json"
HOLDOUT_SUMMARY = ROOT / "results/weak_steps/rgb_train_holdout_all.summary.json"

# Locked to the fixed-seed CPU run documented in docs/STATUS.md.
LINEAR_HOLD_HITS = 2551
MLP_HOLD_HITS = 2688
HOLD_ROWS = 2771
MLP_RECALL_HITS = {
    "RETRIEVE": 1273,
    "RE_PORTION": 917,
    "RE_DECOMPOSE": 0,
    "ABSTAIN": 0,
    "STOP": 356,
    "CORRECT_CODE": 142,
}
HOLD_SUPPORT = {
    "RETRIEVE": 1285,
    "RE_PORTION": 949,
    "RE_DECOMPOSE": 0,
    "ABSTAIN": 24,
    "STOP": 356,
    "CORRECT_CODE": 157,
}


def check(name: str, ok: bool, detail: str = "") -> bool:
    print(f"  {'OK' if ok else 'FAIL'} {name}" + (f" — {detail}" if detail else ""))
    return ok


def _recall(y_true: np.ndarray, y_pred: list[str]) -> dict[str, tuple[int, int]]:
    truth = [str(item) for item in y_true.tolist()]
    out: dict[str, tuple[int, int]] = {}
    for action in BC_ACTIONS:
        idx = [i for i, label in enumerate(truth) if label == action]
        hit = sum(1 for i in idx if y_pred[i] == action)
        out[action] = (hit, len(idx))
    return out


def _fmt_recall(recall: dict[str, tuple[int, int]]) -> str:
    bits = []
    for action, (hit, count) in recall.items():
        if count == 0:
            bits.append(f"{action}:n=0")
        else:
            bits.append(f"{action}:{hit}/{count}={hit / count:.4f}")
    return " ".join(bits)


def test_contract() -> bool:
    ok = True
    ok &= check("feature dim is 31", FEATURE_DIM == 31, str(FEATURE_DIM))
    ok &= check(
        "MLP shape is 31-64-64-n",
        MLP_HIDDEN_SIZES == (64, 64) and MLP_DROPOUT == 0.0 and MLP_L2 == 1e-4,
        f"hidden={MLP_HIDDEN_SIZES} dropout={MLP_DROPOUT} l2={MLP_L2} lr={MLP_LR} "
        f"epochs={MLP_EPOCHS} batch={MLP_BATCH_SIZE} seed={MLP_SEED}",
    )
    ok &= check("volume action is not in the BC set", "VOLUME_ESTIMATE" not in BC_ACTIONS, str(BC_ACTIONS))
    ok &= check("default MLP head has six names", PolicyMLP().n_actions == len(BC_ACTIONS) == 6)
    refused_narrow = False
    try:
        PolicyMLP(n_actions=5)
    except ValueError:
        refused_narrow = True
    ok &= check("n_actions=5 does not drop CORRECT_CODE", refused_narrow)
    refused_volume = False
    try:
        PolicyMLP(actions=("RETRIEVE", "VOLUME_ESTIMATE"))
    except ValueError:
        refused_volume = True
    ok &= check("VOLUME_ESTIMATE cannot be an MLP action", refused_volume)
    unfitted = False
    try:
        PolicyMLP().predict_action(np.zeros(FEATURE_DIM))
    except RuntimeError:
        unfitted = True
    ok &= check("predict before fit raises", unfitted)

    base = {
        "hyp_i": "brown rice",
        "code_i": "AAA",
        "cands_i": [{"code": "AAA", "score": 0.8, "name": "rice"}],
        "portion_g_i": 100.0,
        "verify_codes": ["CODE_CONF_GAP"],
        "step_i": 2,
        "action_t": "RE_PORTION",
        "obs_t": {"portion_g": 150.0, "to": "SECRET", "privileged": True},
        "train_gt_energy_kcal": 999.0,
    }
    leaked = dict(base)
    leaked["obs_t"] = {"portion_g": 1.0, "to": "OTHER", "privileged": True}
    leaked["action_t"] = "VOLUME_ESTIMATE"
    leaked["train_gt_energy_kcal"] = 0.0
    ok &= check(
        "obs_t does not change h",
        np.array_equal(build_features(base), build_features(leaked)),
    )

    samples = [
        ROOT / "results" / "n5k_rgb_test_overhead_b0_b1.jsonl",
        ROOT / "results" / "weak_steps" / "rgb_test_steps.jsonl",
        ROOT / "results" / "weak_steps" / "depth_test_all.jsonl",
        ROOT / "results" / "n5k_subset_b0_b1.jsonl",
    ]
    for path in samples:
        blocked = False
        try:
            assert_bc_jsonl_path(path)
        except BCDataRefused:
            blocked = True
        ok &= check(f"protect {path.name}", blocked)
    for name in sorted(PROTECTED_RESULT_NAMES):
        if not name.endswith(".jsonl"):
            continue
        blocked = False
        try:
            assert_bc_jsonl_path(ROOT / "results" / name)
        except BCDataRefused:
            blocked = True
        ok &= check(f"protect frozen {name}", blocked)
    return ok


def test_separable_toy() -> bool:
    """A rare class with its own coordinate must be fit. This is not the hold-out."""
    rng = np.random.default_rng(1)
    n_rows = 240
    matrix = rng.normal(scale=0.05, size=(n_rows, FEATURE_DIM))
    labels = np.array(["RE_PORTION"] * n_rows, dtype=object)
    matrix[:30, 0] = 4.0
    matrix[:30, 1] = -1.0
    labels[:30] = "CORRECT_CODE"
    matrix[30:50, 0] = -1.0
    matrix[30:50, 1] = 4.0
    labels[30:50] = "ABSTAIN"
    matrix[50:, 0] = -1.0
    matrix[50:, 1] = -1.0
    policy = PolicyMLP()
    try:
        policy.fit(matrix, labels, holdout=matrix)
        return check("fit rejects a hold-out matrix", False, "kwargs were accepted")
    except TypeError:
        pass
    policy.fit(matrix, labels)
    pred = policy.predict_actions(matrix)
    recall = _recall(labels, pred)
    cc_hit, cc_n = recall["CORRECT_CODE"]
    ab_hit, ab_n = recall["ABSTAIN"]
    ok = True
    ok &= check(
        "separable toy recovers CORRECT_CODE and ABSTAIN",
        cc_hit == cc_n == 30 and ab_hit == ab_n == 20 and "RE_DECOMPOSE" not in pred and "VOLUME_ESTIMATE" not in pred,
        _fmt_recall(recall),
    )
    return ok


def _shared_with(X: np.ndarray, y: np.ndarray, left: str, right: str) -> tuple[int, int]:
    """Rows of ``left`` whose rounded feature vector also appears with ``right``."""
    groups: dict[tuple[float, ...], set[str]] = {}
    keys: list[tuple[float, ...]] = []
    labels = [str(item) for item in y.tolist()]
    for row, label in zip(X, labels):
        key = tuple(np.round(row, 6))
        keys.append(key)
        groups.setdefault(key, set()).add(label)
    total = 0
    shared = 0
    for key, label in zip(keys, labels):
        if label != left:
            continue
        total += 1
        if right in groups[key]:
            shared += 1
    return shared, total


def test_holdout_accuracy() -> bool:
    if not FIT_PATH.is_file() or not HOLDOUT_PATH.is_file():
        return check(
            "fit and hold-out jsonl present",
            False,
            f"fit={FIT_PATH.is_file()} holdout={HOLDOUT_PATH.is_file()}",
        )
    started = time.perf_counter()
    fit_rows = load_bc_rows(FIT_PATH, expect_pool="fit")
    hold_rows = load_bc_rows(HOLDOUT_PATH, expect_pool="holdout")
    fit_ids = set(fit_rows.dish_ids)
    hold_ids = set(hold_rows.dish_ids)
    fit_labels = [str(item) for item in fit_rows.y.tolist()]
    weights = inverse_sqrt_class_weights(fit_labels, BC_ACTIONS)

    linear = PolicyBC()
    linear.fit(fit_rows.X, fit_rows.y)
    mlp = PolicyMLP()
    mlp.fit(fit_rows.X, fit_rows.y)
    mlp_again = PolicyMLP().fit(fit_rows.X, fit_rows.y)

    linear_pred = linear.predict_actions(hold_rows.X)
    mlp_pred = mlp.predict_actions(hold_rows.X)
    mlp_pred_again = mlp_again.predict_actions(hold_rows.X)
    truth = [str(item) for item in hold_rows.y.tolist()]
    linear_hits = sum(left == right for left, right in zip(truth, linear_pred))
    mlp_hits = sum(left == right for left, right in zip(truth, mlp_pred))
    linear_recall = _recall(hold_rows.y, linear_pred)
    mlp_recall = _recall(hold_rows.y, mlp_pred)
    majority = Counter(fit_labels).most_common(1)[0][0]
    majority_hits = sum(label == majority for label in truth)
    shared, abstain_n = _shared_with(fit_rows.X, fit_rows.y, "ABSTAIN", "STOP")
    elapsed = time.perf_counter() - started

    print(
        f"feature_dim={FEATURE_DIM} actions={list(BC_ACTIONS)} "
        f"fit_rows={fit_rows.n_rows} fit_dishes={fit_rows.n_dishes} "
        f"holdout_rows={hold_rows.n_rows} holdout_dishes={hold_rows.n_dishes} "
        f"linear_holdout_acc={linear_hits / hold_rows.n_rows:.4f} ({linear_hits}/{hold_rows.n_rows}) "
        f"mlp_holdout_acc={mlp_hits / hold_rows.n_rows:.4f} ({mlp_hits}/{hold_rows.n_rows}) "
        f"majority={majority} majority_holdout_acc={majority_hits / hold_rows.n_rows:.4f} "
        f"fit_actions={action_counts(fit_rows.y)} "
        f"class_weight={{{', '.join(f'{name}:{weights[name]:.4f}' for name in BC_ACTIONS)}}} "
        f"linear_holdout_recall={_fmt_recall(linear_recall)} "
        f"mlp_holdout_recall={_fmt_recall(mlp_recall)} "
        f"fit_abstain_rows_sharing_stop_vector={shared}/{abstain_n} "
        f"seconds={elapsed:.2f}"
    )

    ok = True
    ok &= check("fit and hold-out dishes are disjoint", fit_ids.isdisjoint(hold_ids))
    ok &= check("both pools are 31-d", fit_rows.feature_dim == hold_rows.feature_dim == 31)
    ok &= check(
        "class weights come from the fit labels",
        mlp._class_weight is not None
        and all(abs(mlp._class_weight[name] - weights[name]) < 1e-12 for name in BC_ACTIONS),
    )
    ok &= check("RE_DECOMPOSE weight is zero", weights["RE_DECOMPOSE"] == 0.0)
    ok &= check("second MLP fit matches the first on hold-out", mlp_pred == mlp_pred_again)
    ok &= check(
        "linear hold-out stays at 2551/2771",
        linear_hits == LINEAR_HOLD_HITS and hold_rows.n_rows == HOLD_ROWS,
        f"{linear_hits}/{hold_rows.n_rows}",
    )
    ok &= check(
        "MLP hold-out accuracy is 2688/2771 and beats linear",
        mlp_hits == MLP_HOLD_HITS and mlp_hits > linear_hits,
        f"{mlp_hits}/{hold_rows.n_rows}",
    )
    ok &= check(
        "MLP hold-out accuracy beats the fit-set majority baseline",
        mlp_hits > majority_hits,
        f"mlp={mlp_hits} majority={majority_hits}",
    )
    for action, hit in MLP_RECALL_HITS.items():
        got, support = mlp_recall[action]
        expect_support = HOLD_SUPPORT[action]
        ok &= check(
            f"MLP hold-out {action}",
            got == hit and support == expect_support,
            f"{got}/{support}",
        )
    ok &= check(
        "majority classes did not collapse",
        mlp_recall["RETRIEVE"][0] >= 1200
        and mlp_recall["RE_PORTION"][0] >= 850
        and mlp_recall["STOP"][0] == HOLD_SUPPORT["STOP"],
    )
    ok &= check(
        "CORRECT_CODE recall is above the linear toy",
        mlp_recall["CORRECT_CODE"][0] > linear_recall["CORRECT_CODE"][0],
        f"mlp={mlp_recall['CORRECT_CODE'][0]} linear={linear_recall['CORRECT_CODE'][0]}",
    )
    pred_names = set(mlp_pred)
    ok &= check(
        "MLP predictions stay inside the six BC actions",
        pred_names.issubset(set(BC_ACTIONS)) and "VOLUME_ESTIMATE" not in pred_names,
        str(sorted(pred_names)),
    )
    single = mlp.predict_action(hold_rows.X[0])
    ok &= check("predict_action matches the first hold-out row", single == mlp_pred[0], single)
    ok &= check(
        "fit ABSTAIN rows mostly share a STOP feature vector",
        shared == 235 and abstain_n == 236,
        f"{shared}/{abstain_n}",
    )
    return ok


def test_summary_flags() -> bool:
    ok = True
    for path in (FIT_SUMMARY, HOLDOUT_SUMMARY):
        payload = json.loads(path.read_text())
        ok &= check(
            f"{path.name} uses_test_gt false and volume off",
            payload.get("uses_test_gt") is False
            and payload.get("volume_tool") is False
            and payload.get("vlm") is False
            and payload.get("legacy_regression_weights") is False,
            f"uses_test_gt={payload.get('uses_test_gt')} volume_tool={payload.get('volume_tool')}",
        )
    return ok


def main() -> int:
    print("bc mlp smoke")
    ok = True
    ok &= test_contract()
    ok &= test_separable_toy()
    ok &= test_summary_flags()
    ok &= test_holdout_accuracy()
    if not ok:
        print("FAILED")
        return 1
    print("All BC MLP smoke checks passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
