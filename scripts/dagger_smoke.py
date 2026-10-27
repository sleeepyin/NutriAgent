#!/usr/bin/env python3
"""CPU smoke for one DAgger round on the weak-step teacher.

Fits the pure-BC ``PolicyMLP`` on ``rgb_train_fit_all.jsonl`` (the same fixed
hyperparameters as ``scripts/bc_mlp_smoke.py``), rolls that policy out on a
fit-pool prefix, labels visited states with the rule teacher, and retrains
``PolicyMLP`` on the aggregate. Hold-out accuracy is printed from
``rgb_train_holdout_all.jsonl`` only. Nothing is written.

``--n`` is a prefix of the official fit order, not a sample. ``--all-fit``
rolls the whole fit pool. The default ``--n 8`` is the short check.

    python scripts/dagger_smoke.py
    python scripts/dagger_smoke.py --n 32
    python scripts/dagger_smoke.py --all-fit
"""

from __future__ import annotations

import argparse
import hashlib
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np  # noqa: E402

from vdr.controller.bc_data import action_counts, load_bc_rows  # noqa: E402
from vdr.controller.dagger import (  # noqa: E402
    DAGGER_MAX_STEPS,
    DAGGER_STUCK_LIMIT,
    DaggerPoolRefused,
    WeakTeacherEnv,
    action_hits,
    assert_dagger_output,
    expert_ir_steps,
    fit_pool_dishes,
    run_dagger_round,
    stack_aggregate,
)
from vdr.controller.features import FEATURE_DIM, PRE_ACTION_FIELDS, build_features  # noqa: E402
from vdr.controller.policy_bc import BC_ACTIONS, PolicyBC  # noqa: E402
from vdr.controller.policy_mlp import (  # noqa: E402
    MLP_DROPOUT,
    MLP_EPOCHS,
    MLP_HIDDEN_SIZES,
    MLP_L2,
    MLP_LR,
    MLP_SEED,
    PolicyMLP,
)
from vdr.weak.teacher import (  # noqa: E402
    UnsafeOutputPath,
    generate_dish_steps,
    select_train_dishes,
)

FIT_PATH = ROOT / "results/weak_steps/rgb_train_fit_all.jsonl"
HOLDOUT_PATH = ROOT / "results/weak_steps/rgb_train_holdout_all.jsonl"
FROZEN_JSONL = ROOT / "results/n5k_rgb_test_overhead_b0_b1.jsonl"

# Locked pure-BC numbers from scripts/bc_mlp_smoke.py. DAgger must not move them.
LINEAR_HOLD_HITS = 2551
MLP_HOLD_HITS = 2688
HOLD_ROWS = 2771

# Measured on the assistant CPU with the fixed MLP seed. Not a hold-out search.
DEFAULT_N = 8
N8_NEW_ROWS = 5
N8_ROLLED = 3
N8_DAGGER_HITS = 2677
N8_RECALL = {
    "RETRIEVE": (1280, 1285),
    "RE_PORTION": (896, 949),
    "RE_DECOMPOSE": (0, 0),
    "ABSTAIN": (0, 24),
    "STOP": (356, 356),
    "CORRECT_CODE": (145, 157),
}
N32_NEW_ROWS = 11
N32_NOVEL_ROWS = 1
N32_ROLLED = 8
N32_DAGGER_HITS = 2671
N32_RECALL = {
    "RETRIEVE": (1278, 1285),
    "RE_PORTION": (889, 949),
    "RE_DECOMPOSE": (0, 0),
    "ABSTAIN": (0, 24),
    "STOP": (356, 356),
    "CORRECT_CODE": (148, 157),
}
# Whole fit pool, one round. Overall accuracy is below pure BC. Do not retune.
ALL_FIT_NEW_ROWS = 1490
ALL_FIT_NOVEL_ROWS = 117
ALL_FIT_COPIES = 1373
ALL_FIT_ROLLED = 766
ALL_FIT_SKIPPED = 2913
ALL_FIT_MISMATCH = 1401
ALL_FIT_CONFLICTS = 4
ALL_FIT_DAGGER_HITS = 2658
ALL_FIT_RECALL = {
    "RETRIEVE": (1285, 1285),
    "RE_PORTION": (900, 949),
    "RE_DECOMPOSE": (0, 0),
    "ABSTAIN": (0, 24),
    "STOP": (356, 356),
    "CORRECT_CODE": (117, 157),
}
ALL_FIT_NEW_ACTIONS = {
    "RETRIEVE": 145,
    "RE_PORTION": 945,
    "RE_DECOMPOSE": 0,
    "ABSTAIN": 196,
    "STOP": 3,
    "CORRECT_CODE": 201,
}


def check(name: str, ok: bool, detail: str = "") -> bool:
    print(f"  {'OK' if ok else 'FAIL'} {name}" + (f" — {detail}" if detail else ""))
    return ok


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _fmt_hits(hits: dict[str, tuple[int, int]]) -> str:
    bits = []
    for action, (hit, count) in hits.items():
        if count == 0:
            bits.append(f"{action}:n=0")
        else:
            bits.append(f"{action}:{hit}/{count}={hit / count:.4f}")
    return " ".join(bits)


def test_contract() -> bool:
    ok = True
    ok &= check("feature dim is 31", FEATURE_DIM == 31, str(FEATURE_DIM))
    ok &= check(
        "MLP hyperparameters are the BC defaults",
        MLP_HIDDEN_SIZES == (64, 64) and MLP_DROPOUT == 0.0 and MLP_L2 == 1e-4 and MLP_LR == 1e-2,
        f"hidden={MLP_HIDDEN_SIZES} dropout={MLP_DROPOUT} l2={MLP_L2} lr={MLP_LR} "
        f"epochs={MLP_EPOCHS} seed={MLP_SEED}",
    )
    ok &= check(
        "DAgger caps are the fixed safety limits",
        DAGGER_MAX_STEPS == 48 and DAGGER_STUCK_LIMIT == 3,
        f"max_steps={DAGGER_MAX_STEPS} stuck={DAGGER_STUCK_LIMIT}",
    )
    ok &= check("volume action is not in the BC set", "VOLUME_ESTIMATE" not in BC_ACTIONS)
    ok &= check("obs_t is not a pre-action field", "obs_t" not in PRE_ACTION_FIELDS)
    refused = False
    try:
        PolicyMLP(actions=("RETRIEVE", "VOLUME_ESTIMATE"))
    except ValueError:
        refused = True
    ok &= check("VOLUME_ESTIMATE cannot be an MLP action", refused)

    blocked = []
    samples = [
        ROOT / "results" / "dagger" / "n5k_rgb_test_overhead_b0_b1.jsonl",
        ROOT / "results" / "dagger" / "rgb_test_steps.jsonl",
        ROOT / "results" / "dagger" / "rgb_train_fit_all.jsonl",
        ROOT / "results" / "dagger" / "rgb_train_holdout_round.jsonl",
        ROOT / "results" / "weak_steps" / "dagger_round.jsonl",
    ]
    for path in samples:
        try:
            assert_dagger_output(path)
        except (DaggerPoolRefused, UnsafeOutputPath):
            blocked.append(path.name)
        else:
            ok &= check(f"protect {path.name}", False, "write was allowed")
    ok &= check(
        "dagger output refuses frozen, test, and weak-step names",
        len(blocked) == len(samples),
        str(blocked),
    )
    allowed = False
    try:
        assert_dagger_output(ROOT / "results" / "dagger" / "round1_fit_n8.jsonl")
        allowed = True
    except (DaggerPoolRefused, UnsafeOutputPath):
        allowed = False
    ok &= check("results/dagger/round1_fit_n8.jsonl is an allowed name", allowed)

    hold = select_train_dishes(pool="holdout", n=1)[0]
    refused_hold = False
    try:
        WeakTeacherEnv(hold)
    except DaggerPoolRefused:
        refused_hold = True
    ok &= check(f"env refuses hold-out dish {hold.dish_id}", refused_hold)
    return ok


def test_expert_matches_teacher(dishes: list) -> bool:
    """Expert rollout equals generate_dish_steps. This is the env regression."""
    ok = True
    for dish in dishes:
        got = expert_ir_steps(dish)
        expected, _audit = generate_dish_steps(dish)
        match = got == expected
        actions = [step["action_t"] for step in expected]
        ok &= check(
            f"expert IR matches teacher {dish.dish_id}",
            match and "VOLUME_ESTIMATE" not in actions,
            " ".join(actions) if match else f"got {len(got)} teacher {len(expected)}",
        )
        if not match:
            break
        step = got[0]
        state = {field: step.get(field) for field in PRE_ACTION_FIELDS}
        leaked = dict(step)
        leaked["obs_t"] = {"portion_g": 1.0, "to": "OTHER", "privileged": True}
        leaked["train_gt_energy_kcal"] = 0.0
        same = np.array_equal(build_features(state), build_features(leaked))
        ok &= check(f"{dish.dish_id} obs_t does not change h", same)
    return ok


def test_round(n: int | None) -> bool:
    started = time.perf_counter()
    before = {
        "fit": _sha256(FIT_PATH),
        "holdout": _sha256(HOLDOUT_PATH),
        "frozen": _sha256(FROZEN_JSONL),
    }
    fit_rows = load_bc_rows(FIT_PATH, expect_pool="fit")
    hold_rows = load_bc_rows(HOLDOUT_PATH, expect_pool="holdout")
    dishes = fit_pool_dishes(n)
    if n is not None and len(dishes) != n:
        return check("fit prefix has the requested dishes", False, f"{len(dishes)} != {n}")

    linear = PolicyBC().fit(fit_rows.X, fit_rows.y)
    pure = PolicyMLP().fit(fit_rows.X, fit_rows.y)
    pure_again = PolicyMLP().fit(fit_rows.X, fit_rows.y)
    linear_pred = linear.predict_actions(hold_rows.X)
    pure_pred = pure.predict_actions(hold_rows.X)
    pure_pred_again = pure_again.predict_actions(hold_rows.X)
    truth = [str(item) for item in hold_rows.y.tolist()]
    linear_hits = sum(left == right for left, right in zip(truth, linear_pred))
    pure_hits = sum(left == right for left, right in zip(truth, pure_pred))
    pure_recall = action_hits(truth, pure_pred)

    result = run_dagger_round(
        pure,
        dishes,
        fit_rows.X,
        fit_rows.y,
        fit_rows.dish_ids,
    )
    agg_X, agg_y = stack_aggregate(fit_rows.X, fit_rows.y, result)
    dagger = PolicyMLP().fit(agg_X, agg_y)
    dagger_pred = dagger.predict_actions(hold_rows.X)
    dagger_hits = sum(left == right for left, right in zip(truth, dagger_pred))
    dagger_recall = action_hits(truth, dagger_pred)
    elapsed = time.perf_counter() - started

    fit_ids = []
    seen = set()
    for dish_id in fit_rows.dish_ids:
        if dish_id not in seen:
            seen.add(dish_id)
            fit_ids.append(dish_id)
    prefix_ids = fit_ids if n is None else fit_ids[:n]
    dish_ids = [dish.dish_id for dish in dishes]
    new_outside = sorted({dish_id for dish_id in result.new_dish_ids if dish_id not in set(dish_ids)})
    new_actions = sorted(set(str(item) for item in result.new_y.tolist()))
    after = {
        "fit": _sha256(FIT_PATH),
        "holdout": _sha256(HOLDOUT_PATH),
        "frozen": _sha256(FROZEN_JSONL),
    }

    print(
        f"dagger_round n_dishes={result.n_dishes} rolled={result.n_dishes_rolled} "
        f"skipped_agree={result.n_dishes_skipped_agree} visited={result.n_visited} "
        f"mismatch={result.n_mismatch} new_rows={result.n_new_rows} "
        f"novel_rows={result.n_novel_rows} mismatch_copies={result.n_mismatch_copies} "
        f"new_actions={action_counts(result.new_y)} "
        f"duplicate_states={result.n_duplicate_states} label_conflicts={result.n_label_conflicts} "
        f"bc_rows={fit_rows.n_rows} aggregate_rows={agg_X.shape[0]} "
        f"pure_mlp_holdout_acc={pure_hits / hold_rows.n_rows:.4f} ({pure_hits}/{hold_rows.n_rows}) "
        f"dagger_mlp_holdout_acc={dagger_hits / hold_rows.n_rows:.4f} ({dagger_hits}/{hold_rows.n_rows}) "
        f"linear_holdout_acc={linear_hits / hold_rows.n_rows:.4f} ({linear_hits}/{hold_rows.n_rows}) "
        f"pure_mlp_holdout_recall={_fmt_hits(pure_recall)} "
        f"dagger_mlp_holdout_recall={_fmt_hits(dagger_recall)} "
        f"uses_test_gt={result.uses_test_gt} volume_tool={result.volume_tool} "
        f"max_steps={result.max_steps} stuck_limit={result.stuck_limit} "
        f"seconds={elapsed:.2f}"
    )

    ok = True
    ok &= check("fit prefix matches the jsonl dish order", dish_ids == prefix_ids)
    ok &= check(
        "pure MLP hold-out stays 2688/2771",
        pure_hits == MLP_HOLD_HITS and hold_rows.n_rows == HOLD_ROWS and pure_pred == pure_pred_again,
        f"{pure_hits}/{hold_rows.n_rows}",
    )
    ok &= check(
        "linear hold-out stays 2551/2771",
        linear_hits == LINEAR_HOLD_HITS,
        f"{linear_hits}/{hold_rows.n_rows}",
    )
    ok &= check(
        "rollout accounting covers every requested dish",
        result.n_dishes_rolled + result.n_dishes_skipped_agree == result.n_dishes == len(dishes),
    )
    ok &= check("new rows stay inside the rolled fit dishes", not new_outside, str(new_outside[:5]))
    ok &= check(
        "new labels stay inside the BC actions",
        set(new_actions).issubset(set(BC_ACTIONS)) and "VOLUME_ESTIMATE" not in new_actions,
        str(new_actions),
    )
    ok &= check(
        "aggregate is BC plus the new rows",
        agg_X.shape == (fit_rows.n_rows + result.n_new_rows, FEATURE_DIM)
        and len(agg_y) == agg_X.shape[0],
        str(agg_X.shape),
    )
    ok &= check("DAgger did not read test GT or the volume tool", not result.uses_test_gt and not result.volume_tool)
    ok &= check(
        "DAgger predictions stay inside the six BC actions",
        set(dagger_pred).issubset(set(BC_ACTIONS)) and "VOLUME_ESTIMATE" not in dagger_pred,
    )
    ok &= check(
        "fit, hold-out, and frozen jsonl bytes are unchanged",
        before == after,
        "fit=" + before["fit"][:16] + " frozen=" + before["frozen"][:16],
    )
    ok &= check(
        "new rows are novel states plus mistake copies",
        result.n_new_rows == result.n_novel_rows + result.n_mismatch_copies,
        f"{result.n_new_rows}={result.n_novel_rows}+{result.n_mismatch_copies}",
    )
    if n == DEFAULT_N:
        ok &= _check_locked(
            "n=8",
            result,
            dagger_hits,
            dagger_recall,
            new_rows=N8_NEW_ROWS,
            rolled=N8_ROLLED,
            hits=N8_DAGGER_HITS,
            recall=N8_RECALL,
        )
    elif n == 32:
        ok &= _check_locked(
            "n=32",
            result,
            dagger_hits,
            dagger_recall,
            new_rows=N32_NEW_ROWS,
            rolled=N32_ROLLED,
            hits=N32_DAGGER_HITS,
            recall=N32_RECALL,
            novel=N32_NOVEL_ROWS,
        )
    elif n is None:
        ok &= _check_locked(
            "all-fit",
            result,
            dagger_hits,
            dagger_recall,
            new_rows=ALL_FIT_NEW_ROWS,
            rolled=ALL_FIT_ROLLED,
            hits=ALL_FIT_DAGGER_HITS,
            recall=ALL_FIT_RECALL,
            novel=ALL_FIT_NOVEL_ROWS,
            copies=ALL_FIT_COPIES,
            skipped=ALL_FIT_SKIPPED,
            mismatch=ALL_FIT_MISMATCH,
            conflicts=ALL_FIT_CONFLICTS,
            new_actions=ALL_FIT_NEW_ACTIONS,
        )
    return ok


def _check_locked(
    tag: str,
    result,
    dagger_hits: int,
    dagger_recall: dict[str, tuple[int, int]],
    *,
    new_rows: int,
    rolled: int,
    hits: int,
    recall: dict[str, tuple[int, int]],
    novel: int | None = None,
    copies: int | None = None,
    skipped: int | None = None,
    mismatch: int | None = None,
    conflicts: int | None = None,
    new_actions: dict[str, int] | None = None,
) -> bool:
    ok = True
    ok &= check(f"{tag} new rows", result.n_new_rows == new_rows, str(result.n_new_rows))
    ok &= check(f"{tag} rolled dishes", result.n_dishes_rolled == rolled, str(result.n_dishes_rolled))
    ok &= check(f"{tag} hold-out hits", dagger_hits == hits, f"{dagger_hits}/{HOLD_ROWS}")
    if novel is not None:
        ok &= check(f"{tag} novel rows", result.n_novel_rows == novel, str(result.n_novel_rows))
    if copies is not None:
        ok &= check(f"{tag} mistake copies", result.n_mismatch_copies == copies, str(result.n_mismatch_copies))
    if skipped is not None:
        ok &= check(f"{tag} skipped dishes", result.n_dishes_skipped_agree == skipped, str(result.n_dishes_skipped_agree))
    if mismatch is not None:
        ok &= check(f"{tag} mismatches", result.n_mismatch == mismatch, str(result.n_mismatch))
    if conflicts is not None:
        ok &= check(f"{tag} label conflicts", result.n_label_conflicts == conflicts, str(result.n_label_conflicts))
    for action, expect in recall.items():
        ok &= check(f"{tag} hold-out {action}", dagger_recall[action] == expect, str(dagger_recall[action]))
    if new_actions is not None:
        got = action_counts(result.new_y)
        ok &= check(f"{tag} new-row action counts", got == new_actions, str(got))
    return ok


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--n",
        type=int,
        default=DEFAULT_N,
        help="fit-pool prefix. Official order, no seed. Ignored with --all-fit.",
    )
    parser.add_argument(
        "--all-fit",
        action="store_true",
        help="roll the whole rgb_train fit pool (3679 dishes). CPU, no GPU.",
    )
    args = parser.parse_args(argv)
    if args.all_fit:
        n: int | None = None
    else:
        if args.n < 1:
            print("refused: --n must be >= 1", file=sys.stderr)
            return 2
        n = args.n

    print(f"dagger smoke n={n if n is not None else 'all-fit'}")
    ok = True
    ok &= test_contract()
    dishes = fit_pool_dishes(None if n is None else min(n, 8))
    # Expert equality reruns the teacher tools. Keep it to 8 dishes even on a long prefix.
    ok &= test_expert_matches_teacher(dishes[:8])
    ok &= test_round(n)
    if not ok:
        print("FAILED")
        return 1
    print("All DAgger smoke checks passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
