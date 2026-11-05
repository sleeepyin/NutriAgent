#!/usr/bin/env python3
"""Hold-out full-set energy MAE, Coverage, Selective MAE, and AURC.

Compares the saved closed-loop policy (Ours, mask none / zero / shuffle)
with the open-loop B5 MLP. Does not train the closed-loop policy, does not
write a B5 checkpoint, and does not search a threshold.

    python scripts/closed_loop_mae_aurc.py --self-check
    python scripts/closed_loop_mae_aurc.py

rgb_test is not rolled out here. A later IR jsonl of the same shape can be
scored without fitting a threshold::

    python scripts/closed_loop_mae_aurc.py \\
        --trajectories path/to/rollout.jsonl \\
        --expect-pool any \\
        --allow-test-split

That flag is off by default. It still does not read ``results/n5k_*`` and
does not tune on labels. This command is not part of the hold-out run.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np  # noqa: E402

from vdr.controller.bc_data import FIT_JSONL, assert_bc_jsonl_path, load_bc_rows  # noqa: E402
from vdr.controller.closed_loop_data import (  # noqa: E402
    HOLDOUT_JSONL,
    _belief_at_step,
    load_closed_loop_rows,
)
from vdr.controller.features import build_features  # noqa: E402
from vdr.controller.policy_bc import BC_ACTIONS  # noqa: E402
from vdr.controller.policy_mlp import PolicyMLP  # noqa: E402
from vdr.controller.policy_state_mlp import PolicyStateMLP  # noqa: E402
from vdr.controller.state_features import (  # noqa: E402
    OBS_SLICE,
    build_state_features,
    mask_observation_block,
)
from vdr.data.n5k import load_labels, test_split_ids  # noqa: E402
from vdr.eval.energy_selective import (  # noqa: E402
    SECONDARY_ABSTAIN_THRESHOLD,
    EnergyEvalRefused,
    dish_readings,
    evaluate_energy_table,
    format_energy_report,
    run_self_check,
)
from vdr.ir.io import read_ir_jsonl  # noqa: E402
from vdr.weak.teacher import PROTECTED_RESULT_NAMES, pool_of_dish  # noqa: E402

OUT_DIR = ROOT / "results" / "closed_loop"
WEIGHTS = OUT_DIR / "policy_state_mlp.npz"
TRAIN_META = OUT_DIR / "train_meta.json"
SAVED_METRICS = OUT_DIR / "metrics_holdout.json"
JSON_NAME = "mae_aurc_holdout.json"
REPORT_NAME = "report_mae_aurc.txt"
TEST_JSON_NAME = "mae_aurc_testsplit.json"
TEST_REPORT_NAME = "report_mae_aurc_testsplit.txt"

# closed_loop_bc_train.py default --shuffle-seed. Not chosen from energy error.
MASK_SHUFFLE_SEED = 42
# Published B5 hold-out hits from scripts/bc_mlp_smoke.py / docs/STATUS.md.
B5_PUBLISHED_HOLD_HITS = 2688
B5_PUBLISHED_HOLD_ROWS = 2771
CANONICAL_N_DISHES = 380
CANONICAL_AUDIT_N = 356
CANONICAL_AUDIT_MAE = "30.7085"
METHOD_ORDER = (
    "ours_mask_none",
    "b5_open_loop",
    "ours_mask_zero",
    "ours_mask_shuffle",
)
FROZEN_OUTPUT_NAMES = {
    "metrics_holdout.json",
    "policy_state_mlp.npz",
    "report_holdout.txt",
    "train_meta.json",
    "fixrate_holdout.json",
    "report_fixrate.txt",
    "rgb_train_fit_all.jsonl",
    "rgb_train_holdout_all.jsonl",
    "rgb_train_fit_all.summary.json",
    "rgb_train_holdout_all.summary.json",
}
DOCUMENTED_PREFIXES = {
    "rgb_train_fit_all.jsonl": "ed47f9239bfecb3c",
    "n5k_rgb_test_overhead_b0_b1.jsonl": "a4dfdd82e3a595f4",
}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _frozen_paths() -> list[Path]:
    names = [
        ROOT / "results" / "weak_steps" / "rgb_train_fit_all.jsonl",
        ROOT / "results" / "weak_steps" / "rgb_train_holdout_all.jsonl",
        ROOT / "results" / "weak_steps" / "rgb_train_fit_all.summary.json",
        ROOT / "results" / "weak_steps" / "rgb_train_holdout_all.summary.json",
        WEIGHTS,
        TRAIN_META,
        SAVED_METRICS,
        OUT_DIR / "report_holdout.txt",
        OUT_DIR / "fixrate_holdout.json",
        OUT_DIR / "report_fixrate.txt",
        ROOT / "results" / "n5k_rgb_test_overhead_b0_b1.jsonl",
        ROOT / "results" / "n5k_rgb_test_overhead_b0_b1.json",
        ROOT / "results" / "n5k_subset_b0_b1.jsonl",
        ROOT / "results" / "n5k_subset_b0_b1.json",
        ROOT / "results" / "n5k_subset_b0_b1_n50_seed42.jsonl",
        ROOT / "results" / "n5k_subset_b0_b1_n50_seed42.json",
    ]
    return [path for path in names if path.is_file()]


def _assert_output_targets(out_dir: Path, *, test_split: bool) -> tuple[Path, Path]:
    resolved = out_dir.resolve()
    allowed = (ROOT / "results" / "closed_loop").resolve()
    if resolved != allowed:
        raise EnergyEvalRefused(f"refusing to write outside {allowed}")
    json_name = TEST_JSON_NAME if test_split else JSON_NAME
    text_name = TEST_REPORT_NAME if test_split else REPORT_NAME
    json_path = resolved / json_name
    text_path = resolved / text_name
    for path in (json_path, text_path):
        if path.name in FROZEN_OUTPUT_NAMES or path.name in PROTECTED_RESULT_NAMES or path.name.startswith("n5k_"):
            raise EnergyEvalRefused(f"refusing to write frozen artifact {path.name}")
    return json_path, text_path


def _group_steps(path: Path, *, expect_pool: str, allow_test: bool) -> dict[str, list[dict]]:
    if expect_pool not in {"fit", "holdout", "any"}:
        raise EnergyEvalRefused(f"expect_pool must be fit, holdout, or any, got {expect_pool!r}")
    if not allow_test:
        assert_bc_jsonl_path(path)
    else:
        if path.name in PROTECTED_RESULT_NAMES or "n5k_" in path.name or path.name in FROZEN_OUTPUT_NAMES:
            raise EnergyEvalRefused(f"refusing frozen artifact {path.name}")
        if "weak_steps" in path.resolve().parts and path.name.startswith("rgb_train_"):
            raise EnergyEvalRefused(f"refusing to reread a frozen weak-step file via --allow-test-split: {path.name}")
    forbidden = test_split_ids()
    grouped: dict[str, list[dict]] = {}
    for record in read_ir_jsonl(path):
        dish_id = record.get("dish_id")
        if not isinstance(dish_id, str) or not dish_id:
            raise EnergyEvalRefused(f"{path.name}: row missing dish_id")
        if dish_id in forbidden and not allow_test:
            raise EnergyEvalRefused(f"{path.name}: refusing test dish {dish_id}")
        if expect_pool != "any":
            slot = pool_of_dish(dish_id)
            if slot != expect_pool:
                raise EnergyEvalRefused(f"{path.name}: {dish_id} is in the {slot} pool, not {expect_pool}")
        if record.get("action_t") == "VOLUME_ESTIMATE":
            raise EnergyEvalRefused(f"{path.name}: VOLUME_ESTIMATE on {dish_id}")
        grouped.setdefault(dish_id, []).append(record)
    if not grouped:
        raise EnergyEvalRefused(f"{path.name}: no rows")
    ordered: dict[str, list[dict]] = {}
    for dish_id in sorted(grouped):
        ordered[dish_id] = sorted(grouped[dish_id], key=lambda row: int(row["step_i"]))
    return ordered


def _stack_state(steps_by_dish: dict[str, list[dict]]) -> tuple[np.ndarray, list[tuple[str, int]], dict[tuple[str, int], str]]:
    rows: list[np.ndarray] = []
    keys: list[tuple[str, int]] = []
    truth: dict[tuple[str, int], str] = {}
    for dish_id, steps in steps_by_dish.items():
        prev: dict | None = None
        for row in steps:
            step_i = int(row["step_i"])
            rows.append(build_state_features(row, prev, _belief_at_step(step_i), mask_obs="none"))
            key = (dish_id, step_i)
            keys.append(key)
            truth[key] = str(row["action_t"])
            prev = row
    return np.vstack(rows), keys, truth


def _apply_mask(matrix: np.ndarray, mode: str, *, shuffle_seed: int) -> np.ndarray:
    if mode == "none":
        return matrix
    if mode == "zero":
        return np.vstack(
            [mask_observation_block(row, mode="zero", obs_slice=OBS_SLICE) for row in matrix]
        )
    if mode == "shuffle":
        return np.vstack(
            [
                mask_observation_block(
                    row,
                    mode="shuffle",
                    obs_slice=OBS_SLICE,
                    rng=np.random.default_rng(shuffle_seed + index),
                )
                for index, row in enumerate(matrix)
            ]
        )
    raise EnergyEvalRefused(f"unknown mask {mode}")


def _b5_matrix(steps_by_dish: dict[str, list[dict]], keys: list[tuple[str, int]]) -> np.ndarray:
    lookup: dict[tuple[str, int], dict] = {}
    for dish_id, steps in steps_by_dish.items():
        for row in steps:
            lookup[(dish_id, int(row["step_i"]))] = row
    return np.vstack([build_features(lookup[key]) for key in keys])


def _predict(policy: PolicyMLP | PolicyStateMLP, matrix: np.ndarray) -> tuple[list[str], np.ndarray]:
    actions = policy.predict_actions(matrix)
    proba = np.asarray(policy.predict_proba(matrix), dtype=np.float64)
    if proba.shape != (len(actions), len(policy.actions)):
        raise EnergyEvalRefused(
            f"predict_proba shape {proba.shape} != ({len(actions)}, {len(policy.actions)})"
        )
    via = [policy.actions[int(index)] for index in np.argmax(proba, axis=1)]
    if via != list(actions):
        raise EnergyEvalRefused("predict_proba argmax disagrees with predict_actions")
    if "ABSTAIN" not in policy.actions:
        raise EnergyEvalRefused("policy has no ABSTAIN column")
    column = proba[:, policy.actions.index("ABSTAIN")]
    if not np.isfinite(column).all():
        raise EnergyEvalRefused("ABSTAIN probability is not finite")
    return list(actions), column


def _maps(
    keys: list[tuple[str, int]],
    actions: list[str],
    scores: np.ndarray,
) -> tuple[dict[tuple[str, int], str], dict[tuple[str, int], float]]:
    if len(keys) != len(actions) or len(keys) != len(scores):
        raise EnergyEvalRefused("prediction length does not match trajectory rows")
    predictions = {key: action for key, action in zip(keys, actions)}
    abstain = {key: float(score) for key, score in zip(keys, scores)}
    return predictions, abstain


def _hits(keys: list[tuple[str, int]], actions: list[str], truth: dict[tuple[str, int], str]) -> int:
    return sum(action == truth[key] for key, action in zip(keys, actions))


def _saved_hits(path: Path) -> dict[str, int]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    hits: dict[str, int] = {}
    for mode in ("none", "zero", "shuffle"):
        per_class = payload["evaluations"][mode]["per_class"]
        hits[mode] = int(sum(int(block["tp"]) for block in per_class.values()))
    return hits


def _load_policy(weights: Path, meta_path: Path) -> PolicyStateMLP:
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    actions = tuple(meta.get("actions") or ())
    if actions != tuple(BC_ACTIONS):
        raise EnergyEvalRefused("train_meta action order does not match the BC head")
    class_weight = meta.get("class_weight")
    if not isinstance(class_weight, dict):
        raise EnergyEvalRefused("train_meta is missing class_weight")
    return PolicyStateMLP.from_npz(weights, actions=actions, class_weight=class_weight)


def _cafe_labels(dish_ids: set[str]) -> dict[str, float]:
    table = load_labels()
    missing = sorted(dish_id for dish_id in dish_ids if dish_id not in table)
    if missing:
        raise EnergyEvalRefused(
            "cafe energy label missing for "
            + ", ".join(missing[:5])
            + (" ..." if len(missing) > 5 else "")
        )
    return {dish_id: float(table[dish_id]["energy_kcal"]) for dish_id in dish_ids}


def _replay_b5(fit_path: Path, holdout_path: Path) -> PolicyMLP:
    """Reproduce the published open-loop MLP. Does not write weights.

    Constants stay on ``PolicyMLP`` (seed 0, 80 epochs, lr 1e-2, L2 1e-4,
    batch 512, dropout 0). The fit pool is the only supervision. The
    published hold-out hit count is a replay check, not a search.
    """
    fit = load_bc_rows(fit_path, expect_pool="fit")
    policy = PolicyMLP()
    policy.fit(fit.X, fit.y)
    hold = load_bc_rows(holdout_path, expect_pool="holdout")
    if len(hold.y) != B5_PUBLISHED_HOLD_ROWS:
        raise EnergyEvalRefused(
            f"B5 hold-out rows {len(hold.y)} != published {B5_PUBLISHED_HOLD_ROWS}"
        )
    predicted = policy.predict_actions(hold.X)
    hits = sum(pred == str(label) for pred, label in zip(predicted, hold.y.tolist()))
    if hits != B5_PUBLISHED_HOLD_HITS:
        raise EnergyEvalRefused(
            f"B5 replay hits {hits} != published {B5_PUBLISHED_HOLD_HITS}; refusing to score energy"
        )
    return policy


def _jsonable(payload: dict) -> str:
    return json.dumps(payload, indent=2, ensure_ascii=False) + "\n"


def _rel(path: Path) -> str:
    return str(path.relative_to(ROOT)) if path.is_relative_to(ROOT) else str(path)


def main() -> int:
    parser = argparse.ArgumentParser(description="Closed-loop vs B5 energy MAE / Coverage / Selective MAE / AURC")
    parser.add_argument("--self-check", action="store_true", help="Synthetic contract only; writes nothing")
    parser.add_argument("--trajectories", type=Path, default=HOLDOUT_JSONL)
    parser.add_argument("--expect-pool", choices=("fit", "holdout", "any"), default="holdout")
    parser.add_argument(
        "--allow-test-split",
        action="store_true",
        help="Score rgb_test/depth_test dish ids. Does not fit a threshold. Writes mae_aurc_testsplit.*, not the hold-out files.",
    )
    parser.add_argument("--fit", type=Path, default=FIT_JSONL)
    parser.add_argument("--weights", type=Path, default=WEIGHTS)
    parser.add_argument("--train-meta", type=Path, default=TRAIN_META)
    parser.add_argument("--saved-metrics", type=Path, default=SAVED_METRICS)
    parser.add_argument("--out-dir", type=Path, default=OUT_DIR)
    parser.add_argument("--shuffle-seed", type=int, default=MASK_SHUFFLE_SEED)
    args = parser.parse_args()

    if args.self_check:
        run_self_check()
        print(
            "closed_loop_mae_aurc self-check ok "
            f"threshold={SECONDARY_ABSTAIN_THRESHOLD} (secondary, not fit)"
        )
        return 0

    if args.shuffle_seed != MASK_SHUFFLE_SEED:
        raise EnergyEvalRefused(
            f"shuffle seed {args.shuffle_seed} is not the published closed-loop seed {MASK_SHUFFLE_SEED}"
        )

    run_self_check()
    started = time.perf_counter()
    trajectories = args.trajectories if args.trajectories.is_absolute() else ROOT / args.trajectories
    fit_path = args.fit if args.fit.is_absolute() else ROOT / args.fit
    weights = args.weights if args.weights.is_absolute() else ROOT / args.weights
    meta_path = args.train_meta if args.train_meta.is_absolute() else ROOT / args.train_meta
    saved_metrics = args.saved_metrics if args.saved_metrics.is_absolute() else ROOT / args.saved_metrics
    out_dir = args.out_dir if args.out_dir.is_absolute() else ROOT / args.out_dir
    canonical_holdout = (ROOT / HOLDOUT_JSONL).resolve()
    is_canonical = trajectories.resolve() == canonical_holdout and not args.allow_test_split
    json_path, text_path = _assert_output_targets(out_dir, test_split=args.allow_test_split)

    for path in (trajectories, fit_path, weights, meta_path, saved_metrics, canonical_holdout):
        if not path.is_file():
            raise EnergyEvalRefused(f"missing {path}")

    frozen_paths = _frozen_paths()
    if len({path.name for path in frozen_paths}) != len(frozen_paths):
        raise EnergyEvalRefused("frozen paths share a file name; refusing an ambiguous hash map")
    frozen_before = {path.name: _sha256(path) for path in frozen_paths}
    for name, prefix in DOCUMENTED_PREFIXES.items():
        digest = frozen_before.get(name)
        if digest is None or not digest.startswith(prefix):
            raise EnergyEvalRefused(f"{name} sha256 does not start with the documented prefix {prefix}")

    steps_by_dish = _group_steps(trajectories, expect_pool=args.expect_pool, allow_test=args.allow_test_split)
    state, keys, truth = _stack_state(steps_by_dish)
    if is_canonical:
        loaded = load_closed_loop_rows(trajectories, expect_pool="holdout")
        loaded_keys = list(zip(loaded.dish_ids, (int(step) for step in loaded.step_indices)))
        if loaded_keys != keys or not np.array_equal(loaded.X, state):
            raise EnergyEvalRefused("closed-loop feature stack does not match load_closed_loop_rows")
        if len(steps_by_dish) != CANONICAL_N_DISHES:
            raise EnergyEvalRefused(f"hold-out dishes {len(steps_by_dish)} != {CANONICAL_N_DISHES}")

    policy = _load_policy(weights, meta_path)
    masked = {
        "ours_mask_none": _apply_mask(state, "none", shuffle_seed=args.shuffle_seed),
        "ours_mask_zero": _apply_mask(state, "zero", shuffle_seed=args.shuffle_seed),
        "ours_mask_shuffle": _apply_mask(state, "shuffle", shuffle_seed=args.shuffle_seed),
    }
    predicted: dict[str, tuple[list[str], np.ndarray]] = {
        name: _predict(policy, matrix) for name, matrix in masked.items()
    }
    saved_hits = _saved_hits(saved_metrics)
    ours_hits = {
        "none": _hits(keys, predicted["ours_mask_none"][0], truth),
        "zero": _hits(keys, predicted["ours_mask_zero"][0], truth),
        "shuffle": _hits(keys, predicted["ours_mask_shuffle"][0], truth),
    }
    if is_canonical:
        for mode, hits in ours_hits.items():
            if hits != saved_hits[mode]:
                raise EnergyEvalRefused(
                    f"mask {mode} hits {hits} != saved metrics tp sum {saved_hits[mode]}"
                )

    print(
        f"replaying B5 PolicyMLP.fit on {_rel(fit_path)} "
        "(published constants, fit pool only, checkpoint not written)",
        flush=True,
    )
    b5 = _replay_b5(fit_path, canonical_holdout)
    b5_actions, b5_scores = _predict(b5, _b5_matrix(steps_by_dish, keys))
    b5_hits_on_target = _hits(keys, b5_actions, truth)
    if is_canonical and b5_hits_on_target != B5_PUBLISHED_HOLD_HITS:
        raise EnergyEvalRefused(
            f"B5 hits on stacked hold-out rows {b5_hits_on_target} != {B5_PUBLISHED_HOLD_HITS}"
        )
    predicted["b5_open_loop"] = (b5_actions, b5_scores)

    labels = _cafe_labels(set(steps_by_dish))
    readings = {}
    for name in METHOD_ORDER:
        actions, scores = predicted[name]
        pred_map, score_map = _maps(keys, actions, scores)
        readings[name] = dish_readings(steps_by_dish, pred_map, score_map, labels)

    payload = evaluate_energy_table(readings)
    payload["method_order"] = list(METHOD_ORDER)
    audit = payload["teacher_terminal_audit"]
    if is_canonical:
        rounded = audit["mae_rounded_like_weak_step_summary"]
        if audit["n"] != CANONICAL_AUDIT_N or f"{rounded:.4f}" != CANONICAL_AUDIT_MAE:
            raise EnergyEvalRefused(
                "teacher terminal audit "
                f"n={audit['n']} mae={rounded} does not match the frozen weak-step summary "
                f"n={CANONICAL_AUDIT_N} MAE={CANONICAL_AUDIT_MAE}"
            )

    elapsed = time.perf_counter() - started
    payload["elapsed_seconds"] = round(elapsed, 2)
    payload["owner"] = "营养论文代码编写"
    payload["status_file"] = "docs/STATUS.md was not modified. STATUS ownership stays 方案执行."
    payload["scope"] = {
        "split": (
            "rgb_train hold-out (380 dishes, disjoint from the fit pool). "
            "Not rgb_test and not the 507-dish overhead pool."
            if is_canonical
            else f"trajectories={_rel(trajectories)} expect_pool={args.expect_pool} allow_test_split={args.allow_test_split}"
        ),
        "labels": (
            "Nutrition5k cafe dish energy_kcal from dish_metadata_cafe1/cafe2 via "
            "vdr.data.n5k.load_labels. Same labels as the weak-step train-energy audit. "
            "Labels are used only to score. They are not features and no threshold is fit on them."
        ),
        "rgb_test": (
            "Not run. A full-set rgb_test number needs a VLM rollout that writes this IR shape. "
            "Pass --trajectories with --expect-pool any --allow-test-split later. "
            "That path does not search a threshold and does not read results/n5k_*. "
            "This run did not pass --allow-test-split."
            if not args.allow_test_split
            else "allow-test-split is on. Thresholds were still not fit. results/n5k_* was not read."
        ),
        "b0_b1": (
            "B0 energy MAE 97.03 is the rgb_test Direct-VLM anchor (n=507) and is not a row. "
            "B1 110.72 is not faithful DietAI24 and is not a row."
        ),
        "fixrate_and_accuracy": (
            "FixRate and overall action accuracy are not this table. "
            "Action hits below are a weight-loader check."
        ),
        "b5": (
            "Open-loop 31-d PolicyMLP replayed in memory on the fit pool with the module "
            f"constants. Hold-out action hits {B5_PUBLISHED_HOLD_HITS}/{B5_PUBLISHED_HOLD_ROWS}. "
            "No checkpoint was written and no constant was changed."
        ),
        "closed_loop_weights": f"Loaded {_rel(weights)}. PolicyStateMLP.fit was not called.",
        "mask": (
            f"mask-zero nulls the observation block. mask-shuffle permutes it with "
            f"np.random.default_rng({MASK_SHUFFLE_SEED} + row_index), row_index in sorted "
            "dish_id then step_i. Same rule as scripts/closed_loop_bc_train.py --shuffle-seed 42. "
            "B5 has no observation block, so it is not masked."
        ),
    }
    payload["loader_sanity"] = {
        "ours_mask_none_hits": ours_hits["none"],
        "ours_mask_zero_hits": ours_hits["zero"],
        "ours_mask_shuffle_hits": ours_hits["shuffle"],
        "saved_metrics_tp_sum": saved_hits,
        "canonical_holdout_matches_saved_metrics": (
            {mode: ours_hits[mode] == saved_hits[mode] for mode in ours_hits} if is_canonical else None
        ),
        "b5_published_hold_hits": B5_PUBLISHED_HOLD_HITS,
        "b5_hits_on_scored_rows": b5_hits_on_target,
        "n_rows": len(keys),
        "n_dishes": len(steps_by_dish),
        "note": "Action agreement checks the loader. It is not Full-set MAE / AURC proof.",
    }
    payload["inputs"] = {
        "trajectories": _rel(trajectories),
        "fit_jsonl": _rel(fit_path),
        "weights": _rel(weights),
        "train_meta": _rel(meta_path),
        "saved_metrics": _rel(saved_metrics),
        "canonical_holdout": is_canonical,
        "allow_test_split": bool(args.allow_test_split),
        "shuffle_seed": MASK_SHUFFLE_SEED,
        "secondary_abstain_threshold": SECONDARY_ABSTAIN_THRESHOLD,
        "frozen_sha256_prefix": {name: digest[:16] for name, digest in sorted(frozen_before.items())},
        "documented_prefix_match": {
            name: frozen_before.get(name, "").startswith(prefix) for name, prefix in DOCUMENTED_PREFIXES.items()
        },
        "frozen_unchanged": True,
    }
    text = format_energy_report(payload)
    out_dir.mkdir(parents=True, exist_ok=True)
    json_path.write_text(_jsonable(payload), encoding="utf-8")
    text_path.write_text(text, encoding="utf-8")
    frozen_after = {path.name: _sha256(path) for path in frozen_paths}
    changed = sorted(name for name, digest in frozen_before.items() if frozen_after.get(name) != digest)
    if changed:
        json_path.unlink(missing_ok=True)
        text_path.unlink(missing_ok=True)
        raise EnergyEvalRefused("frozen files changed while writing the energy table: " + ", ".join(changed))

    print(text, end="")
    print(f"wrote {_rel(json_path)} and {_rel(text_path)} in {elapsed:.1f}s")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except EnergyEvalRefused as exc:
        print(f"ENERGY EVAL REFUSED: {exc}", file=sys.stderr)
        raise SystemExit(1) from exc
