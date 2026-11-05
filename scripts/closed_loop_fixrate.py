#!/usr/bin/env python3
"""Hold-out FixRate − WorsenRate and tool cost under budget K.

Loads the saved closed-loop policy. Does not train. Writes only
``results/closed_loop/fixrate_holdout.json`` and ``report_fixrate.txt``.

    python scripts/closed_loop_fixrate.py --self-check
    python scripts/closed_loop_fixrate.py
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

from vdr.controller.bc_data import assert_bc_jsonl_path  # noqa: E402
from vdr.controller.closed_loop_data import HOLDOUT_JSONL, load_closed_loop_rows  # noqa: E402
from vdr.controller.policy_bc import BC_ACTIONS  # noqa: E402
from vdr.controller.policy_state_mlp import PolicyStateMLP  # noqa: E402
from vdr.data.n5k import load_labels, test_split_ids  # noqa: E402
from vdr.eval.fixrate import (  # noqa: E402
    BUDGET_K,
    FIXRATE_EPSILON_KCAL,
    FixRateRefused,
    evaluate_fixrate,
    format_fixrate_report,
    run_self_check,
)
from vdr.ir.io import read_ir_jsonl  # noqa: E402
from vdr.weak.teacher import pool_of_dish  # noqa: E402

OUT_DIR = ROOT / "results" / "closed_loop"
WEIGHTS = OUT_DIR / "policy_state_mlp.npz"
TRAIN_META = OUT_DIR / "train_meta.json"
SAVED_METRICS = OUT_DIR / "metrics_holdout.json"
REPORT_NAME = "report_fixrate.txt"
JSON_NAME = "fixrate_holdout.json"
FROZEN_OUTPUT_NAMES = {
    "metrics_holdout.json",
    "policy_state_mlp.npz",
    "report_holdout.txt",
    "train_meta.json",
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
        ROOT / "results" / "n5k_rgb_test_overhead_b0_b1.jsonl",
        ROOT / "results" / "n5k_rgb_test_overhead_b0_b1.json",
        ROOT / "results" / "n5k_subset_b0_b1.jsonl",
        ROOT / "results" / "n5k_subset_b0_b1.json",
        ROOT / "results" / "n5k_subset_b0_b1_n50_seed42.jsonl",
        ROOT / "results" / "n5k_subset_b0_b1_n50_seed42.json",
    ]
    return [path for path in names if path.is_file()]


def _assert_output_targets(out_dir: Path) -> tuple[Path, Path]:
    resolved = out_dir.resolve()
    if "weak_steps" in resolved.parts or resolved.name.startswith("n5k"):
        raise FixRateRefused(f"refusing to write under {resolved}")
    json_path = resolved / JSON_NAME
    text_path = resolved / REPORT_NAME
    for path in (json_path, text_path):
        if path.name in FROZEN_OUTPUT_NAMES or path.name.startswith("n5k_"):
            raise FixRateRefused(f"refusing to write frozen artifact {path.name}")
    return json_path, text_path


def _group_steps(path: Path) -> dict[str, list[dict]]:
    assert_bc_jsonl_path(path)
    forbidden = test_split_ids()
    grouped: dict[str, list[dict]] = {}
    for record in read_ir_jsonl(path):
        dish_id = record.get("dish_id")
        if not isinstance(dish_id, str) or not dish_id:
            raise FixRateRefused(f"{path.name}: row missing dish_id")
        if dish_id in forbidden:
            raise FixRateRefused(f"{path.name}: refusing test dish {dish_id}")
        slot = pool_of_dish(dish_id)
        if slot != "holdout":
            raise FixRateRefused(f"{path.name}: {dish_id} is in the {slot} pool, not holdout")
        if record.get("action_t") == "VOLUME_ESTIMATE":
            raise FixRateRefused(f"{path.name}: VOLUME_ESTIMATE on {dish_id}")
        grouped.setdefault(dish_id, []).append(record)
    if not grouped:
        raise FixRateRefused(f"{path.name}: no hold-out rows")
    return grouped


def _train_energy(dish_ids: set[str]) -> dict[str, float]:
    labels = load_labels()
    missing = sorted(dish_id for dish_id in dish_ids if dish_id not in labels)
    if missing:
        raise FixRateRefused(
            "train cafe label missing for "
            + ", ".join(missing[:5])
            + (" ..." if len(missing) > 5 else "")
        )
    return {dish_id: float(labels[dish_id]["energy_kcal"]) for dish_id in dish_ids}


def _load_policy(weights: Path, meta_path: Path) -> PolicyStateMLP:
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    actions = tuple(meta.get("actions") or ())
    if actions != tuple(BC_ACTIONS):
        raise FixRateRefused(
            "train_meta action order does not match the BC head; refusing to score a permuted policy"
        )
    class_weight = meta.get("class_weight")
    if not isinstance(class_weight, dict):
        raise FixRateRefused("train_meta is missing class_weight")
    return PolicyStateMLP.from_npz(weights, actions=actions, class_weight=class_weight)


def _saved_accuracy(path: Path) -> float:
    payload = json.loads(path.read_text(encoding="utf-8"))
    return float(payload["evaluations"]["none"]["overall_accuracy"])


def _jsonable(payload: dict) -> str:
    return json.dumps(payload, indent=2, ensure_ascii=False) + "\n"


def main() -> int:
    parser = argparse.ArgumentParser(description="Closed-loop hold-out FixRate / WorsenRate / tool cost")
    parser.add_argument("--self-check", action="store_true", help="Synthetic contract only; writes nothing")
    parser.add_argument("--holdout", type=Path, default=HOLDOUT_JSONL)
    parser.add_argument("--weights", type=Path, default=WEIGHTS)
    parser.add_argument("--train-meta", type=Path, default=TRAIN_META)
    parser.add_argument("--saved-metrics", type=Path, default=SAVED_METRICS)
    parser.add_argument("--out-dir", type=Path, default=OUT_DIR)
    args = parser.parse_args()

    if args.self_check:
        run_self_check()
        print(
            f"closed_loop_fixrate self-check ok epsilon={FIXRATE_EPSILON_KCAL} K={BUDGET_K}"
        )
        return 0

    run_self_check()
    started = time.perf_counter()
    holdout = args.holdout if args.holdout.is_absolute() else ROOT / args.holdout
    weights = args.weights if args.weights.is_absolute() else ROOT / args.weights
    meta_path = args.train_meta if args.train_meta.is_absolute() else ROOT / args.train_meta
    saved_metrics = args.saved_metrics if args.saved_metrics.is_absolute() else ROOT / args.saved_metrics
    out_dir = args.out_dir if args.out_dir.is_absolute() else ROOT / args.out_dir
    json_path, text_path = _assert_output_targets(out_dir)

    for path in (holdout, weights, meta_path, saved_metrics):
        if not path.is_file():
            raise FixRateRefused(f"missing {path}")

    frozen_paths = _frozen_paths()
    if len({path.name for path in frozen_paths}) != len(frozen_paths):
        raise FixRateRefused("frozen paths share a file name; refusing an ambiguous hash map")
    frozen_before = {path.name: _sha256(path) for path in frozen_paths}
    steps_by_dish = _group_steps(holdout)
    dataset = load_closed_loop_rows(holdout, expect_pool="holdout")
    raw_keys = {
        (dish_id, int(step["step_i"]))
        for dish_id, steps in steps_by_dish.items()
        for step in steps
    }
    feature_keys = set(zip(dataset.dish_ids, (int(step) for step in dataset.step_indices)))
    if raw_keys != feature_keys:
        raise FixRateRefused("hold-out feature rows and IR rows do not name the same steps")

    policy = _load_policy(weights, meta_path)
    predicted = policy.predict_actions(dataset.X)
    truth = [str(item) for item in dataset.y.tolist()]
    if len(predicted) != len(truth):
        raise FixRateRefused("prediction length does not match hold-out rows")
    agreement = sum(pred == label for pred, label in zip(predicted, truth)) / len(truth)
    saved_accuracy = _saved_accuracy(saved_metrics)
    sanity_ok = abs(agreement - saved_accuracy) <= 1e-12
    predictions = {
        (str(dish_id), int(step_i)): action
        for dish_id, step_i, action in zip(dataset.dish_ids, dataset.step_indices, predicted)
    }
    payload = evaluate_fixrate(
        steps_by_dish,
        predictions,
        _train_energy(set(steps_by_dish)),
    )
    payload["loader_sanity"] = {
        "logged_action_agreement": agreement,
        "saved_holdout_accuracy": saved_accuracy,
        "agreement_matches_saved_metrics": sanity_ok,
        "n_rows": len(truth),
        "note": "Weight-loader check against metrics_holdout.json. Overall accuracy is not method proof.",
    }
    if payload["primary"]["n_pairs"] <= 0:
        raise FixRateRefused("primary FixRate has no scored pairs")

    elapsed = time.perf_counter() - started

    def _rel(path: Path) -> str:
        return str(path.relative_to(ROOT)) if path.is_relative_to(ROOT) else str(path)

    payload["elapsed_seconds"] = round(elapsed, 2)
    payload["inputs"] = {
        "holdout_jsonl": _rel(holdout),
        "weights": _rel(weights),
        "train_meta": _rel(meta_path),
        "saved_metrics": _rel(saved_metrics),
        "holdout_sha256": frozen_before.get(holdout.name, _sha256(holdout)),
        "weights_sha256": frozen_before.get(weights.name, _sha256(weights)),
        "frozen_sha256_prefix": {name: digest[:16] for name, digest in sorted(frozen_before.items())},
        "documented_prefix_match": {
            name: frozen_before.get(name, "").startswith(prefix)
            for name, prefix in DOCUMENTED_PREFIXES.items()
        },
        "frozen_unchanged": True,
    }
    text = format_fixrate_report(payload)
    out_dir.mkdir(parents=True, exist_ok=True)
    json_path.write_text(_jsonable(payload), encoding="utf-8")
    text_path.write_text(text, encoding="utf-8")
    frozen_after = {path.name: _sha256(path) for path in frozen_paths}
    changed = sorted(name for name, digest in frozen_before.items() if frozen_after.get(name) != digest)
    if changed:
        json_path.unlink(missing_ok=True)
        text_path.unlink(missing_ok=True)
        raise FixRateRefused("frozen files changed while writing FixRate: " + ", ".join(changed))

    print(text, end="")
    print(
        f"wrote {json_path.relative_to(ROOT)} and {text_path.relative_to(ROOT)} "
        f"in {elapsed:.1f}s loader_ok={sanity_ok}"
    )
    if not sanity_ok:
        print(
            f"loader agreement {agreement:.6f} != saved accuracy {saved_accuracy:.6f}",
            file=sys.stderr,
        )
        return 1
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except FixRateRefused as exc:
        print(f"FIXRATE REFUSED: {exc}", file=sys.stderr)
        raise SystemExit(1) from exc
