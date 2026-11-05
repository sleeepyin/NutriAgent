#!/usr/bin/env python3
"""On-policy rgb_test rollout for the closed-loop energy table.

The saved policy drives every action. Local Qwen3.8-27B-VL (llama.cpp on
GPU 0, port 18080) writes component hypotheses. FNDDS and the portion grid
write the tool IR. Cafe labels are not read here; scoring is a separate
command that does not fit a threshold.

    python scripts/closed_loop_on_policy_rgb_test.py --self-check
    python scripts/closed_loop_on_policy_rgb_test.py --limit 1
    python scripts/closed_loop_on_policy_rgb_test.py

``--limit`` is an official-order prefix for a smoke, not the main table.
The full command writes ``results/closed_loop/on_policy_rgb_test.jsonl`` and
then scores it with ``closed_loop_mae_aurc.py --on-policy-vlm``. That score
file is not ``mae_aurc_holdout.json``.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
import time
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from vdr.controller.on_policy import (  # noqa: E402
    OnPolicyRefused,
    load_closed_loop_policy,
    rollout_dish,
    run_self_check,
)
from vdr.data.n5k import overhead_rgb, read_ids  # noqa: E402
from vdr.perception.qwen38 import Qwen38Error, server_ready  # noqa: E402
from vdr.weak.teacher import PROTECTED_RESULT_NAMES  # noqa: E402

OUT_DIR = ROOT / "results" / "closed_loop"
WEIGHTS = OUT_DIR / "policy_state_mlp.npz"
TRAIN_META = OUT_DIR / "train_meta.json"
FULL_JSONL = OUT_DIR / "on_policy_rgb_test.jsonl"
FULL_SUMMARY = OUT_DIR / "on_policy_rgb_test_summary.json"
SMOKE_JSONL = OUT_DIR / "on_policy_rgb_test_smoke.jsonl"
SMOKE_SUMMARY = OUT_DIR / "on_policy_rgb_test_smoke_summary.json"
FAILURES = OUT_DIR / "on_policy_rgb_test_failures.jsonl"
WEIGHT_PREFIX = "f0ac0b41f99cbd72"
DOCUMENTED_PREFIXES = {
    "rgb_train_fit_all.jsonl": "ed47f9239bfecb3c",
    "n5k_rgb_test_overhead_b0_b1.jsonl": "a4dfdd82e3a595f4",
    "policy_state_mlp.npz": WEIGHT_PREFIX,
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
        OUT_DIR / "metrics_holdout.json",
        OUT_DIR / "report_holdout.txt",
        OUT_DIR / "fixrate_holdout.json",
        OUT_DIR / "report_fixrate.txt",
        OUT_DIR / "mae_aurc_holdout.json",
        OUT_DIR / "report_mae_aurc.txt",
        OUT_DIR / "abstain_calibration.json",
        ROOT / "results" / "n5k_rgb_test_overhead_b0_b1.jsonl",
        ROOT / "results" / "n5k_rgb_test_overhead_b0_b1.json",
        ROOT / "results" / "n5k_subset_b0_b1.jsonl",
        ROOT / "results" / "n5k_subset_b0_b1.json",
        ROOT / "results" / "n5k_subset_b0_b1_n50_seed42.jsonl",
        ROOT / "results" / "n5k_subset_b0_b1_n50_seed42.json",
    ]
    return [path for path in names if path.is_file()]


def _assert_frozen(before: dict[str, str] | None = None) -> dict[str, str]:
    paths = _frozen_paths()
    if len({path.name for path in paths}) != len(paths):
        raise OnPolicyRefused("frozen paths share a file name")
    digests = {path.name: _sha256(path) for path in paths}
    for name, prefix in DOCUMENTED_PREFIXES.items():
        digest = digests.get(name)
        if digest is None or not digest.startswith(prefix):
            raise OnPolicyRefused(f"{name} sha256 does not start with {prefix}")
    if before is not None:
        changed = sorted(name for name, digest in before.items() if digests.get(name) != digest)
        if changed:
            raise OnPolicyRefused("frozen files changed: " + ", ".join(changed))
    return digests


def _assert_output(path: Path) -> None:
    if path.parent.resolve() != OUT_DIR.resolve():
        raise OnPolicyRefused(f"refusing to write outside {OUT_DIR}")
    if path.name in PROTECTED_RESULT_NAMES or path.name.startswith("n5k_") or "holdout" in path.name:
        raise OnPolicyRefused(f"refusing to write {path.name}")


def _overhead_ids() -> list[str]:
    return [dish_id for dish_id in read_ids("rgb_test") if overhead_rgb(dish_id).is_file()]


def _done_ids(path: Path) -> set[str]:
    if not path.is_file():
        return set()
    done: set[str] = set()
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        dish_id = row.get("dish_id")
        if isinstance(dish_id, str):
            done.add(dish_id)
    return done


def _append(path: Path, steps: list[dict]) -> None:
    _assert_output(path)
    with path.open("a", encoding="utf-8") as handle:
        for step in steps:
            handle.write(json.dumps(step, ensure_ascii=False) + "\n")
        handle.flush()


def _terminal_kcal(steps: list[dict]) -> float | None:
    nutri = steps[-1].get("nutri_dish") if steps else None
    if not isinstance(nutri, dict):
        return None
    value = nutri.get("energy_kcal")
    if value is None:
        return None
    return float(value)


def _summary(
    *,
    jsonl: Path,
    attempted: list[str],
    failures: list[dict],
    vlm: dict,
    frozen: dict[str, str],
    elapsed: float,
    complete: bool,
    full_pool: bool,
) -> dict:
    actions: Counter[str] = Counter()
    terminals: Counter[str] = Counter()
    n_steps = 0
    n_dishes = 0
    if jsonl.is_file():
        by_dish: dict[str, list[dict]] = {}
        for line in jsonl.read_text(encoding="utf-8").splitlines():
            if line.strip():
                row = json.loads(line)
                by_dish.setdefault(row["dish_id"], []).append(row)
        for steps in by_dish.values():
            n_dishes += 1
            n_steps += len(steps)
            for step in steps:
                actions[str(step.get("action_t"))] += 1
            terminals[str(steps[-1].get("action_t"))] += 1
    return {
        "protocol": "on_policy_vlm_closed_loop_v1",
        "split": "rgb_test",
        "full_pool": full_pool,
        "complete": complete,
        "n_official_rgb_test": len(read_ids("rgb_test")),
        "n_with_overhead_rgb": len(_overhead_ids()),
        "n_attempted_this_process": len(attempted),
        "n_dishes_in_jsonl": n_dishes,
        "n_steps": n_steps,
        "n_failures": len(failures),
        "failures": failures,
        "action_counts": dict(sorted(actions.items())),
        "terminal_action_counts": dict(sorted(terminals.items())),
        "trajectories": str(jsonl.relative_to(ROOT)),
        "trajectories_sha256": _sha256(jsonl) if jsonl.is_file() else None,
        "vlm": vlm,
        "policy_weights": "results/closed_loop/policy_state_mlp.npz",
        "policy_weights_sha256_prefix": frozen.get("policy_state_mlp.npz", "")[:16],
        "uses_test_gt": False,
        "rl": False,
        "volume_tool": False,
        "not_faithful_b1_or_b2": True,
        "b5_and_masks": (
            "Not separate rollouts. closed_loop_mae_aurc.py --on-policy-vlm re-reads this "
            "trace. Mask-none matches the executed actions. B5 and observation masks are "
            "counterfactual predictions on these states."
        ),
        "offline_holdout_not_this_file": (
            "results/closed_loop/mae_aurc_holdout.json is the #28/#31 offline logged "
            "hold-out table. It is not this rollout."
        ),
        "labels": "This file does not contain cafe labels and does not report MAE.",
        "elapsed_seconds": round(elapsed, 2),
        "frozen_sha256_prefix": {name: digest[:16] for name, digest in sorted(frozen.items())},
        "status_file": "docs/STATUS.md was not modified.",
        "owner": "营养论文代码编写",
    }


def _write_summary(path: Path, payload: dict) -> None:
    _assert_output(path)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def _score(jsonl: Path) -> None:
    command = [
        sys.executable,
        str(ROOT / "scripts" / "closed_loop_mae_aurc.py"),
        "--trajectories",
        str(jsonl),
        "--expect-pool",
        "any",
        "--allow-test-split",
        "--on-policy-vlm",
    ]
    subprocess.run(command, cwd=ROOT, check=True)


def main() -> int:
    parser = argparse.ArgumentParser(description="On-policy Qwen closed-loop rollout on rgb_test")
    parser.add_argument("--self-check", action="store_true")
    parser.add_argument("--limit", type=int, default=None, help="Official-order prefix. Not the main table.")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--no-score", action="store_true")
    args = parser.parse_args()
    if args.self_check:
        run_self_check()
        print("on-policy self-check ok")
        return 0
    if args.limit is not None and args.limit < 1:
        raise OnPolicyRefused("--limit must be >= 1")

    frozen = _assert_frozen()
    vlm = server_ready()
    ids = _overhead_ids()
    if args.limit is not None:
        ids = ids[: args.limit]
    jsonl = SMOKE_JSONL if args.limit is not None else FULL_JSONL
    summary_path = SMOKE_SUMMARY if args.limit is not None else FULL_SUMMARY
    _assert_output(jsonl)
    _assert_output(summary_path)
    if jsonl.is_file() and not args.resume and args.limit is None:
        raise OnPolicyRefused(f"{jsonl.name} already exists; pass --resume to continue")
    if args.limit is not None and jsonl.is_file() and not args.resume:
        jsonl.unlink()
    done = _done_ids(jsonl) if args.resume else set()
    policy = load_closed_loop_policy(WEIGHTS, TRAIN_META)
    failures: list[dict] = []
    attempted: list[str] = []
    started = time.perf_counter()
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    pending = [dish_id for dish_id in ids if dish_id not in done]
    print(
        f"on-policy rgb_test n={len(pending)} skipped={len(done)} "
        f"model={vlm.get('model_alias')} base={vlm.get('base_url')}",
        flush=True,
    )
    for index, dish_id in enumerate(pending, start=1):
        image = overhead_rgb(dish_id)
        t0 = time.perf_counter()
        try:
            result = rollout_dish(dish_id, policy, image_path=image)
        except (OnPolicyRefused, Qwen38Error, OSError, ValueError) as exc:
            failures.append({"dish_id": dish_id, "error": f"{type(exc).__name__}: {exc}"})
            print(f"[{index}/{len(pending)}] {dish_id} FAIL {exc}", flush=True)
            if index == 1 and not done:
                raise
            continue
        _append(jsonl, result.steps)
        attempted.append(dish_id)
        print(
            f"[{index}/{len(pending)}] {dish_id} steps={len(result.steps)} "
            f"end={result.steps[-1]['action_t']} reason={result.ended_reason} "
            f"hyps={result.n_hypotheses} kcal={_terminal_kcal(result.steps)} "
            f"{time.perf_counter() - t0:.1f}s",
            flush=True,
        )
    elapsed = time.perf_counter() - started
    frozen_after = _assert_frozen(frozen)
    payload = _summary(
        jsonl=jsonl,
        attempted=attempted,
        failures=failures,
        vlm=vlm,
        frozen=frozen_after,
        elapsed=elapsed,
        complete=not pending or (len(attempted) + len(failures) == len(pending)),
        full_pool=args.limit is None and len(ids) == len(_overhead_ids()),
    )
    _write_summary(summary_path, payload)
    print(f"wrote {summary_path.relative_to(ROOT)} failures={len(failures)}", flush=True)
    if failures and args.limit is None:
        _assert_output(FAILURES)
        with FAILURES.open("a", encoding="utf-8") as handle:
            for row in failures:
                handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    if args.limit is not None:
        print("smoke prefix was not scored as the rgb_test main table", flush=True)
    elif args.no_score:
        print("scoring skipped by --no-score", flush=True)
    elif jsonl.is_file() and jsonl.stat().st_size > 0:
        if failures:
            print(
                f"scoring {jsonl.name}; {len(failures)} failed dishes were not imputed",
                flush=True,
            )
        _score(jsonl)
        _assert_frozen(frozen)
    else:
        print("no trajectories were written; refusing to emit an energy table", flush=True)
        return 1
    return 0 if not failures else 1


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except OnPolicyRefused as exc:
        print(f"ON-POLICY REFUSED: {exc}", file=sys.stderr)
        raise SystemExit(1) from exc
