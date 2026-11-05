#!/usr/bin/env python3
"""P0a audit: whether evidence tools change dish energy and mass.

Reads the logged hold-out weak steps used by the #28/#31 offline table and
the #34 on-policy rgb_test trajectories. Does not train, does not write
weights, and does not read cafe or rgb_test nutrition labels.

    python scripts/closed_loop_tool_state_audit.py --self-check
    python scripts/closed_loop_tool_state_audit.py

Writes only:

    results/closed_loop/tool_state_audit.json
    results/closed_loop/report_tool_state_audit.txt
    results/closed_loop/report_tool_state_audit.md
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

from vdr.controller.bc_data import assert_bc_jsonl_path  # noqa: E402
from vdr.controller.closed_loop_data import HOLDOUT_JSONL  # noqa: E402
from vdr.controller.on_policy import OnPolicyRefused, assert_on_policy_marker  # noqa: E402
from vdr.data.n5k import test_split_ids  # noqa: E402
from vdr.eval.tool_state_audit import (  # noqa: E402
    AUDIT_ACTIONS,
    EPSILON_LABEL,
    ToolStateAuditRefused,
    build_payload,
    format_markdown_report,
    format_text_report,
    run_self_check,
)
from vdr.ir.io import read_ir_jsonl  # noqa: E402
from vdr.weak.teacher import pool_of_dish  # noqa: E402

OUT_DIR = ROOT / "results" / "closed_loop"
ON_POLICY_JSONL = OUT_DIR / "on_policy_rgb_test.jsonl"
JSON_NAME = "tool_state_audit.json"
TXT_NAME = "report_tool_state_audit.txt"
MD_NAME = "report_tool_state_audit.md"

# File-identity checks from the #28/#31 and #34 records. Not thresholds.
DOCUMENTED_HOLDOUT = {
    "n_dishes": 380,
    "n_rows": 2771,
    "n_stop_pass_corrected": 356,
    "actions": {
        "RETRIEVE": 1285,
        "RE_PORTION": 949,
        "STOP": 356,
        "CORRECT_CODE": 157,
        "ABSTAIN": 24,
        "RE_DECOMPOSE": 0,
    },
    "last_action": {"STOP": 356, "ABSTAIN": 24},
}
DOCUMENTED_ON_POLICY = {
    "n_dishes": 507,
    "n_rows": 3822,
    "n_stop_pass_corrected": 488,
    "actions": {
        "RETRIEVE": 1530,
        "RE_PORTION": 1469,
        "STOP": 488,
        "CORRECT_CODE": 326,
        "ABSTAIN": 9,
        "RE_DECOMPOSE": 0,
    },
    "last_action": {"STOP": 488, "ABSTAIN": 9, "RETRIEVE": 10},
}
DOCUMENTED_PREFIXES = {
    "rgb_train_fit_all.jsonl": "ed47f9239bfecb3c",
    "n5k_rgb_test_overhead_b0_b1.jsonl": "a4dfdd82e3a595f4",
}
FROZEN_OUTPUT_NAMES = {
    "metrics_holdout.json",
    "policy_state_mlp.npz",
    "report_holdout.txt",
    "train_meta.json",
    "fixrate_holdout.json",
    "report_fixrate.txt",
    "mae_aurc_holdout.json",
    "report_mae_aurc.txt",
    "mae_aurc_testsplit.json",
    "report_mae_aurc_testsplit.txt",
    "abstain_calibration.json",
    "on_policy_rgb_test.jsonl",
    "on_policy_rgb_test_summary.json",
    "rgb_train_fit_all.jsonl",
    "rgb_train_holdout_all.jsonl",
    "rgb_train_fit_all.summary.json",
    "rgb_train_holdout_all.summary.json",
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
        OUT_DIR / "policy_state_mlp.npz",
        OUT_DIR / "train_meta.json",
        OUT_DIR / "metrics_holdout.json",
        OUT_DIR / "report_holdout.txt",
        OUT_DIR / "fixrate_holdout.json",
        OUT_DIR / "report_fixrate.txt",
        OUT_DIR / "mae_aurc_holdout.json",
        OUT_DIR / "report_mae_aurc.txt",
        OUT_DIR / "mae_aurc_testsplit.json",
        OUT_DIR / "report_mae_aurc_testsplit.txt",
        OUT_DIR / "abstain_calibration.json",
        OUT_DIR / "on_policy_rgb_test.jsonl",
        OUT_DIR / "on_policy_rgb_test_summary.json",
        ROOT / "results" / "n5k_rgb_test_overhead_b0_b1.jsonl",
        ROOT / "results" / "n5k_rgb_test_overhead_b0_b1.json",
        ROOT / "results" / "n5k_subset_b0_b1.jsonl",
        ROOT / "results" / "n5k_subset_b0_b1.json",
        ROOT / "results" / "n5k_subset_b0_b1_n50_seed42.jsonl",
        ROOT / "results" / "n5k_subset_b0_b1_n50_seed42.json",
    ]
    return [path for path in names if path.is_file()]


def _assert_output_targets(out_dir: Path) -> tuple[Path, Path, Path]:
    resolved = out_dir.resolve()
    if "weak_steps" in resolved.parts or resolved.name.startswith("n5k"):
        raise ToolStateAuditRefused(f"refusing to write under {resolved}")
    paths = (resolved / JSON_NAME, resolved / TXT_NAME, resolved / MD_NAME)
    for path in paths:
        if path.name in FROZEN_OUTPUT_NAMES or path.name.startswith("n5k_"):
            raise ToolStateAuditRefused(f"refusing to write frozen artifact {path.name}")
    return paths


def _group_rows(path: Path) -> dict[str, list[dict]]:
    grouped: dict[str, list[dict]] = {}
    for record in read_ir_jsonl(path):
        dish_id = record.get("dish_id")
        if not isinstance(dish_id, str) or not dish_id:
            raise ToolStateAuditRefused(f"{path.name}: row missing dish_id")
        grouped.setdefault(dish_id, []).append(record)
    if not grouped:
        raise ToolStateAuditRefused(f"{path.name}: no rows")
    return grouped


def _census(grouped: dict[str, list[dict]]) -> dict[str, Counter[str]]:
    actions: Counter[str] = Counter()
    last_action: Counter[str] = Counter()
    for steps in grouped.values():
        for row in steps:
            action = row.get("action_t")
            actions[str(action)] += 1
        last = max(steps, key=lambda row: int(row["step_i"]))
        last_action[str(last.get("action_t"))] += 1
    return {"actions": actions, "last_action": last_action}


def _require_census(name: str, grouped: dict[str, list[dict]], documented: dict) -> None:
    census = _census(grouped)
    n_rows = sum(census["actions"].values())
    problems: list[str] = []
    if len(grouped) != int(documented["n_dishes"]):
        problems.append(f"dishes {len(grouped)} != {documented['n_dishes']}")
    if n_rows != int(documented["n_rows"]):
        problems.append(f"rows {n_rows} != {documented['n_rows']}")
    for action, count in documented["actions"].items():
        seen = int(census["actions"].get(action, 0))
        if seen != int(count):
            problems.append(f"{action} {seen} != {count}")
    extra = sorted(set(census["actions"]) - set(documented["actions"]))
    if extra:
        problems.append("unexpected actions " + ", ".join(extra))
    for action, count in documented["last_action"].items():
        seen = int(census["last_action"].get(action, 0))
        if seen != int(count):
            problems.append(f"last {action} {seen} != {count}")
    if problems:
        raise ToolStateAuditRefused(f"{name} is not the documented trajectory file: " + "; ".join(problems))


def _stop_pass_corrected(grouped: dict[str, list[dict]]) -> dict[str, list[dict]]:
    kept: dict[str, list[dict]] = {}
    for dish_id, steps in grouped.items():
        last = max(steps, key=lambda row: int(row["step_i"]))
        if last.get("action_t") == "STOP" and last.get("s") in {"pass", "corrected"}:
            kept[dish_id] = steps
    return kept


def _load_holdout(path: Path) -> dict[str, list[dict]]:
    assert_bc_jsonl_path(path)
    forbidden = test_split_ids()
    grouped = _group_rows(path)
    for dish_id, steps in grouped.items():
        if dish_id in forbidden:
            raise ToolStateAuditRefused(f"{path.name}: refusing test dish {dish_id}")
        slot = pool_of_dish(dish_id)
        if slot != "holdout":
            raise ToolStateAuditRefused(f"{path.name}: {dish_id} is in the {slot} pool, not holdout")
        for row in steps:
            obs = row.get("obs_t")
            if isinstance(obs, dict) and obs.get("rollout") == "on_policy_vlm":
                raise ToolStateAuditRefused(f"{path.name}: {dish_id} is an on-policy trajectory")
            if isinstance(obs, dict) and obs.get("uses_test_gt") is True:
                raise ToolStateAuditRefused(f"{path.name}: {dish_id} marks uses_test_gt")
    return grouped


def _load_on_policy(path: Path) -> dict[str, list[dict]]:
    if "holdout" in path.name or path.name.startswith("rgb_train"):
        raise ToolStateAuditRefused(f"{path.name}: refusing to read a weak-step log as on-policy")
    grouped = _group_rows(path)
    try:
        assert_on_policy_marker(grouped)
    except OnPolicyRefused as exc:
        raise ToolStateAuditRefused(str(exc)) from exc
    return grouped


def _rel(path: Path) -> str:
    return str(path.relative_to(ROOT)) if path.is_relative_to(ROOT) else str(path)


def _jsonable(payload: dict) -> str:
    return json.dumps(payload, indent=2, ensure_ascii=False) + "\n"


def _git_head() -> str | None:
    try:
        out = subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            cwd=ROOT,
            text=True,
            stderr=subprocess.DEVNULL,
        )
    except (OSError, subprocess.CalledProcessError):
        return None
    text = out.strip()
    return text or None


def main() -> int:
    parser = argparse.ArgumentParser(description="P0a audit of dish energy and mass around tool steps")
    parser.add_argument("--self-check", action="store_true", help="Synthetic contract only; writes nothing")
    parser.add_argument("--holdout", type=Path, default=HOLDOUT_JSONL)
    parser.add_argument("--on-policy", type=Path, default=ON_POLICY_JSONL)
    parser.add_argument("--out-dir", type=Path, default=OUT_DIR)
    args = parser.parse_args()

    if args.self_check:
        run_self_check()
        print(f"closed_loop_tool_state_audit self-check ok epsilon={EPSILON_LABEL}")
        return 0

    run_self_check()
    started = time.perf_counter()
    holdout = args.holdout if args.holdout.is_absolute() else ROOT / args.holdout
    on_policy = args.on_policy if args.on_policy.is_absolute() else ROOT / args.on_policy
    out_dir = args.out_dir if args.out_dir.is_absolute() else ROOT / args.out_dir
    json_path, text_path, md_path = _assert_output_targets(out_dir)
    for path in (holdout, on_policy):
        if not path.is_file():
            raise ToolStateAuditRefused(f"missing {path}")

    frozen_paths = _frozen_paths()
    if len({path.name for path in frozen_paths}) != len(frozen_paths):
        raise ToolStateAuditRefused("frozen paths share a file name; refusing an ambiguous hash map")
    frozen_before = {path.name: _sha256(path) for path in frozen_paths}
    for name, prefix in DOCUMENTED_PREFIXES.items():
        digest = frozen_before.get(name)
        if digest is not None and not digest.startswith(prefix):
            raise ToolStateAuditRefused(f"{name} sha256 prefix {digest[:16]} != documented {prefix}")

    logged = _load_holdout(holdout)
    rolled = _load_on_policy(on_policy)
    _require_census("logged_holdout", logged, DOCUMENTED_HOLDOUT)
    _require_census("on_policy_rgb_test", rolled, DOCUMENTED_ON_POLICY)
    logged_primary = _stop_pass_corrected(logged)
    rolled_primary = _stop_pass_corrected(rolled)
    if len(logged_primary) != int(DOCUMENTED_HOLDOUT["n_stop_pass_corrected"]):
        raise ToolStateAuditRefused(
            f"logged STOP pass/corrected dishes {len(logged_primary)} != {DOCUMENTED_HOLDOUT['n_stop_pass_corrected']}"
        )
    if len(rolled_primary) != int(DOCUMENTED_ON_POLICY["n_stop_pass_corrected"]):
        raise ToolStateAuditRefused(
            f"on-policy STOP pass/corrected dishes {len(rolled_primary)} != {DOCUMENTED_ON_POLICY['n_stop_pass_corrected']}"
        )

    payload = build_payload(
        {
            "logged_holdout": logged,
            "on_policy_rgb_test": rolled,
            "logged_holdout_stop_pass_corrected": logged_primary,
            "on_policy_stop_pass_corrected": rolled_primary,
        },
        overall_of=("logged_holdout", "on_policy_rgb_test"),
    )
    elapsed = time.perf_counter() - started
    payload["elapsed_seconds"] = round(elapsed, 2)
    payload["inputs"] = {
        "logged_holdout_jsonl": _rel(holdout),
        "on_policy_jsonl": _rel(on_policy),
        "logged_holdout_sha256": frozen_before.get(holdout.name, _sha256(holdout)),
        "on_policy_sha256": frozen_before.get(on_policy.name, _sha256(on_policy)),
        "git_head": _git_head(),
        "documented_population": {
            "logged_holdout": "rgb_train hold-out weak steps; #28/#31 offline family. Primary subset is STOP and s in {pass, corrected}.",
            "on_policy_rgb_test": "results/closed_loop/on_policy_rgb_test.jsonl; #34 on-policy family. Primary subset is STOP and s in {pass, corrected}.",
        },
        "frozen_sha256_prefix": {name: digest[:16] for name, digest in sorted(frozen_before.items())},
        "documented_prefix_match": {
            name: frozen_before.get(name, "").startswith(prefix) if name in frozen_before else None
            for name, prefix in DOCUMENTED_PREFIXES.items()
        },
        "labels_read": False,
        "mass_estimator_trained": False,
        "frozen_unchanged": True,
    }
    payload["audited_actions"] = list(AUDIT_ACTIONS)
    text = format_text_report(payload)
    markdown = format_markdown_report(payload)
    out_dir.mkdir(parents=True, exist_ok=True)
    json_path.write_text(_jsonable(payload), encoding="utf-8")
    text_path.write_text(text, encoding="utf-8")
    md_path.write_text(markdown, encoding="utf-8")
    frozen_after = {path.name: _sha256(path) for path in frozen_paths}
    changed = sorted(name for name, digest in frozen_before.items() if frozen_after.get(name) != digest)
    if changed:
        json_path.unlink(missing_ok=True)
        text_path.unlink(missing_ok=True)
        md_path.unlink(missing_ok=True)
        raise ToolStateAuditRefused("frozen files changed while writing the audit: " + ", ".join(changed))

    print(text, end="")
    print(
        f"wrote {_rel(json_path)}, {_rel(text_path)}, and {_rel(md_path)} in {elapsed:.1f}s"
    )
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except ToolStateAuditRefused as exc:
        print(f"TOOL STATE AUDIT REFUSED: {exc}", file=sys.stderr)
        raise SystemExit(1) from exc
