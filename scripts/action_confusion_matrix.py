#!/usr/bin/env python3
"""Offline action confusion matrix + per-class PR/F1 for VDR controller.

JSONL rows need gold ``action_t`` (or ``gold_action``) and prediction
``pred_action`` (or ``pred`` / ``action_pred``).

Demo mode reconstructs the documented BC MLP hold-out recalls from
docs/STATUS.md as a synthetic check. Overall accuracy is not the story.
ABSTAIN F1 must be 0 for that case (0/24, all predicted STOP). Acc ≈ 0.9700.
This does not train a policy and does not read frozen weak-step jsonl.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Sequence

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from vdr.eval.action_metrics import (  # noqa: E402
    ACTIONS,
    confusion_matrix,
    format_report,
    per_class_prf,
)

# Documented BC MLP hold-out recalls (STATUS): hits / support.
BC_MLP_HOLDOUT_RECALL = {
    "RETRIEVE": (1273, 1285),
    "RE_PORTION": (917, 949),
    "STOP": (356, 356),
    "CORRECT_CODE": (142, 157),
    "ABSTAIN": (0, 24),
    "RE_DECOMPOSE": (0, 0),
}


def load_jsonl(path: Path) -> tuple[list[str], list[str]]:
    gold: list[str] = []
    pred: list[str] = []
    with path.open(encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, 1):
            line = line.strip()
            if not line:
                continue
            row: dict[str, Any] = json.loads(line)
            gold_action = row.get("action_t", row.get("gold_action", row.get("gold")))
            pred_action = row.get("pred_action", row.get("pred", row.get("action_pred")))
            if gold_action is None or pred_action is None:
                raise ValueError(
                    f"{path}:{line_no}: need action_t/gold_action and pred_action/pred"
                )
            gold.append(str(gold_action))
            pred.append(str(pred_action))
    return gold, pred


def synthesize_bc_mlp_holdout() -> tuple[list[str], list[str]]:
    """Synthetic pairs matching documented BC MLP hold-out recalls.

    Off-diagonal mass sits on the STATUS-documented colliding classes:
    ABSTAIN all go to STOP (24); RETRIEVE misses (12) go to RE_PORTION;
    RE_PORTION misses (32) are 31 CORRECT_CODE and 1 RETRIEVE; CORRECT_CODE
    misses (15) go to RE_PORTION; STOP is perfect; RE_DECOMPOSE is empty.

    The exact unpublished matrix may differ on non-ABSTAIN precision. This
    demo locks the recall numbers, ABSTAIN F1 = 0, and accuracy 2688/2771.
    """
    gold: list[str] = []
    pred: list[str] = []

    def add(gold_action: str, pred_action: str, n: int) -> None:
        gold.extend([gold_action] * n)
        pred.extend([pred_action] * n)

    add("RETRIEVE", "RETRIEVE", 1273)
    add("RETRIEVE", "RE_PORTION", 12)
    add("RE_PORTION", "RE_PORTION", 917)
    add("RE_PORTION", "CORRECT_CODE", 31)
    add("RE_PORTION", "RETRIEVE", 1)
    add("STOP", "STOP", 356)
    add("CORRECT_CODE", "CORRECT_CODE", 142)
    add("CORRECT_CODE", "RE_PORTION", 15)
    add("ABSTAIN", "STOP", 24)

    if len(gold) != 2771:
        raise RuntimeError(f"synthetic hold-out length {len(gold)} != 2771")
    for name, (hits, support) in BC_MLP_HOLDOUT_RECALL.items():
        if support == 0:
            continue
        got = sum(1 for g, p in zip(gold, pred) if g == name and p == name)
        if got != hits:
            raise RuntimeError(f"{name} hits {got} != {hits}")
        if sum(1 for g in gold if g == name) != support:
            raise RuntimeError(f"{name} support mismatch")
    return gold, pred


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--jsonl", type=Path, help="JSONL with gold + pred actions")
    parser.add_argument(
        "--demo",
        action="store_true",
        help="Synthetic BC MLP hold-out recall check (ABSTAIN F1=0, Acc≈0.9700)",
    )
    parser.add_argument("--out", type=Path, help="Write report to this path")
    parser.add_argument(
        "--expected-abstain-f1",
        type=float,
        default=None,
        help="If set, exit 1 when ABSTAIN F1 differs (demo uses 0.0)",
    )
    args = parser.parse_args(list(argv) if argv is not None else None)

    if args.demo:
        gold, pred = synthesize_bc_mlp_holdout()
        title = "BC MLP hold-out confusion (SYNTHETIC demo from STATUS recalls)"
        if args.expected_abstain_f1 is None:
            args.expected_abstain_f1 = 0.0
    elif args.jsonl:
        gold, pred = load_jsonl(args.jsonl)
        title = f"Action confusion matrix ({args.jsonl})"
    else:
        parser.error("provide --jsonl or --demo")
        return 2

    mat = confusion_matrix(gold, pred, ACTIONS)
    report = format_report(mat, ACTIONS, title=title)
    sys.stdout.write(report)

    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(report, encoding="utf-8")

    if args.expected_abstain_f1 is not None:
        got = per_class_prf(mat, ACTIONS)["ABSTAIN"]["f1"]
        if abs(got - args.expected_abstain_f1) > 1e-9:
            sys.stderr.write(
                f"ABSTAIN F1={got:.4f} != expected {args.expected_abstain_f1:.4f}\n"
            )
            return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
