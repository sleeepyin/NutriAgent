#!/usr/bin/env python3
"""Selective prediction CLI: Coverage, Selective MAE, AURC.

Formulae are defined in ``vdr.eval.selective``. This script does not train.
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

from vdr.eval.selective import (  # noqa: E402
    aurc,
    coverage_and_selective_mae,
    risk_coverage_curve,
)


def _mean(values: Sequence[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def demo_arrays() -> tuple[list[float], list[float], list[int]]:
    """Synthetic: low abstain_score lines up with low error."""
    errors = [1.0, 2.0, 3.0, 4.0, 10.0, 12.0, 20.0, 30.0]
    scores = [0.1, 0.2, 0.25, 0.3, 0.6, 0.7, 0.85, 0.95]
    flags = [0, 0, 0, 0, 1, 1, 1, 1]
    return errors, scores, flags


def format_selective_report(
    *,
    flag_metrics: dict[str, float],
    thr_metrics: dict[str, float],
    aurc_metrics: dict[str, float],
    threshold: float,
) -> str:
    lines = [
        "Selective metrics",
        "=================",
        "",
        "By abstain flag (keep where flag=0):",
        f"  coverage      : {flag_metrics['coverage']:.4f}",
        f"  selective MAE : {flag_metrics['selective_mae']:.4f}",
        f"  n_kept/n      : {int(flag_metrics['n_kept'])}/{int(flag_metrics['n'])}",
        "",
        f"By abstain_score < {threshold}:",
        f"  coverage      : {thr_metrics['coverage']:.4f}",
        f"  selective MAE : {thr_metrics['selective_mae']:.4f}",
        "",
        "AURC (trapezoid risk-coverage; lower better):",
        f"  aurc                : {aurc_metrics['aurc']:.6f}",
        f"  risk@full_coverage  : {aurc_metrics['risk_at_full_coverage']:.4f}",
        f"  risk@min_coverage   : {aurc_metrics['risk_at_min_coverage']:.4f}",
        "",
        "Formula: sort by confidence = ascending abstain_score;",
        "  c_k = k/N, r_k = mean error of the top-k confident;",
        "  AURC = trapz(r, c) including 0.5 * r_1 * c_1 from the origin.",
        "  See vdr/eval/selective.py.",
    ]
    return "\n".join(lines) + "\n"


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--demo", action="store_true")
    parser.add_argument("--jsonl", type=Path, help="rows with error + abstain_score/flag")
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--out", type=Path)
    args = parser.parse_args(list(argv) if argv is not None else None)

    if args.demo:
        errors, scores, flags = demo_arrays()
    elif args.jsonl:
        errors = []
        scores = []
        flags = []
        has_flags = False
        with args.jsonl.open(encoding="utf-8") as handle:
            for line in handle:
                row: dict[str, Any] = json.loads(line)
                errors.append(float(row["error"]))
                scores.append(float(row.get("abstain_score", row.get("score", 0.0))))
                if "abstain" in row or "abstain_flag" in row:
                    has_flags = True
                    flags.append(int(bool(row.get("abstain", row.get("abstain_flag")))))
        if not has_flags:
            flags = [1 if score >= args.threshold else 0 for score in scores]
    else:
        parser.error("provide --jsonl or --demo")
        return 2

    flag_metrics = coverage_and_selective_mae(errors, abstain_flags=flags)
    thr_metrics = coverage_and_selective_mae(
        errors, abstain_scores=scores, threshold=args.threshold
    )
    aurc_metrics = aurc(errors, scores)
    text = format_selective_report(
        flag_metrics=flag_metrics,
        thr_metrics=thr_metrics,
        aurc_metrics=aurc_metrics,
        threshold=args.threshold,
    )
    sys.stdout.write(text)
    payload = {
        "by_flag": flag_metrics,
        "by_threshold": thr_metrics,
        "threshold": args.threshold,
        "aurc": aurc_metrics,
        "curve_head": risk_coverage_curve(errors, scores)[:5],
    }
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(text, encoding="utf-8")
        args.out.with_suffix(".json").write_text(json.dumps(payload, indent=2), encoding="utf-8")

    if args.demo:
        full_mae = _mean(errors)
        if flag_metrics["selective_mae"] >= full_mae:
            sys.stderr.write("demo: expected selective MAE < full MAE\n")
            return 1
        if aurc_metrics["aurc"] <= 0:
            sys.stderr.write("demo: AURC should be positive for nonzero errors\n")
            return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
