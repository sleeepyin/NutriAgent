#!/usr/bin/env python3
"""Phase 8 skeleton. Fill only numbers that exist. Never invent results."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from .common import add_common_args, load_json, out_dir, save_status
from agent_extension.src.paths import Paths


def main(argv=None) -> int:
    ap = add_common_args(argparse.ArgumentParser())
    args = ap.parse_args(argv)
    paths = Paths.resolve()
    d = out_dir("e12_paper_draft", args)
    e03 = load_json(paths.results / "e03_tool_value" / "status.json", {})
    e07 = load_json(paths.results / "e07_main_benchmark" / "benchmark.json", {})
    e08 = load_json(paths.results / "e08_cross_domain" / "cross_domain.json", {})
    gates = (e03.get("gates") or {})
    primary = (e07.get("primary") or {})
    zero = (e08.get("zero_target") or {})

    md = []
    md.append("# CAIA-MM draft integration notes (auto-filled, Level A)\n")
    md.append("Narrative structure: **Estimate → Diagnose → Act**.\n")
    md.append("Recommended title: Mechanism-Aware Active Food Measurement with Post-Policy Conformal Risk Control.\n")
    md.append("## What this run is\n")
    md.append("- Public-data feasibility / mechanism verification (Level A).")
    md.append("- Not a unified Agent vs fixed-protocol causal claim across datasets.")
    md.append("- No first-claim language.\n")
    md.append("## Gates\n")
    md.append("```json\n" + json.dumps(gates, indent=2) + "\n```\n")
    md.append("## Primary endpoint (test, shared terminal q)\n")
    if primary:
        md.append("```json\n" + json.dumps(primary, indent=2) + "\n```\n")
    else:
        md.append("Not available in this run.\n")
    md.append("## Cross-domain\n")
    if zero:
        md.append(f"- Zero-target empirical coverage: {zero.get('interval', {}).get('coverage')}")
        md.append(f"- 90% claim allowed: {zero.get('claim_90')}")
        md.append(f"- Reason: {zero.get('reason')}\n")
    md.append("## Abstract template (do not fill fake numbers)\n")
    md.append(
        "Existing food analysis systems passively estimate weight and nutrients from a fixed input, "
        "even when the uncertainty is caused by missing physical or semantic information. "
        "We study a cost-aware conformal agent that decides whether to stop, request a reference object, "
        "acquire an additional view or depth, ask a structured semantic question, or abstain. "
        "The policy is trained independently of calibration. "
        f"On this Level A run the locked primary comparison is: {primary or '[pending]'}.\n"
    )
    (d / "paper_integration.md").write_text("\n".join(md), encoding="utf-8")
    save_status(d, True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
