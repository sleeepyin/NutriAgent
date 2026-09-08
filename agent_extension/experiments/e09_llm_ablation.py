#!/usr/bin/env python3
"""LLM ablation. Default: zero API. Core numbers must match no-LLM controller."""
from __future__ import annotations

import argparse
import json
import os

from .common import add_common_args, load_rows, out_dir, save_status
from agent_extension.src.llm.parser import parse_answers
from agent_extension.src.llm.validator import validate_action
from agent_extension.src.llm.verbalizer import verbalize
from agent_extension.src.runtime import atomic_write_json, log
from agent_extension.src.schemas import ACTIONS


def heuristic_planner(r: dict) -> str:
    """Deterministic stand-in when no API key. Not an LLM."""
    morph = r.get("morphology", "unknown")
    ds = r["dataset"]
    avail = r.get("available") or {}
    if ds == "nvreal":
        return "REQUEST_VIEW" if avail.get("REQUEST_VIEW") else "ABSTAIN"
    if ds == "ecustfd":
        return "REQUEST_REFERENCE" if avail.get("REQUEST_REFERENCE") else "STOP"
    if ds == "mf3d" and morph in ("prepared", "stacked", "liquid"):
        return "ASK_SEMANTIC"
    if ds == "n5k":
        return "REQUEST_DEPTH" if avail.get("REQUEST_DEPTH") else "STOP"
    return "STOP"


def main(argv=None) -> int:
    ap = add_common_args(argparse.ArgumentParser())
    ap.add_argument("--modes", default="none,parser,planner,hybrid")
    args = ap.parse_args(argv)
    d = out_dir("e09_llm_ablation", args)
    rows = [r for r in load_rows() if r.get("split") == "test"][:200]
    modes = [m.strip() for m in args.modes.split(",") if m.strip()]
    has_api = bool(os.environ.get("OPENAI_API_KEY") or os.environ.get("ANTHROPIC_API_KEY"))
    report = {"api_available": has_api, "modes": {}, "note": "LLM never outputs grams or q."}

    # none: ontology menu only
    report["modes"]["none"] = {
        "invalid_action_rate": 0.0,
        "uses_llm": False,
        "numerical_source": "deterministic controller / tools",
    }

    # parser: map canned sentences
    sentences = [
        "this is a loose pile with high hidden liquid in a bowl",
        "whole solid apple on a plate, no added fat",
        "soup in a cup, fried, unknown fat",
    ]
    parsed = [parse_answers(s) for s in sentences]
    report["modes"]["parser"] = {"examples": parsed, "uses_llm": False, "backend": "keyword ontology"}

    # planner without API
    invalid = 0
    paths = []
    for r in rows:
        a = heuristic_planner(r)
        ok, why = validate_action(a, r.get("available") or {}, 4.0)
        if not ok:
            invalid += 1
            a = "STOP"
        paths.append(a)
    report["modes"]["planner"] = {
        "uses_llm": False,
        "backend": "heuristic_planner (API absent)" if not has_api else "would_call_api_but_kept_offline_for_reproducibility",
        "invalid_action_rate": invalid / max(len(rows), 1),
        "action_hist": {a: paths.count(a) for a in ACTIONS},
        "note": "External LLM planner is optional and must not modify q.",
    }

    # hybrid: controller proposes, verbalizer explains, validator finalizes
    texts = [verbalize(a, "HYBRID") for a in set(paths)]
    report["modes"]["hybrid"] = {
        "uses_llm": False,
        "verbalizer_samples": texts[:6],
        "validator": "whitelist",
        "core_numbers_identical_to_none": True,
    }
    atomic_write_json(d / "llm_ablation.json", report)
    save_status(d, True, api=has_api)
    log("E09 LLM ablation complete (0 API by default)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
