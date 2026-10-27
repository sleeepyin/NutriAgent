#!/usr/bin/env python3
"""Generate train-only weak controller steps. CPU, no VLM.

Refuses every split other than ``rgb_train``. Does not read the 507-dish
test pool and will not write the frozen B0/B1 result files.

Examples:

    python scripts/weak_steps_smoke.py
    python scripts/gen_weak_steps_train.py --split rgb_train --pool fit --n 8 \\
        --out results/weak_steps/rgb_train_fit_n8.jsonl
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from vdr.ir.io import write_ir_jsonl  # noqa: E402
from vdr.weak.teacher import (  # noqa: E402
    TRAIN_SPLIT,
    TestSplitRefused,
    UnsafeOutputPath,
    assert_safe_output,
    default_output_path,
    generate_split,
    summary_path_for,
)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--split", default=TRAIN_SPLIT, help="only rgb_train is accepted")
    parser.add_argument("--pool", choices=("fit", "holdout", "all"), default="fit")
    parser.add_argument("--n", type=int, default=None, help="cap the number of train dishes")
    parser.add_argument("--seed", type=int, default=None, help="with --n, sample instead of a prefix")
    parser.add_argument("--dish-id", action="append", default=None, dest="dish_ids")
    parser.add_argument("--catalog", default=None, help="FNDDS foods.json; default is the repo catalog")
    parser.add_argument("--out", type=Path, default=None, help="JSONL path for IR steps")
    args = parser.parse_args(argv)

    out = args.out
    if out is None:
        out = ROOT / default_output_path(pool=args.pool, n=args.n, seed=args.seed)
    elif not out.is_absolute():
        out = ROOT / out
    summary_path = summary_path_for(out)

    try:
        assert_safe_output(out)
        assert_safe_output(summary_path)
        steps, summary = generate_split(
            split=args.split,
            pool=args.pool,
            n=args.n,
            seed=args.seed,
            dish_ids=args.dish_ids,
            catalog_path=args.catalog,
        )
    except (TestSplitRefused, UnsafeOutputPath, ValueError) as exc:
        print(f"refused: {exc}", file=sys.stderr)
        return 2

    n_written = write_ir_jsonl(out, steps)
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    actions = summary.get("actions") or {}
    terminal = summary.get("terminal_s") or {}
    print(
        f"split={args.split} pool={args.pool} dishes={summary['n_samples']} "
        f"steps={n_written} actions={actions} terminal={terminal}"
    )
    print(f"wrote {out}")
    print(f"wrote {summary_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
