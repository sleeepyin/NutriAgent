#!/usr/bin/env python3
"""Train closed-loop BC on weak steps; evaluate hold-out with mask ablations.

Fit ``PolicyStateMLP`` on ``rgb_train_fit_all.jsonl``. Metrics and confusion
on ``rgb_train_holdout_all.jsonl`` only. Writes under ``results/closed_loop/``.

    python scripts/closed_loop_bc_train.py
    python scripts/closed_loop_bc_train.py --limit 512   # smoke-sized fit
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np  # noqa: E402

from vdr.controller.closed_loop_data import (  # noqa: E402
    FIT_JSONL,
    HOLDOUT_JSONL,
    load_closed_loop_rows,
    observation_block_norm,
)
from vdr.controller.features import FEATURE_DIM  # noqa: E402
from vdr.controller.policy_state_mlp import PolicyStateMLP  # noqa: E402
from vdr.controller.state_features import (  # noqa: E402
    OBS_SLICE,
    STATE_FEATURE_DIM,
    mask_observation_block,
)
from vdr.eval.action_metrics import (  # noqa: E402
    ACTIONS,
    confusion_matrix,
    format_report,
    overall_accuracy,
    per_class_prf,
)
from vdr.controller.policy_mlp import (  # noqa: E402
    MLP_BATCH_SIZE,
    MLP_EPOCHS,
    MLP_HIDDEN_SIZES,
    MLP_LR,
    MLP_SEED,
)

OUT_DIR = ROOT / "results" / "closed_loop"


def _eval_split(
    policy: PolicyStateMLP,
    hold: np.ndarray,
    y_true: np.ndarray,
    *,
    mask_mode: str,
    shuffle_seed: int | None,
) -> dict:
    if mask_mode == "none":
        X_eval = hold
    elif mask_mode == "zero":
        X_eval = np.vstack(
            [mask_observation_block(row, mode="zero", obs_slice=OBS_SLICE) for row in hold]
        )
    elif mask_mode == "shuffle":
        X_eval = np.vstack(
            [
                mask_observation_block(
                    row,
                    mode="shuffle",
                    obs_slice=OBS_SLICE,
                    rng=np.random.default_rng((shuffle_seed or 0) + idx),
                )
                for idx, row in enumerate(hold)
            ]
        )
    else:
        raise ValueError(mask_mode)

    pred = policy.predict_actions(X_eval)
    truth = [str(item) for item in y_true.tolist()]
    mat = confusion_matrix(truth, pred, labels=ACTIONS)
    prf = per_class_prf(mat, labels=ACTIONS)
    return {
        "mask_obs": mask_mode,
        "overall_accuracy": overall_accuracy(mat),
        "per_class": prf,
        "confusion_matrix": mat,
        "report": format_report(
            mat,
            labels=ACTIONS,
            title=f"Closed-loop hold-out ({mask_mode})",
        ),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Closed-loop state-aware BC train/eval")
    parser.add_argument("--fit", type=Path, default=FIT_JSONL)
    parser.add_argument("--holdout", type=Path, default=HOLDOUT_JSONL)
    parser.add_argument("--out-dir", type=Path, default=OUT_DIR)
    parser.add_argument("--limit", type=int, default=None, help="Limit fit rows (smoke)")
    parser.add_argument("--shuffle-seed", type=int, default=42)
    args = parser.parse_args()

    started = time.perf_counter()
    fit = load_closed_loop_rows(args.fit, expect_pool="fit", limit=args.limit)
    hold = load_closed_loop_rows(args.holdout, expect_pool="holdout")

    obs_norm = observation_block_norm(fit.X)
    step0 = obs_norm[fit.step_indices == 0] if hasattr(fit, "step_indices") else obs_norm
    later = obs_norm[np.array(fit.step_indices) > 0]
    if later.size and not np.any(later > 0):
        print("WARN: no non-zero obs block on step>0 in fit sample", file=sys.stderr)

    policy = PolicyStateMLP()
    policy.fit(fit.X, fit.y)
    elapsed_fit = time.perf_counter() - started

    results: dict = {
        "policy": "PolicyStateMLP",
        "input_dim": STATE_FEATURE_DIM,
        "open_loop_b5_dim": FEATURE_DIM,
        "hidden": list(MLP_HIDDEN_SIZES),
        "epochs": MLP_EPOCHS,
        "batch_size": MLP_BATCH_SIZE,
        "lr": MLP_LR,
        "seed": MLP_SEED,
        "fit_path": str(args.fit),
        "holdout_path": str(args.holdout),
        "fit_rows": fit.n_rows,
        "fit_dishes": fit.n_dishes,
        "holdout_rows": hold.n_rows,
        "holdout_dishes": hold.n_dishes,
        "fit_limit": args.limit,
        "fit_seconds": round(elapsed_fit, 2),
        "obs_block_step0_mean_norm": float(step0.mean()) if step0.size else 0.0,
        "obs_block_step_gt0_mean_norm": float(later.mean()) if later.size else 0.0,
        "evaluations": {},
        "not_main_table_proof": (
            "Hold-out action metrics only. Not Full-set MAE / AURC main-table proof."
        ),
    }

    for mode in ("none", "zero", "shuffle"):
        payload = _eval_split(
            policy,
            hold.X,
            hold.y,
            mask_mode=mode,
            shuffle_seed=args.shuffle_seed,
        )
        slim = {
            "mask_obs": payload["mask_obs"],
            "overall_accuracy": payload["overall_accuracy"],
            "per_class": payload["per_class"],
            "abstain_f1": payload["per_class"]["ABSTAIN"]["f1"],
            "abstain_recall": payload["per_class"]["ABSTAIN"]["recall"],
            "abstain_support": payload["per_class"]["ABSTAIN"]["support"],
        }
        results["evaluations"][mode] = slim
        print(payload["report"])

    args.out_dir.mkdir(parents=True, exist_ok=True)
    metrics_path = args.out_dir / "metrics_holdout.json"
    metrics_path.write_text(json.dumps(results, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    report_path = args.out_dir / "report_holdout.txt"
    lines = [
        "Closed-loop BC hold-out summary",
        f"fit_rows={fit.n_rows} holdout_rows={hold.n_rows}",
        f"STATE_FEATURE_DIM={STATE_FEATURE_DIM} (B5 open-loop={FEATURE_DIM})",
        "",
    ]
    for mode in ("none", "zero", "shuffle"):
        ev = results["evaluations"][mode]
        lines.append(
            f"mask={mode}: acc={ev['overall_accuracy']:.4f} "
            f"ABSTAIN F1={ev['abstain_f1']:.4f} "
            f"(R={ev['abstain_recall']:.4f}, n={int(ev['abstain_support'])})"
        )
    lines.append("")
    lines.append(results["not_main_table_proof"])
    report_path.write_text("\n".join(lines) + "\n", encoding="utf-8")

    weights_path = args.out_dir / "policy_state_mlp.npz"
    blob = policy.export_numpy()
    np.savez(
        weights_path,
        mean=blob["mean"],
        std=blob["std"],
        W1=blob["W1"],
        b1=blob["b1"],
        W2=blob["W2"],
        b2=blob["b2"],
        W3=blob["W3"],
        b3=blob["b3"],
    )
    meta_path = args.out_dir / "train_meta.json"
    meta_path.write_text(
        json.dumps(
            {
                "weights": str(weights_path.relative_to(ROOT)),
                "metrics": str(metrics_path.relative_to(ROOT)),
                "class_weight": blob["class_weight"],
                "actions": list(blob["actions"]),
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )

    print(
        f"wrote {metrics_path.relative_to(ROOT)} "
        f"main ABSTAIN F1={results['evaluations']['none']['abstain_f1']:.4f} "
        f"mask-zero ABSTAIN F1={results['evaluations']['zero']['abstain_f1']:.4f}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
