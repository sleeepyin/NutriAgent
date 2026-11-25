"""Write the Plan A report from the audit and any metrics files."""

from __future__ import annotations

import json
from pathlib import Path

from digi_nutri.protocol import (
    AUDIT_PATH,
    FROZEN_B1_TRAIN_MAE,
    FROZEN_B1_TRAIN_MASS_MAE,
    HOLDOUT_METRICS,
    NUTRIENTS,
    REPORT_PATH,
    TRAIN_METRICS,
)


def _fmt(value: float) -> str:
    return f"{float(value):.6f}"


def write_report(root: Path) -> None:
    audit = json.loads((root / AUDIT_PATH).read_text())
    n_train = int(audit["n_train"])
    n_zero = int(audit["n_d_table_eq_0"])
    proportion = n_zero / n_train
    border = audit["border_zero_fraction"]
    lines = [
        "# Plan A geometry teacher",
        "",
        "This run retrains only the geometry teacher. Separate encoders and a zero-initialized",
        "residual are a BatchNorm repair. They are not a new method, and predicted mass times",
        "predicted density is the same product as the frozen run.",
        "",
        "## D_table audit",
        "",
        "The count was computed with the border median that includes missing-depth zeros,",
        "before any height-code change. Height is `clip(D_table - D_raw, 0, 150 mm) / 150`,",
        "then one 3×3 closing.",
        "",
        f"Kept train plates: {n_train} of {audit['n_train_file']}.",
        f"`D_table == 0`: {n_zero} / {n_train} = {proportion:.12f}.",
        f"All-zero height maps: {int(audit['n_height_all_zero'])} / {n_train}.",
        "Those two sets are the same plates.",
        "",
        (
            f"Every kept plate has at least one zero on the 1-pixel border "
            f"({border['n_gt_0']} / {n_train}). The median fraction of border zeros is "
            f"{float(border['p50']):.6f}. The median table depth on the full kept set is "
            f"{float(audit['d_table_mm']['p50']):.4f} mm."
        ),
        "",
        audit["decision_note"],
        "",
        f"Training uses `apply_d_table_zero_exclusion` = {str(bool(audit['apply_d_table_zero_exclusion'])).lower()}.",
        "",
        "## Train gate",
        "",
    ]
    train_path = root / TRAIN_METRICS
    if not train_path.exists():
        lines.append("Train evaluation has not been recorded.")
    else:
        payload = json.loads(train_path.read_text())
        lines.append(
            "Eval mode, no augmentation, on the same kept training plates. "
            "A tie with frozen B1 fails. Mass is recorded and is not a pass condition."
        )
        lines.append("")
        lines.append("| | energy | fat | carb | protein | mass |")
        lines.append("|---|---:|---:|---:|---:|---:|")
        frozen_mass = FROZEN_B1_TRAIN_MASS_MAE
        lines.append(
            "| frozen B1 | "
            + " | ".join(_fmt(FROZEN_B1_TRAIN_MAE[name]) for name in NUTRIENTS)
            + f" | {_fmt(frozen_mass)} |"
        )
        metrics = payload["metrics"]
        lines.append(
            "| T_geo Plan A | "
            + " | ".join(_fmt(metrics[name]["mae"]) for name in NUTRIENTS)
            + f" | {_fmt(metrics['mass']['mae'])} |"
        )
        lines.append("")
        lines.append(f"Train gate pass: {payload['train_gate_pass']}.")
        lines.append(f"Verdict: `{payload['verdict']}`.")
        if payload.get("train_gate_failures"):
            lines.append("")
            lines.append("Failures:")
            for row in payload["train_gate_failures"]:
                lines.append(
                    f"- {row['nutrient']}: {_fmt(row['t_geo_mae'])} vs frozen B1 {_fmt(row['b1_mae'])}"
                )
        lines.append("")
        lines.append(
            f"Batch size {payload['batch_size']}. Measured parameters {payload['param_count']}."
        )
    lines.extend(["", "## Holdout", ""])
    holdout_path = root / HOLDOUT_METRICS
    if not holdout_path.exists():
        lines.append("The official 507-plate holdout was not opened.")
    else:
        holdout = json.loads(holdout_path.read_text())
        lines.append(
            f"Official holdout n={holdout['n']}, plate groups={holdout['n_groups']}. "
            "Read once. Distillation was not started."
        )
        lines.append("")
        lines.append(f"Verdict: `{holdout['verdict']}`.")
        if holdout.get("failures"):
            lines.append("")
            lines.append(f"Failing contrasts: {len(holdout['failures'])}.")
    lines.append("")
    (root / REPORT_PATH).write_text("\n".join(lines))
