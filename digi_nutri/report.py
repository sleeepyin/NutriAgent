"""Write the rerun record from the metric files."""

from __future__ import annotations

import json
from pathlib import Path

from digi_nutri.protocol import HOLDOUT_METRICS, MODELS, NUTRIENTS, REPORT_PATH, TRAIN_METRICS


def _fmt(value: float) -> str:
    return f"{float(value):.4f}"


def _mae_table(mae: dict) -> list[str]:
    lines = [
        "| Model | Energy | Fat | Carb | Protein | Mass |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for kind in MODELS:
        row = mae[kind]
        lines.append(
            "| "
            + " | ".join(
                [
                    kind,
                    _fmt(row["energy"]),
                    _fmt(row["fat"]),
                    _fmt(row["carb"]),
                    _fmt(row["protein"]),
                    _fmt(row["mass"]),
                ]
            )
            + " |"
        )
    return lines


def write_report(root: Path) -> None:
    train_path = root / TRAIN_METRICS
    holdout_path = root / HOLDOUT_METRICS
    if not train_path.exists():
        return
    train = json.loads(train_path.read_text())
    lines = [
        "# Same-setting dual-teacher rerun",
        "",
        "One training run of B1, T_geo, and T_ing under the frozen recipe.",
        "The comparator is the B1 trained in this run.",
        "Geometric augmentation, head dropout, the split learning rate, and the",
        "explicit height aggregates are the pre-registered regularization and",
        "volume readout. This record does not treat them as a new method.",
        "Distillation was not run.",
        "",
        f"Parameter counts: `{json.dumps(train['param_counts'])}`.",
        "",
        "## Train metrics",
        "",
        "Eval mode, no augmentation, kept official train plates.",
        "These numbers are not a Q0 gate.",
        "",
        *_mae_table(train["mae"]),
        "",
    ]
    if holdout_path.exists():
        holdout = json.loads(holdout_path.read_text())
        lines.extend(
            [
                "## Official holdout",
                "",
                f"Split `official_depth_test`, n={holdout['n']}, groups={holdout['n_groups']}.",
                "Eval mode, no augmentation. Plate nutrition is each model's own predicted mass",
                "times its own predicted density.",
                "",
                "### MAE",
                "",
                *_mae_table(holdout["mae"]),
                "",
                "### Dataset PMAE",
                "",
                _pmae_header(),
                *_pmae_rows(holdout, "dataset_pmae"),
                "",
                "### Per-plate PMAE",
                "",
                _pmae_header(),
                *_pmae_rows(holdout, "plate_pmae"),
                "",
                "### Headline intervals",
                "",
                "Each row is the 95% plate-grouped CI of (teacher - this run's B1).",
                "An interval is below zero only when its upper end is strictly below 0.",
                "Mass intervals are omitted here and are not part of the 24.",
                "",
                "| Teacher | Nutrient | Metric | Lo | Hi | Below 0 |",
                "|---|---|---|---:|---:|---|",
            ]
        )
        for row in holdout["intervals"]:
            flag = "yes" if row["below_zero"] else "no"
            lines.append(
                f"| {row['teacher']} | {row['nutrient']} | {row['metric']} | "
                f"{row['lo']:.6f} | {row['hi']:.6f} | {flag} |"
            )
        lines.extend(
            [
                "",
                f"Intervals below zero: {holdout['n_intervals_below_zero']} / {holdout['n_intervals']}.",
                "",
                f"Verdict: `{holdout['verdict']}`.",
                "",
                "The run stops on this verdict. Distillation stays locked.",
                "",
            ]
        )
    else:
        lines.extend(["## Official holdout", "", "Not run yet.", ""])
    path = root / REPORT_PATH
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines))


def _pmae_header() -> str:
    return "\n".join(
        [
            "| Model | Energy | Fat | Carb | Protein | Mass |",
            "|---|---:|---:|---:|---:|---:|",
        ]
    )


def _pmae_rows(holdout: dict, metric: str) -> list[str]:
    rows = []
    for kind in MODELS:
        metrics = holdout["metrics"][kind]
        values = [_fmt(metrics[name][metric]) for name in list(NUTRIENTS) + ["mass"]]
        rows.append("| " + " | ".join([kind, *values]) + " |")
    return rows
