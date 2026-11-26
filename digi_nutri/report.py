"""Write the T_multi record from the metric files."""

from __future__ import annotations

import json
from pathlib import Path

from digi_nutri.protocol import HOLDOUT_METRICS, MODEL, NUTRIENTS, REPORT_PATH, TRAIN_METRICS


def _fmt(value: float) -> str:
    return f"{float(value):.4f}"


def _mae_row(label: str, row: dict) -> str:
    cells = [label] + [_fmt(row[name]) for name in list(NUTRIENTS) + ["mass"]]
    return "| " + " | ".join(cells) + " |"


def _mae_table(rows: list[tuple[str, dict]]) -> list[str]:
    lines = [
        "| Model | Energy | Fat | Carb | Protein | Mass |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    lines.extend(_mae_row(label, row) for label, row in rows)
    return lines


def write_report(root: Path) -> None:
    train_path = root / TRAIN_METRICS
    holdout_path = root / HOLDOUT_METRICS
    if not train_path.exists():
        return
    train = json.loads(train_path.read_text())
    lines = [
        "# Unified privileged teacher T_multi",
        "",
        "One training run of T_multi under the frozen #91 recipe.",
        "The comparator is the frozen #91 same-setting B1. B1 was not retrained.",
        "Geometric augmentation, head dropout, the split learning rate, and the",
        "explicit height readout are the pre-registered regularization and volume",
        "features. This record does not treat T_multi as a new method.",
        "Distillation was not run.",
        "",
        f"T_multi parameters: {train['param_counts'][MODEL]}.",
        "",
        "## Train metrics",
        "",
        "Eval mode, no augmentation, kept official train plates.",
        "These numbers are not a Q0 gate. A train advantage is not a pass.",
        "",
        *_mae_table(
            [
                ("frozen #91 B1", train["frozen_b1_mae"]),
                (MODEL, train["mae"][MODEL]),
            ]
        ),
        "",
    ]
    if holdout_path.exists():
        holdout = json.loads(holdout_path.read_text())
        lines.extend(
            [
                "## Official holdout",
                "",
                f"Split `official_depth_test`, n={holdout['n']}, groups={holdout['n_groups']}.",
                "Eval mode, no augmentation. Plate nutrition is predicted mass times predicted density.",
                "The B1 column is the frozen #91 holdout prediction, not a new run.",
                "",
                "### MAE",
                "",
                *_mae_table(
                    [
                        ("frozen #91 B1", holdout["frozen_b1_mae"]),
                        (MODEL, holdout["mae"][MODEL]),
                    ]
                ),
                "",
                "### Dataset PMAE",
                "",
                *_pmae_table(holdout, "dataset_pmae"),
                "",
                "### Per-plate PMAE",
                "",
                *_pmae_table(holdout, "plate_pmae"),
                "",
                "### Headline intervals",
                "",
                "Each row is the 95% plate-grouped CI of (T_multi - frozen #91 B1).",
                "An interval is below zero only when its upper end is strictly below 0.",
                "Mass is recorded after this table and is not one of the 12.",
                "",
                "| Nutrient | Metric | Lo | Hi | Below 0 |",
                "|---|---|---:|---:|---|",
            ]
        )
        for row in holdout["intervals"]:
            flag = "yes" if row["below_zero"] else "no"
            lines.append(
                f"| {row['nutrient']} | {row['metric']} | {row['lo']:.6f} | {row['hi']:.6f} | {flag} |"
            )
        lines.extend(
            [
                "",
                f"Intervals below zero: {holdout['n_intervals_below_zero']} / {holdout['n_intervals']}.",
                "",
                "### Mass intervals (not a pass condition)",
                "",
                "| Metric | Lo | Hi | Below 0 |",
                "|---|---:|---:|---|",
            ]
        )
        for row in holdout["mass_intervals"]:
            flag = "yes" if row["below_zero"] else "no"
            lines.append(f"| {row['metric']} | {row['lo']:.6f} | {row['hi']:.6f} | {flag} |")
        lines.extend(
            [
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


def _pmae_table(holdout: dict, metric: str) -> list[str]:
    names = list(NUTRIENTS) + ["mass"]
    frozen = {name: holdout["frozen_b1_metrics"][name][metric] for name in names}
    model = {name: holdout["metrics"][MODEL][name][metric] for name in names}
    return _mae_table([("frozen #91 B1", frozen), (MODEL, model)])
