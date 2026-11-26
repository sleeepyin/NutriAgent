"""Write the v3 step 1 record from the metric files. STATUS is not edited."""

from __future__ import annotations

import json
from pathlib import Path

from digi_nutri.protocol import B1_MODEL, HOLDOUT_METRICS, MODEL, NUTRIENTS, REPORT_PATH, TRAIN_METRICS


def _fmt(value: float) -> str:
    return f"{float(value):.4f}"


def _row(label: str, values: dict, names: list[str]) -> str:
    cells = [label] + [_fmt(values[name]) for name in names]
    return "| " + " | ".join(cells) + " |"


def _table(rows: list[tuple[str, dict]], names: list[str]) -> list[str]:
    header = "| Model | " + " | ".join(name.capitalize() for name in names) + " |"
    align = "|---|" + "|".join(["---:"] * len(names)) + "|"
    lines = [header, align]
    lines.extend(_row(label, values, names) for label, values in rows)
    return lines


def write_report(root: Path) -> None:
    train_path = root / TRAIN_METRICS
    holdout_path = root / HOLDOUT_METRICS
    if not train_path.exists():
        return
    train = json.loads(train_path.read_text())
    names = list(NUTRIENTS) + ["mass"]
    lines = [
        "# DiGI-Nutri v3 step 1",
        "",
        "One new ConvNeXt-Small RGB baseline, then one residual-Atwater T_multi.",
        "The comparator is that new B1. Mass times density stays the plate estimate.",
        "Energy density is 9 fat + 4 carb + 4 protein plus a learned residual.",
        "The residual penalty is the frozen 0.1 SmoothL1 term. Labels are not forced onto Atwater.",
        "Train metrics are not a gate. Student training was not started.",
        "docs/STATUS.md and README.md were not edited.",
        "",
        f"B1 parameters: {train['param_counts']['b1']}.",
        f"T_multi parameters: {train['param_counts'][MODEL]}.",
        "",
        "## Train metrics",
        "",
        "Eval mode, no augmentation, kept official train plates.",
        "A train advantage is not a Q0 pass.",
        "",
        "### MAE",
        "",
        *_table(
            [
                ("B1", train["mae"][B1_MODEL]),
                ("T_multi", train["mae"][MODEL]),
            ],
            names,
        ),
        "",
    ]
    if holdout_path.exists():
        holdout = json.loads(holdout_path.read_text())
        absolute = holdout["absolute_gate"]
        lines.extend(
            [
                "## Official holdout",
                "",
                f"Split `official_depth_test`, n={holdout['n']}, groups={holdout['n_groups']}.",
                "Eval mode, no augmentation. This file is the single holdout read.",
                "Plate nutrition is predicted mass times predicted density.",
                "The B1 column is the ConvNeXt-Small baseline trained in this round.",
                "",
                "### MAE",
                "",
                *_table(
                    [
                        ("B1", holdout["mae"][B1_MODEL]),
                        ("T_multi", holdout["mae"][MODEL]),
                    ],
                    names,
                ),
                "",
                "### Dataset PMAE",
                "",
                *_table(
                    [
                        ("B1", holdout["dataset_pmae"][B1_MODEL]),
                        ("T_multi", holdout["dataset_pmae"][MODEL]),
                    ],
                    names,
                ),
                "",
                "### Per-plate PMAE",
                "",
                *_pmae_table(holdout, "plate_pmae", names),
                "",
                "### Relative gate",
                "",
                "Each row is the 95% plate-grouped CI of (T_multi - this round's B1).",
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
                f"Relative gate: {'pass' if holdout['relative_pass'] else 'fail'}.",
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
                "### Absolute gate",
                "",
                "Dataset PMAE only. Per-plate PMAE is not an absolute gate.",
                "",
                f"T_multi energy dataset PMAE: {absolute['energy_dataset_pmae']:.4f} "
                f"(limit {absolute['energy_limit']}).",
                f"Mean of energy, mass, fat, carb, and protein: {absolute['mean5_dataset_pmae']:.4f} "
                f"(limit {absolute['mean5_limit']}).",
                f"Absolute gate: {'pass' if absolute['pass'] else 'fail'}.",
                "",
                f"Verdict: `{holdout['verdict']}`.",
                "",
                "The run stops on this verdict. Student training stays locked.",
                "",
            ]
        )
    else:
        lines.extend(["## Official holdout", "", "Not run yet.", ""])
    path = root / REPORT_PATH
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n")


def _pmae_table(holdout: dict, metric: str, names: list[str]) -> list[str]:
    baseline = {name: holdout["metrics"][B1_MODEL][name][metric] for name in names}
    teacher = {name: holdout["metrics"][MODEL][name][metric] for name in names}
    return _table([("B1", baseline), ("T_multi", teacher)], names)
