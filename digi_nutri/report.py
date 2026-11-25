"""Write the Q0 report from frozen artifacts."""

from __future__ import annotations

import json
from pathlib import Path

from digi_nutri.protocol import NUTRIENTS


def _fmt(value: float) -> str:
    return f"{value:.4f}"


def _ci(row: dict) -> str:
    return f"{_fmt(row['point_diff'])} [{_fmt(row['lo'])}, {_fmt(row['hi'])}]"


def render(root: Path = Path(".")) -> str:
    config = json.loads((root / "configs" / "q0_frozen.json").read_text())
    hygiene = json.loads((root / "results" / "v2_q0" / "split_hygiene.json").read_text())
    lines = [
        "# DiGI-Nutri v2 Q0",
        "",
        f"Status commit `{config['status_commit']}`. Backbone `{config['backbone']}`. "
        f"Image {config['image_size']}, epochs {config['epochs']}, AdamW lr {config['lr']}, "
        f"weight decay {config['weight_decay']}, cosine eta_min {config['eta_min']}, "
        f"batch {config['batch_size']}.",
        "",
        f"Parameter counts: {json.dumps(config['param_counts'])}.",
        "",
        "Train exclusions:",
    ]
    for row in hygiene["train_exclusions"]:
        lines.append(f"- `{row['dish_id']}`: {', '.join(row['reasons'])}")
    lines.append("")
    lines.append(
        f"Kept train {config['n_train']} of {config['n_train_file']}. "
        f"Official test {config['n_test']} dishes in {config['n_test_groups']} plate groups. "
        f"Reshoot rule: exact nutrition within {config['reshoot_nutrition_atol']} and "
        f"timestamp gap at most {config['reshoot_max_gap_seconds']} seconds, inside the evaluated split."
    )
    crosses = config.get("cross_split_reshoots") or []
    lines.append(f"Cross-split pairs under that rule: {len(crosses)}. The official split was not rewritten.")
    lines.append("")
    lines.append(
        "USDA name collisions with different per-gram vectors are not resolved by picking a row. "
        f"Ambiguous names: {', '.join(sorted(config['usda_ambiguous_names'])) or 'none'}. "
        "A dish uses the unique ingredient id."
    )
    lines.append("")
    train_path = root / "results" / "v2_q0" / "train_metrics.json"
    holdout_path = root / "results" / "v2_q0" / "holdout_metrics.json"
    if train_path.exists():
        train = json.loads(train_path.read_text())
        lines.append("## Train MAE")
        lines.append("")
        lines.append("| model | energy | fat | carb | protein | mass |")
        lines.append("|---|---:|---:|---:|---:|---:|")
        for model in ("b1", "t_geo", "t_ing"):
            metrics = train["metrics"][model]
            cells = [_fmt(metrics[name]["mae"]) for name in NUTRIENTS]
            cells.append(_fmt(metrics["mass"]["mae"]))
            lines.append("| " + model + " | " + " | ".join(cells) + " |")
        lines.append("")
        lines.append(f"Train gate pass: {train['train_gate_pass']}.")
        if train.get("train_gate_failures"):
            lines.append("")
            lines.append("Train-gate failures:")
            for row in train["train_gate_failures"]:
                lines.append(
                    f"- {row['teacher']} {row['nutrient']}: {row['teacher_mae']:.4f} vs B1 {row['b1_mae']:.4f}"
                )
    if not holdout_path.exists():
        lines.extend(
            [
                "",
                "Official holdout was not evaluated.",
                "",
                "Distillation was not started.",
            ]
        )
        return "\n".join(lines) + "\n"
    holdout = json.loads(holdout_path.read_text())
    lines.extend(["", "## Holdout", ""])
    lines.append("| model | nutrient | MAE | dataset PMAE | plate PMAE |")
    lines.append("|---|---|---:|---:|---:|")
    for model in ("b1", "t_geo", "t_ing"):
        for nutrient in list(NUTRIENTS) + ["mass"]:
            row = holdout["metrics"][model][nutrient]
            lines.append(
                f"| {model} | {nutrient} | {_fmt(row['mae'])} | {_fmt(row['dataset_pmae'])} | {_fmt(row['plate_pmae'])} |"
            )
    lines.extend(["", "## Plate-grouped 95% CI of (teacher - B1)", ""])
    for teacher in ("t_geo", "t_ing"):
        lines.append(f"### {teacher}")
        lines.append("")
        for nutrient in NUTRIENTS:
            bits = []
            for metric in ("mae", "dataset_pmae", "plate_pmae"):
                bits.append(f"{metric} {_ci(holdout['contrasts'][teacher][nutrient][metric])}")
            lines.append(f"- {nutrient}: " + "; ".join(bits))
        lines.append("")
    lines.append(f"Verdict: `{holdout['verdict']}`.")
    lines.append("")
    lines.append("Distillation was not started.")
    return "\n".join(lines) + "\n"


def main() -> None:
    text = render(Path("."))
    path = Path("results/v2_q0/REPORT.md")
    path.write_text(text)
    print(path, flush=True)


if __name__ == "__main__":
    main()
