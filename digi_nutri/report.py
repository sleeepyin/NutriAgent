"""Markdown report for the single Phase 1 holdout read."""

from __future__ import annotations

from digi_nutri.protocol import HEADLINE, NUTRIENTS, UNITS

MODEL_ORDER = (
    "geometry_teacher",
    "ingredient_teacher",
    "b1",
    "b2a",
    "b2b",
    "b3",
)
MODEL_TITLE = {
    "geometry_teacher": "geometry teacher (privileged)",
    "ingredient_teacher": "ingredient teacher (privileged, names only)",
    "b1": "B1 RGB",
    "b2a": "B2a RGB",
    "b2b": "B2b RGB",
    "b3": "B3 RGB",
}
PAIRS = (
    "geometry_teacher_minus_b1",
    "ingredient_teacher_minus_b1",
    "b3_minus_b1",
    "b3_minus_b2a",
    "b3_minus_b2b",
)


def render_not_trained(reason: str) -> str:
    return "\n".join(
        [
            "# DiGI-Nutri Phase 1",
            "",
            "Training has not started. No official holdout evaluation was run, and no Q0/Q1/Q2 verdict is claimed.",
            "",
            f"Reason: {reason}",
            "",
            "STATUS.md was not edited. Phase 2 was not started. No pull request is opened against main.",
            "",
        ]
    )


def _num(value: float) -> str:
    return f"{value:.4f}"


def _ci(row: dict, kind: str) -> str:
    if kind == "mae":
        return (
            f"{_num(row['mae_mean'])} "
            f"[{_num(row['mae_ci_low'])}, {_num(row['mae_ci_high'])}]"
        )
    return (
        f"{_num(row['pmae_mean'])} "
        f"[{_num(row['pmae_ci_low'])}, {_num(row['pmae_ci_high'])}]"
    )


def render_report(metrics: dict) -> str:
    decision = metrics["decision"]
    lines = [
        "# DiGI-Nutri Phase 1",
        "",
        f"Verdict: `{decision['verdict']}`",
        "",
        decision["message"],
        "",
        "This file is the single official holdout read on the Nutrition5k depth test split. "
        "Student numbers use one overhead RGB image. Teacher numbers use that teacher's privileged inputs and are only a Q0 headroom check. "
        "Mass is not a pass condition. "
        "STATUS.md was not edited. Phase 2 was not started. The pull request base is digi-nutri, not main.",
        "",
        "## Rules applied",
        "",
        metrics["rules"]["q0"],
        "",
        metrics["rules"]["stable"],
        "",
        f"Q0 pass: {decision['q0_pass']}. Q1 opened: {decision['q1_opened']}. "
        f"Q1 pass: {decision['q1_pass']}. Q2 opened: {decision['q2_opened']}. "
        f"Q2 pass: {decision['q2_pass']}.",
        "",
    ]
    if decision["verdict"] == "q0_fail_stop_wait_triste":
        lines.append(
            "Q1 and Q2 were not opened. Any descriptive contrast below is not a license to keep distilling or to change the loss."
        )
        lines.append("")
    lines.extend(
        [
            "## Holdout",
            "",
            f"Dishes: {metrics['n_test']}. Plate groups: {metrics['n_plate_groups']}. "
            f"PMAE is `{metrics['pmae_definition']}`.",
            "",
            "### MAE",
            "",
            _metric_table(metrics, "mae"),
            "",
            "### PMAE percent",
            "",
            _metric_table(metrics, "pmae_percent"),
            "",
            "## Plate-grouped 95% intervals of (method − baseline)",
            "",
        ]
    )
    for key in PAIRS:
        lines.append(f"### {key}")
        lines.append("")
        lines.append("| nutrient | role | MAE difference | PMAE difference | stable |")
        lines.append("|---|---|---|---|---|")
        block = metrics["comparisons"][key]
        for nutrient in NUTRIENTS:
            role = "headline" if nutrient in HEADLINE else "recorded"
            row = block[nutrient]
            lines.append(
                f"| {nutrient} ({UNITS[nutrient]}) | {role} | {_ci(row, 'mae')} | {_ci(row, 'pmae')} | {row['stable']} |"
            )
        lines.append("")
    config = metrics.get("config") or {}
    if config:
        lines.extend(
            [
                "## Frozen training setup",
                "",
                f"Backbone `{config.get('backbone')}`, fusion `{config.get('fusion')}`, "
                f"image {config.get('image_size')}, batch {config.get('batch_size')}, "
                f"teacher epochs {config.get('epochs_teachers')}, student epochs {config.get('epochs_students')}, "
                f"AdamW lr {config.get('lr')}, weight decay {config.get('weight_decay')}, cosine eta_min {config.get('eta_min')}.",
                "",
                f"Warmup epochs {config.get('warmup_epochs')}, distillation ramp epochs {config.get('ramp_epochs')}. "
                f"Loss weights geo {config.get('lambda_geo')}, semantic {config.get('lambda_sem')}, aux {config.get('lambda_aux')}. "
                f"Depth fill `{config.get('depth_fill')}`. Text model `{config.get('text_model')}`. "
                f"Gram weighting: {config.get('gram_weight')}.",
                "",
                config.get("batch_size_note") or "",
                "",
            ]
        )
    last = metrics.get("train_last_epoch") or {}
    if last:
        lines.append("## Last training epoch (train fit, not the holdout)")
        lines.append("")
        for name, row in last.items():
            lines.append(f"- {name}: task/loss {row.get('loss')} MAE {row.get('mae')}")
        lines.append("")
    git = metrics.get("git") or {}
    if git:
        lines.extend(
            [
                "## Git",
                "",
                f"HEAD `{git.get('head')}`. Branch `{git.get('branch')}`. "
                f"main `{git.get('main') or git.get('origin_main')}`. "
                f"digi-nutri `{git.get('digi_nutri') or git.get('origin_digi_nutri')}`.",
                "",
            ]
        )
    lines.append("The metrics JSON next to this file is the holdout record. A second evaluation must refuse to overwrite it.")
    lines.append("")
    return "\n".join(lines)


def _metric_table(metrics: dict, field: str) -> str:
    header = "| nutrient | " + " | ".join(MODEL_TITLE[name] for name in MODEL_ORDER) + " |"
    rule = "|---|" + "|".join(["---"] * len(MODEL_ORDER)) + "|"
    body = [header, rule]
    for nutrient in NUTRIENTS:
        cells = [_num(metrics["models"][name][field][nutrient]) for name in MODEL_ORDER]
        role = "headline" if nutrient in HEADLINE else "recorded"
        body.append(f"| {nutrient} ({role}) | " + " | ".join(cells) + " |")
    return "\n".join(body)
