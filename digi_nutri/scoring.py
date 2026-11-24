"""Holdout summaries and plate-grouped contrasts. No model code."""

from __future__ import annotations

import numpy as np

from digi_nutri.bootstrap import plate_grouped_ci
from digi_nutri.protocol import HEADLINE, NUTRIENTS, PROTOCOL_VERSION, UNITS
from digi_nutri.verdict import Q0_RULE, STABLE_RULE, decide

COMPARISONS = (
    ("geometry_teacher_minus_b1", "geometry_teacher", "b1"),
    ("ingredient_teacher_minus_b1", "ingredient_teacher", "b1"),
    ("b3_minus_b1", "b3", "b1"),
    ("b3_minus_b2a", "b3", "b2a"),
    ("b3_minus_b2b", "b3", "b2b"),
)


def per_dish_errors(targets, preds, eps: float) -> tuple[np.ndarray, np.ndarray]:
    y = np.asarray(targets, dtype=np.float64)
    pred = np.asarray(preds, dtype=np.float64)
    absolute = np.abs(pred - y)
    percent = 100.0 * absolute / np.maximum(np.abs(y), eps)
    return absolute, percent


def _mean_by_nutrient(values: np.ndarray) -> dict[str, float]:
    return {name: float(values[:, index].mean()) for index, name in enumerate(NUTRIENTS)}


def comparison_block(
    method_abs,
    method_percent,
    base_abs,
    base_percent,
    groups,
    n_boot: int,
    seed: int,
    alpha: float,
) -> dict:
    block = {}
    for index, name in enumerate(NUTRIENTS):
        mae = plate_grouped_ci(
            method_abs[:, index] - base_abs[:, index], groups, n_boot, seed, alpha
        )
        pmae = plate_grouped_ci(
            method_percent[:, index] - base_percent[:, index],
            groups,
            n_boot,
            seed,
            alpha,
        )
        block[name] = {
            "mae_mean": mae["mean"],
            "mae_ci_low": mae["ci_low"],
            "mae_ci_high": mae["ci_high"],
            "mae_stable": mae["stable_below_zero"],
            "pmae_mean": pmae["mean"],
            "pmae_ci_low": pmae["ci_low"],
            "pmae_ci_high": pmae["ci_high"],
            "pmae_stable": pmae["stable_below_zero"],
            "stable": bool(mae["stable_below_zero"] and pmae["stable_below_zero"]),
            "n_groups": mae["n_groups"],
            "n_dishes": mae["n_dishes"],
            "n_boot": mae["n_boot"],
        }
    return block


def assemble_holdout_metrics(
    labels: list[dict],
    predictions: dict,
    n_boot: int,
    seed: int,
    eps: float,
    alpha: float = 0.05,
) -> dict:
    dish_ids = [row["dish_id"] for row in labels]
    if len(set(dish_ids)) != len(dish_ids):
        raise ValueError("duplicate dish ids in the holdout labels")
    targets = np.asarray([row["y"] for row in labels], dtype=np.float64)
    groups = np.asarray([row["plate_group"] for row in labels])
    if targets.ndim != 2 or targets.shape[1] != len(NUTRIENTS):
        raise ValueError("labels must contain five nutrients")
    summaries = {}
    errors = {}
    for name, preds in predictions.items():
        absolute, percent = per_dish_errors(targets, preds, eps)
        if absolute.shape != targets.shape:
            raise ValueError(f"{name} predictions do not match the holdout")
        errors[name] = (absolute, percent)
        summaries[name] = {
            "mae": _mean_by_nutrient(absolute),
            "pmae_percent": _mean_by_nutrient(percent),
        }
    comparisons = {}
    for key, method, baseline in COMPARISONS:
        method_abs, method_percent = errors[method]
        base_abs, base_percent = errors[baseline]
        comparisons[key] = comparison_block(
            method_abs,
            method_percent,
            base_abs,
            base_percent,
            groups,
            n_boot,
            seed,
            alpha,
        )
    decision = decide(comparisons)
    return {
        "protocol_version": PROTOCOL_VERSION,
        "split": "official_depth_test",
        "n_test": len(labels),
        "n_plate_groups": len(set(groups.tolist())),
        "nutrients": list(NUTRIENTS),
        "headline": list(HEADLINE),
        "units": dict(UNITS),
        "pmae_eps": eps,
        "pmae_definition": "100 * |pred-y| / max(|y|, eps)",
        "rules": {"q0": Q0_RULE, "stable": STABLE_RULE},
        "models": summaries,
        "comparisons": comparisons,
        "decision": decision,
        "verdict": decision["verdict"],
        "labels": labels,
        "predictions": {name: np.asarray(preds, dtype=np.float64).tolist() for name, preds in predictions.items()},
        "official_holdout_evaluated_once": True,
        "student_test_input": "overhead_rgb_only",
        "teacher_eval": "privileged_inputs_for_q0_only",
    }
