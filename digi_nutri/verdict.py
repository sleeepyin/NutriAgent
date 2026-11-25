"""Train gate against the frozen B1 numbers, and the later holdout rule."""

from __future__ import annotations

from digi_nutri.protocol import FAIL, FROZEN_B1_TRAIN_MAE, HEADLINE_METRICS, NUTRIENTS, PASS


def tgeo_train_gate(mae: dict) -> tuple[bool, list[dict]]:
    """Energy and the three macros must be strictly below frozen B1.

    Mass is ignored. A tie fails.
    """
    failures = []
    for nutrient, limit in FROZEN_B1_TRAIN_MAE.items():
        value = float(mae[nutrient])
        baseline = float(limit)
        if not value < baseline:
            failures.append(
                {
                    "nutrient": nutrient,
                    "t_geo_mae": value,
                    "b1_mae": baseline,
                }
            )
    return len(failures) == 0, failures


def q0_verdict(contrasts: dict) -> tuple[str, list[dict]]:
    """Both teachers, four nutrients, three headline metrics. Mass is not a gate."""
    failures = []
    for teacher in ("t_geo", "t_ing"):
        for nutrient in NUTRIENTS:
            for metric in HEADLINE_METRICS:
                row = contrasts[teacher][nutrient][metric]
                hi = float(row["hi"])
                lo = float(row["lo"])
                if not hi < 0.0:
                    failures.append(
                        {
                            "teacher": teacher,
                            "nutrient": nutrient,
                            "metric": metric,
                            "lo": lo,
                            "hi": hi,
                        }
                    )
    if failures:
        return FAIL, failures
    return PASS, []


def holdout_block_reason(
    train_gate_pass: bool,
    holdout_exists: bool,
    train_metrics_exist: bool,
) -> str | None:
    if holdout_exists:
        return "holdout metrics already exist"
    if not train_metrics_exist:
        return "train metrics are missing"
    if not train_gate_pass:
        return "train gate did not pass"
    return None
