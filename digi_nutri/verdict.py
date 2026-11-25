"""Pre-registered Q0 decision. Mass is recorded and does not decide the gate."""

from __future__ import annotations

from digi_nutri.protocol import HEADLINE_METRICS, NUTRIENTS, TEACHERS

PASS = "q0_pass"
FAIL = "q0_fail_stop_wait_triste"


def q0_verdict(contrasts: dict) -> tuple[str, list[dict]]:
    """Pass only when every headline CI lies entirely below 0.

    `contrasts[teacher][nutrient][metric]` has numeric `lo` and `hi`.
    The upper endpoint must be strictly below 0. Touching 0 fails.
    """
    failures = []
    for teacher in TEACHERS:
        for nutrient in NUTRIENTS:
            for metric in HEADLINE_METRICS:
                row = contrasts[teacher][nutrient][metric]
                hi = float(row["hi"])
                lo = float(row["lo"])
                if not (hi < 0.0):
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


def train_gate(train_mae: dict) -> tuple[bool, list[dict]]:
    """Both teachers must have strictly lower train MAE than B1 on every headline nutrient."""
    failures = []
    for teacher in TEACHERS:
        for nutrient in NUTRIENTS:
            teacher_mae = float(train_mae[teacher][nutrient])
            baseline_mae = float(train_mae["b1"][nutrient])
            if not (teacher_mae < baseline_mae):
                failures.append(
                    {
                        "teacher": teacher,
                        "nutrient": nutrient,
                        "teacher_mae": teacher_mae,
                        "b1_mae": baseline_mae,
                    }
                )
    return len(failures) == 0, failures
