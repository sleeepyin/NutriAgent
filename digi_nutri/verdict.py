"""Holdout Q0 against this run's B1. There is no train gate."""

from __future__ import annotations

from digi_nutri.protocol import FAIL, HEADLINE_METRICS, NUTRIENTS, PASS


def q0_verdict(contrasts: dict) -> tuple[str, list[dict]]:
    """Both teachers, four nutrients, three headline metrics. Mass is not a gate."""
    failures = []
    checked = 0
    for teacher in ("t_geo", "t_ing"):
        for nutrient in NUTRIENTS:
            for metric in HEADLINE_METRICS:
                row = contrasts[teacher][nutrient][metric]
                hi = float(row["hi"])
                lo = float(row["lo"])
                checked += 1
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
    if checked != 24:
        raise RuntimeError(f"Q0 checked {checked} intervals, expected 24")
    if failures:
        return FAIL, failures
    return PASS, []


def holdout_block_reason(holdout_exists: bool, train_metrics_exist: bool) -> str | None:
    if holdout_exists:
        return "holdout metrics already exist"
    if not train_metrics_exist:
        return "train metrics are missing"
    return None
