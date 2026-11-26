"""Official holdout Q0 for one T_multi. There is no train gate."""

from __future__ import annotations

from digi_nutri.protocol import FAIL, HEADLINE_METRICS, MODEL, NUTRIENTS, N_HEADLINE_INTERVALS, PASS


def q0_verdict(contrasts: dict) -> tuple[str, list[dict]]:
    """Energy and three macros, three headline metrics. Mass is not a gate."""
    failures = []
    checked = 0
    block = contrasts[MODEL]
    for nutrient in NUTRIENTS:
        for metric in HEADLINE_METRICS:
            row = block[nutrient][metric]
            hi = float(row["hi"])
            lo = float(row["lo"])
            checked += 1
            if not hi < 0.0:
                failures.append(
                    {
                        "model": MODEL,
                        "nutrient": nutrient,
                        "metric": metric,
                        "lo": lo,
                        "hi": hi,
                    }
                )
    if checked != N_HEADLINE_INTERVALS:
        raise RuntimeError(f"Q0 checked {checked} intervals, expected {N_HEADLINE_INTERVALS}")
    if failures:
        return FAIL, failures
    return PASS, []


def holdout_block_reason(holdout_exists: bool, train_metrics_exist: bool) -> str | None:
    if holdout_exists:
        return "holdout metrics already exist; refusing another official evaluation"
    if not train_metrics_exist:
        return "train metrics are missing"
    return None
