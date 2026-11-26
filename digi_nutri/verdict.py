"""Official holdout Q0. Relative intervals and the dataset-PMAE absolute gate."""

from __future__ import annotations

from digi_nutri.protocol import (
    ABSOLUTE_PMAE_KEYS,
    ENERGY_DATASET_PMAE_MAX,
    FAIL,
    HEADLINE_METRICS,
    MEAN5_DATASET_PMAE_MAX,
    MODEL,
    NUTRIENTS,
    N_HEADLINE_INTERVALS,
    PASS,
)


def q0_verdict(contrasts: dict, dataset_pmae: dict[str, float]) -> tuple[str, dict]:
    """Both gates. Mass intervals are not read here. Plate PMAE is not absolute."""
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
    missing = [name for name in ABSOLUTE_PMAE_KEYS if name not in dataset_pmae]
    if missing:
        raise RuntimeError(f"absolute gate is missing dataset PMAE for {missing}")
    components = {name: float(dataset_pmae[name]) for name in ABSOLUTE_PMAE_KEYS}
    energy = components["energy"]
    mean5 = float(sum(components.values()) / len(ABSOLUTE_PMAE_KEYS))
    energy_ok = energy <= ENERGY_DATASET_PMAE_MAX
    mean_ok = mean5 <= MEAN5_DATASET_PMAE_MAX
    absolute = {
        "energy_dataset_pmae": energy,
        "energy_limit": ENERGY_DATASET_PMAE_MAX,
        "energy_ok": energy_ok,
        "mean5_dataset_pmae": mean5,
        "mean5_limit": MEAN5_DATASET_PMAE_MAX,
        "mean5_ok": mean_ok,
        "components": components,
        "pass": bool(energy_ok and mean_ok),
    }
    details = {
        "relative_failures": failures,
        "relative_pass": not failures,
        "absolute": absolute,
    }
    if failures or not absolute["pass"]:
        return FAIL, details
    return PASS, details


def holdout_block_reason(holdout_exists: bool, train_metrics_exist: bool) -> str | None:
    if holdout_exists:
        return "holdout metrics already exist; refusing another official evaluation"
    if not train_metrics_exist:
        return "train metrics are missing"
    return None
