"""Read the written D_table audit. This file does not choose a cutoff."""

from __future__ import annotations


def use_zero_exclusion(audit: dict) -> bool:
    required = (
        "n_train",
        "n_d_table_eq_0",
        "proportion_d_table_eq_0",
        "n_height_all_zero",
        "apply_d_table_zero_exclusion",
    )
    missing = [key for key in required if key not in audit]
    if missing:
        raise RuntimeError(f"D_table audit is missing {missing}")
    n_train = int(audit["n_train"])
    n_zero = int(audit["n_d_table_eq_0"])
    if n_train <= 0:
        raise RuntimeError("D_table audit has no training plates")
    proportion = n_zero / n_train
    if abs(proportion - float(audit["proportion_d_table_eq_0"])) > 1e-12:
        raise RuntimeError("D_table proportion does not match the count")
    if int(audit["n_height_all_zero"]) != n_zero:
        raise RuntimeError("all-zero height count does not match D_table == 0")
    return bool(audit["apply_d_table_zero_exclusion"])
