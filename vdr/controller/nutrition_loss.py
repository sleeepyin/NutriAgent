"""Nutrition loss whose change is the supervised target of ``Q_φ``.

``L`` is the absolute energy error in kcal, clipped at ``ENERGY_CAP_KCAL``
(2500) and divided by that cap. The cap is the existing rule-verify ceiling,
not a threshold fit on hold-out or on rgb_test. A step with no finite
committed energy has no ``L``: treating "missing" as a full-cap error made
the first commitment dwarf later kcal edits, including ``RE_PORTION``.

``ΔL = L_before - L_after`` is positive when absolute kcal error fell.
``STOP`` and ``ABSTAIN`` do not edit ``n``, so their target is 0.

The dish label is the train-split cafe total. This module does not open a
label file. Callers must refuse rgb_test.
"""

from __future__ import annotations

import math
from typing import Any, Mapping

from vdr.controller.features import _finite
from vdr.controller.state_features import ENERGY_CAP_KCAL

LOSS_DEFINITION = (
    "L = min(|energy_hat - energy_label|, 2500) / 2500. "
    "The 2500 kcal cap is ENERGY_CAP_KCAL from the closed-loop features "
    "(the same ceiling as rule_verify.ENERGY_MAX). It was not fit on hold-out "
    "or on rgb_test. Delta L = L(pre) - L(post) on nutri_dish.energy_kcal, "
    "and only when both sides have a finite energy. Positive means absolute "
    "kcal error fell. A missing energy is not scored as a full-cap miss. "
    "STOP and ABSTAIN do not edit n, so their target is 0. "
    "Mass and macro errors stay in U_mass / U_nutrient; they are not mixed "
    "into this target. rgb_test labels are not a legal reference."
)


def nutrition_loss(nutri: Mapping[str, Any] | None, label: Mapping[str, float]) -> float | None:
    """Clipped normalized absolute energy error, or ``None`` if energy is absent."""
    err = energy_abs_error(nutri, label)
    if err is None:
        return None
    if ENERGY_CAP_KCAL <= 0.0:
        raise ValueError("ENERGY_CAP_KCAL must be positive")
    return float(min(err, ENERGY_CAP_KCAL) / ENERGY_CAP_KCAL)


def energy_abs_error(nutri: Mapping[str, Any] | None, label: Mapping[str, float]) -> float | None:
    """Unclipped absolute kcal error, or ``None`` when the commitment has no energy."""
    truth = _finite(label.get("energy_kcal"))
    if truth is None:
        raise ValueError("label is missing energy_kcal")
    hat = None if nutri is None else _finite(nutri.get("energy_kcal"))
    if hat is None:
        return None
    error = abs(hat - truth)
    if not math.isfinite(error):
        return None
    return float(error)


def run_self_check() -> None:
    label = {
        "energy_kcal": 100.0,
        "mass_g": 100.0,
        "protein_g": 10.0,
        "carb_g": 10.0,
        "fat_g": 2.0,
    }
    perfect = {"energy_kcal": 100.0}
    if nutrition_loss(perfect, label) != 0.0:
        raise AssertionError("perfect energy should have loss 0")
    if nutrition_loss(None, label) is not None:
        raise AssertionError("missing energy is not a full-cap loss")
    off = nutrition_loss({"energy_kcal": 100.0 + 250.0}, label)
    if off is None or abs(off - 250.0 / ENERGY_CAP_KCAL) > 1e-12:
        raise AssertionError(off)
    huge = nutrition_loss({"energy_kcal": 100.0 + 9000.0}, label)
    if huge != 1.0:
        raise AssertionError("errors above the cap must clip at 1")
    if energy_abs_error(None, label) is not None:
        raise AssertionError("missing energy should stay unscored in kcal")
    if energy_abs_error(perfect, label) != 0.0:
        raise AssertionError("energy error")
