"""Q0 / Q1 / Q2, registered before the holdout read."""

from __future__ import annotations

from digi_nutri.protocol import HEADLINE

Q0_RULE = (
    "A teacher is clearly better than B1 on a nutrient when the plate-grouped "
    "95% CI of (teacher - B1) lies entirely below 0 for both per-dish MAE and "
    "per-dish PMAE. Q0 passes only when the geometry teacher and the names-only "
    "ingredient teacher both clear that bar on energy, fat, carb, and protein. "
    "Mass is recorded and is not required."
)
STABLE_RULE = (
    "Stably better means the plate-grouped 95% CI of (method - baseline) is "
    "strictly below 0 on both per-dish MAE and per-dish PMAE. Resamples draw "
    "physical plates, using dish_id unless a reshoot file groups ids. A CI "
    "that touches or crosses 0 is not a pass. Mass is not a pass condition."
)

VERDICT_TEXT = {
    "q0_fail_stop_wait_triste": (
        "Q0 failed. The privileged teachers are not clearly better than B1 on "
        "energy and the three macros. Stop. Do not keep distilling. Wait for Triste."
    ),
    "q1_fail_stop_wait_triste": (
        "Q1 failed. B3 is not stably better than B1 on energy and all three "
        "macros. Stop. Do not add decoupling, gating, or confidence weights. "
        "Wait for Triste."
    ),
    "q2_fail_stop_wait_triste": (
        "Q2 failed. B3 is not stably better than both single-teacher students "
        "on energy and all three macros. Stop. Do not use decoupling to rescue "
        "it. Wait for Triste."
    ),
    "q1_q2_pass_stop_wait_triste": (
        "Q1 and Q2 passed on the single official holdout read. Stop and wait "
        "for Triste. Do not start Phase 2."
    ),
}


def _all_stable(comparison: dict) -> bool:
    for nutrient in HEADLINE:
        try:
            stable = comparison[nutrient]["stable"]
        except KeyError as exc:
            raise KeyError(f"comparison is missing headline nutrient {nutrient}") from exc
        if not stable:
            return False
    return True


def decide(comparisons: dict) -> dict:
    q0 = _all_stable(comparisons["geometry_teacher_minus_b1"]) and _all_stable(
        comparisons["ingredient_teacher_minus_b1"]
    )
    q1 = _all_stable(comparisons["b3_minus_b1"])
    q2 = _all_stable(comparisons["b3_minus_b2a"]) and _all_stable(comparisons["b3_minus_b2b"])
    if not q0:
        verdict = "q0_fail_stop_wait_triste"
        return {
            "q0_pass": False,
            "q1_pass": None,
            "q2_pass": None,
            "q1_opened": False,
            "q2_opened": False,
            "verdict": verdict,
            "message": VERDICT_TEXT[verdict],
            "descriptive_q1": q1,
            "descriptive_q2": q2,
        }
    if not q1:
        verdict = "q1_fail_stop_wait_triste"
        return {
            "q0_pass": True,
            "q1_pass": False,
            "q2_pass": None,
            "q1_opened": True,
            "q2_opened": False,
            "verdict": verdict,
            "message": VERDICT_TEXT[verdict],
            "descriptive_q1": False,
            "descriptive_q2": q2,
        }
    if not q2:
        verdict = "q2_fail_stop_wait_triste"
        return {
            "q0_pass": True,
            "q1_pass": True,
            "q2_pass": False,
            "q1_opened": True,
            "q2_opened": True,
            "verdict": verdict,
            "message": VERDICT_TEXT[verdict],
            "descriptive_q1": True,
            "descriptive_q2": False,
        }
    verdict = "q1_q2_pass_stop_wait_triste"
    return {
        "q0_pass": True,
        "q1_pass": True,
        "q2_pass": True,
        "q1_opened": True,
        "q2_opened": True,
        "verdict": verdict,
        "message": VERDICT_TEXT[verdict],
        "descriptive_q1": True,
        "descriptive_q2": True,
    }
