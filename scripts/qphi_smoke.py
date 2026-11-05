#!/usr/bin/env python3
"""CPU checks for U_*, heuristic utility, supervised Q_φ, and the rollout hook.

Does not read cafe labels, does not fit on hold-out, and does not write
``results/``. The closed-loop BC self-check runs last because it replaces
the label loader with a guard.
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from vdr.controller.nutrition_loss import run_self_check as check_loss  # noqa: E402
from vdr.controller.q_phi import run_self_check as check_q  # noqa: E402
from vdr.controller.q_state import run_self_check as check_z  # noqa: E402
from vdr.controller.selector import HeuristicUtilityPolicy  # noqa: E402
from vdr.controller.utility import run_self_check as check_utility  # noqa: E402
from vdr.tools.mass_estimator import run_self_check as check_mass  # noqa: E402


def _heuristic_rollout() -> None:
    """Stub tools, one rgb_test id, no cafe label. Mass is called only by RE_PORTION."""
    from vdr.controller.on_policy import rgb_test_ids, rollout_dish

    dish_id = sorted(rgb_test_ids())[0]
    calls: list[dict[str, object]] = []

    def mass_fn(image, candidates, component_index=None, component_hints_g=None, catalog_path=None, **kwargs):
        del component_index, component_hints_g, catalog_path, kwargs
        calls.append({"image": None if image is None else str(image), "n": len(list(candidates or []))})
        return {
            "portion_g": 210.0,
            "mass_g": 210.0,
            "sigma": {"mass_g": 12.0, "energy_kcal": 40.0},
            "nutrients": {"energy_kcal": 180.0, "protein_g": 2.0, "carb_g": 20.0, "fat_g": 1.0},
            "rgb_only": True,
            "depth_used": False,
            "portion_src": "mass_estimator",
            "confidence": 12.0 / 112.0,
        }

    def retrieve_fn(query, top_k=5, catalog_path=None):
        del top_k, catalog_path

        class Result:
            cands = [
                {"code": "111", "score": 0.91, "name": "Apple raw"},
                {"code": "222", "score": 0.80, "name": "Apple juice"},
            ]

        result = Result()
        result.query = query  # type: ignore[attr-defined]
        return result

    def lookup_fn(code, catalog_path=None):
        del code, catalog_path
        return {"energy_kcal": 100.0, "protein_g": 1.0, "carb_g": 10.0, "fat_g": 1.0}

    def portion_fn(code, hint_g=100.0, catalog_path=None):
        del code, hint_g, catalog_path

        class Result:
            portion_g = 100.0

        return Result()

    rolled = rollout_dish(
        dish_id,
        HeuristicUtilityPolicy(),
        image_path="overhead/rgb.png",
        perceive_fn=lambda _path: ([{"name": "apple", "portion_g": 180.0}], {"model_alias": "qwen3.8-27b", "serving": "self-check"}),
        retrieve_fn=retrieve_fn,
        lookup_fn=lookup_fn,
        portion_fn=portion_fn,
        mass_fn=mass_fn,
        check_replay=True,
    )
    actions = [step["action_t"] for step in rolled.steps]
    if actions != ["RETRIEVE", "RE_PORTION", "STOP"]:
        raise AssertionError(actions)
    if len(calls) != 1:
        raise AssertionError(calls)
    if rolled.steps[1]["obs_t"].get("portion_src") != "mass_estimator":
        raise AssertionError(rolled.steps[1]["obs_t"])
    if rolled.steps[1]["obs_t"].get("rgb_only") is not True:
        raise AssertionError("RE_PORTION obs is not marked RGB-only")
    sigma = rolled.steps[1]["obs_t"].get("sigma")
    if not isinstance(sigma, dict) or sigma.get("mass_g") != 12.0:
        raise AssertionError(sigma)


def main() -> int:
    check_z()
    check_loss()
    check_utility()
    check_q()
    check_mass()
    _heuristic_rollout()
    # This guard replaces the label loader. Keep it last.
    from vdr.controller.on_policy import run_self_check as check_on_policy

    check_on_policy()
    print("qphi smoke ok")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
