#!/usr/bin/env python3
"""CPU contract for the RGB Mass Estimator and the RE_PORTION wiring.

Does not train, does not read rgb_test nutrition, and does not write results/.
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from vdr.controller.on_policy import run_self_check as run_on_policy_self_check  # noqa: E402
from vdr.tools.mass_estimator import (  # noqa: E402
    MassEstimatorError,
    run_self_check,
)


def _default_reportion_calls_estimate_mass() -> None:
    """RE_PORTION with no injected mass_fn must call estimate_mass, not the grid."""
    import vdr.tools.mass_estimator as mass
    from vdr.controller.on_policy import rollout_dish
    from vdr.data.n5k import read_ids

    calls: list[dict] = []
    hints: list[float] = []

    def estimate_mass(image, candidates, **kwargs):
        calls.append({"image": None if image is None else str(image), "n": len(list(candidates)), "kwargs": kwargs})
        return {
            "portion_g": 210.0,
            "mass_g": 210.0,
            "sigma": {"mass_g": 10.0},
            "intervals": {"mass_g": {"low": 180.0, "high": 240.0, "coverage": 0.95}},
            "confidence": 0.9,
            "nutrient_refs": {"model_rgb": {}, "fndds": []},
            "nutrients": {"energy_kcal": 180.0},
            "rgb_only": True,
            "depth_used": False,
            "backbone": "stub",
            "portion_src": "mass_estimator",
        }

    def portion_fn(code, hint_g=100.0, catalog_path=None):
        del code, catalog_path
        hints.append(float(hint_g))

        class Result:
            portion_g = 100.0

        return Result()

    def retrieve_fn(query, top_k=5, catalog_path=None):
        del top_k, catalog_path

        class Result:
            cands = [{"code": "111", "score": 0.91, "name": "Apple raw"}]
            selected_code = "111"

        result = Result()
        result.query = query
        return result

    def lookup_fn(code, catalog_path=None):
        del code, catalog_path
        return {"energy_kcal": 100.0, "protein_g": 1.0, "carb_g": 10.0, "fat_g": 1.0}

    class ListPolicy:
        def __init__(self) -> None:
            self.i = 0
            self.actions = ["RETRIEVE", "RE_PORTION", "STOP"]

        def predict_actions(self, _x):
            action = self.actions[self.i]
            self.i += 1
            return [action]

    original = mass.estimate_mass
    mass.estimate_mass = estimate_mass
    try:
        dish_id = read_ids("rgb_test")[0]
        rolled = rollout_dish(
            dish_id,
            ListPolicy(),
            image_path="overhead/rgb.png",
            perceive_fn=lambda _path: ([{"name": "apple", "portion_g": 180}], {"model_alias": "stub", "serving": "smoke"}),
            retrieve_fn=retrieve_fn,
            lookup_fn=lookup_fn,
            portion_fn=portion_fn,
            mass_fn=None,
            check_replay=False,
        )
    finally:
        mass.estimate_mass = original
    if len(calls) != 1:
        raise MassEstimatorError(f"RE_PORTION did not call estimate_mass: {calls}")
    if hints != [100.0]:
        raise MassEstimatorError(f"RE_PORTION consulted the portion grid: {hints}")
    obs = rolled.steps[1]["obs_t"]
    if obs.get("portion_src") != "mass_estimator" or abs(float(obs["portion_g"]) - 210.0) > 1e-6:
        raise MassEstimatorError(f"RE_PORTION did not keep the mass grams: {obs}")


def _torch_forward() -> None:
    try:
        import torch
    except ImportError:
        print("torch not installed; skipped the random forward")
        return
    from vdr.tools.mass_estimator import build_mass_net

    for backbone in ("vit_b_16", "resnet101"):
        model = build_mass_net(backbone, pretrained=False)
        model.eval()
        images = torch.zeros(2, 3, 224, 224)
        mean, scale = model(images)
        if mean.shape != (2, 5) or scale.shape != (2, 5):
            raise MassEstimatorError((backbone, mean.shape, scale.shape))
        if torch.any(scale <= 0):
            raise MassEstimatorError(f"{backbone} scale was not positive")
    print("random forward ok")


def main() -> int:
    run_self_check()
    run_on_policy_self_check()
    _default_reportion_calls_estimate_mass()
    _torch_forward()
    print("mass estimator smoke ok")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
