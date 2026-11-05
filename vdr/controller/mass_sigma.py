"""Dish-level RGB σ from the P0b Mass Estimator.

One forward per overhead ``rgb.png``. The network does not see depth, and
the returned numbers are the specialist's mean and Laplace scale, not the
cafe label. rgb_test ids are refused. The live controller can instead read
σ off a previous ``estimate_mass`` observation; this helper exists so logged
traces from before that tool still carry a real scale.
"""

from __future__ import annotations

from typing import Any, Mapping, Sequence

from vdr.data.n5k import overhead_rgb, test_split_ids
from vdr.tools.mass_estimator import (
    DEFAULT_CHECKPOINT,
    TARGET_NAMES,
    MassEstimatorError,
    assert_rgb_path,
    coerce_rgb_image,
    eval_transform,
    load_mass_model,
    predict_original,
)


def estimate_from_sigma_record(record: Mapping[str, Any]) -> dict[str, Any]:
    """Turn a cached mean/σ record into the payload ``uncertainty_parts`` reads."""
    mean = record.get("mean")
    sigma = record.get("sigma")
    if not isinstance(mean, Mapping) or not isinstance(sigma, Mapping):
        raise MassEstimatorError("sigma record needs mean and sigma objects")
    return {
        "rgb_only": True,
        "depth_used": False,
        "sigma": {str(key): sigma[key] for key in sigma},
        "mass_g": mean.get("mass_g"),
        "nutrients": {
            "energy_kcal": mean.get("energy_kcal"),
            "protein_g": mean.get("protein_g"),
            "carb_g": mean.get("carb_g"),
            "fat_g": mean.get("fat_g"),
        },
    }


def collect_dish_sigma(
    dish_ids: Sequence[str],
    *,
    batch_size: int = 16,
) -> tuple[dict[str, dict[str, Any]], str]:
    """``(dish_id -> {mean, sigma}, backbone)`` for dishes with overhead ``rgb.png``.

    Dishes without an image are omitted. A depth path or an rgb_test id raises.
    """
    if batch_size < 1:
        raise ValueError("batch_size must be >= 1")
    banned = test_split_ids()
    wanted: list[str] = []
    seen: set[str] = set()
    for dish_id in dish_ids:
        if dish_id in banned:
            raise MassEstimatorError(f"refusing rgb_test or depth_test id {dish_id}")
        if not isinstance(dish_id, str) or not dish_id:
            raise MassEstimatorError(f"bad dish id {dish_id!r}")
        if dish_id in seen:
            continue
        seen.add(dish_id)
        wanted.append(dish_id)

    packed, meta = load_mass_model(DEFAULT_CHECKPOINT)
    transform = eval_transform()
    model = packed["model"]
    target_mean = packed["mean"]
    target_std = packed["std"]
    backbone = str(meta.get("backbone"))
    out: dict[str, dict[str, Any]] = {}
    batch_ids: list[str] = []
    batch_tensors: list[Any] = []
    flushed = 0

    def _flush() -> None:
        nonlocal flushed
        if not batch_ids:
            return
        import torch

        stacked = torch.stack(batch_tensors)
        means, scales = predict_original(model, stacked, target_mean, target_std)
        for index, dish_id in enumerate(batch_ids):
            point: dict[str, float] = {}
            sigma: dict[str, float] = {}
            for key_i, key in enumerate(TARGET_NAMES):
                point[key] = round(max(0.0, float(means[index, key_i])), 4)
                sigma[key] = round(max(0.0, float(scales[index, key_i])), 4)
            out[dish_id] = {"mean": point, "sigma": sigma}
        flushed += len(batch_ids)
        batch_ids.clear()
        batch_tensors.clear()
        if flushed % 320 == 0 or flushed == len(wanted):
            print(f"mass sigma forwards {flushed}", flush=True)

    for dish_id in wanted:
        path = overhead_rgb(dish_id)
        if not path.is_file():
            continue
        assert_rgb_path(path)
        if path.name != "rgb.png" or path.parent.parent.name != "realsense_overhead":
            raise MassEstimatorError(f"sigma image is not overhead rgb.png: {path}")
        batch_tensors.append(transform(coerce_rgb_image(path)))
        batch_ids.append(dish_id)
        if len(batch_ids) >= batch_size:
            _flush()
    _flush()
    if any(dish_id in banned for dish_id in out):
        raise MassEstimatorError("sigma map contains a test id")
    return out, backbone
