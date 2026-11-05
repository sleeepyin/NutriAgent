"""RGB-only Mass Estimator (P0b).

Dish-level multi-task head for Nutrition5K overhead RGB:

    [M, E, P, C, F] = mass_g, energy_kcal, protein_g, carb_g, fat_g

plus a per-target Laplace scale σ. The tool entry point is
``estimate_mass(image, candidates)``. It does not read depth, it does not
load the legacy files under ``weights/``, and it does not score ``rgb_test``.

RE_PORTION in the closed-loop rollout calls this tool. The gram grid remains
the blind first guess inside RETRIEVE; it is not a substitute for RE_PORTION.
"""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from vdr.data.n5k import load_labels, overhead_rgb, read_ids, test_split_ids
from vdr.tools.nutrition_lookup import lookup_per_100g, scale_by_portion
from vdr.tools.rule_verify import PORTION_MAX_G, PORTION_MIN_G

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CHECKPOINT = ROOT / "results" / "mass_estimator"

# M, E, P, C, F. This order is the head order. It is not LABEL_FIELDS.
TARGET_NAMES: tuple[str, ...] = (
    "mass_g",
    "energy_kcal",
    "protein_g",
    "carb_g",
    "fat_g",
)
# STATUS / PLAN list ViT-B/16 first, then ResNet101. Tie-break follows that order.
BACKBONE_ORDER: tuple[str, ...] = ("vit_b_16", "resnet101")
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)
IMAGE_SIZE = 224
EVAL_RESIZE = 256
# Central 95% of a Laplace: each tail is 2.5%, so 0.5*exp(-t/b) = 0.025.
LAPLACE_Z95 = math.log(20.0)
# Fixed grams in the confidence map. Not fit on hold-out or rgb_test.
CONF_SCALE_G = 100.0
SCALE_FLOOR = 1e-3
# sha256(dish_id)[:8] % 10 == 0 is the existing rgb_train hold-out.
# Residue 1 is inside fit and is the only probe-val slice.
HOLDOUT_RESIDUE = 0
PROBE_VAL_RESIDUE = 1
PROTOCOL_ID = "p0b_mass_rgb_train_fit_holdout"
SHARD_BYTES = 80 * 1024 * 1024
RGB_SUFFIXES = {".png", ".jpg", ".jpeg", ".webp"}
_CACHE: dict[str, Any] = {}


class MassEstimatorError(ValueError):
    """The mass tool was asked to do something the RGB protocol forbids."""


@dataclass(frozen=True)
class RgbDish:
    """One official rgb_train dish with an overhead rgb.png and cafe totals."""

    dish_id: str
    image: Path
    pool: str
    residue: int
    mass_g: float
    energy_kcal: float
    protein_g: float
    carb_g: float
    fat_g: float

    def targets(self) -> np.ndarray:
        return np.array(
            [self.mass_g, self.energy_kcal, self.protein_g, self.carb_g, self.fat_g],
            dtype=np.float64,
        )


def dish_residue(dish_id: str) -> int:
    """Same 8 hex digits as ``vdr.weak.teacher.pool_of_dish``. Zero is hold-out."""
    digest = hashlib.sha256(dish_id.encode("utf-8")).hexdigest()
    return int(digest[:8], 16) % 10


def pool_of_rgb_train(dish_id: str) -> str:
    """rgb_train fit/hold-out. Matches the weak-step teacher; it does not touch rgb_test."""
    if dish_residue(dish_id) == HOLDOUT_RESIDUE:
        return "holdout"
    return "fit"


def assert_rgb_path(path: str | Path) -> None:
    """Refuse depth paths. A non-png name is allowed for a caller-supplied RGB file."""
    image = Path(path)
    folded = [part.casefold() for part in image.parts]
    if any("depth" in part for part in folded):
        raise MassEstimatorError(f"RGB-only main table refuses a depth path: {image}")
    if image.suffix.casefold() not in RGB_SUFFIXES:
        raise MassEstimatorError(f"RGB-only main table refuses {image.name}")


def _legacy_path(path: Path) -> bool:
    folded = [part.casefold() for part in path.parts]
    name = path.name.casefold()
    if any(part in {"nutrition5k", "metafood3d", "nvreal"} and "weights" in folded for part in folded):
        return True
    markers = ("density_", "deep_weight", "weight_cqr", "efficientnet")
    return any(marker in name for marker in markers)


def assert_not_legacy_checkpoint(path: str | Path) -> None:
    """The specialist is trained here. Old density / deep-weight files stay unloaded."""
    if _legacy_path(Path(path)):
        raise MassEstimatorError(f"refusing legacy regression weights: {path}")


def assert_safe_result_dir(path: str | Path) -> None:
    """Do not write the frozen n5k tables or the weak-step jsonl."""
    directory = Path(path)
    if directory.suffix:
        directory = directory.parent
    parts = {part.casefold() for part in directory.parts}
    name = directory.name.casefold()
    if "weak_steps" in parts or name.startswith("n5k_") or "n5k_rgb_test" in name:
        raise MassEstimatorError(f"refusing to write frozen results at {directory}")


def component_portion_g(
    dish_mass_g: float,
    component_index: int | None,
    component_hints_g: Sequence[float | None] | None,
) -> float:
    """Split the specialist's dish mass across components, then clip to portion bounds.

    Hints are the VLM gram guesses already on the hypotheses. They set only the
    relative split. A missing or non-positive hint has weight 1, so it does not
    take the plate. No coefficient here is fit on hold-out or rgb_test.
    """
    mass = max(0.0, float(dish_mass_g))
    if component_hints_g is None or component_index is None:
        portion = mass
    else:
        hints = list(component_hints_g)
        if component_index < 0 or component_index >= len(hints):
            raise MassEstimatorError(
                f"component_index {component_index} outside {len(hints)} hints"
            )
        weights = [1.0 if hint is None or float(hint) <= 0.0 else float(hint) for hint in hints]
        portion = mass * weights[component_index] / sum(weights)
    return float(min(PORTION_MAX_G, max(PORTION_MIN_G, portion)))


def iter_rgb_train_dishes() -> list[RgbDish]:
    """Official rgb_train dishes that have overhead rgb.png. No rgb_test, no depth."""
    labels = load_labels()
    banned = test_split_ids()
    rows: list[RgbDish] = []
    for dish_id in read_ids("rgb_train"):
        if dish_id in banned:
            raise MassEstimatorError(f"{dish_id} is in rgb_test or depth_test")
        image = overhead_rgb(dish_id)
        if not image.is_file():
            continue
        assert_rgb_path(image)
        if image.name != "rgb.png" or image.parent.parent.name != "realsense_overhead":
            raise MassEstimatorError(f"training image is not overhead rgb.png: {image}")
        label = labels.get(dish_id)
        if label is None:
            continue
        residue = dish_residue(dish_id)
        pool = pool_of_rgb_train(dish_id)
        rows.append(
            RgbDish(
                dish_id=dish_id,
                image=image,
                pool=pool,
                residue=residue,
                mass_g=float(label["mass_g"]),
                energy_kcal=float(label["energy_kcal"]),
                protein_g=float(label["protein_g"]),
                carb_g=float(label["carb_g"]),
                fat_g=float(label["fat_g"]),
            )
        )
    return rows


def partition_rgb_train(
    rows: Sequence[RgbDish],
) -> tuple[list[RgbDish], list[RgbDish], list[RgbDish], list[RgbDish]]:
    """Return probe-train, probe-val, full fit, hold-out.

    Probe-val is fit residue 1. Hold-out is residue 0 and is not in the probe.
    """
    fit = [row for row in rows if row.pool == "fit"]
    holdout = [row for row in rows if row.pool == "holdout"]
    probe_val = [row for row in fit if row.residue == PROBE_VAL_RESIDUE]
    probe_train = [row for row in fit if row.residue != PROBE_VAL_RESIDUE]
    _assert_partitions(probe_train, probe_val, fit, holdout)
    return probe_train, probe_val, fit, holdout


def _assert_partitions(
    probe_train: Sequence[RgbDish],
    probe_val: Sequence[RgbDish],
    fit: Sequence[RgbDish],
    holdout: Sequence[RgbDish],
) -> None:
    names = {
        "probe_train": {row.dish_id for row in probe_train},
        "probe_val": {row.dish_id for row in probe_val},
        "holdout": {row.dish_id for row in holdout},
    }
    if names["probe_train"] & names["probe_val"]:
        raise MassEstimatorError("probe-train and probe-val overlap")
    if names["probe_train"] & names["holdout"] or names["probe_val"] & names["holdout"]:
        raise MassEstimatorError("hold-out leaked into the fit probe")
    if {row.dish_id for row in fit} & names["holdout"]:
        raise MassEstimatorError("fit and hold-out overlap")
    if not probe_train or not probe_val or not holdout:
        raise MassEstimatorError("rgb_train RGB partition is empty")
    banned = test_split_ids()
    for row in list(probe_train) + list(probe_val) + list(holdout):
        if row.dish_id in banned:
            raise MassEstimatorError(f"{row.dish_id} is a test dish")
        if "depth" in str(row.image).casefold():
            raise MassEstimatorError(f"depth path in the RGB table: {row.image}")


def target_mean_std(rows: Sequence[RgbDish]) -> tuple[np.ndarray, np.ndarray]:
    """Per-target mean and std. Caller must pass the train rows only."""
    if not rows:
        raise MassEstimatorError("cannot standardize an empty split")
    matrix = np.stack([row.targets() for row in rows], axis=0)
    mean = matrix.mean(axis=0)
    std = matrix.std(axis=0)
    std = np.maximum(std, 1e-3)
    return mean.astype(np.float64), std.astype(np.float64)


def select_backbone(rows: Sequence[Mapping[str, Any]]) -> str:
    """Lower probe-val mass MAE, then energy MAE, then STATUS order.

    ``rows`` are fit-internal probe scores. Hold-out and rgb_test are not inputs.
    """
    if not rows:
        raise MassEstimatorError("backbone probe is empty")

    def sort_key(row: Mapping[str, Any]) -> tuple[float, float, int]:
        name = str(row["backbone"])
        if name not in BACKBONE_ORDER:
            raise MassEstimatorError(f"unknown backbone {name}")
        return (float(row["mass_mae"]), float(row["energy_mae"]), BACKBONE_ORDER.index(name))

    return str(min(rows, key=sort_key)["backbone"])


def confidence_from_mass_sigma(sigma_g: float) -> float:
    """Map the mass Laplace scale in grams to (0, 1]. Fixed 100 g, not a searched cut."""
    sigma = max(0.0, float(sigma_g))
    return float(1.0 / (1.0 + sigma / CONF_SCALE_G))


def laplace_interval(mean: float, scale: float) -> dict[str, float]:
    """Central 95% Laplace interval. The point estimate itself is clipped at 0 later."""
    width = LAPLACE_Z95 * max(float(scale), 0.0)
    low = float(mean) - width
    high = float(mean) + width
    low = max(0.0, low)
    high = max(low, high)
    return {"low": low, "high": high, "coverage": 0.95, "family": "laplace"}


def coerce_candidates(candidates: Any) -> list[dict[str, Any]]:
    if candidates is None:
        return []
    if isinstance(candidates, Mapping):
        candidates = [candidates]
    if isinstance(candidates, str):
        candidates = [candidates]
    out: list[dict[str, Any]] = []
    for cand in candidates:
        if isinstance(cand, str):
            out.append({"code": cand, "score": None, "name": None})
            continue
        if not isinstance(cand, Mapping) or "code" not in cand:
            raise MassEstimatorError("each candidate needs a code")
        score = cand.get("score")
        name = cand.get("name")
        out.append(
            {
                "code": str(cand["code"]),
                "score": None if score is None else float(score),
                "name": None if name is None else str(name),
            }
        )
    return out


def _round(value: float, digits: int = 4) -> float:
    return round(float(value), digits)


def nutrient_refs(
    candidates: Sequence[Mapping[str, Any]],
    portion_g: float,
    *,
    catalog_path: str | Path | None = None,
    lookup_fn: Any = lookup_per_100g,
) -> list[dict[str, Any]]:
    """FNDDS per-100g scaled by the component grams. Missing codes stay unavailable."""
    refs: list[dict[str, Any]] = []
    for cand in candidates:
        code = str(cand["code"])
        try:
            per = lookup_fn(code, catalog_path=catalog_path)
        except KeyError:
            refs.append({"code": code, "name": cand.get("name"), "available": False})
            continue
        keep = ("energy_kcal", "protein_g", "carb_g", "fat_g")
        per_keep = {key: float(per[key]) for key in keep if key in per}
        scaled = scale_by_portion(per_keep, portion_g)
        refs.append(
            {
                "code": code,
                "name": cand.get("name"),
                "score": cand.get("score"),
                "available": True,
                "source": "fndds_per_100g_times_portion",
                "portion_g": _round(portion_g),
                "per_100g": {key: _round(val) for key, val in per_keep.items()},
                "nutrients": {key: _round(val) for key, val in scaled.items()},
            }
        )
    return refs


def assemble_estimate(
    means: Mapping[str, float],
    sigmas: Mapping[str, float],
    *,
    backbone: str,
    candidates: Any = None,
    component_index: int | None = None,
    component_hints_g: Sequence[float | None] | None = None,
    catalog_path: str | Path | None = None,
    checkpoint: str | Path | None = None,
    lookup_fn: Any = lookup_per_100g,
) -> dict[str, Any]:
    """Build the tool payload from original-unit means and Laplace scales."""
    missing = [name for name in TARGET_NAMES if name not in means or name not in sigmas]
    if missing:
        raise MassEstimatorError(f"prediction is missing {missing}")
    point = {name: max(0.0, float(means[name])) for name in TARGET_NAMES}
    sigma = {name: max(0.0, float(sigmas[name])) for name in TARGET_NAMES}
    intervals = {
        name: {
            key: (_round(val) if isinstance(val, float) else val)
            for key, val in laplace_interval(float(means[name]), sigma[name]).items()
        }
        for name in TARGET_NAMES
    }
    cands = coerce_candidates(candidates)
    portion = component_portion_g(point["mass_g"], component_index, component_hints_g)
    model_rgb = {name: _round(point[name]) for name in TARGET_NAMES}
    return {
        "tool": "estimate_mass",
        "modality": "rgb",
        "rgb_only": True,
        "depth_used": False,
        "backbone": backbone,
        "checkpoint": None if checkpoint is None else str(checkpoint),
        "mass_g": model_rgb["mass_g"],
        "nutrients": {
            "energy_kcal": model_rgb["energy_kcal"],
            "protein_g": model_rgb["protein_g"],
            "carb_g": model_rgb["carb_g"],
            "fat_g": model_rgb["fat_g"],
        },
        "sigma": {name: _round(sigma[name]) for name in TARGET_NAMES},
        "intervals": intervals,
        "confidence": _round(confidence_from_mass_sigma(sigma["mass_g"])),
        "nutrient_refs": {
            "model_rgb": model_rgb,
            "fndds": nutrient_refs(cands, portion, catalog_path=catalog_path, lookup_fn=lookup_fn),
        },
        "portion_g": _round(portion),
        "portion_src": "mass_estimator",
        "candidates_n": len(cands),
    }


def _vision_stack() -> tuple[Any, ...]:
    import torch
    from torch import nn
    from torchvision import transforms
    from torchvision.models import ResNet101_Weights, ViT_B_16_Weights, resnet101, vit_b_16

    return torch, nn, transforms, resnet101, vit_b_16, ResNet101_Weights, ViT_B_16_Weights


def build_mass_net(backbone: str, *, pretrained: bool) -> Any:
    """ViT-B/16 or ResNet101, ImageNet-1K V1 when ``pretrained``, then a 10-d head.

    The head emits 5 means and 5 positive Laplace scales in standardized space.
    """
    if backbone not in BACKBONE_ORDER:
        raise MassEstimatorError(f"backbone must be one of {BACKBONE_ORDER}")
    torch, nn, _transforms, resnet101, vit_b_16, ResNet101_Weights, ViT_B_16_Weights = _vision_stack()

    class MassNet(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.backbone_name = backbone
            if backbone == "vit_b_16":
                weights = ViT_B_16_Weights.IMAGENET1K_V1 if pretrained else None
                encoder = vit_b_16(weights=weights)
                feat = int(encoder.heads.head.in_features)
                encoder.heads = nn.Identity()
            else:
                weights = ResNet101_Weights.IMAGENET1K_V1 if pretrained else None
                encoder = resnet101(weights=weights)
                feat = int(encoder.fc.in_features)
                encoder.fc = nn.Identity()
            self.encoder = encoder
            self.head = nn.Linear(feat, len(TARGET_NAMES) * 2)
            nn.init.zeros_(self.head.weight)
            bias = torch.zeros(len(TARGET_NAMES) * 2)
            # softplus(bias) ≈ 1 so the initial scale is about one standardized unit.
            bias[len(TARGET_NAMES) :] = math.log(math.e - 1.0)
            self.head.bias.data.copy_(bias)

        def forward(self, images: Any) -> tuple[Any, Any]:
            feat = self.encoder(images)
            if feat.ndim > 2:
                feat = feat.flatten(1)
            raw = self.head(feat)
            mean = raw[:, : len(TARGET_NAMES)]
            scale = torch.nn.functional.softplus(raw[:, len(TARGET_NAMES) :]) + SCALE_FLOOR
            return mean, scale

    return MassNet()


def eval_transform() -> Any:
    """ImageNet V1 eval geometry shared by ViT-B/16 and ResNet101 V1: resize 256, crop 224."""
    _torch, _nn_mod, transforms, *_rest = _vision_stack()
    return transforms.Compose(
        [
            transforms.Resize(EVAL_RESIZE, interpolation=transforms.InterpolationMode.BILINEAR),
            transforms.CenterCrop(IMAGE_SIZE),
            transforms.ToTensor(),
            transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        ]
    )


def train_transform() -> Any:
    _torch, _nn_mod, transforms, *_rest = _vision_stack()
    return transforms.Compose(
        [
            transforms.RandomResizedCrop(
                IMAGE_SIZE,
                scale=(0.7, 1.0),
                interpolation=transforms.InterpolationMode.BILINEAR,
            ),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        ]
    )


def laplace_nll(mean: Any, scale: Any, target: Any) -> Any:
    """Mean Laplace negative log-likelihood. ``scale`` is σ, already positive."""
    return (torch_log_term(scale) + (target - mean).abs() / scale).mean()


def torch_log_term(scale: Any) -> Any:
    import torch

    return torch.log(2.0 * scale)


def _rgb_array(image: np.ndarray) -> np.ndarray:
    """HWC with 3 channels. Extra channels are refused before any decoder import."""
    if image.ndim != 3 or image.shape[2] != 3:
        raise MassEstimatorError(
            f"RGB-only refuses array shape {image.shape}; "
            "depth and extra channels are not the main table"
        )
    if np.issubdtype(image.dtype, np.floating):
        copied = np.array(image, dtype=np.float64, copy=True)
        if copied.size and float(np.max(copied)) <= 1.0:
            copied *= 255.0
        return np.clip(copied, 0, 255).astype(np.uint8)
    if image.dtype != np.uint8:
        return np.clip(image, 0, 255).astype(np.uint8)
    return image


def coerce_rgb_image(image: Any) -> Any:
    """Accept a path, a PIL RGB/RGBA image, or an HWC uint8/float array with 3 channels."""
    if image is None:
        raise MassEstimatorError("estimate_mass requires an RGB image")
    if isinstance(image, np.ndarray):
        array = _rgb_array(image)
        from PIL import Image

        return Image.fromarray(array, mode="RGB")
    if isinstance(image, (str, Path)):
        from PIL import Image

        path = Path(image)
        assert_rgb_path(path)
        if not path.is_file():
            raise MassEstimatorError(f"RGB image not found: {path}")
        with Image.open(path) as handle:
            if handle.mode not in {"RGB", "RGBA"}:
                raise MassEstimatorError(f"RGB-only refuses mode {handle.mode}")
            return handle.convert("RGB")
    mode = getattr(image, "mode", None)
    if mode is not None and hasattr(image, "convert"):
        if mode not in {"RGB", "RGBA"}:
            raise MassEstimatorError(f"RGB-only refuses mode {mode}")
        return image.convert("RGB")
    raise MassEstimatorError(f"RGB-only refuses image type {type(image).__name__}")


def _state_arrays(model: Any) -> dict[str, np.ndarray]:
    arrays: dict[str, np.ndarray] = {}
    for name, value in model.state_dict().items():
        array = value.detach().cpu().numpy()
        if np.issubdtype(array.dtype, np.floating):
            array = array.astype(np.float16)
        arrays[f"w.{name}"] = array
    return arrays


def save_checkpoint(
    directory: str | Path,
    model: Any,
    *,
    mean: np.ndarray,
    std: np.ndarray,
    meta: Mapping[str, Any],
) -> dict[str, Any]:
    """Write float16 weight shards plus ``train_meta.json``. No ``.pth``."""
    assert_not_legacy_checkpoint(directory)
    assert_safe_result_dir(directory)
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    payload = dict(meta)
    if payload.get("rgb_only") is not True or payload.get("depth_used") is not False:
        raise MassEstimatorError("checkpoint meta must say rgb_only and depth_used false")
    if payload.get("uses_rgb_test_gt") is not False:
        raise MassEstimatorError("checkpoint meta must say rgb_test GT was not used")
    arrays = _state_arrays(model)
    arrays["target_mean"] = np.asarray(mean, dtype=np.float32)
    arrays["target_std"] = np.asarray(std, dtype=np.float32)
    for stale in directory.glob("weights-part-*.npz"):
        stale.unlink()
    parts: list[str] = []
    current: dict[str, np.ndarray] = {}
    size = 0
    part_index = 0

    def flush() -> None:
        nonlocal current, size, part_index
        if not current:
            return
        name = f"weights-part-{part_index:02d}.npz"
        np.savez(directory / name, **current)
        parts.append(name)
        part_index += 1
        current = {}
        size = 0

    for key, array in arrays.items():
        nbytes = int(array.nbytes)
        if current and size + nbytes > SHARD_BYTES:
            flush()
        current[key] = array
        size += nbytes
    flush()
    payload = dict(payload)
    payload["weights"] = parts
    payload["targets"] = list(TARGET_NAMES)
    payload["target_mean"] = [float(v) for v in np.asarray(mean).tolist()]
    payload["target_std"] = [float(v) for v in np.asarray(std).tolist()]
    (directory / "train_meta.json").write_text(
        json.dumps(payload, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    return payload


def load_arrays(directory: str | Path) -> dict[str, np.ndarray]:
    directory = Path(directory)
    assert_not_legacy_checkpoint(directory)
    parts = sorted(directory.glob("weights-part-*.npz"))
    if not parts:
        raise MassEstimatorError(f"no weight shards in {directory}")
    arrays: dict[str, np.ndarray] = {}
    for part in parts:
        data = np.load(part)
        for key in data.files:
            arrays[key] = data[key]
    return arrays


def load_train_meta(directory: str | Path | None = None) -> dict[str, Any]:
    directory = DEFAULT_CHECKPOINT if directory is None else Path(directory)
    meta_path = directory / "train_meta.json"
    if not meta_path.is_file():
        raise MassEstimatorError(f"missing {meta_path}")
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    if meta.get("rgb_only") is not True or meta.get("depth_used") is not False:
        raise MassEstimatorError("refusing a checkpoint that is not RGB-only")
    if meta.get("uses_rgb_test_gt") is not False:
        raise MassEstimatorError("refusing a checkpoint that used rgb_test GT")
    return meta


def load_mass_model(directory: str | Path | None = None, *, device: Any = None) -> tuple[Any, dict[str, Any]]:
    """Load the fine-tuned RGB specialist. Does not download ImageNet weights again."""
    import torch

    directory = DEFAULT_CHECKPOINT if directory is None else Path(directory)
    meta = load_train_meta(directory)
    backbone = str(meta["backbone"])
    arrays = load_arrays(directory)
    mean = np.asarray(arrays["target_mean"], dtype=np.float64)
    std = np.asarray(arrays["target_std"], dtype=np.float64)
    model = build_mass_net(backbone, pretrained=False)
    state: dict[str, Any] = {}
    for key, array in arrays.items():
        if not key.startswith("w."):
            continue
        tensor = torch.from_numpy(np.array(array))
        if tensor.dtype == torch.float16:
            tensor = tensor.float()
        state[key[2:]] = tensor
    missing, unexpected = model.load_state_dict(state, strict=True)
    if missing or unexpected:
        raise MassEstimatorError(f"checkpoint mismatch missing={missing} unexpected={unexpected}")
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()
    packed = {"model": model, "mean": mean, "std": std, "meta": meta, "device": device}
    return packed, meta


def clear_model_cache() -> None:
    _CACHE.clear()


def _cached_model(directory: Path) -> dict[str, Any]:
    key = str(directory.resolve())
    cached = _CACHE.get(key)
    if cached is None:
        cached, _meta = load_mass_model(directory)
        _CACHE[key] = cached
    return cached


def predict_original(model: Any, batch: Any, mean: np.ndarray, std: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Return unclipped means and positive scales in original units. Shape (N, 5).

    The tool clips the reported point at 0. Intervals stay centered on this mean.
    """
    import torch

    device = next(model.parameters()).device
    with torch.no_grad():
        pred_mean, pred_scale = model(batch.to(device))
    mean_z = pred_mean.float().cpu().numpy()
    scale_z = pred_scale.float().cpu().numpy()
    mean_y = mean_z * std.reshape(1, -1) + mean.reshape(1, -1)
    scale_y = np.maximum(scale_z * std.reshape(1, -1), 0.0)
    return mean_y, scale_y


def estimate_mass(
    image: Any,
    candidates: Any = None,
    *,
    component_index: int | None = None,
    component_hints_g: Sequence[float | None] | None = None,
    catalog_path: str | Path | None = None,
    checkpoint: str | Path | None = None,
    lookup_fn: Any = lookup_per_100g,
) -> dict[str, Any]:
    """Predict dish mass, nutrients, σ, and a component portion from one RGB image.

    ``candidates`` are FNDDS codes (or ``{code, score, name}`` rows) for the
    component being portioned. Nutrient refs scale those codes by the allocated
    grams. The network itself sees only RGB.
    """
    import torch

    directory = DEFAULT_CHECKPOINT if checkpoint is None else Path(checkpoint)
    packed = _cached_model(directory)
    pil = coerce_rgb_image(image)
    batch = eval_transform()(pil).unsqueeze(0)
    means, scales = predict_original(packed["model"], batch, packed["mean"], packed["std"])
    mean_map = {name: float(means[0, index]) for index, name in enumerate(TARGET_NAMES)}
    sigma_map = {name: float(scales[0, index]) for index, name in enumerate(TARGET_NAMES)}
    # Silence an unused import if torch is only needed for the model.
    del torch
    return assemble_estimate(
        mean_map,
        sigma_map,
        backbone=str(packed["meta"]["backbone"]),
        candidates=candidates,
        component_index=component_index,
        component_hints_g=component_hints_g,
        catalog_path=catalog_path,
        checkpoint=directory,
        lookup_fn=lookup_fn,
    )


def run_self_check() -> None:
    """CPU checks. No checkpoint, no rgb_test labels, no legacy weights."""
    import tempfile

    if abs(component_portion_g(250.0, None, None) - 250.0) > 1e-9:
        raise AssertionError("single-dish portion")
    if component_portion_g(0.0, None, None) != PORTION_MIN_G:
        raise AssertionError("non-positive mass still has to be a legal portion")
    if component_portion_g(5000.0, None, None) != PORTION_MAX_G:
        raise AssertionError("portion ceiling")
    left = component_portion_g(300.0, 0, [100.0, 200.0])
    right = component_portion_g(300.0, 1, [100.0, 200.0])
    if abs(left - 100.0) > 1e-6 or abs(right - 200.0) > 1e-6:
        raise AssertionError((left, right))
    missing = component_portion_g(101.0, 0, [None, 100.0])
    if abs(missing - 1.0) > 1e-6:
        raise AssertionError(missing)

    try:
        assert_rgb_path(Path("imagery/depth/dish_1/rgb.png"))
    except MassEstimatorError:
        pass
    else:
        raise AssertionError("depth path was accepted")
    try:
        assert_not_legacy_checkpoint(Path("weights/nutrition5k/density_resnet18_final.pth"))
    except MassEstimatorError:
        pass
    else:
        raise AssertionError("legacy weights were accepted")
    try:
        assert_safe_result_dir(Path("results/weak_steps"))
    except MassEstimatorError:
        pass
    else:
        raise AssertionError("weak_steps write was accepted")

    chosen = select_backbone(
        [
            {"backbone": "resnet101", "mass_mae": 10.0, "energy_mae": 5.0},
            {"backbone": "vit_b_16", "mass_mae": 10.0, "energy_mae": 5.0},
        ]
    )
    if chosen != "vit_b_16":
        raise AssertionError(chosen)
    chosen = select_backbone(
        [
            {"backbone": "vit_b_16", "mass_mae": 11.0, "energy_mae": 1.0},
            {"backbone": "resnet101", "mass_mae": 10.0, "energy_mae": 50.0},
        ]
    )
    if chosen != "resnet101":
        raise AssertionError(chosen)

    def lookup_fn(code: str, catalog_path: str | Path | None = None) -> dict[str, float]:
        del catalog_path
        if code != "111":
            raise KeyError(code)
        return {"energy_kcal": 50.0, "protein_g": 1.0, "carb_g": 10.0, "fat_g": 0.5}

    estimate = assemble_estimate(
        {"mass_g": 200.0, "energy_kcal": 180.0, "protein_g": 6.0, "carb_g": 20.0, "fat_g": 7.0},
        {"mass_g": 100.0, "energy_kcal": 40.0, "protein_g": 2.0, "carb_g": 5.0, "fat_g": 2.0},
        backbone="vit_b_16",
        candidates=[{"code": "111", "score": 0.9, "name": "Apple"}, {"code": "missing"}],
        component_index=0,
        component_hints_g=[100.0, 100.0],
        lookup_fn=lookup_fn,
    )
    if estimate["rgb_only"] is not True or estimate["depth_used"] is not False:
        raise AssertionError("estimate is not RGB-only")
    if estimate["portion_src"] != "mass_estimator" or abs(estimate["portion_g"] - 100.0) > 1e-6:
        raise AssertionError(estimate["portion_g"])
    if abs(estimate["confidence"] - 0.5) > 1e-9:
        raise AssertionError(estimate["confidence"])
    if estimate["nutrient_refs"]["fndds"][0]["nutrients"]["energy_kcal"] != 50.0:
        raise AssertionError(estimate["nutrient_refs"])
    if estimate["nutrient_refs"]["fndds"][1]["available"] is not False:
        raise AssertionError("missing code should stay unavailable")
    try:
        coerce_rgb_image(np.zeros((4, 4, 4), dtype=np.uint8))
    except MassEstimatorError:
        pass
    else:
        raise AssertionError("4-channel array was accepted")

    from vdr.weak.teacher import pool_of_dish

    rows = iter_rgb_train_dishes()
    for row in rows[:8]:
        if pool_of_dish(row.dish_id) != row.pool:
            raise AssertionError(f"{row.dish_id} disagrees with the weak-step pool")
    probe_train, probe_val, fit, holdout = partition_rgb_train(rows)
    if len(fit) < 1000 or len(holdout) < 50:
        raise AssertionError((len(fit), len(holdout)))
    if {row.dish_id for row in fit} & {row.dish_id for row in holdout}:
        raise AssertionError("fit/hold-out overlap")
    mean, std = target_mean_std(probe_train)
    if mean.shape != (5,) or np.any(std <= 0):
        raise AssertionError("bad standardization")

    with tempfile.TemporaryDirectory() as tmp:
        directory = Path(tmp) / "mass"
        # A tiny fake module is unnecessary: round-trip the numpy shard helper via a real state
        # only when torch is installed. Always round-trip raw arrays.
        arrays = {
            "w.example": np.ones((4,), dtype=np.float16),
            "target_mean": np.zeros(5, dtype=np.float32),
            "target_std": np.ones(5, dtype=np.float32),
        }
        directory.mkdir()
        np.savez(directory / "weights-part-00.npz", **arrays)
        (directory / "train_meta.json").write_text(
            json.dumps(
                {
                    "rgb_only": True,
                    "depth_used": False,
                    "uses_rgb_test_gt": False,
                    "backbone": "vit_b_16",
                    "weights": ["weights-part-00.npz"],
                }
            ),
            encoding="utf-8",
        )
        loaded = load_arrays(directory)
        if loaded["w.example"].dtype != np.float16 or loaded["w.example"].shape != (4,):
            raise AssertionError("shard reload")
        meta = load_train_meta(directory)
        if meta["backbone"] != "vit_b_16":
            raise AssertionError(meta)
        try:
            bad = directory / "train_meta.json"
            bad.write_text(
                json.dumps({"rgb_only": False, "depth_used": False, "uses_rgb_test_gt": False}),
                encoding="utf-8",
            )
            load_train_meta(directory)
        except MassEstimatorError:
            pass
        else:
            raise AssertionError("non-rgb checkpoint was loaded")
