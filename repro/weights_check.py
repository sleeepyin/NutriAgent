"""Accept a Food2K classification ResNet-101. Reject a nutrition checkpoint."""

from __future__ import annotations

import hashlib
from pathlib import Path

import torch


def _flatten(blob: object) -> dict[str, torch.Tensor]:
    if isinstance(blob, dict) and "state_dict" in blob and isinstance(blob["state_dict"], dict):
        blob = blob["state_dict"]
    elif isinstance(blob, dict) and "model" in blob and isinstance(blob["model"], dict):
        blob = blob["model"]
    if not isinstance(blob, dict):
        raise RuntimeError("checkpoint is not a state dict")
    flat: dict[str, torch.Tensor] = {}
    for key, value in blob.items():
        if not torch.is_tensor(value):
            continue
        name = key
        for prefix in ("module.", "backbone.", "model.", "encoder."):
            if name.startswith(prefix):
                name = name[len(prefix) :]
        flat[name] = value
    return flat


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def inspect_food2k(path: str | Path, expected_sha256: str | None = None) -> dict:
    path = Path(path)
    actual_sha256 = file_sha256(path)
    if expected_sha256 is not None and actual_sha256 != expected_sha256:
        raise RuntimeError(
            f"Food2K sha256 {actual_sha256} != frozen {expected_sha256}. "
            "Fusion will not train, and ImageNet will not be substituted."
        )
    flat = _flatten(torch.load(path, map_location="cpu"))
    names = " ".join(flat)
    lowered = names.lower()
    for banned in ("calorie", "nutrition_head", "dpf", "protein_head"):
        if banned in lowered:
            raise RuntimeError(f"Food2K file looks like a nutrition checkpoint ({banned})")
    fc = None
    for key, value in flat.items():
        if key in ("fc.weight", "classifier.weight") or key.endswith(".fc.weight"):
            fc = value
            break
    conv1 = flat.get("conv1.weight")
    if fc is None or tuple(fc.shape) != (2000, 2048):
        found = None if fc is None else tuple(fc.shape)
        raise RuntimeError(f"Food2K fc.weight shape is {found}, required (2000, 2048)")
    if conv1 is None or tuple(conv1.shape) != (64, 3, 7, 7):
        found = None if conv1 is None else tuple(conv1.shape)
        raise RuntimeError(f"Food2K conv1.weight shape is {found}, required (64, 3, 7, 7)")
    usable = {key: value for key, value in flat.items() if not key.startswith("fc.")}
    return {
        "state": usable,
        "fc_shape": [2000, 2048],
        "conv1_shape": [64, 3, 7, 7],
        "tensors": len(usable),
        "sha256": actual_sha256,
    }
