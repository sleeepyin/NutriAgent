"""Read-only identity checks for the frozen #91 same-setting B1."""

from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path

import numpy as np
import torch

from digi_nutri.metrics import mae
from digi_nutri.models import TMulti, apply_frozen_b1, count_parameters
from digi_nutri.protocol import (
    EXPECTED_TEST_N,
    FROZEN_B1_BYTES,
    FROZEN_B1_CHECKPOINT,
    FROZEN_B1_CONFIG_SHA256,
    FROZEN_B1_EPOCH,
    FROZEN_B1_HOLDOUT_MAE,
    FROZEN_B1_HOLDOUT_MAE_ROUNDED,
    FROZEN_B1_PARAMS,
    FROZEN_B1_PLATE_GROUPS,
    FROZEN_B1_PREDICTIONS,
    FROZEN_B1_SHA256,
    FROZEN_B1_TRAIN_MAE,
    FROZEN_B1_TRAIN_METRICS,
    NUTRIENTS,
    RGB_ENCODER_PARAMS,
)


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _rounded(value: float) -> float:
    return round(float(value) + 0.0, 4)


def assert_mae_block(found: dict[str, float], expected: dict[str, float], rounded: dict[str, float], label: str) -> None:
    for name in list(NUTRIENTS) + ["mass"]:
        got = float(found[name])
        if abs(got - float(expected[name])) > 1e-6:
            raise RuntimeError(f"{label} {name} MAE {got} != frozen {expected[name]}")
        if name in rounded and _rounded(got) != float(rounded[name]):
            raise RuntimeError(f"{label} {name} does not round to {rounded[name]}")


def verify_checkpoint_file() -> None:
    path = Path(FROZEN_B1_CHECKPOINT)
    if path.name != "b1.pt":
        raise RuntimeError("refusing a checkpoint whose file name is not b1.pt")
    if not path.is_file():
        raise RuntimeError(f"frozen #91 B1 checkpoint is missing: {path}")
    if path.stat().st_size != FROZEN_B1_BYTES:
        raise RuntimeError(f"frozen B1 size {path.stat().st_size} != {FROZEN_B1_BYTES}")
    found = sha256_file(path)
    if found != FROZEN_B1_SHA256:
        raise RuntimeError("frozen B1 sha256 does not match the #91 same-setting checkpoint")


def load_b1_weights() -> dict[str, torch.Tensor]:
    verify_checkpoint_file()
    saved = torch.load(FROZEN_B1_CHECKPOINT, map_location="cpu", weights_only=False)
    if saved.get("kind") != "b1":
        raise RuntimeError(f"checkpoint kind is {saved.get('kind')}, expected b1")
    if int(saved.get("epoch", -1)) != FROZEN_B1_EPOCH:
        raise RuntimeError(f"checkpoint epoch is {saved.get('epoch')}, expected {FROZEN_B1_EPOCH}")
    if saved.get("config_sha256") != FROZEN_B1_CONFIG_SHA256:
        raise RuntimeError("checkpoint config hash is not the #91 same-setting run")
    state = saved["model"]
    prefixes = {key.split(".", 1)[0] for key in state}
    if prefixes != {"backbone", "mass_head", "density_head"}:
        raise RuntimeError(f"refusing B1 state with prefixes {sorted(prefixes)}")
    parameter_numel = 0
    for key, value in state.items():
        if "running_mean" in key or "running_var" in key or key.endswith("num_batches_tracked"):
            continue
        parameter_numel += int(value.numel())
    if parameter_numel != FROZEN_B1_PARAMS:
        raise RuntimeError(f"B1 parameter numel {parameter_numel} != {FROZEN_B1_PARAMS}")
    return state


def initialize_from_frozen_b1(model: TMulti) -> str:
    state = load_b1_weights()
    before_height = {key: value.detach().clone() for key, value in model.height_encoder.state_dict().items()}
    before_encoder = {key: value.detach().clone() for key, value in model.encoder.state_dict().items()}
    before_attention = {key: value.detach().clone() for key, value in model.attention.state_dict().items()}
    before_cab = {key: value.detach().clone() for key, value in model.cab.state_dict().items()}
    apply_frozen_b1(model, state)
    if count_parameters(model.rgb_encoder) != RGB_ENCODER_PARAMS:
        raise RuntimeError("RGB encoder parameter count changed")
    for name, value in model.height_encoder.state_dict().items():
        if not torch.equal(value, before_height[name]):
            raise RuntimeError("height encoder was overwritten by the B1 checkpoint")
    for name, value in model.encoder.state_dict().items():
        if not torch.equal(value, before_encoder[name]):
            raise RuntimeError("ingredient encoder was overwritten by the B1 checkpoint")
    for name, value in model.attention.state_dict().items():
        if not torch.equal(value, before_attention[name]):
            raise RuntimeError("ingredient attention was overwritten by the B1 checkpoint")
    for name, value in model.cab.state_dict().items():
        if not torch.equal(value, before_cab[name]):
            raise RuntimeError("CAB was overwritten by the B1 checkpoint")
    volume = model.mass_head.fc1.weight[:, model.rgb_encoder.out_channels :]
    if torch.count_nonzero(volume).item() != 0:
        raise RuntimeError("volume columns of the mass head are not zero")
    return FROZEN_B1_SHA256


def load_frozen_holdout_rows() -> dict[str, np.ndarray]:
    path = Path(FROZEN_B1_PREDICTIONS)
    if not path.is_file():
        raise RuntimeError(f"frozen #91 holdout predictions are missing: {path}")
    dish_ids = []
    groups = []
    mass = []
    nutrition = []
    mass_hat = []
    yhat = []
    with path.open(newline="") as handle:
        reader = csv.DictReader(handle)
        required = {
            "dish_id",
            "group_id",
            "mass",
            "energy_gt",
            "fat_gt",
            "carb_gt",
            "protein_gt",
            "b1_mass_hat",
            "b1_energy_hat",
            "b1_fat_hat",
            "b1_carb_hat",
            "b1_protein_hat",
        }
        if reader.fieldnames is None or not required.issubset(set(reader.fieldnames)):
            raise RuntimeError("frozen holdout predictions are missing B1 columns")
        for row in reader:
            dish_ids.append(row["dish_id"])
            groups.append(row["group_id"])
            mass.append(float(row["mass"]))
            nutrition.append([float(row[f"{name}_gt"]) for name in NUTRIENTS])
            mass_hat.append(float(row["b1_mass_hat"]))
            yhat.append([float(row[f"b1_{name}_hat"]) for name in NUTRIENTS])
    if len(dish_ids) != EXPECTED_TEST_N:
        raise RuntimeError(f"frozen holdout file has {len(dish_ids)} rows, expected {EXPECTED_TEST_N}")
    packed = {
        "dish_id": np.asarray(dish_ids),
        "group_id": np.asarray(groups),
        "mass": np.asarray(mass, dtype=np.float64),
        "nutrition": np.asarray(nutrition, dtype=np.float64),
        "mass_hat": np.asarray(mass_hat, dtype=np.float64),
        "yhat": np.asarray(yhat, dtype=np.float64),
    }
    found = {name: mae(packed["yhat"][:, index], packed["nutrition"][:, index]) for index, name in enumerate(NUTRIENTS)}
    found["mass"] = mae(packed["mass_hat"], packed["mass"])
    assert_mae_block(found, FROZEN_B1_HOLDOUT_MAE, FROZEN_B1_HOLDOUT_MAE_ROUNDED, "holdout")
    return packed


def verify_frozen_train_mae() -> dict[str, float]:
    path = Path(FROZEN_B1_TRAIN_METRICS)
    payload = json.loads(path.read_text())
    if payload.get("config_sha256") != FROZEN_B1_CONFIG_SHA256:
        raise RuntimeError("frozen train metrics are not from the #91 same-setting run")
    if int(payload.get("n", -1)) != 2754:
        raise RuntimeError("frozen train metrics are not the 2754-plate eval")
    block = payload["metrics"]["b1"]
    found = {name: float(block[name]["mae"]) for name in list(NUTRIENTS) + ["mass"]}
    for name, expected in FROZEN_B1_TRAIN_MAE.items():
        if abs(found[name] - float(expected)) > 1e-9:
            raise RuntimeError(f"frozen train {name} MAE {found[name]} != {expected}")
    return found


def verify_saved_plate_groups(groups: dict[str, str]) -> None:
    saved = json.loads(Path(FROZEN_B1_PLATE_GROUPS).read_text())
    if saved != groups:
        raise RuntimeError("recomputed test plate groups do not match the frozen #91 map")
