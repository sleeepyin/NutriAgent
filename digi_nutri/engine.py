"""Train B1 once, then T_multi once, and predict with augmentation off."""

from __future__ import annotations

import hashlib
import os
import random
import subprocess
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from digi_nutri.data import PlateDataset, collate_plates
from digi_nutri.loss import TaskLoss
from digi_nutri.models import (
    apply_trained_b1,
    assert_encoder_counts,
    build_b1,
    build_tmulti,
    count_parameters,
    parameter_groups,
)
from digi_nutri.protocol import (
    ADAM_BETAS,
    ATWATER_CARB,
    ATWATER_FAT,
    ATWATER_PRO,
    BACKBONE_LR,
    BATCH_SIZE,
    B1_PARAMS,
    EPOCHS,
    ETA_MIN,
    HEAD_LR,
    HOLDOUT_METRICS,
    PARAM_LIMIT,
    PHYSICAL_GPU,
    T_MULTI_PARAMS,
    WEIGHT_DECAY,
)


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def config_digest(path: str | Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def gpu_memory_mib() -> dict[int, int]:
    output = subprocess.check_output(
        ["nvidia-smi", "--query-gpu=index,memory.used", "--format=csv,noheader,nounits"],
        text=True,
    )
    found = {}
    for line in output.strip().splitlines():
        index, used = (part.strip() for part in line.split(","))
        found[int(index)] = int(used)
    return found


def require_physical_gpu1() -> torch.device:
    visible = os.environ.get("CUDA_VISIBLE_DEVICES", "")
    if visible != PHYSICAL_GPU:
        raise SystemExit(f"CUDA_VISIBLE_DEVICES must be {PHYSICAL_GPU} so GPU 0 stays untouched")
    if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise SystemExit("expected one visible CUDA device, the idle physical GPU 1")
    return torch.device("cuda:0")


def assert_gpu0_not_grown(baseline: int) -> None:
    used = gpu_memory_mib()[0]
    if used > baseline + 800:
        raise SystemExit(
            f"GPU 0 memory rose from {baseline} MiB to {used} MiB; leaving that process untouched"
        )


def assert_gpu1_idle(limit_mib: int = 500) -> dict[int, int]:
    memory = gpu_memory_mib()
    if memory.get(1, 1) > limit_mib:
        raise SystemExit(f"GPU 1 is not idle ({memory.get(1)} MiB); not starting")
    return memory


def assert_no_holdout() -> None:
    if Path(HOLDOUT_METRICS).exists():
        raise SystemExit("holdout metrics already exist; refusing another official evaluation")


def _seed_worker(worker_id: int) -> None:
    del worker_id
    worker_seed = torch.initial_seed() % 2**32
    random.seed(worker_seed)
    np.random.seed(worker_seed)


def make_loader(dataset: PlateDataset, batch_size: int, train: bool, seed: int) -> DataLoader:
    generator = torch.Generator()
    generator.manual_seed(seed)
    workers = 4 if train else 2
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=train,
        num_workers=workers,
        pin_memory=True,
        collate_fn=collate_plates,
        worker_init_fn=_seed_worker,
        generator=generator,
        persistent_workers=False,
    )


def _checkpoint_path(config: dict, kind: str) -> Path:
    return Path(config["output_dir"]) / "checkpoints" / f"{kind}.pt"


def _save_checkpoint(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".pt.tmp")
    torch.save(payload, temporary)
    temporary.replace(path)


def _check_budget(config: dict) -> None:
    if int(config["epochs"]) != EPOCHS or int(config["batch_size"]) != BATCH_SIZE:
        raise RuntimeError("frozen budget does not match protocol.py")
    if float(config["head_lr"]) != HEAD_LR or float(config["backbone_lr"]) != BACKBONE_LR:
        raise RuntimeError("frozen learning rates do not match protocol.py")
    if float(config["fusion_lr"]) != HEAD_LR:
        raise RuntimeError("fusion learning rate does not match the head learning rate")
    if float(config["residual_loss_weight"]) != 0.1:
        raise RuntimeError("residual coefficient is not the frozen 0.1")
    if config["rgb_backbone"] != "convnext_small" or config["height_backbone"] != "resnet18":
        raise RuntimeError("backbones do not match the frozen pair")
    if int(config["param_counts"]["t_multi"]) != T_MULTI_PARAMS:
        raise RuntimeError("config parameter count does not match the frozen constant")
    if int(config["param_counts"]["t_multi"]) >= PARAM_LIMIT:
        raise RuntimeError("refusing to train at or above 70M parameters")
    if int(config["param_counts"]["b1"]) != B1_PARAMS:
        raise RuntimeError("config B1 parameter count does not match the frozen constant")


def _forward(kind: str, model: torch.nn.Module, batch: dict, device: torch.device) -> dict:
    rgb = batch["rgb"].to(device, non_blocking=True)
    if kind == "b1":
        if "height" in batch:
            raise RuntimeError("B1 received a height tensor")
        return model(rgb)
    if kind != "t_multi":
        raise RuntimeError(f"unknown model {kind}")
    return model(
        rgb,
        batch["height"].to(device, non_blocking=True),
        batch["height01"].to(device, non_blocking=True),
        batch["name_ids"].to(device, non_blocking=True),
        batch["priors"].to(device, non_blocking=True),
        batch["ing_mask"].to(device, non_blocking=True),
    )


def _check_output(kind: str, output: dict) -> None:
    mass = output["mass"]
    density = output["density"]
    if mass.ndim != 2 or density.shape != (mass.shape[0], 4):
        raise RuntimeError("mass and density shapes left the product")
    if not torch.allclose(output["yhat"], mass * density, rtol=1e-4, atol=1e-4):
        raise RuntimeError("yhat is not predicted mass times predicted density")
    if kind == "b1":
        if "delta_cal" in output:
            raise RuntimeError("B1 must not emit a calorie residual")
        return
    fat = density[:, 1]
    carb = density[:, 2]
    protein = density[:, 3]
    expected = ATWATER_FAT * fat + ATWATER_CARB * carb + ATWATER_PRO * protein
    expected = expected + output["delta_cal"].reshape(-1)
    if not torch.allclose(density[:, 0], expected, rtol=1e-4, atol=1e-4):
        raise RuntimeError("energy density left the residual Atwater formula")


def _build(kind: str, config: dict, pretrained: bool):
    if kind == "b1":
        model = build_b1(
            float(config["init_mass"]),
            list(config["init_density"]),
            pretrained=pretrained,
            enforce_count=True,
        )
        if count_parameters(model) != B1_PARAMS:
            raise RuntimeError("B1 parameter count changed after the freeze")
        return model
    model = build_tmulti(
        float(config["init_mass"]),
        list(config["init_density"]),
        int(config["vocab_size"]),
        pretrained_height=pretrained,
        enforce_count=True,
    )
    count = count_parameters(model)
    if count != T_MULTI_PARAMS or count >= PARAM_LIMIT:
        raise RuntimeError(f"T_multi parameter count {count} is not the frozen under-70M count")
    assert_encoder_counts(model)
    return model


def _dataset(kind: str, records: list[dict], train: bool, config: dict) -> PlateDataset:
    return PlateDataset(
        records,
        train=train,
        cache_dir=config["cache_dir"],
        load_height=(kind == "t_multi"),
    )


def train_model(
    kind: str,
    config: dict,
    records: list[dict],
    device: torch.device,
    config_path: str | Path,
    gpu0_baseline: int,
    smoke_steps: int = 0,
) -> Path | None:
    if kind not in ("b1", "t_multi"):
        raise RuntimeError(f"this step trains only b1 and t_multi, not {kind}")
    assert_no_holdout()
    _check_budget(config)
    set_seed(int(config["seed"]))
    torch.backends.cudnn.benchmark = True
    dataset = _dataset(kind, records, True, config)
    loader = make_loader(dataset, int(config["batch_size"]), train=True, seed=int(config["seed"]))
    checkpoint_path = _checkpoint_path(config, kind)
    digest = config_digest(config_path)
    loss_fn = TaskLoss(config["sigma_mean"], float(config["sigma_mass"]))
    resumed = False
    start = 0
    b1_sha = None
    if smoke_steps > 0:
        # Memory check only. Do not read or write the official checkpoint.
        model = _build(kind, config, pretrained=True)
    elif checkpoint_path.exists():
        model = _build(kind, config, pretrained=False)
        saved = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        if saved.get("config_sha256") != digest or saved.get("kind") != kind:
            raise RuntimeError(f"checkpoint does not match the frozen {kind} config")
        model.load_state_dict(saved["model"])
        start = int(saved["epoch"])
        resumed = True
        b1_sha = saved.get("b1_sha256")
        print(f"{kind} resume at epoch {start}", flush=True)
        del saved
    else:
        model = _build(kind, config, pretrained=(kind == "b1" or kind == "t_multi"))
        if kind == "t_multi":
            b1_path = _checkpoint_path(config, "b1")
            if not b1_path.exists():
                raise RuntimeError("train B1 to 60 epochs before starting T_multi")
            saved_b1 = torch.load(b1_path, map_location="cpu", weights_only=False)
            if saved_b1.get("kind") != "b1" or int(saved_b1["epoch"]) != EPOCHS:
                raise RuntimeError("T_multi can only start from this round's finished B1")
            if saved_b1.get("config_sha256") != digest:
                raise RuntimeError("B1 checkpoint does not match the frozen config")
            apply_trained_b1(model, saved_b1["model"])
            if not model.b1_initialized:
                raise RuntimeError("T_multi was not initialized from this round's B1")
            b1_sha = sha256_file(b1_path)
            del saved_b1
            print("initialized T_multi RGB encoder and regression heads from this round's B1", flush=True)
    if kind == "t_multi" and count_parameters(model) >= PARAM_LIMIT:
        raise RuntimeError("refusing to train T_multi at or above 70M parameters")
    model = model.to(device)
    loss_fn = loss_fn.to(device)
    assert_gpu0_not_grown(gpu0_baseline)
    optimizer = torch.optim.AdamW(
        parameter_groups(model, kind),
        weight_decay=WEIGHT_DECAY,
        betas=ADAM_BETAS,
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=int(config["epochs"]), eta_min=ETA_MIN
    )
    if resumed:
        saved = torch.load(checkpoint_path, map_location=device, weights_only=False)
        optimizer.load_state_dict(saved["optimizer"])
        scheduler.load_state_dict(saved["scheduler"])
        del saved
    if smoke_steps <= 0 and start >= int(config["epochs"]):
        print(f"{kind} already finished {start} epochs", flush=True)
        return checkpoint_path

    steps_left = smoke_steps if smoke_steps > 0 else None
    epoch_end = 1 if smoke_steps > 0 else int(config["epochs"])
    for epoch in range(start, epoch_end):
        model.train()
        total = 0.0
        seen = 0
        checked = False
        for batch in loader:
            optimizer.zero_grad(set_to_none=True)
            output = _forward(kind, model, batch, device)
            if not checked:
                _check_output(kind, output)
                checked = True
            loss = loss_fn(
                output["mass"],
                output["density"],
                batch["mass"].to(device, non_blocking=True),
                batch["nutrition"].to(device, non_blocking=True),
                delta_cal=None if kind == "b1" else output["delta_cal"],
            )
            if not torch.isfinite(loss):
                raise RuntimeError(f"{kind} non-finite loss at epoch {epoch + 1}")
            loss.backward()
            optimizer.step()
            total += float(loss.item()) * batch["rgb"].shape[0]
            seen += batch["rgb"].shape[0]
            if steps_left is not None:
                steps_left -= 1
                if steps_left <= 0:
                    print(f"{kind} smoke steps done loss {total / max(seen, 1):.6f}", flush=True)
                    return None
        scheduler.step()
        assert_gpu0_not_grown(gpu0_baseline)
        if smoke_steps > 0:
            continue
        payload = {
            "epoch": epoch + 1,
            "kind": kind,
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(),
            "config_sha256": digest,
        }
        if b1_sha is not None:
            payload["b1_sha256"] = b1_sha
        _save_checkpoint(checkpoint_path, payload)
        lrs = [group["lr"] for group in optimizer.param_groups]
        print(
            f"{kind} epoch {epoch + 1}/{config['epochs']} loss {total / max(seen, 1):.6f} "
            f"lr_backbone {lrs[0]:.8f} lr_head {lrs[1]:.8f}",
            flush=True,
        )
    return checkpoint_path


@torch.no_grad()
def predict(
    kind: str,
    config: dict,
    records: list[dict],
    checkpoint: str | Path,
    device: torch.device,
) -> dict[str, np.ndarray]:
    if kind not in ("b1", "t_multi"):
        raise RuntimeError(f"this step predicts only b1 and t_multi, not {kind}")
    dataset = _dataset(kind, records, False, config)
    loader = make_loader(dataset, int(config["batch_size"]), train=False, seed=int(config["seed"]))
    model = _build(kind, config, pretrained=False).to(device)
    saved = torch.load(checkpoint, map_location=device, weights_only=False)
    if saved.get("kind") != kind:
        raise RuntimeError(f"prediction checkpoint is not {kind}")
    if int(saved["epoch"]) != int(config["epochs"]):
        raise RuntimeError(f"checkpoint epoch is {saved['epoch']}, expected {config['epochs']}")
    model.load_state_dict(saved["model"])
    model.eval()
    dish_ids: list[str] = []
    groups: list[str] = []
    yhat = []
    mass = []
    density = []
    nutrition = []
    mass_gt = []
    delta = []
    for batch in loader:
        output = _forward(kind, model, batch, device)
        _check_output(kind, output)
        dish_ids.extend(batch["dish_id"])
        groups.extend(batch["group_id"])
        yhat.append(output["yhat"].detach().cpu().numpy())
        mass.append(output["mass"].detach().cpu().numpy().reshape(-1))
        density.append(output["density"].detach().cpu().numpy())
        nutrition.append(batch["nutrition"].numpy())
        mass_gt.append(batch["mass"].numpy().reshape(-1))
        if kind == "t_multi":
            delta.append(output["delta_cal"].detach().cpu().numpy().reshape(-1))
    packed = {
        "dish_id": np.asarray(dish_ids),
        "group_id": np.asarray(groups),
        "yhat": np.concatenate(yhat, axis=0),
        "mass_hat": np.concatenate(mass, axis=0),
        "density_hat": np.concatenate(density, axis=0),
        "nutrition": np.concatenate(nutrition, axis=0),
        "mass": np.concatenate(mass_gt, axis=0),
    }
    product = packed["mass_hat"][:, None] * packed["density_hat"]
    if not np.allclose(packed["yhat"], product, rtol=1e-4, atol=1e-4):
        raise RuntimeError("yhat is not predicted mass times predicted density")
    if kind == "t_multi":
        packed["delta_cal"] = np.concatenate(delta, axis=0)
        energy = packed["density_hat"][:, 0]
        macros = packed["density_hat"][:, 1:]
        expected = (
            ATWATER_FAT * macros[:, 0]
            + ATWATER_CARB * macros[:, 1]
            + ATWATER_PRO * macros[:, 2]
            + packed["delta_cal"]
        )
        if not np.allclose(energy, expected, rtol=1e-4, atol=1e-4):
            raise RuntimeError("saved energy density left the residual Atwater formula")
    return packed
