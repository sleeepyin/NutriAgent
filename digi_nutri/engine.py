"""Train T_multi once and predict with augmentation off."""

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
from digi_nutri.frozen_b1 import initialize_from_frozen_b1
from digi_nutri.loss import TaskLoss
from digi_nutri.models import build_model, count_parameters, parameter_groups
from digi_nutri.protocol import (
    ADAM_BETAS,
    BACKBONE_LR,
    BATCH_SIZE,
    EPOCHS,
    ETA_MIN,
    HEAD_LR,
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


def forward_batch(model: torch.nn.Module, batch: dict, device: torch.device) -> dict:
    return model(
        batch["rgb"].to(device, non_blocking=True),
        batch["height"].to(device, non_blocking=True),
        batch["height01"].to(device, non_blocking=True),
        batch["name_ids"].to(device, non_blocking=True),
        batch["priors"].to(device, non_blocking=True),
        batch["ing_mask"].to(device, non_blocking=True),
    )


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


def build_from_config(config: dict, pretrained_height: bool) -> torch.nn.Module:
    model = build_model(
        init_mass=float(config["init_mass"]),
        init_density=list(config["init_density"]),
        vocab_size=int(config["vocab_size"]),
        pretrained_height=pretrained_height,
    )
    count = count_parameters(model)
    if count != int(config["param_counts"]["t_multi"]) or count != T_MULTI_PARAMS:
        raise RuntimeError(f"T_multi parameter count {count} does not match the frozen config")
    if count >= PARAM_LIMIT:
        raise RuntimeError(f"T_multi has {count} parameters, limit is {PARAM_LIMIT}")
    return model


def _checkpoint_path(config: dict) -> Path:
    return Path(config["output_dir"]) / "checkpoints" / "t_multi.pt"


def _save_checkpoint(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".pt.tmp")
    torch.save(payload, temporary)
    temporary.replace(path)


def train_model(
    config: dict,
    records: list[dict],
    device: torch.device,
    config_path: str | Path,
    gpu0_baseline: int,
    smoke_steps: int = 0,
) -> Path:
    if int(config["epochs"]) != EPOCHS or int(config["batch_size"]) != BATCH_SIZE:
        raise RuntimeError("frozen budget does not match protocol.py")
    if float(config["head_lr"]) != HEAD_LR or float(config["backbone_lr"]) != BACKBONE_LR:
        raise RuntimeError("frozen learning rates do not match protocol.py")
    if float(config["fusion_lr"]) != HEAD_LR:
        raise RuntimeError("fusion learning rate does not match the head learning rate")
    set_seed(int(config["seed"]))
    torch.backends.cudnn.benchmark = True
    dataset = PlateDataset(records, train=True, cache_dir=config["cache_dir"])
    loader = make_loader(dataset, int(config["batch_size"]), train=True, seed=int(config["seed"]))
    model = build_from_config(config, pretrained_height=True)
    checkpoint_path = _checkpoint_path(config)
    digest = config_digest(config_path)
    loss_fn = TaskLoss(config["sigma_mean"], float(config["sigma_mass"]))
    resumed = False
    start = 0
    if smoke_steps <= 0 and checkpoint_path.exists():
        saved = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        if saved.get("config_sha256") != digest or saved.get("kind") != "t_multi":
            raise RuntimeError("checkpoint does not match the frozen T_multi config")
        model.load_state_dict(saved["model"])
        start = int(saved["epoch"])
        resumed = True
        print(f"t_multi resume at epoch {start}", flush=True)
    else:
        b1_sha = initialize_from_frozen_b1(model)
        if b1_sha != config["frozen_b1_sha256"] or not model.b1_initialized:
            raise RuntimeError("T_multi was not initialized from the frozen #91 B1")
        print("initialized RGB encoder and regression heads from frozen #91 B1", flush=True)
    if count_parameters(model) >= PARAM_LIMIT:
        raise RuntimeError("refusing to train at or above 60M parameters")
    model = model.to(device)
    loss_fn = loss_fn.to(device)
    assert_gpu0_not_grown(gpu0_baseline)
    optimizer = torch.optim.AdamW(
        parameter_groups(model),
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
        print(f"t_multi already finished {start} epochs", flush=True)
        return checkpoint_path

    steps_left = smoke_steps if smoke_steps > 0 else None
    epoch_end = 1 if smoke_steps > 0 else int(config["epochs"])
    for epoch in range(start, epoch_end):
        model.train()
        total = 0.0
        seen = 0
        for batch in loader:
            optimizer.zero_grad(set_to_none=True)
            output = forward_batch(model, batch, device)
            loss = loss_fn(
                output["mass"],
                output["density"],
                batch["mass"].to(device),
                batch["nutrition"].to(device),
            )
            if not torch.isfinite(loss):
                raise RuntimeError(f"t_multi non-finite loss at epoch {epoch + 1}")
            loss.backward()
            optimizer.step()
            total += float(loss.item()) * batch["rgb"].shape[0]
            seen += batch["rgb"].shape[0]
            if steps_left is not None:
                steps_left -= 1
                if steps_left <= 0:
                    print(f"t_multi smoke steps done loss {total / max(seen, 1):.6f}", flush=True)
                    return checkpoint_path
        scheduler.step()
        if smoke_steps > 0:
            continue
        _save_checkpoint(
            checkpoint_path,
            {
                "epoch": epoch + 1,
                "kind": "t_multi",
                "model": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "scheduler": scheduler.state_dict(),
                "config_sha256": digest,
                "init_b1_sha256": config["frozen_b1_sha256"],
            },
        )
        lrs = [group["lr"] for group in optimizer.param_groups]
        print(
            f"t_multi epoch {epoch + 1}/{config['epochs']} loss {total / max(seen, 1):.6f} "
            f"lr_backbone {lrs[0]:.8f} lr_head {lrs[1]:.8f}",
            flush=True,
        )
    return checkpoint_path


@torch.no_grad()
def predict(
    config: dict,
    records: list[dict],
    checkpoint: str | Path,
    device: torch.device,
) -> dict[str, np.ndarray]:
    dataset = PlateDataset(records, train=False, cache_dir=config["cache_dir"])
    loader = make_loader(dataset, int(config["batch_size"]), train=False, seed=int(config["seed"]))
    model = build_from_config(config, pretrained_height=False).to(device)
    saved = torch.load(checkpoint, map_location=device, weights_only=False)
    if saved.get("kind") != "t_multi":
        raise RuntimeError("prediction checkpoint is not t_multi")
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
    for batch in loader:
        output = forward_batch(model, batch, device)
        dish_ids.extend(batch["dish_id"])
        groups.extend(batch["group_id"])
        yhat.append(output["yhat"].detach().cpu().numpy())
        mass.append(output["mass"].detach().cpu().numpy().reshape(-1))
        density.append(output["density"].detach().cpu().numpy())
        nutrition.append(batch["nutrition"].numpy())
        mass_gt.append(batch["mass"].numpy().reshape(-1))
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
    return packed
