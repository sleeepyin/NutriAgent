"""Train one rerun model and predict with augmentation off."""

from __future__ import annotations

import hashlib
import random
import subprocess
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from digi_nutri.data import PlateDataset, collate_plates
from digi_nutri.loss import TaskLoss
from digi_nutri.models import build_model, count_parameters, parameter_groups
from digi_nutri.protocol import (
    ADAM_BETAS,
    BATCH_SIZE,
    EPOCHS,
    ETA_MIN,
    HEAD_LR,
    BACKBONE_LR,
    PARAM_LIMIT,
    SEED,
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


def require_cuda1(device: torch.device) -> None:
    index = 0 if device.index is None else int(device.index)
    if device.type != "cuda" or index != 1:
        raise SystemExit("this rerun uses only cuda:1")


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


def assert_gpu0_not_grown(baseline: int) -> None:
    used = gpu_memory_mib()[0]
    if used > baseline + 800:
        raise SystemExit(
            f"GPU 0 memory rose from {baseline} MiB to {used} MiB; leaving that process untouched"
        )


def forward_batch(model: torch.nn.Module, kind: str, batch: dict, device: torch.device) -> dict:
    rgb = batch["rgb"].to(device, non_blocking=True)
    if kind == "b1":
        return model(rgb)
    if kind == "t_geo":
        return model(
            rgb,
            batch["height"].to(device, non_blocking=True),
            batch["height01"].to(device, non_blocking=True),
        )
    if kind == "t_ing":
        return model(
            rgb,
            batch["name_ids"].to(device, non_blocking=True),
            batch["priors"].to(device, non_blocking=True),
            batch["ing_mask"].to(device, non_blocking=True),
        )
    raise ValueError(kind)


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


def build_from_config(kind: str, config: dict, pretrained: bool) -> torch.nn.Module:
    model = build_model(
        kind,
        init_mass=float(config["init_mass"]),
        init_density=list(config["init_density"]),
        vocab_size=int(config["vocab_size"]),
        pretrained=pretrained,
    )
    count = count_parameters(model)
    if count != int(config["param_counts"][kind]):
        raise RuntimeError(
            f"{kind} parameter count {count} != frozen config {config['param_counts'][kind]}"
        )
    if kind == "t_geo" and count >= PARAM_LIMIT:
        raise RuntimeError(f"T_geo has {count} parameters, limit is {PARAM_LIMIT}")
    return model


def _checkpoint_path(config: dict, kind: str) -> Path:
    return Path(config["output_dir"]) / "checkpoints" / f"{kind}.pt"


def _save_checkpoint(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".pt.tmp")
    torch.save(payload, temporary)
    temporary.replace(path)


def train_model(
    kind: str,
    config: dict,
    records: list[dict],
    device: torch.device,
    config_path: str | Path,
    gpu0_baseline: int | None = None,
) -> Path:
    require_cuda1(device)
    if int(config["epochs"]) != EPOCHS or int(config["batch_size"]) != BATCH_SIZE:
        raise RuntimeError("frozen budget does not match protocol.py")
    if float(config["head_lr"]) != HEAD_LR or float(config["backbone_lr"]) != BACKBONE_LR:
        raise RuntimeError("frozen learning rates do not match protocol.py")
    set_seed(int(config["seed"]))
    torch.backends.cudnn.benchmark = True
    dataset = PlateDataset(records, kind=kind, train=True, cache_dir=config["cache_dir"])
    loader = make_loader(dataset, int(config["batch_size"]), train=True, seed=int(config["seed"]))
    model = build_from_config(kind, config, pretrained=True).to(device)
    if gpu0_baseline is not None:
        assert_gpu0_not_grown(gpu0_baseline)
    loss_fn = TaskLoss(config["sigma_mean"], float(config["sigma_mass"])).to(device)
    optimizer = torch.optim.AdamW(
        parameter_groups(model, kind),
        weight_decay=WEIGHT_DECAY,
        betas=ADAM_BETAS,
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=int(config["epochs"]), eta_min=ETA_MIN
    )
    checkpoint_path = _checkpoint_path(config, kind)
    digest = config_digest(config_path)
    start = 0
    if checkpoint_path.exists():
        saved = torch.load(checkpoint_path, map_location=device, weights_only=False)
        if saved.get("config_sha256") != digest or saved.get("kind") != kind:
            raise RuntimeError(f"checkpoint for {kind} does not match the frozen config")
        model.load_state_dict(saved["model"])
        optimizer.load_state_dict(saved["optimizer"])
        scheduler.load_state_dict(saved["scheduler"])
        start = int(saved["epoch"])
        print(f"{kind} resume at epoch {start}", flush=True)
    if start >= int(config["epochs"]):
        print(f"{kind} already finished {start} epochs", flush=True)
        return checkpoint_path

    for epoch in range(start, int(config["epochs"])):
        model.train()
        total = 0.0
        seen = 0
        for batch in loader:
            optimizer.zero_grad(set_to_none=True)
            output = forward_batch(model, kind, batch, device)
            loss = loss_fn(
                output["mass"],
                output["density"],
                batch["mass"].to(device),
                batch["nutrition"].to(device),
            )
            if not torch.isfinite(loss):
                raise RuntimeError(f"{kind} non-finite loss at epoch {epoch + 1}")
            loss.backward()
            optimizer.step()
            total += float(loss.item()) * batch["rgb"].shape[0]
            seen += batch["rgb"].shape[0]
        scheduler.step()
        _save_checkpoint(
            checkpoint_path,
            {
                "epoch": epoch + 1,
                "kind": kind,
                "model": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "scheduler": scheduler.state_dict(),
                "config_sha256": digest,
            },
        )
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
    require_cuda1(device)
    dataset = PlateDataset(records, kind=kind, train=False, cache_dir=config["cache_dir"])
    loader = make_loader(dataset, int(config["batch_size"]), train=False, seed=int(config["seed"]))
    model = build_from_config(kind, config, pretrained=False).to(device)
    saved = torch.load(checkpoint, map_location=device, weights_only=False)
    if int(saved["epoch"]) != int(config["epochs"]):
        raise RuntimeError(f"{kind} checkpoint epoch is {saved['epoch']}, expected {config['epochs']}")
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
        output = forward_batch(model, kind, batch, device)
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
