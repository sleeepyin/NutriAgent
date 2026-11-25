"""Train and predict one Q0 model."""

from __future__ import annotations

import hashlib
import random
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from digi_nutri.data import PlateDataset, collate_plates
from digi_nutri.loss import TaskLoss
from digi_nutri.models import build_model, count_parameters
from digi_nutri.protocol import (
    ADAM_BETAS,
    EPOCHS,
    ETA_MIN,
    LEARNING_RATE,
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


def forward_batch(model: torch.nn.Module, kind: str, batch: dict, device: torch.device) -> dict:
    rgb = batch["rgb"].to(device, non_blocking=True)
    if kind == "b1":
        return model(rgb)
    if kind == "t_geo":
        return model(rgb, batch["height"].to(device, non_blocking=True))
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
        config["backbone"],
        pretrained=pretrained,
        init_mass=float(config["init_mass"]),
        init_density=list(config["init_density"]),
        vocab_size=int(config["vocab_size"]),
    )
    if kind == "t_geo" and count_parameters(model) >= PARAM_LIMIT:
        raise RuntimeError(f"T_geo has {count_parameters(model)} parameters, limit is {PARAM_LIMIT}")
    return model


def train_model(
    kind: str,
    config: dict,
    records: list[dict],
    device: torch.device,
    config_path: str | Path,
) -> Path:
    if int(config["epochs"]) != EPOCHS:
        raise RuntimeError("frozen epoch budget does not match protocol.py")
    set_seed(int(config["seed"]))
    torch.backends.cudnn.benchmark = True
    dataset = PlateDataset(
        records,
        kind=kind,
        train=True,
        rgb_root=config["rgb_root"],
        depth_root=config["depth_root"],
        cache_dir=config["cache_dir"],
    )
    loader = make_loader(dataset, int(config["batch_size"]), train=True, seed=int(config["seed"]))
    model = build_from_config(kind, config, pretrained=True).to(device)
    loss_fn = TaskLoss(config["sigma_mean"], float(config["sigma_mass"])).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=LEARNING_RATE,
        weight_decay=WEIGHT_DECAY,
        betas=ADAM_BETAS,
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=int(config["epochs"]), eta_min=ETA_MIN
    )
    checkpoint_path = Path(config["output_dir"]) / "checkpoints" / f"{kind}.pt"
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    digest = config_digest(config_path)
    start = 0
    if checkpoint_path.exists():
        saved = torch.load(checkpoint_path, map_location=device, weights_only=False)
        if saved.get("config_sha256") != digest:
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
            loss = loss_fn(output["mass"], output["density"], batch["mass"].to(device), batch["nutrition"].to(device))
            if not torch.isfinite(loss):
                raise RuntimeError(f"{kind} non-finite loss at epoch {epoch + 1}")
            loss.backward()
            optimizer.step()
            total += float(loss.item()) * batch["rgb"].shape[0]
            seen += batch["rgb"].shape[0]
        scheduler.step()
        torch.save(
            {
                "epoch": epoch + 1,
                "model": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "scheduler": scheduler.state_dict(),
                "config_sha256": digest,
            },
            checkpoint_path,
        )
        lr = optimizer.param_groups[0]["lr"]
        print(
            f"{kind} epoch {epoch + 1}/{config['epochs']} loss {total / max(seen, 1):.6f} lr {lr:.8f}",
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
    dataset = PlateDataset(
        records,
        kind=kind,
        train=False,
        rgb_root=config["rgb_root"],
        depth_root=config["depth_root"],
        cache_dir=config["cache_dir"],
    )
    loader = make_loader(dataset, int(config["batch_size"]), train=False, seed=int(config["seed"]))
    model = build_from_config(kind, config, pretrained=False).to(device)
    saved = torch.load(checkpoint, map_location=device, weights_only=False)
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
