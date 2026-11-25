"""Train Plan A T_geo and predict frozen teachers without training them."""

from __future__ import annotations

import hashlib
import random
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from digi_nutri.data import PlateDataset, collate_plates, ensure_cache
from digi_nutri.frozen_teachers import FrozenB1, FrozenTIng
from digi_nutri.loss import TaskLoss
from digi_nutri.models import build_plan_a, count_parameters, load_rgb_from_frozen_b1
from digi_nutri.protocol import (
    ADAM_BETAS,
    EPOCHS,
    ETA_MIN,
    GAMMA_INIT,
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


def build_for_kind(kind: str, config: dict, pretrained_height: bool) -> torch.nn.Module:
    init_mass = float(config["init_mass"])
    init_density = list(config["init_density"])
    if kind == "t_geo":
        model = build_plan_a(init_mass, init_density, pretrained_height=pretrained_height)
        if count_parameters(model) != int(config["param_counts"]["t_geo"]):
            raise RuntimeError("T_geo parameter count does not match the frozen config")
        if count_parameters(model) >= PARAM_LIMIT:
            raise RuntimeError("T_geo is at or above 60M parameters")
        return model
    if kind == "b1":
        return FrozenB1(init_mass, init_density)
    if kind == "t_ing":
        return FrozenTIng(init_mass, init_density, int(config["vocab_size"]))
    raise RuntimeError("Plan A trains T_geo only; B1 and T_ing are loaded, not trained")


def _checkpoint_file(kind: str, config: dict) -> Path:
    if kind == "t_geo":
        return Path(config["output_dir"]) / "checkpoints" / "t_geo.pt"
    if kind == "b1":
        return Path(config["b1_checkpoint"])
    if kind == "t_ing":
        return Path(config["t_ing_checkpoint"])
    raise RuntimeError(kind)


def _assert_epoch0_matches_b1(
    model: torch.nn.Module,
    config: dict,
    records: list[dict],
    device: torch.device,
) -> None:
    baseline = FrozenB1(float(config["init_mass"]), list(config["init_density"])).to(device)
    saved = torch.load(config["b1_checkpoint"], map_location=device, weights_only=False)
    baseline.load_state_dict(saved["model"])
    baseline.eval()
    model.eval()
    dataset = PlateDataset(
        records[:4],
        kind="t_geo",
        train=False,
        rgb_root=config["rgb_root"],
        depth_root=config["depth_root"],
        cache_dir=config["cache_dir"],
        exclude_zeros=bool(config["apply_d_table_zero_exclusion"]),
    )
    loader = make_loader(dataset, batch_size=4, train=False, seed=int(config["seed"]))
    batch = next(iter(loader))
    with torch.no_grad():
        left = forward_batch(model, "t_geo", batch, device)
        right = forward_batch(baseline, "b1", batch, device)
    if not torch.allclose(left["yhat"], right["yhat"], rtol=1e-4, atol=1e-4):
        delta = float((left["yhat"] - right["yhat"]).abs().max())
        raise RuntimeError(f"epoch-0 T_geo does not match frozen B1 (max abs {delta})")
    del baseline
    if device.type == "cuda":
        torch.cuda.empty_cache()


def train_model(
    config: dict,
    records: list[dict],
    device: torch.device,
    config_path: str | Path,
) -> Path:
    if int(config["epochs"]) != EPOCHS or float(config["lr"]) != LEARNING_RATE:
        raise RuntimeError("frozen budget does not match protocol.py")
    if abs(float(config["gamma_init"]) - float(GAMMA_INIT)) > 0:
        raise RuntimeError("gamma init is not 0")
    set_seed(int(config["seed"]))
    torch.backends.cudnn.benchmark = True
    ensure_cache(
        [record["dish_id"] for record in records],
        config["rgb_root"],
        config["depth_root"],
        config["cache_dir"],
        bool(config["apply_d_table_zero_exclusion"]),
    )
    dataset = PlateDataset(
        records,
        kind="t_geo",
        train=True,
        rgb_root=config["rgb_root"],
        depth_root=config["depth_root"],
        cache_dir=config["cache_dir"],
        exclude_zeros=bool(config["apply_d_table_zero_exclusion"]),
    )
    loader = make_loader(dataset, int(config["batch_size"]), train=True, seed=int(config["seed"]))
    model = build_for_kind("t_geo", config, pretrained_height=True)
    checkpoint_path = _checkpoint_file("t_geo", config)
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    digest = config_digest(config_path)
    start = 0
    saved = None
    if checkpoint_path.exists():
        saved = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        if saved.get("config_sha256") != digest:
            raise RuntimeError("checkpoint does not match the frozen config")
        model.load_state_dict(saved["model"])
        start = int(saved["epoch"])
    else:
        frozen = torch.load(config["b1_checkpoint"], map_location="cpu", weights_only=False)
        load_rgb_from_frozen_b1(model, frozen["model"])
        del frozen
    model.to(device)
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
    if saved is not None:
        optimizer.load_state_dict(saved["optimizer"])
        for state in optimizer.state.values():
            for key, value in state.items():
                if torch.is_tensor(value):
                    state[key] = value.to(device)
        scheduler.load_state_dict(saved["scheduler"])
        print(f"t_geo resume at epoch {start}", flush=True)
    else:
        _assert_epoch0_matches_b1(model, config, records, device)
        print("epoch-0 T_geo matches frozen B1", flush=True)
    if start >= int(config["epochs"]):
        print(f"t_geo already finished {start} epochs", flush=True)
        return checkpoint_path

    for epoch in range(start, int(config["epochs"])):
        model.train()
        total = 0.0
        seen = 0
        for batch in loader:
            optimizer.zero_grad(set_to_none=True)
            output = forward_batch(model, "t_geo", batch, device)
            loss = loss_fn(
                output["mass"],
                output["density"],
                batch["mass"].to(device),
                batch["nutrition"].to(device),
            )
            if not torch.isfinite(loss):
                raise RuntimeError(f"non-finite loss at epoch {epoch + 1}")
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
            f"t_geo epoch {epoch + 1}/{config['epochs']} loss {total / max(seen, 1):.6f} lr {lr:.8f}",
            flush=True,
        )
    return checkpoint_path


@torch.no_grad()
def predict(
    kind: str,
    config: dict,
    records: list[dict],
    device: torch.device,
) -> dict[str, np.ndarray]:
    if kind == "t_geo":
        ensure_cache(
            [record["dish_id"] for record in records],
            config["rgb_root"],
            config["depth_root"],
            config["cache_dir"],
            bool(config["apply_d_table_zero_exclusion"]),
        )
    dataset = PlateDataset(
        records,
        kind=kind,
        train=False,
        rgb_root=config["rgb_root"],
        depth_root=config["depth_root"],
        cache_dir=config["cache_dir"],
        exclude_zeros=bool(config["apply_d_table_zero_exclusion"]) if kind == "t_geo" else None,
    )
    loader = make_loader(dataset, int(config["batch_size"]), train=False, seed=int(config["seed"]))
    model = build_for_kind(kind, config, pretrained_height=False).to(device)
    checkpoint = _checkpoint_file(kind, config)
    if not checkpoint.exists():
        raise RuntimeError(f"missing checkpoint for {kind}")
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
