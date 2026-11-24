"""Train teachers to a fixed budget, then students. The holdout is not read here."""

from __future__ import annotations

import logging
import random

import numpy as np
import torch

from digi_nutri.losses import (
    LossWeights,
    aux_bce_loss,
    geo_affinity_loss,
    phase1_total_loss,
    semantic_distill_loss,
    task_loss,
)
from digi_nutri.protocol import needs_geo, needs_sem
from digi_nutri.schedule import distillation_scale

LOGGER = logging.getLogger("digi_nutri")
_FORBIDDEN = {
    "grams",
    "ingredient_calories",
    "ingredient_fat",
    "ingredient_carb",
    "ingredient_protein",
    "names",
}


def seed_all(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def freeze_module(module: torch.nn.Module) -> None:
    module.eval()
    for parameter in module.parameters():
        parameter.requires_grad_(False)


def _reject_forbidden(batch: dict) -> None:
    found = _FORBIDDEN.intersection(batch)
    if found:
        raise RuntimeError(
            f"batch contains fields that must not enter the step: {sorted(found)}"
        )


def _optimizer(module: torch.nn.Module, lr: float, weight_decay: float):
    return torch.optim.AdamW(
        (parameter for parameter in module.parameters() if parameter.requires_grad),
        lr=lr,
        weight_decay=weight_decay,
    )


def _scheduler(optimizer, epochs: int, eta_min: float):
    return torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=epochs, eta_min=eta_min
    )


def train_one_epoch_teacher(
    model: torch.nn.Module,
    batches,
    optimizer,
    device: torch.device,
    target_mean: torch.Tensor,
    target_std: torch.Tensor,
    kind: str,
    eps: float,
) -> dict:
    if kind not in {"geometry", "ingredient"}:
        raise ValueError(kind)
    model.train()
    loss_sum = 0.0
    seen = 0
    absolute = torch.zeros(5, dtype=torch.float64)
    for batch in batches:
        _reject_forbidden(batch)
        optimizer.zero_grad(set_to_none=True)
        rgb = batch["rgb"].to(device, non_blocking=True)
        target = batch["y"].to(device, non_blocking=True)
        if kind == "geometry":
            depth = batch["depth"].to(device, non_blocking=True)
            pred = model(rgb, depth)["nutrients"].float()
        else:
            pred = model(
                rgb,
                batch["name_emb"].to(device, non_blocking=True),
                batch["name_mask"].to(device, non_blocking=True),
            )["nutrients"].float()
        loss, _parts = task_loss(pred, target, target_mean, target_std, eps=eps)
        if not torch.isfinite(loss):
            raise RuntimeError(f"non-finite teacher loss ({kind})")
        loss.backward()
        optimizer.step()
        count = int(target.shape[0])
        seen += count
        loss_sum += float(loss.detach()) * count
        absolute += (pred.detach() - target).abs().sum(dim=0).double().cpu()
    if seen == 0:
        raise RuntimeError("teacher loader was empty")
    return {"loss": loss_sum / seen, "mae": (absolute / seen).tolist(), "seen": seen}


def train_one_epoch_student(
    model: torch.nn.Module,
    batches,
    optimizer,
    device: torch.device,
    variant: str,
    scale: float,
    weights: LossWeights,
    target_mean: torch.Tensor,
    target_std: torch.Tensor,
    eps: float,
    geo_teacher: torch.nn.Module | None = None,
    ing_teacher: torch.nn.Module | None = None,
) -> dict:
    model.train()
    if geo_teacher is not None:
        geo_teacher.eval()
    if ing_teacher is not None:
        ing_teacher.eval()
    loss_sum = 0.0
    task_sum = 0.0
    seen = 0
    absolute = torch.zeros(5, dtype=torch.float64)
    geo_calls = 0
    sem_calls = 0
    for batch in batches:
        _reject_forbidden(batch)
        optimizer.zero_grad(set_to_none=True)
        rgb = batch["rgb"].to(device, non_blocking=True)
        target = batch["y"].to(device, non_blocking=True)
        out = model.forward_train(rgb)
        pred = out["nutrients"].float()
        task, _parts = task_loss(pred, target, target_mean, target_std, eps=eps)
        geo = sem = aux = None
        if scale > 0.0 and needs_geo(variant):
            if geo_teacher is None or "depth" not in batch:
                raise RuntimeError("geometry distillation requires the frozen teacher and depth")
            with torch.no_grad():
                teacher = geo_teacher(rgb, batch["depth"].to(device, non_blocking=True))
            geo_calls += 1
            geo = geo_affinity_loss(
                [feat.float() for feat in out["features"]],
                [feat.float() for feat in teacher["features"]],
            )
        if scale > 0.0 and needs_sem(variant):
            if ing_teacher is None or "name_emb" not in batch or "multihot" not in batch:
                raise RuntimeError("ingredient distillation requires the frozen teacher and names")
            if "aux_logits" not in out:
                raise RuntimeError("L_aux requires the train-only ingredient head")
            with torch.no_grad():
                teacher = ing_teacher(
                    rgb,
                    batch["name_emb"].to(device, non_blocking=True),
                    batch["name_mask"].to(device, non_blocking=True),
                )
            sem_calls += 1
            sem = semantic_distill_loss(out["fused"].float(), teacher["fused"].float())
            aux = aux_bce_loss(
                out["aux_logits"], batch["multihot"].to(device, non_blocking=True)
            )
        loss = phase1_total_loss(task, geo, sem, aux, scale, variant, weights)
        if not torch.isfinite(loss):
            raise RuntimeError(f"non-finite student loss ({variant})")
        loss.backward()
        optimizer.step()
        count = int(target.shape[0])
        seen += count
        loss_sum += float(loss.detach()) * count
        task_sum += float(task.detach()) * count
        absolute += (pred.detach() - target).abs().sum(dim=0).double().cpu()
    if seen == 0:
        raise RuntimeError("student loader was empty")
    return {
        "loss": loss_sum / seen,
        "task": task_sum / seen,
        "mae": (absolute / seen).tolist(),
        "seen": seen,
        "scale": scale,
        "geo_calls": geo_calls,
        "sem_calls": sem_calls,
    }


def _brief(stats: dict) -> str:
    mae = ", ".join(f"{value:.3f}" for value in stats["mae"])
    return f"loss={stats['loss']:.4f} mae=[{mae}] lr={stats['lr']:.6g}"


def _fit_loop(run_epoch, optimizer, scheduler, epochs: int, describe) -> list[dict]:
    history = []
    for epoch in range(epochs):
        stats = run_epoch(epoch)
        stats["epoch"] = epoch
        stats["lr"] = float(optimizer.param_groups[0]["lr"])
        scheduler.step()
        history.append(stats)
        LOGGER.info("%s epoch %s/%s %s", describe, epoch + 1, epochs, _brief(stats))
    return history


def train_teacher(
    model,
    loader,
    device,
    epochs: int,
    lr: float,
    weight_decay: float,
    eta_min: float,
    target_mean: torch.Tensor,
    target_std: torch.Tensor,
    kind: str,
    eps: float,
) -> list[dict]:
    optimizer = _optimizer(model, lr, weight_decay)
    scheduler = _scheduler(optimizer, epochs, eta_min)
    mean = target_mean.to(device=device, dtype=torch.float32)
    std = target_std.to(device=device, dtype=torch.float32)

    def run_epoch(_epoch: int) -> dict:
        return train_one_epoch_teacher(model, loader, optimizer, device, mean, std, kind, eps)

    return _fit_loop(run_epoch, optimizer, scheduler, epochs, kind)


def train_student(
    model,
    loader,
    device,
    epochs: int,
    lr: float,
    weight_decay: float,
    eta_min: float,
    target_mean: torch.Tensor,
    target_std: torch.Tensor,
    variant: str,
    weights: LossWeights,
    warmup_epochs: int,
    ramp_epochs: int,
    eps: float,
    geo_teacher=None,
    ing_teacher=None,
) -> list[dict]:
    optimizer = _optimizer(model, lr, weight_decay)
    scheduler = _scheduler(optimizer, epochs, eta_min)
    mean = target_mean.to(device=device, dtype=torch.float32)
    std = target_std.to(device=device, dtype=torch.float32)

    def run_epoch(epoch: int) -> dict:
        scale = distillation_scale(epoch, warmup_epochs, ramp_epochs)
        return train_one_epoch_student(
            model,
            loader,
            optimizer,
            device,
            variant,
            scale,
            weights,
            mean,
            std,
            eps,
            geo_teacher=geo_teacher,
            ing_teacher=ing_teacher,
        )

    return _fit_loop(run_epoch, optimizer, scheduler, epochs, variant)
