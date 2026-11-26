"""Paper losses. No DiGI task loss and no Atwater term."""

from __future__ import annotations

import torch
import torch.nn as nn


def geometric_l1(pred: torch.Tensor, target: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    """Fifth root of the product of five batch-mean L1 terms."""
    losses = (pred - target).abs().mean(dim=0)
    return torch.exp(torch.log(losses + eps).mean())


def masked_l1(pred: torch.Tensor, target: torch.Tensor, valid: torch.Tensor) -> torch.Tensor:
    mask = valid.to(dtype=pred.dtype)
    denom = mask.sum().clamp_min(1.0)
    return ((pred - target).abs() * mask).sum() / denom


class UncertaintyLoss(nn.Module):
    """Swin-Nutrition equations (1) and (2), with a fixed training-set mean."""

    def __init__(self, train_mean: torch.Tensor):
        super().__init__()
        if train_mean.ndim != 1 or torch.any(train_mean <= 0):
            raise ValueError("training-set nutrient means must be positive")
        train_mean = train_mean.float()
        self.register_buffer("train_mean", train_mean)
        self.c = nn.Parameter(torch.ones(train_mean.numel(), device=train_mean.device, dtype=train_mean.dtype))

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        mae = (pred - target).abs().mean(dim=0)
        subtask = mae / self.train_mean
        c2 = self.c.square()
        return (subtask / (2.0 * c2 + 1e-8) + torch.log1p(c2)).sum()
