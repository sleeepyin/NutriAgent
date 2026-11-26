"""Frozen L_task. Coefficients stay the same as the #91 same-setting run."""

from __future__ import annotations

import torch
import torch.nn.functional as F

from digi_nutri.protocol import (
    LOSS_DENSITY_WEIGHT,
    LOSS_MASS_WEIGHT,
    MASS_FLOOR_G,
    SMOOTH_L1_BETA,
)


class TaskLoss(torch.nn.Module):
    def __init__(self, sigma_mean: list[float], sigma_mass: float):
        super().__init__()
        sigma = torch.tensor(list(sigma_mean), dtype=torch.float32)
        if sigma.numel() != 4 or torch.any(sigma <= 0):
            raise ValueError("sigma_mean must be four positive train-set nutrient means")
        if sigma_mass <= 0:
            raise ValueError("sigma_M must be the positive train-set mean mass")
        self.register_buffer("sigma_mean", sigma)
        self.register_buffer("sigma_mass", torch.tensor(float(sigma_mass), dtype=torch.float32))

    def forward(
        self,
        mass_hat: torch.Tensor,
        density_hat: torch.Tensor,
        mass_gt: torch.Tensor,
        nutrition_gt: torch.Tensor,
    ) -> torch.Tensor:
        """L_task from predicted mass and predicted density.

        yhat is mass_hat * density_hat. The density target uses ground-truth
        mass and nutrition, with the mass denominator clipped at 5 g.
        """
        mass_hat = mass_hat.view(-1, 1)
        mass_gt = mass_gt.view(-1, 1).to(mass_hat.dtype)
        density_hat = density_hat.view(-1, 4)
        nutrition_gt = nutrition_gt.view(-1, 4).to(density_hat.dtype)
        yhat = mass_hat * density_hat

        nutrient = F.smooth_l1_loss(
            yhat, nutrition_gt, beta=SMOOTH_L1_BETA, reduction="none"
        )
        nutrient = (nutrient / self.sigma_mean.view(1, 4)).mean()

        mass_term = F.smooth_l1_loss(
            mass_hat, mass_gt, beta=SMOOTH_L1_BETA, reduction="none"
        )
        mass_term = (mass_term / self.sigma_mass).mean()

        denominator = torch.clamp(mass_gt, min=MASS_FLOOR_G)
        density_target = nutrition_gt / denominator
        density_term = F.smooth_l1_loss(
            density_hat, density_target, beta=SMOOTH_L1_BETA, reduction="none"
        ).mean()
        return nutrient + LOSS_MASS_WEIGHT * mass_term + LOSS_DENSITY_WEIGHT * density_term
