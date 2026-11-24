"""Warm up on the task loss, then turn distillation on linearly."""

from __future__ import annotations


def distillation_scale(epoch: int, warmup_epochs: int = 5, ramp_epochs: int = 5) -> float:
    """Epoch is zero-based.

    The first `warmup_epochs` epochs return 0. The next `ramp_epochs` epochs
    return k/ramp for k = 1..ramp, then 1.
    """

    if epoch < 0:
        raise ValueError("epoch must be >= 0")
    if epoch < warmup_epochs:
        return 0.0
    if ramp_epochs <= 0:
        return 1.0
    step = epoch - warmup_epochs + 1
    return float(min(1.0, step / ramp_epochs))
