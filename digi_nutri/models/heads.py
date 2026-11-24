"""Multi-scale pool, then add or concat. Five nutrient outputs."""

from __future__ import annotations

import torch
import torch.nn as nn


class MultiScaleAddHead(nn.Module):
    def __init__(
        self,
        channels: tuple[int, ...],
        hidden: int = 512,
        fusion: str = "add",
        n_labels: int = 0,
    ) -> None:
        super().__init__()
        if fusion not in {"add", "concat"}:
            raise ValueError("phase 1 fusion must be add or concat")
        self.fusion = fusion
        self.proj = nn.ModuleList(
            [nn.Conv2d(count, hidden, kernel_size=1) for count in channels]
        )
        fused_dim = hidden if fusion == "add" else hidden * len(channels)
        self.mlp = nn.Sequential(
            nn.Linear(fused_dim, hidden),
            nn.ReLU(inplace=True),
            nn.Linear(hidden, 5),
        )
        self.aux = nn.Linear(fused_dim, n_labels) if n_labels > 0 else None

    def fuse(self, feats: list[torch.Tensor]) -> torch.Tensor:
        if len(feats) != len(self.proj):
            raise ValueError("feature count does not match the head")
        vectors = [proj(feat).mean(dim=(2, 3)) for proj, feat in zip(self.proj, feats)]
        if self.fusion == "add":
            fused = vectors[0]
            for vector in vectors[1:]:
                fused = fused + vector
            return fused
        return torch.cat(vectors, dim=-1)

    def forward(self, feats: list[torch.Tensor], with_aux: bool):
        fused = self.fuse(feats)
        pred = self.mlp(fused)
        aux = self.aux(fused) if with_aux and self.aux is not None else None
        return pred, fused, aux
