"""Geometry-teacher fusion and ingredient-teacher cross-attention.

The student does not use this module to mix two subspaces.
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn


class CrossModalChannelSpatialFusion(nn.Module):
    """Channel and spatial attention between an RGB map and a depth map."""

    def __init__(self, channels: int, reduction: int = 16) -> None:
        super().__init__()
        hidden = max(channels // reduction, 8)
        self.channel_mlp = nn.Sequential(
            nn.Linear(channels * 2, hidden),
            nn.ReLU(inplace=True),
            nn.Linear(hidden, channels * 2),
        )
        spatial_hidden = max(channels // reduction, 8)
        self.spatial = nn.Sequential(
            nn.Conv2d(channels * 2, spatial_hidden, kernel_size=7, padding=3, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(spatial_hidden, 2, kernel_size=7, padding=3, bias=False),
        )

    def forward(self, rgb_feat: torch.Tensor, depth_feat: torch.Tensor) -> torch.Tensor:
        if rgb_feat.shape != depth_feat.shape:
            raise RuntimeError(
                f"RGB/depth feature mismatch {tuple(rgb_feat.shape)} vs {tuple(depth_feat.shape)}"
            )
        pooled = torch.cat(
            [rgb_feat.mean(dim=(2, 3)), depth_feat.mean(dim=(2, 3))],
            dim=1,
        )
        channel = torch.sigmoid(self.channel_mlp(pooled))
        channel = channel.view(rgb_feat.shape[0], 2, rgb_feat.shape[1], 1, 1)
        spatial = torch.sigmoid(self.spatial(torch.cat([rgb_feat, depth_feat], dim=1)))
        return (
            channel[:, 0] * spatial[:, 0:1] * rgb_feat
            + channel[:, 1] * spatial[:, 1:2] * depth_feat
        )


class SemanticCrossAttention(nn.Module):
    """Visual queries attend to name tokens plus the dish-level vector."""

    def __init__(
        self,
        vis_channels: int,
        text_dim: int,
        attn_dim: int = 512,
        heads: int = 8,
    ) -> None:
        super().__init__()
        if attn_dim % heads != 0:
            raise ValueError("attn_dim must be divisible by heads")
        self.heads = heads
        self.head_dim = attn_dim // heads
        self.down = nn.Conv2d(vis_channels, attn_dim, kernel_size=1)
        self.up = nn.Conv2d(attn_dim, vis_channels, kernel_size=1)
        self.q = nn.Linear(attn_dim, attn_dim)
        self.k = nn.Linear(text_dim, attn_dim)
        self.v = nn.Linear(text_dim, attn_dim)
        self.out = nn.Linear(attn_dim, attn_dim)

    def forward(
        self,
        vis: torch.Tensor,
        tokens: torch.Tensor,
        mask: torch.Tensor,
    ) -> torch.Tensor:
        batch, _channels, height, width = vis.shape
        hidden = self.down(vis)
        queries = self.q(hidden.flatten(2).transpose(1, 2))
        kept = tokens * mask.unsqueeze(-1).to(dtype=tokens.dtype)
        keys = self.k(kept)
        values = self.v(kept)
        queries = self._split(queries)
        keys = self._split(keys)
        values = self._split(values)
        logits = torch.matmul(queries, keys.transpose(-1, -2)) / math.sqrt(self.head_dim)
        logits = logits.masked_fill(~mask[:, None, None, :], torch.finfo(logits.dtype).min)
        attn = torch.softmax(logits, dim=-1)
        attn = torch.nan_to_num(attn, nan=0.0)
        mixed = torch.matmul(attn, values)
        mixed = mixed.transpose(1, 2).contiguous().view(batch, height * width, -1)
        mixed = self.out(mixed).transpose(1, 2).view(batch, -1, height, width)
        return vis + self.up(mixed)

    def _split(self, tensor: torch.Tensor) -> torch.Tensor:
        batch, length, _ = tensor.shape
        return tensor.view(batch, length, self.heads, self.head_dim).transpose(1, 2)
