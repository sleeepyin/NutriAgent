"""Frozen B1 and T_ing, loaded for evaluation only. They are not trained here."""

from __future__ import annotations

import math

import torch

from digi_nutri.models import ImageEncoder, MLPHead, product
from digi_nutri.protocol import CAB_ATTN_DIM, HEAD_HIDDEN, TOKEN_DIM


def _pool(features: torch.Tensor) -> torch.Tensor:
    return features.mean(dim=(2, 3))


def _biases(init_mass: float, init_density: list[float]) -> tuple[list[float], list[float]]:
    mass_bias = [math.log(math.expm1(init_mass))]
    density_bias = [math.log(math.expm1(max(value, 1e-3))) for value in init_density]
    return mass_bias, density_bias


class FrozenB1(torch.nn.Module):
    def __init__(self, init_mass: float, init_density: list[float]):
        super().__init__()
        self.backbone = ImageEncoder("resnet101", pretrained=False)
        mass_bias, density_bias = _biases(init_mass, init_density)
        channels = self.backbone.out_channels
        self.mass_head = MLPHead(channels, 1, mass_bias, HEAD_HIDDEN)
        self.density_head = MLPHead(channels, 4, density_bias, HEAD_HIDDEN)

    def forward(self, rgb: torch.Tensor) -> dict[str, torch.Tensor]:
        pooled = _pool(self.backbone(rgb))
        return product(self.mass_head(pooled), self.density_head(pooled))


class IngredientEncoder(torch.nn.Module):
    def __init__(self, vocab_size: int, token_dim: int = TOKEN_DIM):
        super().__init__()
        self.embed = torch.nn.Embedding(vocab_size, token_dim, padding_idx=0)
        self.prior = torch.nn.Linear(4, token_dim)

    def forward(self, name_ids: torch.Tensor, priors: torch.Tensor) -> torch.Tensor:
        return self.embed(name_ids) + self.prior(priors)


class LocalSpatialSemanticAttention(torch.nn.Module):
    def __init__(self, channels: int, token_dim: int = TOKEN_DIM, attn_dim: int = CAB_ATTN_DIM):
        super().__init__()
        self.q = torch.nn.Conv2d(channels, attn_dim, kernel_size=1, bias=False)
        self.k = torch.nn.Linear(token_dim, attn_dim, bias=False)
        self.v = torch.nn.Linear(token_dim, attn_dim, bias=False)
        self.proj = torch.nn.Conv2d(attn_dim, channels, kernel_size=1, bias=False)
        self.norm = torch.nn.GroupNorm(32, channels)

    def forward(self, features: torch.Tensor, tokens: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        batch, _, grid_h, grid_w = features.shape
        query = self.q(features).flatten(2).transpose(1, 2)
        key = self.k(tokens)
        value = self.v(tokens)
        logits = torch.bmm(query, key.transpose(1, 2)) * (query.shape[-1] ** -0.5)
        valid = mask[:, None, :].to(dtype=torch.bool)
        logits = logits.masked_fill(~valid, torch.finfo(logits.dtype).min)
        empty = ~mask.to(dtype=torch.bool).any(dim=1)
        if torch.any(empty):
            logits = logits.clone()
            logits[empty] = 0.0
        weights = torch.softmax(logits, dim=-1)
        if torch.any(empty):
            weights = weights.clone()
            weights[empty] = 0.0
        mixed = torch.bmm(weights, value).transpose(1, 2).reshape(batch, -1, grid_h, grid_w)
        return self.norm(features + self.proj(mixed))


class FrozenTIng(torch.nn.Module):
    def __init__(self, init_mass: float, init_density: list[float], vocab_size: int):
        super().__init__()
        self.backbone = ImageEncoder("resnet101", pretrained=False)
        mass_bias, density_bias = _biases(init_mass, init_density)
        channels = self.backbone.out_channels
        self.mass_head = MLPHead(channels, 1, mass_bias, HEAD_HIDDEN)
        self.density_head = MLPHead(channels, 4, density_bias, HEAD_HIDDEN)
        self.encoder = IngredientEncoder(vocab_size)
        self.attention = LocalSpatialSemanticAttention(channels)

    def forward(
        self,
        rgb: torch.Tensor,
        name_ids: torch.Tensor,
        priors: torch.Tensor,
        mask: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        features = self.backbone(rgb)
        tokens = self.encoder(name_ids, priors)
        semantic = self.attention(features, tokens, mask)
        return product(self.mass_head(_pool(features)), self.density_head(_pool(semantic)))
