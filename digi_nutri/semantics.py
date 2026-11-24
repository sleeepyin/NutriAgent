"""Dish-level ingredient vectors. Names-only is the Phase 1 teacher."""

from __future__ import annotations

import torch
import torch.nn.functional as F


def names_only_dish_vector(
    name_embeddings: torch.Tensor,
    name_mask: torch.Tensor,
    grams: torch.Tensor | None = None,
) -> torch.Tensor:
    """Mean of L2-normalized name embeddings. Gram weights are refused."""

    if grams is not None:
        raise ValueError("names-only pooling does not accept gram weights")
    if name_embeddings.ndim != 3:
        raise ValueError("name embeddings must be B,K,D")
    emb = F.normalize(name_embeddings.float(), dim=-1, eps=1e-6)
    weights = name_mask.to(device=emb.device).float().unsqueeze(-1)
    denom = weights.sum(dim=1).clamp_min(1.0)
    pooled = (emb * weights).sum(dim=1) / denom
    pooled = F.normalize(pooled, dim=-1, eps=1e-6)
    empty = name_mask.to(device=emb.device).sum(dim=1) == 0
    return pooled.masked_fill(empty.unsqueeze(-1), 0.0)


def gram_weighted_dish_vector(
    name_embeddings: torch.Tensor,
    name_mask: torch.Tensor,
    grams: torch.Tensor,
) -> torch.Tensor:
    """Optional gram-weighted pool. Not used by the Phase 1 teacher."""

    if grams is None:
        raise ValueError("gram pooling requires grams")
    emb = F.normalize(name_embeddings.float(), dim=-1, eps=1e-6)
    weights = grams.float().clamp_min(0) * name_mask.to(device=emb.device).float()
    denom = weights.sum(dim=1, keepdim=True).clamp_min(1e-6)
    pooled = (emb * weights.unsqueeze(-1)).sum(dim=1) / denom
    return F.normalize(pooled, dim=-1, eps=1e-6)
