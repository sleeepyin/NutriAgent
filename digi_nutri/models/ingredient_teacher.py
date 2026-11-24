"""RGB teacher modulated by a names-only dish vector. No gram weights."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from digi_nutri.models.backbone import RESNET101_CHANNELS, build_resnet101_features
from digi_nutri.models.fusion import SemanticCrossAttention
from digi_nutri.models.heads import MultiScaleAddHead
from digi_nutri.semantics import names_only_dish_vector


class IngredientTeacher(nn.Module):
    def __init__(
        self,
        backbone: nn.Module,
        channels: tuple[int, ...],
        hidden: int = 512,
        text_dim: int = 512,
        attn_dim: int = 512,
        heads: int = 8,
    ) -> None:
        super().__init__()
        self.backbone = backbone
        self.text_dim = text_dim
        self.cross_attn = SemanticCrossAttention(
            channels[-1], text_dim, attn_dim=attn_dim, heads=heads
        )
        self.head = MultiScaleAddHead(channels, hidden=hidden, fusion="add", n_labels=0)

    def forward(
        self,
        rgb: torch.Tensor,
        name_embeddings: torch.Tensor,
        name_mask: torch.Tensor,
    ) -> dict:
        name_embeddings = name_embeddings.to(device=rgb.device, dtype=torch.float32)
        name_mask = name_mask.to(device=rgb.device)
        if name_embeddings.ndim != 3:
            raise ValueError("name_embeddings must be B,K,D")
        if name_embeddings.shape[1] == 0:
            batch = rgb.shape[0]
            name_embeddings = rgb.new_zeros(batch, 0, self.text_dim)
            name_mask = torch.zeros(batch, 0, dtype=torch.bool, device=rgb.device)
        dish = names_only_dish_vector(name_embeddings, name_mask)
        tokens = F.normalize(name_embeddings, dim=-1, eps=1e-6)
        tokens = tokens * name_mask.unsqueeze(-1).to(tokens.dtype)
        memory = torch.cat([dish.unsqueeze(1), tokens], dim=1)
        memory_mask = torch.cat(
            [
                torch.ones(rgb.shape[0], 1, dtype=torch.bool, device=rgb.device),
                name_mask,
            ],
            dim=1,
        )
        feats = list(self.backbone(rgb))
        feats[2] = self.cross_attn(feats[2], memory, memory_mask)
        pred, vector, _aux = self.head(feats, with_aux=False)
        return {
            "nutrients": pred,
            "features": feats,
            "fused": vector,
            "dish_vector": dish,
        }


def build_ingredient_teacher(
    pretrained: bool = True,
    use_checkpoint: bool = True,
    hidden: int = 512,
    text_dim: int = 512,
) -> IngredientTeacher:
    return IngredientTeacher(
        build_resnet101_features(3, pretrained, use_checkpoint),
        RESNET101_CHANNELS,
        hidden=hidden,
        text_dim=text_dim,
    )
