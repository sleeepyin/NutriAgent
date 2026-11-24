"""RGB + official depth teacher. Two streams, then channel-spatial fusion."""

from __future__ import annotations

import torch
import torch.nn as nn

from digi_nutri.models.backbone import RESNET101_CHANNELS, build_resnet101_features
from digi_nutri.models.fusion import CrossModalChannelSpatialFusion
from digi_nutri.models.heads import MultiScaleAddHead


class GeometryTeacher(nn.Module):
    def __init__(
        self,
        rgb_backbone: nn.Module,
        depth_backbone: nn.Module,
        channels: tuple[int, ...],
        hidden: int = 512,
    ) -> None:
        super().__init__()
        self.rgb_backbone = rgb_backbone
        self.depth_backbone = depth_backbone
        self.fusions = nn.ModuleList(
            [CrossModalChannelSpatialFusion(count) for count in channels]
        )
        self.head = MultiScaleAddHead(channels, hidden=hidden, fusion="add", n_labels=0)

    def forward(self, rgb: torch.Tensor, depth: torch.Tensor) -> dict[str, torch.Tensor | list]:
        rgb_feats = self.rgb_backbone(rgb)
        depth_feats = self.depth_backbone(depth)
        fused = [
            fusion(rgb_feat, depth_feat)
            for fusion, rgb_feat, depth_feat in zip(self.fusions, rgb_feats, depth_feats)
        ]
        pred, vector, _aux = self.head(fused, with_aux=False)
        return {"nutrients": pred, "features": fused, "fused": vector}


def build_geometry_teacher(
    pretrained: bool = True,
    use_checkpoint: bool = True,
    hidden: int = 512,
) -> GeometryTeacher:
    return GeometryTeacher(
        build_resnet101_features(3, pretrained, use_checkpoint),
        build_resnet101_features(1, pretrained, use_checkpoint),
        RESNET101_CHANNELS,
        hidden=hidden,
    )
