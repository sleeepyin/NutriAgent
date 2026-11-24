"""RGB student. One feature stream. Test forward reads RGB only."""

from __future__ import annotations

import torch
import torch.nn as nn

from digi_nutri.models.backbone import RESNET101_CHANNELS, build_resnet101_features
from digi_nutri.models.heads import MultiScaleAddHead


class Student(nn.Module):
    def __init__(
        self,
        backbone: nn.Module,
        channels: tuple[int, ...],
        hidden: int = 512,
        n_labels: int = 0,
        fusion: str = "add",
    ) -> None:
        super().__init__()
        self.backbone = backbone
        self.head = MultiScaleAddHead(
            channels, hidden=hidden, fusion=fusion, n_labels=n_labels
        )

    def _maps(self, rgb: torch.Tensor) -> list[torch.Tensor]:
        feats = self.backbone(rgb)
        if not isinstance(feats, (list, tuple)) or len(feats) != 3:
            raise RuntimeError("backbone must return three spatial maps")
        return list(feats)

    def forward_train(self, rgb: torch.Tensor) -> dict:
        feats = self._maps(rgb)
        pred, fused, aux = self.head(feats, with_aux=True)
        out = {"nutrients": pred, "features": feats, "fused": fused}
        if aux is not None:
            out["aux_logits"] = aux
        return out

    def forward_test(self, rgb: torch.Tensor) -> dict[str, torch.Tensor]:
        feats = self._maps(rgb)
        pred, _fused, _aux = self.head(feats, with_aux=False)
        return {"nutrients": pred}

    def disable_aux_head(self) -> None:
        self.head.aux = None


def build_student(
    n_labels: int = 0,
    pretrained: bool = True,
    use_checkpoint: bool = True,
    hidden: int = 512,
    fusion: str = "add",
) -> Student:
    return Student(
        build_resnet101_features(3, pretrained, use_checkpoint),
        RESNET101_CHANNELS,
        hidden=hidden,
        n_labels=n_labels,
        fusion=fusion,
    )
