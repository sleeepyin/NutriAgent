"""ResNet-101 feature maps at layer2, layer3, and layer4."""

from __future__ import annotations

import torch
import torch.nn as nn
from torch.utils.checkpoint import checkpoint
from torchvision.models import ResNet101_Weights, resnet101

RESNET101_CHANNELS = (512, 1024, 2048)


def build_resnet101(in_channels: int, pretrained: bool) -> nn.Module:
    if in_channels not in {1, 3}:
        raise ValueError("backbone channels must be 1 or 3")
    weights = ResNet101_Weights.IMAGENET1K_V1 if pretrained else None
    if in_channels == 3:
        return resnet101(weights=weights)
    net = resnet101(weights=None)
    net.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
    if pretrained:
        donor = resnet101(weights=ResNet101_Weights.IMAGENET1K_V1)
        with torch.no_grad():
            net.conv1.weight.copy_(donor.conv1.weight.mean(dim=1, keepdim=True))
            donor_state = donor.state_dict()
            donor_state.pop("conv1.weight")
            net.load_state_dict(donor_state, strict=False)
    else:
        nn.init.kaiming_normal_(net.conv1.weight, mode="fan_out", nonlinearity="relu")
    return net


class ResNet101Features(nn.Module):
    def __init__(self, net: nn.Module, use_checkpoint: bool = False) -> None:
        super().__init__()
        self.stem = nn.Sequential(net.conv1, net.bn1, net.relu, net.maxpool)
        self.layer1 = net.layer1
        self.layer2 = net.layer2
        self.layer3 = net.layer3
        self.layer4 = net.layer4
        self.use_checkpoint = use_checkpoint

    def _stage(self, fn, x: torch.Tensor) -> torch.Tensor:
        if self.use_checkpoint and self.training and x.requires_grad:
            return checkpoint(fn, x, use_reentrant=False)
        return fn(x)

    def forward(self, x: torch.Tensor) -> list[torch.Tensor]:
        x = self.stem(x)
        x = self._stage(self.layer1, x)
        f2 = self._stage(self.layer2, x)
        f3 = self._stage(self.layer3, f2)
        f4 = self._stage(self.layer4, f3)
        return [f2, f3, f4]


def build_resnet101_features(
    in_channels: int,
    pretrained: bool,
    use_checkpoint: bool = False,
) -> ResNet101Features:
    return ResNet101Features(
        build_resnet101(in_channels, pretrained),
        use_checkpoint=use_checkpoint,
    )
