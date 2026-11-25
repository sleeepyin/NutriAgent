"""Plan A geometry teacher.

ResNet-101 reads RGB. An independent ResNet-18 reads the height map.
Their BatchNorms are separate. A light channel-attention block, in the
style already used by DPF-style fusion, is added after the encoders:

    F = F_rgb + gamma * CAB(F_rgb, F_h)

gamma starts at 0, so the first forward matches the frozen RGB baseline
once that baseline's weights are loaded into the RGB encoder and the heads.
"""

from __future__ import annotations

import math

import torch
import torch.nn.functional as F
from torchvision.models import resnet18, resnet101
from torchvision.models import ResNet18_Weights, ResNet101_Weights

from digi_nutri.protocol import CAB_REDUCTION, GAMMA_INIT, HEAD_HIDDEN, PARAM_LIMIT


def count_parameters(module: torch.nn.Module) -> int:
    return sum(parameter.numel() for parameter in module.parameters())


def _inv_softplus(value: float) -> float:
    return math.log(math.expm1(value))


class ImageEncoder(torch.nn.Module):
    def __init__(self, name: str, pretrained: bool):
        super().__init__()
        if name == "resnet101":
            weights = ResNet101_Weights.IMAGENET1K_V1 if pretrained else None
            net = resnet101(weights=weights)
            out_channels = 2048
        elif name == "resnet18":
            weights = ResNet18_Weights.IMAGENET1K_V1 if pretrained else None
            net = resnet18(weights=weights)
            out_channels = 512
        else:
            raise ValueError(f"unsupported encoder {name}")
        self.name = name
        self.out_channels = out_channels
        self.stem = torch.nn.Sequential(net.conv1, net.bn1, net.relu, net.maxpool)
        self.layer1 = net.layer1
        self.layer2 = net.layer2
        self.layer3 = net.layer3
        self.layer4 = net.layer4

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        hidden = self.stem(images)
        hidden = self.layer1(hidden)
        hidden = self.layer2(hidden)
        hidden = self.layer3(hidden)
        return self.layer4(hidden)


class MLPHead(torch.nn.Module):
    def __init__(self, in_dim: int, out_dim: int, bias: list[float], hidden: int = HEAD_HIDDEN):
        super().__init__()
        self.fc1 = torch.nn.Linear(in_dim, hidden)
        self.fc2 = torch.nn.Linear(hidden, out_dim)
        torch.nn.init.zeros_(self.fc2.weight)
        with torch.no_grad():
            self.fc2.bias.copy_(torch.tensor(bias, dtype=self.fc2.bias.dtype))

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return self.fc2(F.relu(self.fc1(features)))


def product(mass_raw: torch.Tensor, density_raw: torch.Tensor) -> dict[str, torch.Tensor]:
    mass = F.softplus(mass_raw)
    density = F.softplus(density_raw)
    return {"mass": mass, "density": density, "yhat": mass * density}


class LightCAB(torch.nn.Module):
    """Channel gate from both pooled maps. No BatchNorm."""

    def __init__(self, rgb_channels: int, height_channels: int, reduction: int = CAB_REDUCTION):
        super().__init__()
        if reduction < 1 or rgb_channels % reduction != 0:
            raise ValueError("CAB reduction must divide the RGB channels")
        hidden = rgb_channels // reduction
        self.align = torch.nn.Conv2d(height_channels, rgb_channels, kernel_size=1, bias=False)
        self.fc1 = torch.nn.Linear(rgb_channels * 2, hidden, bias=False)
        self.fc2 = torch.nn.Linear(hidden, rgb_channels, bias=False)

    def forward(self, f_rgb: torch.Tensor, f_h: torch.Tensor) -> torch.Tensor:
        aligned = self.align(f_h)
        if aligned.shape[-2:] != f_rgb.shape[-2:]:
            raise RuntimeError(
                f"encoder grids differ: rgb {tuple(f_rgb.shape)} height {tuple(aligned.shape)}"
            )
        pooled = torch.cat([f_rgb.mean(dim=(2, 3)), aligned.mean(dim=(2, 3))], dim=1)
        gate = torch.sigmoid(self.fc2(F.relu(self.fc1(pooled))))
        return gate[:, :, None, None] * aligned


class PlanATGeo(torch.nn.Module):
    def __init__(
        self,
        init_mass: float,
        init_density: list[float],
        pretrained_rgb: bool = False,
        pretrained_height: bool = False,
        reduction: int = CAB_REDUCTION,
    ):
        super().__init__()
        self.rgb_encoder = ImageEncoder("resnet101", pretrained_rgb)
        self.height_encoder = ImageEncoder("resnet18", pretrained_height)
        self.cab = LightCAB(
            self.rgb_encoder.out_channels,
            self.height_encoder.out_channels,
            reduction,
        )
        self.gamma = torch.nn.Parameter(torch.full((1,), float(GAMMA_INIT)))
        mass_bias = [_inv_softplus(init_mass)]
        density_bias = [_inv_softplus(max(value, 1e-3)) for value in init_density]
        channels = self.rgb_encoder.out_channels
        self.mass_head = MLPHead(channels, 1, mass_bias)
        self.density_head = MLPHead(channels, 4, density_bias)

    def fuse(self, f_rgb: torch.Tensor, f_h: torch.Tensor) -> torch.Tensor:
        return f_rgb + self.gamma.view(1, 1, 1, 1) * self.cab(f_rgb, f_h)

    def forward(self, rgb: torch.Tensor, height: torch.Tensor) -> dict[str, torch.Tensor]:
        fused = self.fuse(self.rgb_encoder(rgb), self.height_encoder(height))
        pooled = fused.mean(dim=(2, 3))
        return product(self.mass_head(pooled), self.density_head(pooled))


def build_plan_a(
    init_mass: float,
    init_density: list[float],
    pretrained_height: bool,
) -> PlanATGeo:
    model = PlanATGeo(
        init_mass,
        init_density,
        pretrained_rgb=False,
        pretrained_height=pretrained_height,
    )
    count = count_parameters(model)
    if count >= PARAM_LIMIT:
        raise RuntimeError(f"Plan A T_geo has {count} parameters, limit is {PARAM_LIMIT}")
    return model


def load_rgb_from_frozen_b1(model: PlanATGeo, state: dict) -> None:
    """Copy the frozen RGB baseline into the RGB encoder and both heads."""
    mapped = {}
    for key, value in state.items():
        if key.startswith("backbone."):
            mapped["rgb_encoder." + key[len("backbone.") :]] = value
        elif key.startswith("mass_head.") or key.startswith("density_head."):
            mapped[key] = value
        else:
            raise RuntimeError(f"unexpected frozen B1 key {key}")
    result = model.load_state_dict(mapped, strict=False)
    bad_missing = [
        key
        for key in result.missing_keys
        if not (
            key.startswith("height_encoder.")
            or key.startswith("cab.")
            or key == "gamma"
        )
    ]
    if bad_missing or result.unexpected_keys:
        raise RuntimeError(
            f"B1 load left missing={bad_missing} unexpected={list(result.unexpected_keys)}"
        )
    if float(model.gamma.detach().reshape(-1)[0]) != float(GAMMA_INIT):
        raise RuntimeError("loading B1 changed gamma")
