"""B1, Plan A T_geo with explicit height aggregates, and unchanged T_ing."""

from __future__ import annotations

import math

import torch
import torch.nn.functional as F
from torchvision.models import resnet18, resnet101
from torchvision.models import ResNet18_Weights, ResNet101_Weights

from digi_nutri.protocol import (
    BACKBONE_LR,
    CAB_ATTN_DIM,
    CAB_REDUCTION,
    DROPOUT_P,
    GAMMA_INIT,
    HEAD_HIDDEN,
    HEAD_LR,
    PARAM_LIMIT,
    SPP_GRIDS,
    TOKEN_DIM,
    VOLUME_DIM,
)


def count_parameters(module: torch.nn.Module) -> int:
    return sum(parameter.numel() for parameter in module.parameters())


def _inv_softplus(value: float) -> float:
    return math.log(math.expm1(value))


def volume_feature_names(dim: int = VOLUME_DIM) -> list[str]:
    names = [
        "foreground_area_ratio",
        "height_sum",
        "foreground_mean",
        "foreground_p90",
        "spp_avg_1x1",
    ]
    names.extend(f"spp_avg_2x2_{index}" for index in range(4))
    names.extend(f"spp_avg_4x4_{index}" for index in range(16))
    if len(names) != dim:
        raise RuntimeError(f"volume names {len(names)} != {dim}")
    return names


def volume_features(height01: torch.Tensor) -> torch.Tensor:
    """Parameter-free readout of the current [0, 1] height map.

    Foreground pixels are those with height > 0. Sum is the map sum.
    Mean and P90 use foreground pixels only. Spatial pyramid bins are
    average pools at 1x1, 2x2, and 4x4.
    """
    if height01.dim() == 3:
        height01 = height01.unsqueeze(1)
    if height01.dim() != 4 or height01.shape[1] != 1:
        raise RuntimeError(f"height map must be Bx1xHxW, got {tuple(height01.shape)}")
    positive = height01 > 0
    area = positive.flatten(2).to(dtype=height01.dtype).mean(dim=2).squeeze(1)
    total = height01.flatten(2).sum(dim=2).squeeze(1)
    means = []
    p90s = []
    for index in range(height01.shape[0]):
        values = height01[index, 0][positive[index, 0]]
        if values.numel() == 0:
            means.append(height01.new_zeros(()))
            p90s.append(height01.new_zeros(()))
        else:
            means.append(values.mean())
            p90s.append(torch.quantile(values, 0.9))
    mean = torch.stack(means)
    p90 = torch.stack(p90s)
    pooled = [F.adaptive_avg_pool2d(height01, grid).flatten(1) for grid in SPP_GRIDS]
    features = torch.cat(
        [area[:, None], total[:, None], mean[:, None], p90[:, None], *pooled],
        dim=1,
    )
    if features.shape[1] != VOLUME_DIM:
        raise RuntimeError(f"volume dim {features.shape[1]} != {VOLUME_DIM}")
    return features.detach()


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
        self.dropout = torch.nn.Dropout(DROPOUT_P)
        self.fc2 = torch.nn.Linear(hidden, out_dim)
        torch.nn.init.zeros_(self.fc2.weight)
        with torch.no_grad():
            self.fc2.bias.copy_(torch.tensor(bias, dtype=self.fc2.bias.dtype))

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return self.fc2(self.dropout(F.relu(self.fc1(features))))


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


class LocalSpatialSemanticAttention(torch.nn.Module):
    """Each spatial location attends over the dish's ingredient tokens."""

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


class IngredientEncoder(torch.nn.Module):
    def __init__(self, vocab_size: int, token_dim: int = TOKEN_DIM):
        super().__init__()
        self.embed = torch.nn.Embedding(vocab_size, token_dim, padding_idx=0)
        self.prior = torch.nn.Linear(4, token_dim)

    def forward(self, name_ids: torch.Tensor, priors: torch.Tensor) -> torch.Tensor:
        return self.embed(name_ids) + self.prior(priors)


def _head_bias(init_mass: float, init_density: list[float]) -> tuple[list[float], list[float]]:
    mass_bias = [_inv_softplus(init_mass)]
    density_bias = [_inv_softplus(max(value, 1e-3)) for value in init_density]
    return mass_bias, density_bias


class B1(torch.nn.Module):
    """RGB only. No teacher and no density dictionary."""

    def __init__(self, init_mass: float, init_density: list[float], pretrained: bool):
        super().__init__()
        self.backbone = ImageEncoder("resnet101", pretrained)
        mass_bias, density_bias = _head_bias(init_mass, init_density)
        channels = self.backbone.out_channels
        self.mass_head = MLPHead(channels, 1, mass_bias)
        self.density_head = MLPHead(channels, 4, density_bias)

    def forward(self, rgb: torch.Tensor) -> dict[str, torch.Tensor]:
        pooled = self.backbone(rgb).mean(dim=(2, 3))
        return product(self.mass_head(pooled), self.density_head(pooled))


class TGeo(torch.nn.Module):
    """Plan A encoders plus a parameter-free height readout on the mass head."""

    def __init__(
        self,
        init_mass: float,
        init_density: list[float],
        pretrained_rgb: bool,
        pretrained_height: bool,
    ):
        super().__init__()
        self.rgb_encoder = ImageEncoder("resnet101", pretrained_rgb)
        self.height_encoder = ImageEncoder("resnet18", pretrained_height)
        self.cab = LightCAB(self.rgb_encoder.out_channels, self.height_encoder.out_channels)
        self.gamma = torch.nn.Parameter(torch.full((1,), float(GAMMA_INIT)))
        mass_bias, density_bias = _head_bias(init_mass, init_density)
        channels = self.rgb_encoder.out_channels
        self.volume_dim = VOLUME_DIM
        self.mass_head = MLPHead(channels + VOLUME_DIM, 1, mass_bias)
        self.density_head = MLPHead(channels, 4, density_bias)

    def fuse(self, f_rgb: torch.Tensor, f_h: torch.Tensor) -> torch.Tensor:
        return f_rgb + self.gamma.view(1, 1, 1, 1) * self.cab(f_rgb, f_h)

    def forward(
        self,
        rgb: torch.Tensor,
        height: torch.Tensor,
        height01: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        fused = self.fuse(self.rgb_encoder(rgb), self.height_encoder(height))
        pooled = fused.mean(dim=(2, 3))
        mass_input = torch.cat([pooled, volume_features(height01)], dim=1)
        return product(self.mass_head(mass_input), self.density_head(pooled))


class TIng(torch.nn.Module):
    """RGB, ingredient tokens, and static per-gram priors."""

    def __init__(
        self,
        init_mass: float,
        init_density: list[float],
        vocab_size: int,
        pretrained: bool,
    ):
        super().__init__()
        self.backbone = ImageEncoder("resnet101", pretrained)
        mass_bias, density_bias = _head_bias(init_mass, init_density)
        channels = self.backbone.out_channels
        self.mass_head = MLPHead(channels, 1, mass_bias)
        self.density_head = MLPHead(channels, 4, density_bias)
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
        mass_raw = self.mass_head(features.mean(dim=(2, 3)))
        density_raw = self.density_head(semantic.mean(dim=(2, 3)))
        return product(mass_raw, density_raw)


def build_model(
    kind: str,
    init_mass: float,
    init_density: list[float],
    vocab_size: int = 1,
    pretrained: bool = False,
) -> torch.nn.Module:
    if kind == "b1":
        return B1(init_mass, init_density, pretrained)
    if kind == "t_geo":
        model = TGeo(init_mass, init_density, pretrained, pretrained)
        count = count_parameters(model)
        if count >= PARAM_LIMIT:
            raise RuntimeError(f"T_geo has {count} parameters, limit is {PARAM_LIMIT}")
        return model
    if kind == "t_ing":
        return TIng(init_mass, init_density, vocab_size, pretrained)
    raise ValueError("this rerun builds only b1, t_geo, and t_ing")


def parameter_groups(model: torch.nn.Module, kind: str) -> list[dict]:
    prefixes = {
        "b1": ("backbone.",),
        "t_geo": ("rgb_encoder.", "height_encoder."),
        "t_ing": ("backbone.",),
    }[kind]
    backbone = []
    other = []
    for name, parameter in model.named_parameters():
        if name.startswith(prefixes):
            backbone.append(parameter)
        else:
            other.append(parameter)
    if not backbone or not other:
        raise RuntimeError(f"{kind} is missing a backbone or a head/fusion group")
    return [
        {"params": backbone, "lr": BACKBONE_LR},
        {"params": other, "lr": HEAD_LR},
    ]
