"""B1, T_geo, and T_ing. Plate nutrition is predicted mass times predicted density."""

from __future__ import annotations

import math

import torch
import torch.nn.functional as F
from torchvision.models import resnet101, resnet50
from torchvision.models import ResNet101_Weights, ResNet50_Weights

from digi_nutri.protocol import CAB_ATTN_DIM, HEAD_HIDDEN, PARAM_LIMIT, TOKEN_DIM


def count_parameters(module: torch.nn.Module) -> int:
    return sum(parameter.numel() for parameter in module.parameters())


def _inv_softplus(value: float) -> float:
    return math.log(math.expm1(value))


class ImageBackbone(torch.nn.Module):
    def __init__(self, name: str, pretrained: bool):
        super().__init__()
        if name == "resnet101":
            weights = ResNet101_Weights.IMAGENET1K_V1 if pretrained else None
            net = resnet101(weights=weights)
        elif name == "resnet50":
            weights = ResNet50_Weights.IMAGENET1K_V1 if pretrained else None
            net = resnet50(weights=weights)
        else:
            raise ValueError(f"unsupported backbone {name}")
        self.name = name
        self.stem = torch.nn.Sequential(net.conv1, net.bn1, net.relu, net.maxpool)
        self.layer1 = net.layer1
        self.layer2 = net.layer2
        self.layer3 = net.layer3
        self.layer4 = net.layer4
        self.out_channels = 2048

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        hidden = self.stem(images)
        hidden = self.layer1(hidden)
        hidden = self.layer2(hidden)
        hidden = self.layer3(hidden)
        return self.layer4(hidden)


def _pool(features: torch.Tensor) -> torch.Tensor:
    return features.mean(dim=(2, 3))


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


def _product(mass_raw: torch.Tensor, density_raw: torch.Tensor) -> dict[str, torch.Tensor]:
    mass = F.softplus(mass_raw)
    density = F.softplus(density_raw)
    return {"mass": mass, "density": density, "yhat": mass * density}


class SharedWeightCAB(torch.nn.Module):
    """One set of projections fuses the height stream into the RGB stream."""

    def __init__(self, channels: int, attn_dim: int = CAB_ATTN_DIM):
        super().__init__()
        self.q = torch.nn.Conv2d(channels, attn_dim, kernel_size=1, bias=False)
        self.k = torch.nn.Conv2d(channels, attn_dim, kernel_size=1, bias=False)
        self.v = torch.nn.Conv2d(channels, attn_dim, kernel_size=1, bias=False)
        self.proj = torch.nn.Conv2d(attn_dim, channels, kernel_size=1, bias=False)
        self.norm = torch.nn.GroupNorm(32, channels)

    def forward(self, rgb: torch.Tensor, height: torch.Tensor) -> torch.Tensor:
        batch, _, grid_h, grid_w = rgb.shape
        query = self.q(rgb).flatten(2).transpose(1, 2)
        key = self.k(height).flatten(2)
        value = self.v(height).flatten(2).transpose(1, 2)
        weights = torch.softmax(torch.bmm(query, key) * (query.shape[-1] ** -0.5), dim=-1)
        mixed = torch.bmm(weights, value).transpose(1, 2).reshape(batch, -1, grid_h, grid_w)
        return self.norm(rgb + self.proj(mixed))


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


class _PlateModel(torch.nn.Module):
    def __init__(self, backbone: str, pretrained: bool, init_mass: float, init_density: list[float]):
        super().__init__()
        self.backbone_name = backbone
        self.backbone = ImageBackbone(backbone, pretrained)
        mass_bias = [_inv_softplus(init_mass)]
        density_bias = [_inv_softplus(max(value, 1e-3)) for value in init_density]
        channels = self.backbone.out_channels
        self.mass_head = MLPHead(channels, 1, mass_bias)
        self.density_head = MLPHead(channels, 4, density_bias)


class B1(_PlateModel):
    """RGB only. No teacher and no density dictionary."""

    def forward(self, rgb: torch.Tensor) -> dict[str, torch.Tensor]:
        pooled = _pool(self.backbone(rgb))
        return _product(self.mass_head(pooled), self.density_head(pooled))


class TGeo(_PlateModel):
    """RGB plus a relative-height map. The height stream feeds the mass head."""

    def __init__(self, backbone: str, pretrained: bool, init_mass: float, init_density: list[float]):
        super().__init__(backbone, pretrained, init_mass, init_density)
        self.cab = SharedWeightCAB(self.backbone.out_channels)

    def forward(self, rgb: torch.Tensor, height: torch.Tensor) -> dict[str, torch.Tensor]:
        rgb_features = self.backbone(rgb)
        height_features = self.backbone(height)
        fused = self.cab(rgb_features, height_features)
        mass_raw = self.mass_head(_pool(fused))
        density_raw = self.density_head(_pool(rgb_features))
        return _product(mass_raw, density_raw)


class TIng(_PlateModel):
    """RGB, ingredient tokens, and static per-gram priors. Density reads the attention map."""

    def __init__(
        self,
        backbone: str,
        pretrained: bool,
        init_mass: float,
        init_density: list[float],
        vocab_size: int,
    ):
        super().__init__(backbone, pretrained, init_mass, init_density)
        self.encoder = IngredientEncoder(vocab_size)
        self.attention = LocalSpatialSemanticAttention(self.backbone.out_channels)

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
        mass_raw = self.mass_head(_pool(features))
        density_raw = self.density_head(_pool(semantic))
        return _product(mass_raw, density_raw)


def build_model(
    kind: str,
    backbone: str,
    pretrained: bool,
    init_mass: float,
    init_density: list[float],
    vocab_size: int = 1,
) -> torch.nn.Module:
    if kind == "b1":
        return B1(backbone, pretrained, init_mass, init_density)
    if kind == "t_geo":
        return TGeo(backbone, pretrained, init_mass, init_density)
    if kind == "t_ing":
        return TIng(backbone, pretrained, init_mass, init_density, vocab_size)
    raise ValueError("Q0 builds only b1, t_geo, and t_ing")


def select_backbone(init_mass: float = 200.0, init_density: list[float] | None = None) -> tuple[str, int]:
    """Use ResNet-101 when T_geo stays under 60M, otherwise the same smaller net."""
    if init_density is None:
        init_density = [1.0, 0.05, 0.1, 0.08]
    last_count = None
    for name in ("resnet101", "resnet50"):
        model = build_model("t_geo", name, pretrained=False, init_mass=init_mass, init_density=init_density)
        last_count = count_parameters(model)
        if last_count < PARAM_LIMIT:
            return name, last_count
    raise RuntimeError(f"T_geo is {last_count} parameters, at or above {PARAM_LIMIT}")
