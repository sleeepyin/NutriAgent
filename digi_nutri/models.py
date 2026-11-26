"""Same-backbone RGB B1 and residual-Atwater T_multi.

B1 reads RGB with timm convnext_small. T_multi uses that same RGB encoder,
an independent ResNet-18 on the height map, and the ingredient branch on the
density head only. Plate nutrition is predicted mass times predicted density.
Energy density is the Atwater combination of the three macro densities plus a
learned residual. No hard constraint is applied to the labels.
"""

from __future__ import annotations

import hashlib
import math
from pathlib import Path

import timm
import torch
import torch.nn.functional as F
from timm.models.convnext import checkpoint_filter_fn
from torchvision.models import ResNet18_Weights, resnet18

from digi_nutri.protocol import (
    ATWATER_CARB,
    ATWATER_FAT,
    ATWATER_PRO,
    BACKBONE_LR,
    B1_PARAMS,
    CAB_ATTN_DIM,
    CAB_REDUCTION,
    DROPOUT_P,
    GAMMA_INIT,
    HEAD_HIDDEN,
    HEAD_LR,
    HEIGHT_CHANNELS,
    HEIGHT_ENCODER_PARAMS,
    PARAM_LIMIT,
    RESIDUAL_HIDDEN,
    CONVNEXT_WEIGHT_PATH,
    CONVNEXT_WEIGHT_SHA256,
    RGB_BACKBONE,
    RGB_CHANNELS,
    RGB_ENCODER_PARAMS,
    SPP_GRIDS,
    TOKEN_DIM,
    T_MULTI_PARAMS,
    VOLUME_DIM,
)


def count_parameters(module: torch.nn.Module) -> int:
    return sum(parameter.numel() for parameter in module.parameters())


def parameter_breakdown(module: torch.nn.Module) -> dict[str, int]:
    found: dict[str, int] = {}
    for name, parameter in module.named_parameters():
        top = name.split(".", 1)[0]
        found[top] = found.get(top, 0) + parameter.numel()
    return dict(sorted(found.items()))


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
    average pools at 1x1, 2x2, and 4x4 of the height map itself.
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


def load_convnext_imagenet(net: torch.nn.Module) -> None:
    """Load the frozen local ImageNet checkpoint into timm convnext_small."""
    path = Path(CONVNEXT_WEIGHT_PATH)
    if not path.is_file():
        raise RuntimeError(f"missing ConvNeXt ImageNet weights at {path}")
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    if digest.hexdigest() != CONVNEXT_WEIGHT_SHA256:
        raise RuntimeError("ConvNeXt ImageNet checksum does not match the frozen config")
    mapped = checkpoint_filter_fn(torch.load(path, map_location="cpu", weights_only=False), net)
    incompatible = net.load_state_dict(mapped, strict=False)
    missing = [key for key in incompatible.missing_keys if not key.startswith("head.fc.")]
    unexpected = [key for key in incompatible.unexpected_keys if not key.startswith("head.fc.")]
    if missing or unexpected:
        raise RuntimeError(f"ConvNeXt load missing={missing[:8]} unexpected={unexpected[:8]}")


class ConvNeXtEncoder(torch.nn.Module):
    """timm convnext_small. The classification layer is not part of the encoder."""

    def __init__(self, pretrained: bool):
        super().__init__()
        self.name = RGB_BACKBONE
        self.out_channels = RGB_CHANNELS
        self.net = timm.create_model(
            RGB_BACKBONE,
            pretrained=False,
            num_classes=0,
            global_pool="",
            drop_path_rate=0.0,
        )
        if pretrained:
            load_convnext_imagenet(self.net)
        if self.out_channels != RGB_CHANNELS:
            raise RuntimeError("RGB channel constant does not match the encoder")

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        features = self.net(images)
        if features.dim() != 4 or features.shape[1] != self.out_channels:
            raise RuntimeError(f"convnext_small map is {tuple(features.shape)}")
        return features


class HeightEncoder(torch.nn.Module):
    """Independent ResNet-18. Its BatchNorms are not shared with the RGB encoder."""

    def __init__(self, pretrained: bool):
        super().__init__()
        weights = ResNet18_Weights.IMAGENET1K_V1 if pretrained else None
        net = resnet18(weights=weights)
        self.name = "resnet18"
        self.out_channels = HEIGHT_CHANNELS
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


class ResidualHead(torch.nn.Module):
    """Small calorie-density residual. The last layer starts at zero."""

    def __init__(self, in_dim: int, hidden: int = RESIDUAL_HIDDEN):
        super().__init__()
        self.fc1 = torch.nn.Linear(in_dim, hidden)
        self.dropout = torch.nn.Dropout(DROPOUT_P)
        self.fc2 = torch.nn.Linear(hidden, 1)
        torch.nn.init.zeros_(self.fc2.weight)
        torch.nn.init.zeros_(self.fc2.bias)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return self.fc2(self.dropout(F.relu(self.fc1(features))))


def product(mass_raw: torch.Tensor, density_raw: torch.Tensor) -> dict[str, torch.Tensor]:
    mass = F.softplus(mass_raw)
    density = F.softplus(density_raw)
    return {"mass": mass, "density": density, "yhat": mass * density}


def atwater_density(macro_raw: torch.Tensor, delta_raw: torch.Tensor) -> dict[str, torch.Tensor]:
    """rho_cal = 9*rho_fat + 4*rho_carb + 4*rho_pro + delta. Delta is not squashed."""
    if macro_raw.shape[-1] != 3:
        raise RuntimeError(f"macro head must output fat, carb, protein, got {tuple(macro_raw.shape)}")
    macros = F.softplus(macro_raw)
    fat = macros[:, 0]
    carb = macros[:, 1]
    protein = macros[:, 2]
    delta = delta_raw.reshape(-1)
    energy = ATWATER_FAT * fat + ATWATER_CARB * carb + ATWATER_PRO * protein + delta
    density = torch.stack([energy, fat, carb, protein], dim=1)
    return {"density": density, "delta_cal": delta.reshape(-1, 1)}


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
    """Each spatial location attends over the dish's ingredient tokens.

    The projection starts at zero. The block is features + GroupNorm(update),
    so the update is exactly zero at initialization and the copied B1 density
    head still sees the RGB map.
    """

    def __init__(self, channels: int, token_dim: int = TOKEN_DIM, attn_dim: int = CAB_ATTN_DIM):
        super().__init__()
        self.q = torch.nn.Conv2d(channels, attn_dim, kernel_size=1, bias=False)
        self.k = torch.nn.Linear(token_dim, attn_dim, bias=False)
        self.v = torch.nn.Linear(token_dim, attn_dim, bias=False)
        self.proj = torch.nn.Conv2d(attn_dim, channels, kernel_size=1, bias=False)
        self.norm = torch.nn.GroupNorm(32, channels)
        torch.nn.init.zeros_(self.proj.weight)

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
        return features + self.norm(self.proj(mixed))


class IngredientEncoder(torch.nn.Module):
    def __init__(self, vocab_size: int, token_dim: int = TOKEN_DIM):
        super().__init__()
        self.embed = torch.nn.Embedding(vocab_size, token_dim, padding_idx=0)
        self.prior = torch.nn.Linear(4, token_dim)

    def forward(self, name_ids: torch.Tensor, priors: torch.Tensor) -> torch.Tensor:
        return self.embed(name_ids) + self.prior(priors)


def _mass_bias(init_mass: float) -> list[float]:
    return [_inv_softplus(init_mass)]


def _density_bias(init_density: list[float]) -> list[float]:
    return [_inv_softplus(max(value, 1e-3)) for value in init_density]


class B1(torch.nn.Module):
    """RGB only. Mass and four densities, then the product. No height, no dictionary."""

    def __init__(self, init_mass: float, init_density: list[float], pretrained: bool):
        super().__init__()
        if len(init_density) != 4:
            raise ValueError("B1 density init is energy, fat, carb, protein")
        self.backbone = ConvNeXtEncoder(pretrained)
        channels = self.backbone.out_channels
        self.mass_head = MLPHead(channels, 1, _mass_bias(init_mass))
        self.density_head = MLPHead(channels, 4, _density_bias(init_density))

    def forward(self, rgb: torch.Tensor) -> dict[str, torch.Tensor]:
        pooled = self.backbone(rgb).mean(dim=(2, 3))
        return product(self.mass_head(pooled), self.density_head(pooled))


class TMulti(torch.nn.Module):
    """Shared ConvNeXt-Small, Plan A height branch, ingredient density branch."""

    def __init__(
        self,
        init_mass: float,
        init_density: list[float],
        vocab_size: int,
        pretrained_height: bool,
    ):
        super().__init__()
        if len(init_density) != 4:
            raise ValueError("density init must have energy, fat, carb, and protein")
        self.rgb_encoder = ConvNeXtEncoder(pretrained=False)
        self.height_encoder = HeightEncoder(pretrained_height)
        self.cab = LightCAB(self.rgb_encoder.out_channels, self.height_encoder.out_channels)
        self.gamma = torch.nn.Parameter(torch.full((1,), float(GAMMA_INIT)))
        channels = self.rgb_encoder.out_channels
        self.volume_dim = VOLUME_DIM
        self.mass_head = MLPHead(channels + VOLUME_DIM, 1, _mass_bias(init_mass))
        # Fat, carb, protein. Energy is the Atwater residual, not a fourth output.
        macro_bias = _density_bias(init_density[1:])
        self.density_head = MLPHead(channels, 3, macro_bias)
        self.residual_head = ResidualHead(channels)
        self.encoder = IngredientEncoder(vocab_size)
        self.attention = LocalSpatialSemanticAttention(channels)
        self.b1_initialized = False

    def fuse(self, f_rgb: torch.Tensor, f_h: torch.Tensor) -> torch.Tensor:
        return f_rgb + self.gamma.view(1, 1, 1, 1) * self.cab(f_rgb, f_h)

    def forward(
        self,
        rgb: torch.Tensor,
        height: torch.Tensor,
        height01: torch.Tensor,
        name_ids: torch.Tensor,
        priors: torch.Tensor,
        mask: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        f_rgb = self.rgb_encoder(rgb)
        fused = self.fuse(f_rgb, self.height_encoder(height))
        semantic = self.attention(f_rgb, self.encoder(name_ids, priors), mask)
        semantic_pooled = semantic.mean(dim=(2, 3))
        mass_input = torch.cat([fused.mean(dim=(2, 3)), volume_features(height01)], dim=1)
        mass = F.softplus(self.mass_head(mass_input))
        built = atwater_density(self.density_head(semantic_pooled), self.residual_head(semantic_pooled))
        density = built["density"]
        return {
            "mass": mass,
            "density": density,
            "yhat": mass * density,
            "delta_cal": built["delta_cal"],
        }


def _require_count(count: int, expected: int, label: str) -> None:
    if expected <= 0:
        raise RuntimeError(f"{label} has {count} parameters; freeze that count before training")
    if count != expected:
        raise RuntimeError(f"{label} has {count} parameters, frozen count is {expected}")


def build_b1(init_mass: float, init_density: list[float], pretrained: bool, enforce_count: bool = True) -> B1:
    model = B1(init_mass, init_density, pretrained)
    if enforce_count:
        _require_count(count_parameters(model), B1_PARAMS, "B1")
    return model


def build_tmulti(
    init_mass: float,
    init_density: list[float],
    vocab_size: int,
    pretrained_height: bool,
    enforce_count: bool = True,
) -> TMulti:
    model = TMulti(init_mass, init_density, vocab_size, pretrained_height)
    count = count_parameters(model)
    if enforce_count:
        _require_count(count, T_MULTI_PARAMS, "T_multi")
        if count >= PARAM_LIMIT:
            raise RuntimeError(f"T_multi has {count} parameters, limit is {PARAM_LIMIT}")
    elif count >= PARAM_LIMIT:
        raise RuntimeError(f"T_multi has {count} parameters, at or above {PARAM_LIMIT}; narrow and recount")
    return model


def parameter_groups(model: torch.nn.Module, kind: str) -> list[dict]:
    prefixes = {
        "b1": ("backbone.",),
        "t_multi": ("rgb_encoder.", "height_encoder."),
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
    assigned = len(backbone) + len(other)
    total = sum(1 for _ in model.parameters())
    if assigned != total:
        raise RuntimeError("a parameter was left out of the learning-rate groups")
    return [
        {"params": backbone, "lr": BACKBONE_LR},
        {"params": other, "lr": HEAD_LR},
    ]


def apply_trained_b1(model: TMulti, state: dict[str, torch.Tensor]) -> None:
    """Copy this round's B1 RGB encoder and regression heads. New branches stay new."""
    prefixes = {name.split(".", 1)[0] for name in state}
    if prefixes != {"backbone", "mass_head", "density_head"}:
        raise RuntimeError(f"B1 state prefixes are {sorted(prefixes)}")
    rgb_state = {
        key[len("backbone.") :]: value for key, value in state.items() if key.startswith("backbone.")
    }
    model.rgb_encoder.load_state_dict(rgb_state, strict=True)
    weight = state["mass_head.fc1.weight"]
    channels = model.rgb_encoder.out_channels
    if tuple(weight.shape) != (HEAD_HIDDEN, channels):
        raise RuntimeError(f"B1 mass weight shape is {tuple(weight.shape)}")
    density_fc1 = state["density_head.fc1.weight"]
    density_out = state["density_head.fc2.weight"]
    if tuple(density_fc1.shape) != (HEAD_HIDDEN, channels):
        raise RuntimeError(f"B1 density fc1 shape is {tuple(density_fc1.shape)}")
    if tuple(density_out.shape) != (4, HEAD_HIDDEN):
        raise RuntimeError(f"B1 density fc2 shape is {tuple(density_out.shape)}")
    with torch.no_grad():
        model.mass_head.fc1.weight.zero_()
        model.mass_head.fc1.weight[:, :channels].copy_(weight)
        model.mass_head.fc1.bias.copy_(state["mass_head.fc1.bias"])
        model.mass_head.fc2.weight.copy_(state["mass_head.fc2.weight"])
        model.mass_head.fc2.bias.copy_(state["mass_head.fc2.bias"])
        model.density_head.fc1.weight.copy_(density_fc1)
        model.density_head.fc1.bias.copy_(state["density_head.fc1.bias"])
        # B1 rows are energy, fat, carb, protein. Keep the three macro rows.
        model.density_head.fc2.weight.copy_(density_out[1:])
        model.density_head.fc2.bias.copy_(state["density_head.fc2.bias"][1:])
        if float(model.gamma.detach()) != float(GAMMA_INIT):
            raise RuntimeError("gamma moved while the B1 weights were copied")
        if int(torch.count_nonzero(model.residual_head.fc2.weight)) != 0:
            raise RuntimeError("residual head was not left at zero")
        if int(torch.count_nonzero(model.residual_head.fc2.bias)) != 0:
            raise RuntimeError("residual bias was not left at zero")
        if int(torch.count_nonzero(model.attention.proj.weight)) != 0:
            raise RuntimeError("ingredient projection was not left at zero")
    model.b1_initialized = True


def assert_encoder_counts(model: TMulti) -> None:
    if HEIGHT_ENCODER_PARAMS <= 0 or RGB_ENCODER_PARAMS <= 0:
        return
    rgb_count = count_parameters(model.rgb_encoder)
    height_count = count_parameters(model.height_encoder)
    if rgb_count != RGB_ENCODER_PARAMS or height_count != HEIGHT_ENCODER_PARAMS:
        raise RuntimeError(
            f"encoder counts rgb={rgb_count} height={height_count} "
            f"do not match the frozen config"
        )
