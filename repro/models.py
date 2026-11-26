"""DPF-Nutrition and Swin-Nutrition modules."""

from __future__ import annotations

from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
import timm
import torchvision
from timm.models.swin_transformer import checkpoint_filter_fn as swin_filter
from timm.models.vision_transformer import checkpoint_filter_fn as vit_filter

from third_party.dpt.models import DPTDepthModel


class CrossModalAttention(nn.Module):
    """DPF CAB, equations (3)-(5). One conv, no channel reduction."""

    def __init__(self, channels: int):
        super().__init__()
        self.channel = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(channels),
            nn.ReLU(inplace=True),
        )
        self.spatial = nn.Sequential(
            nn.Conv2d(1, 1, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(1),
            nn.ReLU(inplace=True),
        )
        self.project = nn.Conv2d(channels * 2, channels, kernel_size=1, bias=False)

    def forward(self, rgb: torch.Tensor, depth: torch.Tensor) -> torch.Tensor:
        added = rgb + depth
        channel = torch.sigmoid(self.channel(added.mean(dim=(2, 3), keepdim=True)))
        spatial = torch.sigmoid(self.spatial(added.mean(dim=1, keepdim=True)))
        rgb = rgb * channel * spatial
        depth = depth * channel * spatial
        return self.project(torch.cat((rgb, depth), dim=1))


class ResNetTaps(nn.Module):
    def __init__(self, net: nn.Module):
        super().__init__()
        self.conv1 = net.conv1
        self.bn1 = net.bn1
        self.relu = net.relu
        self.maxpool = net.maxpool
        self.layer1 = net.layer1
        self.layer2 = net.layer2
        self.layer3 = net.layer3
        self.layer4 = net.layer4

    def forward(self, x: torch.Tensor) -> list[torch.Tensor]:
        x = self.relu(self.bn1(self.conv1(x)))
        c0 = x
        x = self.maxpool(x)
        c1 = self.layer1(x)
        c2 = self.layer2(c1)
        c3 = self.layer3(c2)
        c4 = self.layer4(c3)
        return [c0, c1, c2, c3, c4]


class DPFNutrition(nn.Module):
    """RGB ResNet-101, depth ResNet-101, CAB, third ResNet-101, five heads."""

    def __init__(self):
        super().__init__()
        rgb = torchvision.models.resnet101(weights=None)
        depth = torchvision.models.resnet101(weights=None)
        fuse = torchvision.models.resnet101(weights=None)
        self.rgb = ResNetTaps(rgb)
        self.depth = ResNetTaps(depth)
        self.fuse_maxpool = fuse.maxpool
        self.fuse_layer1 = fuse.layer1
        self.fuse_layer2 = fuse.layer2
        self.fuse_layer3 = fuse.layer3
        self.fuse_layer4 = fuse.layer4
        channels = (64, 256, 512, 1024, 2048)
        self.cabs = nn.ModuleList(CrossModalAttention(count) for count in channels)
        self.fc = nn.Linear(2048, 2048)
        self.heads = nn.ModuleList(nn.Linear(2048, 1) for _ in range(5))

    def load_food2k(self, state: dict[str, torch.Tensor]) -> None:
        cleaned = {key: value for key, value in state.items() if not key.startswith("fc.")}
        for module in (self.rgb, self.depth):
            missing, unexpected = module.load_state_dict(cleaned, strict=False)
            bad = [key for key in missing if "num_batches_tracked" not in key]
            if bad or unexpected:
                raise RuntimeError(f"Food2K load mismatch missing={bad[:8]} unexpected={unexpected[:8]}")
        fuse_state = {
            key: value
            for key, value in cleaned.items()
            if key.startswith(("maxpool.", "layer1.", "layer2.", "layer3.", "layer4."))
        }
        own = {
            **{f"fuse_layer1.{key[7:]}": value for key, value in cleaned.items() if key.startswith("layer1.")},
            **{f"fuse_layer2.{key[7:]}": value for key, value in cleaned.items() if key.startswith("layer2.")},
            **{f"fuse_layer3.{key[7:]}": value for key, value in cleaned.items() if key.startswith("layer3.")},
            **{f"fuse_layer4.{key[7:]}": value for key, value in cleaned.items() if key.startswith("layer4.")},
        }
        missing, unexpected = self.load_state_dict(own, strict=False)
        layer_missing = [key for key in missing if key.startswith("fuse_layer")]
        if layer_missing or unexpected:
            raise RuntimeError(f"fusion Food2K load mismatch {layer_missing[:8]} {unexpected[:8]}")
        del fuse_state

    def forward(self, rgb: torch.Tensor, depth: torch.Tensor) -> torch.Tensor:
        rgb_maps = self.rgb(rgb)
        depth_maps = self.depth(depth)
        fused = [cab(left, right) for cab, left, right in zip(self.cabs, rgb_maps, depth_maps)]
        feature = fused[0]
        feature = _add(fused[1], self.fuse_layer1(self.fuse_maxpool(feature)))
        feature = _add(fused[2], self.fuse_layer2(feature))
        feature = _add(fused[3], self.fuse_layer3(feature))
        feature = _add(fused[4], self.fuse_layer4(feature))
        pooled = F.adaptive_avg_pool2d(feature, 1).flatten(1)
        shared = self.fc(pooled)
        return torch.cat([head(shared) for head in self.heads], dim=1)


def _add(left: torch.Tensor, right: torch.Tensor) -> torch.Tensor:
    if left.shape != right.shape:
        raise RuntimeError(f"fusion add shape {tuple(left.shape)} vs {tuple(right.shape)}")
    return left + right


class DepthModule(nn.Module):
    def __init__(self):
        super().__init__()
        self.dpt = DPTDepthModel(path=None, non_negative=True, backbone="vitb_rn50_384")

    def load_encoder(self, path: str | Path) -> int:
        blob = torch.load(path, map_location="cpu")
        state = blob.get("model", blob) if isinstance(blob, dict) and "model" in blob else blob
        if not isinstance(state, dict):
            raise RuntimeError("encoder checkpoint is not a state dict")
        try:
            state = vit_filter(state, self.dpt.pretrained.model)
        except Exception:
            state = {key.replace("module.", ""): value for key, value in state.items()}
        target = self.dpt.pretrained.model.state_dict()
        usable = {key: value for key, value in state.items() if key in target and target[key].shape == value.shape}
        if len(usable) < 100:
            raise RuntimeError(f"encoder checkpoint matched only {len(usable)} tensors")
        self.dpt.pretrained.model.load_state_dict(usable, strict=False)
        return len(usable)

    def forward(self, rgb: torch.Tensor) -> torch.Tensor:
        pred = self.dpt(rgb)
        if pred.shape[-2:] != rgb.shape[-2:]:
            pred = F.interpolate(
                pred.unsqueeze(1), size=rgb.shape[-2:], mode="bilinear", align_corners=True
            ).squeeze(1)
        return pred


class SwinNutrition(nn.Module):
    def __init__(self, drop_path_rate: float = 0.0, ffm_channels: int = 256):
        super().__init__()
        self.backbone = timm.create_model(
            "swin_base_patch4_window7_224",
            pretrained=False,
            features_only=True,
            out_indices=(0, 1, 2, 3),
            drop_path_rate=drop_path_rate,
            strict_img_size=False,
            dynamic_img_size=True,
        )
        channels = self.backbone.feature_info.channels()
        if tuple(channels) != (128, 256, 512, 1024):
            raise RuntimeError(f"unexpected Swin-B channels {channels}")
        self.lateral = nn.ModuleList(nn.Conv2d(count, ffm_channels, kernel_size=1) for count in channels)
        self.head = nn.Linear(ffm_channels * 4, 5)

    def load_imagenet(self, path: str | Path) -> int:
        blob = torch.load(path, map_location="cpu")
        state = blob["model"] if isinstance(blob, dict) and "model" in blob else blob
        donor = timm.create_model(
            "swin_base_patch4_window7_224",
            pretrained=False,
            num_classes=0,
            drop_path_rate=0.0,
            strict_img_size=False,
            dynamic_img_size=True,
        )
        state = swin_filter(state, donor)
        donor.load_state_dict(state, strict=False)
        renamed = {key.replace("layers.", "layers_"): value for key, value in donor.state_dict().items()}
        target = self.backbone.state_dict()
        usable = {
            key: value
            for key, value in renamed.items()
            if key in target and tuple(target[key].shape) == tuple(value.shape)
        }
        if len(usable) < 100:
            raise RuntimeError(f"Swin checkpoint matched only {len(usable)} tensors")
        self.backbone.load_state_dict(usable, strict=False)
        return len(usable)

    def forward(self, rgb: torch.Tensor) -> torch.Tensor:
        maps = [_as_nchw(tensor) for tensor in self.backbone(rgb)]
        laterals = [layer(tensor) for layer, tensor in zip(self.lateral, maps)]
        fused = [None] * 4
        fused[3] = laterals[3]
        for index in (2, 1, 0):
            up = F.interpolate(fused[index + 1], size=laterals[index].shape[-2:], mode="nearest")
            fused[index] = laterals[index] + up
        pooled = [F.adaptive_avg_pool2d(tensor, 1).flatten(1) for tensor in fused]
        return self.head(torch.cat(pooled, dim=1))


def _as_nchw(tensor: torch.Tensor) -> torch.Tensor:
    if tensor.ndim != 4:
        raise RuntimeError(f"expected NCHW or NHWC, got {tuple(tensor.shape)}")
    if tensor.shape[1] in (128, 256, 512, 1024):
        return tensor
    if tensor.shape[-1] in (128, 256, 512, 1024):
        return tensor.permute(0, 3, 1, 2).contiguous()
    raise RuntimeError(f"cannot locate Swin channels in {tuple(tensor.shape)}")
