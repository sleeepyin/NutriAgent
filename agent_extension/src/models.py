"""Weight model classes used for inference only. Import-safe: no experiment on import."""
from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
from torchvision import models


class WeightResNet4(nn.Module):
    def __init__(self, dropout=0.3):
        super().__init__()
        self.backbone = models.resnet18(weights=None)
        old = self.backbone.conv1
        new = nn.Conv2d(4, 64, 7, 2, 3, bias=False)
        new.weight.data[:, :3] = old.weight.data
        new.weight.data[:, 3] = old.weight.data.mean(1)
        self.backbone.conv1 = new
        f = self.backbone.fc.in_features
        self.backbone.fc = nn.Sequential(
            nn.Dropout(dropout), nn.Linear(f, 256), nn.ReLU(),
            nn.Dropout(dropout * 0.5), nn.Linear(256, 64), nn.ReLU(), nn.Linear(64, 1),
        )

    def forward(self, x):
        return self.backbone(x).squeeze(-1)

    def features(self, x):
        b = self.backbone
        x = b.relu(b.bn1(b.conv1(x)))
        x = b.maxpool(x)
        x = b.layer4(b.layer3(b.layer2(b.layer1(x))))
        x = b.avgpool(x)
        return x.squeeze(-1).squeeze(-1)


class WeightEffNet4(nn.Module):
    def __init__(self, dropout=0.3):
        super().__init__()
        self.backbone = models.efficientnet_b0(weights=None)
        old = self.backbone.features[0][0]
        new = nn.Conv2d(4, 32, 3, 2, 1, bias=False)
        new.weight.data[:, :3] = old.weight.data
        new.weight.data[:, 3] = old.weight.data.mean(1)
        self.backbone.features[0][0] = new
        f = self.backbone.classifier[1].in_features
        self.backbone.classifier = nn.Sequential(
            nn.Dropout(dropout), nn.Linear(f, 128), nn.ReLU(),
            nn.Dropout(dropout * 0.5), nn.Linear(128, 1),
        )

    def forward(self, x):
        return self.backbone(x).squeeze(-1)


class WeightNet3ch(nn.Module):
    def __init__(self, backbone="resnet18", dropout=0.3):
        super().__init__()
        if backbone == "resnet18":
            self.backbone = models.resnet18(weights=None)
            nf = self.backbone.fc.in_features
            self.backbone.fc = nn.Sequential(
                nn.Dropout(dropout), nn.Linear(nf, 256), nn.ReLU(),
                nn.Dropout(dropout * 0.5), nn.Linear(256, 64), nn.ReLU(), nn.Linear(64, 1),
            )
        else:
            self.backbone = models.efficientnet_b0(weights=None)
            nf = self.backbone.classifier[1].in_features
            self.backbone.classifier = nn.Sequential(
                nn.Dropout(dropout), nn.Linear(nf, 128), nn.ReLU(),
                nn.Dropout(dropout * 0.5), nn.Linear(128, 1),
            )

    def forward(self, x):
        return self.backbone(x).squeeze(-1)


@torch.no_grad()
def predict_np(model, X, device, bs=64) -> np.ndarray:
    model.eval()
    outs = []
    n = len(X)
    for i in range(0, n, bs):
        x = torch.tensor(X[i:i + bs], dtype=torch.float32, device=device)
        outs.append(model(x).detach().cpu().numpy())
    return np.clip(np.concatenate(outs), 0, None)


@torch.no_grad()
def features_np(model, X, device, bs=64) -> np.ndarray:
    model.eval()
    outs = []
    for i in range(0, len(X), bs):
        x = torch.tensor(X[i:i + bs], dtype=torch.float32, device=device)
        if hasattr(model, "features"):
            outs.append(model.features(x).detach().cpu().numpy())
        else:
            outs.append(model(x).detach().cpu().numpy().reshape(-1, 1))
    return np.concatenate(outs, axis=0)


def load_state(model, path, device):
    sd = torch.load(path, map_location=device, weights_only=True)
    model.load_state_dict(sd)
    model.to(device)
    model.eval()
    return model
