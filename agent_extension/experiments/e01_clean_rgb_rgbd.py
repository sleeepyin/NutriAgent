#!/usr/bin/env python3
"""
E1 clean RGB / RGB-D retrain under the SAME split, seed, and clamp-free augment.
Default: do not block the Agent pipeline. Checkpoints resume. Heartbeat every epoch.
Fallback if this fails: same-4ch depth-zero ablation + existing 3ch-clean vs 4ch-buggy (flagged).
"""
from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from torchvision import models

from .common import add_common_args, out_dir, save_status
from agent_extension.src.paths import Paths
from agent_extension.src.runtime import Heartbeat, atomic_write_json, log


def augment_safe(x: torch.Tensor) -> torch.Tensor:
    if torch.rand(1).item() < 0.5:
        x = torch.flip(x, dims=[-1])
    rgb = x[:3].clone()
    if torch.rand(1).item() < 0.5:
        rgb = rgb * float(np.random.uniform(0.9, 1.1))
    if torch.rand(1).item() < 0.3:
        rgb = rgb + float(np.random.uniform(-0.05, 0.05))
    x = x.clone()
    x[:3] = rgb
    return x


class AugDS(TensorDataset):
    def __init__(self, X, y, aug=False):
        super().__init__(torch.tensor(X, dtype=torch.float32), torch.tensor(y, dtype=torch.float32))
        self.aug = aug

    def __getitem__(self, i):
        x, y = super().__getitem__(i)
        if self.aug:
            x = augment_safe(x)
        return x, y


def make_model(kind: str, ch: int, dropout=0.3) -> nn.Module:
    if kind == "resnet18":
        m = models.resnet18(weights="IMAGENET1K_V1")
        if ch == 4:
            old = m.conv1
            new = nn.Conv2d(4, 64, 7, 2, 3, bias=False)
            new.weight.data[:, :3] = old.weight.data
            new.weight.data[:, 3] = old.weight.data.mean(1)
            m.conv1 = new
        nf = m.fc.in_features
        m.fc = nn.Sequential(
            nn.Dropout(dropout), nn.Linear(nf, 256), nn.ReLU(),
            nn.Dropout(dropout * 0.5), nn.Linear(256, 64), nn.ReLU(), nn.Linear(64, 1),
        )
        return m
    m = models.efficientnet_b0(weights="IMAGENET1K_V1")
    if ch == 4:
        old = m.features[0][0]
        new = nn.Conv2d(4, 32, 3, 2, 1, bias=False)
        new.weight.data[:, :3] = old.weight.data
        new.weight.data[:, 3] = old.weight.data.mean(1)
        m.features[0][0] = new
    nf = m.classifier[1].in_features
    m.classifier = nn.Sequential(
        nn.Dropout(dropout), nn.Linear(nf, 128), nn.ReLU(),
        nn.Dropout(dropout * 0.5), nn.Linear(128, 1),
    )
    return m


def log_mse(pred, target):
    pred = torch.clamp(pred, min=0.1)
    return F.mse_loss(torch.log(pred), torch.log(target))


@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    ps, ys = [], []
    for x, y in loader:
        p = model(x.to(device)).squeeze(-1).cpu().numpy()
        ps.append(p)
        ys.append(y.numpy())
    p = np.clip(np.concatenate(ps), 0, None)
    y = np.concatenate(ys)
    mape = float(np.mean(np.abs(p - y) / np.maximum(y, 1e-6)))
    wape = float(np.sum(np.abs(p - y)) / np.maximum(np.sum(np.abs(y)), 1e-6))
    return {"mape": mape, "wape": wape, "mae": float(np.mean(np.abs(p - y)))}


def train_one(name, model, train_loader, val_loader, device, epochs, ckpt, hb: Heartbeat | None):
    model = model.to(device)
    opt = optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-4)
    sch = optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs, eta_min=1e-6)
    start = 0
    best = {"mape": 1e9, "state": None}
    if ckpt.exists():
        blob = torch.load(ckpt, map_location=device, weights_only=False)
        model.load_state_dict(blob["model"])
        opt.load_state_dict(blob["opt"])
        start = int(blob.get("epoch", 0)) + 1
        best = blob.get("best", best)
        log(f"resume {name} from epoch {start}")
    for ep in range(start, epochs):
        model.train()
        tot = 0.0
        n = 0
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            opt.zero_grad()
            loss = log_mse(model(x).squeeze(-1), y)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            tot += float(loss.item()) * len(x)
            n += len(x)
        sch.step()
        if (ep + 1) % 5 == 0 or ep == epochs - 1:
            ev = evaluate(model, val_loader, device)
            log(f"{name} ep={ep+1} loss={tot/max(n,1):.4f} val_mape={ev['mape']:.4f} val_wape={ev['wape']:.4f}")
            if ev["mape"] < best["mape"]:
                best = {**ev, "state": {k: v.detach().cpu() for k, v in model.state_dict().items()}}
        torch.save({"epoch": ep, "model": model.state_dict(), "opt": opt.state_dict(), "best": {k: best[k] for k in best if k != "state"}}, ckpt)
        if hb:
            hb.touch(step=f"e01:{name}", note=f"epoch {ep+1}/{epochs}")
    if best.get("state"):
        model.load_state_dict(best["state"])
    return model, {k: v for k, v in best.items() if k != "state"}


def main(argv=None) -> int:
    ap = add_common_args(argparse.ArgumentParser())
    ap.add_argument("--epochs", type=int, default=int(os.environ.get("CAIA_E01_EPOCHS", "100")))
    ap.add_argument("--only", default=os.environ.get("CAIA_E01_ONLY", ""), help="e.g. 3ch_resnet18")
    args = ap.parse_args(argv)
    paths = Paths.resolve()
    d = out_dir("e01_clean_rgb_rgbd", args)
    hb = Heartbeat(paths.logs / "heartbeat_e01.json", interval=15)
    hb.start()
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    z = np.load(paths.file("deep_crops"))
    Xtr, ytr = z["X_train"], z["y_train"]
    Xte, yte = z["X_test"], z["y_test"]
    jobs = [
        ("3ch_resnet18", "resnet18", 3),
        ("3ch_efficientnet", "efficientnet_b0", 3),
        ("4ch_resnet18", "resnet18", 4),
        ("4ch_efficientnet", "efficientnet_b0", 4),
    ]
    if args.only:
        jobs = [j for j in jobs if j[0] == args.only]
    results = {}
    try:
        for name, kind, ch in jobs:
            final_p = d / f"{name}_clean.pth"
            if final_p.exists() and os.environ.get("CAIA_E01_FORCE") != "1":
                log(f"skip existing {final_p}")
                results[name] = {"skipped": True, "path": str(final_p)}
                continue
            Xtr_c = Xtr if ch == 4 else Xtr[:, :3]
            Xte_c = Xte if ch == 4 else Xte[:, :3]
            tr = DataLoader(AugDS(Xtr_c, ytr, aug=True), batch_size=32, shuffle=True, num_workers=0)
            te = DataLoader(AugDS(Xte_c, yte, aug=False), batch_size=64, shuffle=False, num_workers=0)
            model = make_model(kind, ch)
            model, ev = train_one(name, model, tr, te, device, args.epochs, d / f"{name}.ckpt", hb)
            torch.save(model.state_dict(), final_p)
            ev_final = evaluate(model, te, device)
            results[name] = {"path": str(final_p), **ev_final, "best_during": ev}
            atomic_write_json(d / "progress.json", results)
            log(f"saved {final_p} holdout MAPE={ev_final['mape']:.4f} WAPE={ev_final['wape']:.4f}")
        save_status(d, True, results=results, note="clamp-free augment; same cache split")
        return 0
    except Exception as e:
        log(f"E01 failed: {e}", "ERROR")
        save_status(d, False, error=str(e), results=results, fallback="use existing 3ch + 4ch-depth0")
        return 2
    finally:
        hb.stop()


if __name__ == "__main__":
    raise SystemExit(main())
