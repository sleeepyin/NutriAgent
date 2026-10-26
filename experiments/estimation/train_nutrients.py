#!/usr/bin/env python3
"""
Nutrient Density v4 — GBML: Gradient-Balanced Multi-task Learning
==================================================================
Problem: v2's plain MSE on log1p-density is dominated by Calories
(cal loss ≈ 0.06 vs fat ≈ 0.001 → 60× gradient imbalance; fat barely trains).

Fix: homoscedastic uncertainty weighting (Kendall et al.) with loss-scale init:
    L = Σ_i [ exp(-s_i) · MSE_i + s_i ],   s_i = ½·log σ_i² learnable
Init s_i = ½·log(MSE_i(first batch)) so every nutrient starts with equal
effective gradient contribution, then weights adapt during training.

Everything else matches v2 exactly (architecture, epochs, LR, WD, no aug).
Standalone runs write density_*_v4.pth under backup/v3_excluded/weights/nutrition5k/.
The checkpoints used by C3 stay at density_*_final.pth.
"""
import os, sys, warnings
from pathlib import Path
import numpy as np
import torch, torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from torchvision import models
from sklearn.metrics import mean_absolute_error, r2_score
warnings.filterwarnings('ignore')

SEED = 42
np.random.seed(SEED); torch.manual_seed(SEED)
DEVICE = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
def _boot():
    here = Path(__file__).resolve()
    for parent in here.parents:
        if (parent / "src" / "mmcfa" / "paths.py").is_file():
            for entry in (str(parent), str(parent / "src")):
                if entry not in sys.path:
                    sys.path.insert(0, entry)
            return
    raise RuntimeError("找不到项目根目录（缺少 src/mmcfa/paths.py）")


_boot()
from mmcfa.paths import Paths, n5k

OUT_DIR = str(Paths.resolve().root / "backup" / "v3_excluded" / "results" / "estimation" / "e5")

BATCH, EPOCHS, LR, WD = 32, 100, 1e-4, 1e-4
NAMES = ['Calories', 'Fat', 'Carbs', 'Protein']


class DensityResNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.backbone = models.resnet18(weights=None)
        old = self.backbone.conv1
        new = nn.Conv2d(4, 64, 7, 2, 3, bias=False)
        new.weight.data[:, :3] = old.weight.data
        new.weight.data[:, 3] = old.weight.data.mean(1)
        self.backbone.conv1 = new
        f = self.backbone.fc.in_features
        self.backbone.fc = nn.Sequential(
            nn.Dropout(0.3), nn.Linear(f, 256), nn.ReLU(),
            nn.Dropout(0.15), nn.Linear(256, 64), nn.ReLU(),
            nn.Linear(64, 4))
    def forward(self, x): return self.backbone(x)


class DensityEffNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.backbone = models.efficientnet_b0(weights=None)
        old = self.backbone.features[0][0]
        new = nn.Conv2d(4, 32, 3, 2, 1, bias=False)
        new.weight.data[:, :3] = old.weight.data
        new.weight.data[:, 3] = old.weight.data.mean(1)
        self.backbone.features[0][0] = new
        f = self.backbone.classifier[1].in_features
        self.backbone.classifier = nn.Sequential(
            nn.Dropout(0.3), nn.Linear(f, 128), nn.ReLU(),
            nn.Dropout(0.15), nn.Linear(128, 4))
    def forward(self, x): return self.backbone(x)


def train_gbml(model, loader, name):
    """Train with gradient-balanced multi-task loss (uncertainty weighting)."""
    model = model.to(DEVICE)
    log_s = torch.zeros(4, device=DEVICE, requires_grad=True)
    opt = optim.AdamW([
        {'params': model.parameters(), 'weight_decay': WD},
        {'params': [log_s], 'weight_decay': 0.0}], lr=LR)
    sched = optim.lr_scheduler.CosineAnnealingLR(opt, EPOCHS)

    header = (f"{'Ep':>4s} | {'Cal':>7s} {'Fat':>7s} {'Carb':>7s} {'Prot':>7s} | "
              f"{'w_cal':>6s} {'w_fat':>6s} {'w_carb':>6s} {'w_prot':>6s} | {'total':>7s} | {'best':>5s}")
    print(f"\n  [{name}] " + header)
    print("  " + "-" * len(header))

    best_loss, best_state, best_ep = float('inf'), None, 0
    inited = False

    for ep in range(EPOCHS):
        model.train()
        acc = np.zeros(4); tot = 0.0; n = 0
        for x, y in loader:
            x, y = x.to(DEVICE), y.to(DEVICE)
            opt.zero_grad()
            pred = model(x)
            per = ((pred - y) ** 2).mean(dim=0)                  # (4,) raw per-task MSE
            if not inited:                                       # loss-scale init
                with torch.no_grad():
                    log_s.data = 0.5 * torch.log(per.detach() + 1e-12)
                inited = True
            loss = (per * torch.exp(-log_s) + log_s).sum()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            with torch.no_grad():                                # keep weights finite
                log_s.clamp_(-6.0, 6.0)
            opt.step()
            acc += per.detach().cpu().numpy() * len(x)
            tot += loss.item() * len(x); n += len(x)
        sched.step()
        acc /= n; tot /= n
        wts = torch.exp(-log_s).detach().cpu().numpy()

        if tot < best_loss:
            best_loss, best_ep = tot, ep + 1
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}

        flag = "  ★" if ep + 1 == best_ep else ""
        if (ep + 1) % 5 == 0 or ep == 0:
            print(f"  {ep+1:>4d} | " + " ".join(f"{v:>7.4f}" for v in acc) + " | " +
                  " ".join(f"{v:>6.2f}" for v in wts) + f" | {tot:>7.4f} |{flag}",
                  flush=True)

    model.load_state_dict(best_state)
    print(f"  [{name}] best epoch {best_ep}, total={best_loss:.4f}")
    return model


@torch.no_grad()
def predict(model, X, bs=BATCH):
    model.eval()
    preds = []
    for i in range(0, len(X), bs):
        x = torch.tensor(X[i:i+bs], dtype=torch.float32).to(DEVICE)
        preds.append(model(x).cpu().numpy())
    return np.concatenate(preds, axis=0)


def main():
    print("=" * 70)
    print("Density v4 — GBML (gradient-balanced multi-task loss)")
    print("=" * 70)
    print(f"Device: {DEVICE}")

    data = np.load(n5k('deep_crops_cache.npz'))
    X_train, X_test = data['X_train'], data['X_test']
    y_train_w, y_test_w = data['y_train'], data['y_test']
    dens = np.load(n5k('nutrient_density.npz'))
    y_train_d, y_test_d = dens['y_train_density'], dens['y_test_density']
    nutr = np.load(n5k('nutrient_labels.npz'))
    y_test_n = nutr['y_test_nutr']
    print(f"Train {len(X_train)}, Test {len(X_test)}")

    # weight prediction (frozen ensemble)
    from test_100_samples import WeightResNet, WeightEffNet
    w_rn = WeightResNet().to(DEVICE)
    w_rn.load_state_dict(torch.load(n5k('deep_weight_resnet18_final.pth'),
                                    map_location=DEVICE, weights_only=True)); w_rn.eval()
    w_eff = WeightEffNet().to(DEVICE)
    w_eff.load_state_dict(torch.load(n5k('deep_weight_efficientnet_final.pth'),
                                     map_location=DEVICE, weights_only=True)); w_eff.eval()
    w_pred = np.maximum((predict(w_rn, X_test) + predict(w_eff, X_test)) / 2, 0)
    W = w_pred.reshape(-1, 1)

    # ---- train v4 ----
    y_train_log = np.log1p(y_train_d)
    loader = DataLoader(TensorDataset(torch.tensor(X_train, dtype=torch.float32),
                                      torch.tensor(y_train_log, dtype=torch.float32)),
                        batch_size=BATCH, shuffle=True, num_workers=0, pin_memory=True)

    print("\n[1/3] Training DensityResNet18 (GBML)...")
    rn = train_gbml(DensityResNet(), loader, "RN18")
    v4_dir = Paths.resolve().root / "backup" / "v3_excluded" / "weights" / "nutrition5k"
    v4_dir.mkdir(parents=True, exist_ok=True)
    torch.save(rn.state_dict(), v4_dir / "density_resnet18_v4.pth")

    print("\n[2/3] Training DensityEffNet (GBML)...")
    eff = train_gbml(DensityEffNet(), loader, "Eff")
    torch.save(eff.state_dict(), v4_dir / "density_efficientnet_v4.pth")

    # ---- evaluate v4 vs v2 ----
    print("\n[3/3] Evaluation — v4 vs v2 (ensemble, weight × density)")
    d_v4 = np.maximum(np.expm1((predict(rn, X_test) + predict(eff, X_test)) / 2), 0)

    d_rn2 = DensityResNet().to(DEVICE)
    d_rn2.load_state_dict(torch.load(n5k('density_resnet18_final.pth'),
                                     map_location=DEVICE, weights_only=True)); d_rn2.eval()
    d_eff2 = DensityEffNet().to(DEVICE)
    d_eff2.load_state_dict(torch.load(n5k('density_efficientnet_final.pth'),
                                      map_location=DEVICE, weights_only=True)); d_eff2.eval()
    d_v2 = np.maximum(np.expm1((predict(d_rn2, X_test) + predict(d_eff2, X_test)) / 2), 0)

    print("\n" + "=" * 78)
    print(f"{'Nutrient':<10s} {'v2 PMAE':>9s} {'v4 PMAE':>9s} {'Δ':>8s} | "
          f"{'v2 densMAE':>10s} {'v4 densMAE':>10s} | {'v2 R²':>7s} {'v4 R²':>7s}")
    print("-" * 78)
    for i, name in enumerate(NAMES):
        m2 = mean_absolute_error(y_test_n[:, i], (W * d_v2)[:, i])
        m4 = mean_absolute_error(y_test_n[:, i], (W * d_v4)[:, i])
        p2, p4 = m2 / y_test_n[:, i].mean() * 100, m4 / y_test_n[:, i].mean() * 100
        dm2 = mean_absolute_error(y_test_d[:, i], d_v2[:, i])
        dm4 = mean_absolute_error(y_test_d[:, i], d_v4[:, i])
        r2 = r2_score(y_test_n[:, i], (W * d_v2)[:, i])
        r4 = r2_score(y_test_n[:, i], (W * d_v4)[:, i])
        print(f"{name:<10s} {p2:>8.1f}% {p4:>8.1f}% {p4-p2:>+7.1f}% | "
              f"{dm2:>10.4f} {dm4:>10.4f} | {r2:>7.3f} {r4:>7.3f}")

    os.makedirs(OUT_DIR, exist_ok=True)
    np.savez(os.path.join(OUT_DIR, 'nutrient_v4_results.npz'),
             y_test_n=y_test_n, d_v2=d_v2, d_v4=d_v4, w_pred=w_pred)
    print(f"\nSaved: {OUT_DIR}/nutrient_v4_results.npz")
    print("Done!")


if __name__ == '__main__':
    main()
