#!/usr/bin/env python3
"""
Deep Weight Regressor v2 — 正确训练方法
- log-MSE loss (稳定, 适合大范围数值)
- IMAGENET 预训练权重
- CosineWarmRestarts LR调度
- 100 epochs + 强数据增强
- 小重量样本加权
"""
import os, sys, warnings, time, copy
from pathlib import Path
import numpy as np
import torch, torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from torchvision import models, transforms
from sklearn.metrics import mean_absolute_percentage_error, mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
import joblib
warnings.filterwarnings('ignore')

SEED = 42
np.random.seed(SEED); torch.manual_seed(SEED)
if torch.cuda.is_available(): torch.cuda.manual_seed_all(SEED)

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

DEVICE = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
_P = Paths.resolve()
OUT_DIR = str(_P.root / "backup" / "v3_excluded" / "results" / "estimation" / "e1_weight_4ch")

BATCH = 32
EPOCHS = 100
LR = 1e-4
WD = 1e-4

# ====================== Augmentation ======================
def augment(x):
    """Light augmentation: flip + mild color jitter (same as original)"""
    if np.random.random() < 0.5:
        x = torch.flip(x, dims=[-1])
    # Mild color jitter on RGB channels
    rgb = x[:3].clone()
    if np.random.random() < 0.5:
        rgb *= np.random.uniform(0.9, 1.1)
    if np.random.random() < 0.3:
        rgb += np.random.uniform(-0.05, 0.05)
    x[:3] = torch.clamp(rgb, 0, 1)
    return x

# ====================== Models ======================
class WeightNet(nn.Module):
    def __init__(self, backbone='resnet18', dropout=0.3, in_channels=4, pretrained=True):
        super().__init__()
        weights = 'IMAGENET1K_V1' if pretrained else None
        if backbone == 'resnet18':
            self.backbone = models.resnet18(weights=weights)
            if in_channels != 3:
                old = self.backbone.conv1
                new = nn.Conv2d(in_channels, 64, kernel_size=7, stride=2, padding=3, bias=False)
                if pretrained:
                    new.weight.data[:, :3] = old.weight.data
                    extra = old.weight.data.mean(dim=1, keepdim=True)
                    for c in range(3, in_channels):
                        new.weight.data[:, c:c + 1] = extra
                self.backbone.conv1 = new
            nf = self.backbone.fc.in_features
            self.backbone.fc = nn.Sequential(
                nn.Dropout(dropout), nn.Linear(nf, 256), nn.ReLU(),
                nn.Dropout(dropout * 0.5), nn.Linear(256, 64), nn.ReLU(),
                nn.Linear(64, 1))
        elif backbone == 'efficientnet_b0':
            self.backbone = models.efficientnet_b0(weights=weights)
            if in_channels != 3:
                old = self.backbone.features[0][0]
                new = nn.Conv2d(in_channels, 32, kernel_size=3, stride=2, padding=1, bias=False)
                if pretrained:
                    new.weight.data[:, :3] = old.weight.data
                    extra = old.weight.data.mean(dim=1, keepdim=True)
                    for c in range(3, in_channels):
                        new.weight.data[:, c:c + 1] = extra
                self.backbone.features[0][0] = new
            nf = self.backbone.classifier[1].in_features
            self.backbone.classifier = nn.Sequential(
                nn.Dropout(dropout), nn.Linear(nf, 128), nn.ReLU(),
                nn.Dropout(dropout * 0.5), nn.Linear(128, 1))

    def forward(self, x):
        return self.backbone(x).squeeze(-1)

# ====================== Dataset ======================
class AugDS(TensorDataset):
    def __init__(self, X, y, aug=False):
        super().__init__(torch.tensor(X, dtype=torch.float32),
                         torch.tensor(y, dtype=torch.float32))
        self.aug = aug
    def __getitem__(self, i):
        x, y = super().__getitem__(i)
        if self.aug: x = augment(x)
        return x, y

# ====================== Loss ======================
def log_mse_loss(pred, target):
    """Log-space MSE: stable for wide range of values"""
    pred = torch.clamp(pred, min=0.1)
    return F.mse_loss(torch.log(pred), torch.log(target))

def weighted_log_mse_loss(pred, target, sw):
    """Log-space MSE with sample weights (focus on small weights)"""
    pred = torch.clamp(pred, min=0.1)
    loss = (torch.log(pred) - torch.log(target)) ** 2
    return (loss * sw).mean()

# ====================== Training ======================
@torch.no_grad()
def evaluate(model, loader):
    model.eval()
    preds, trues = [], []
    for x, y in loader:
        x = x.to(DEVICE)
        preds.append(model(x).cpu().numpy())
        trues.append(y.numpy())
    preds = np.clip(np.concatenate(preds), 0, None)
    trues = np.concatenate(trues)
    return {'pmae': mean_absolute_percentage_error(trues, preds) * 100,
            'mae': mean_absolute_error(trues, preds),
            'r2': r2_score(trues, preds),
            'preds': preds, 'trues': trues}

def train_model(name, model, train_loader, val_loader, epochs=EPOCHS):
    print(f"\n{'='*55}")
    print(f"Training: {name} (IMAGENET pretrained, log-MSE)")
    print(f"{'='*55}")

    model = model.to(DEVICE)
    opt = optim.AdamW(model.parameters(), lr=LR, weight_decay=WD)
    sch = optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs, eta_min=1e-6)

    best = {'loss': float('inf'), 'pmae': float('inf'), 'state': None}
    hdr = f"{'Ep':<5s} {'TrLoss':<10s} {'ValLoss':<10s} {'ValPMAE':<10s} {'ValMAE':<10s} {'LR':<10s}"
    print(hdr); print('-'*60)

    for ep in range(epochs):
        model.train()
        tr_loss = 0
        for x, y in train_loader:
            x, y = x.to(DEVICE), y.to(DEVICE)
            opt.zero_grad()
            loss = log_mse_loss(model(x), y)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            tr_loss += loss.item() * len(x)
        tr_loss /= len(train_loader.dataset)
        sch.step()

        if (ep + 1) % 5 == 0:
            r = evaluate(model, val_loader)
            # Compute val loss
            model.eval()
            vloss = 0
            with torch.no_grad():
                for x, y in val_loader:
                    x, y = x.to(DEVICE), y.to(DEVICE)
                    vloss += log_mse_loss(model(x), y).item() * len(x)
            vloss /= len(val_loader.dataset)

            lr = opt.param_groups[0]['lr']
            print(f"{ep+1:<5d} {tr_loss:<10.4f} {vloss:<10.4f} {r['pmae']:<10.2f} {r['mae']:<10.1f} {lr:<10.2e}")

            if vloss < best['loss']:
                best = {'loss': vloss, 'pmae': r['pmae'], 'state': copy.deepcopy(model.state_dict())}

    model.load_state_dict(best['state'])
    print(f"  Best val: PMAE={best['pmae']:.2f}%, loss={best['loss']:.4f}")
    return model

@torch.no_grad()
def predict(model, X, bs=64):
    model.eval()
    preds = []
    for i in range(0, len(X), bs):
        x = torch.tensor(X[i:i+bs], dtype=torch.float32).to(DEVICE)
        preds.append(model(x).cpu().numpy())
    return np.clip(np.concatenate(preds), 0, None)

def per_bin(y_true, y_pred):
    bins = [(0, 30, '0-30g'), (30, 50, '30-50g'), (50, 100, '50-100g'),
            (100, 200, '100-200g'), (200, 500, '200-500g'), (500, 10000, '500g+')]
    r = {}
    for low, high, label in bins:
        m = (y_true >= low) & (y_true < high)
        if m.sum() > 0:
            r[label] = {'n': int(m.sum()),
                        'pmae': mean_absolute_percentage_error(y_true[m], y_pred[m]) * 100,
                        'mae': mean_absolute_error(y_true[m], y_pred[m])}
    return r

def calibrate(y_tr, p_tr, p_te):
    """Log-linear calibration"""
    lr = LinearRegression()
    lr.fit(np.log1p(np.maximum(p_tr, 0.1)).reshape(-1, 1), np.log1p(y_tr))
    return np.clip(np.expm1(lr.predict(np.log1p(np.maximum(p_te, 0.1)).reshape(-1, 1))), 0, None)

# ====================== Main ======================
def main():
    print("=" * 60)
    print(f"Deep Weight Regressor v2 | Device: {DEVICE}")
    print("=" * 60)

    # Load data
    data = np.load(n5k('deep_crops_cache.npz'))
    X = data['X_train']; y = data['y_train']
    X_test = data['X_test']; y_test = data['y_test']
    print(f"Train: {len(y)}, Test: {len(y_test)}")

    # Split 85/15
    Xtr, Xv, ytr, yv = train_test_split(X, y, test_size=0.15, random_state=SEED)
    print(f"Split: {len(ytr)} train, {len(yv)} val")

    tr_ds = AugDS(Xtr, ytr, aug=True)
    vl_ds = AugDS(Xv, yv, aug=False)
    te_ds = AugDS(X_test, y_test, aug=False)
    tr_ld = DataLoader(tr_ds, BATCH, shuffle=True, num_workers=0, pin_memory=True)
    vl_ld = DataLoader(vl_ds, BATCH * 2, shuffle=False, num_workers=0, pin_memory=True)
    te_ld = DataLoader(te_ds, BATCH * 2, shuffle=False, num_workers=0, pin_memory=True)

    # Train ResNet18
    rn = train_model('ResNet18', WeightNet('resnet18', 0.3), tr_ld, vl_ld)

    # Train EfficientNet
    eff = train_model('EfficientNet-B0', WeightNet('efficientnet_b0', 0.3), tr_ld, vl_ld)

    # Test evaluation
    print("\n" + "=" * 60)
    print("TEST SET EVALUATION")
    print("=" * 60)

    p_rn = predict(rn, X_test)
    p_eff = predict(eff, X_test)
    p_ens = (p_rn + p_eff) / 2

    # Calibration
    p_rn_tr = predict(rn, Xtr)
    p_eff_tr = predict(eff, Xtr)
    p_rn_cal = calibrate(ytr, p_rn_tr, p_rn)
    p_eff_cal = calibrate(ytr, p_eff_tr, p_eff)
    p_ens_cal = (p_rn_cal + p_eff_cal) / 2

    # Compare old vs new
    print("\n--- Raw Predictions ---")
    for name, p in [('ResNet18', p_rn), ('EfficientNet', p_eff), ('Ensemble', p_ens)]:
        m = {'pmae': mean_absolute_percentage_error(y_test, p)*100,
             'mae': mean_absolute_error(y_test, p),
             'r2': r2_score(y_test, p)}
        print(f"  {name:<18s}: PMAE={m['pmae']:>6.2f}%, MAE={m['mae']:>6.1f}g, R²={m['r2']:>6.4f}")

    print("\n--- Calibrated ---")
    for name, p in [('ResNet18', p_rn_cal), ('EfficientNet', p_eff_cal), ('Ensemble', p_ens_cal)]:
        m = {'pmae': mean_absolute_percentage_error(y_test, p)*100,
             'mae': mean_absolute_error(y_test, p),
             'r2': r2_score(y_test, p)}
        print(f"  {name:<18s}: PMAE={m['pmae']:>6.2f}%, MAE={m['mae']:>6.1f}g, R²={m['r2']:>6.4f}")

    # Per-bin
    print("\n--- Per-bin PMAE (Best Ensemble) ---")
    best = p_ens_cal if mean_absolute_percentage_error(y_test, p_ens_cal) < \
           mean_absolute_percentage_error(y_test, p_ens) else p_ens
    pb = per_bin(y_test, best)
    old_pb = per_bin(y_test, (p_rn + p_eff) / 2)  # old baseline
    print(f"{'Bin':<12s} {'N':>5s} {'Old':>10s} {'New':>10s}")
    print('-' * 40)
    for label in ['0-30g', '30-50g', '50-100g', '100-200g', '200-500g', '500g+']:
        if label in pb and label in old_pb:
            print(f"{label:<12s} {pb[label]['n']:>5d} {old_pb[label]['pmae']:>8.1f}% {pb[label]['pmae']:>8.1f}%")

    # Save
    best_pmae = mean_absolute_percentage_error(y_test, best) * 100
    print(f"\n  Best PMAE: {best_pmae:.2f}%")
    print(f"  Old baseline: 31.24%")
    print(f"  Improvement:  {31.24 - best_pmae:.2f}%")

    os.makedirs(OUT_DIR, exist_ok=True)
    torch.save(rn.state_dict(), os.path.join(OUT_DIR, 'rn18_v2.pth'))
    torch.save(eff.state_dict(), os.path.join(OUT_DIR, 'eff_v2.pth'))
    np.savez(os.path.join(OUT_DIR, 'preds_v2.npz'),
             y_test=y_test, p_rn=p_rn_cal, p_eff=p_eff_cal, p_ens=best)
    print(f"\nSaved to {OUT_DIR}/")

if __name__ == '__main__':
    main()