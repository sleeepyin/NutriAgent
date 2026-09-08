#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
CQR 密度模型训练 — Conformalized Quantile Regression (2026-08-20)
=================================================================
目的: 让密度模型直接输出分位数 (τ=0.1/0.5/0.9) 而非点估计, 区间形状随
样本自适应; 再共形化 (nutrient_cqr_experiment.py) 与 AEC-S2m-N (2.42) 对比。

设计 (全部继承 v4 已验证的配置, 最小 diff):
  - log1p 空间训练: log1p 单调 ⇒ 分位数在变换下保持 (log 空间的 0.1 分位
    expm1 回来就是原空间的 0.1 分位), 同时保留 v4 的数值稳定性
  - pinball loss ρ_τ(u) = max(τu, (τ-1)u), u = y - q, 每营养素×3 分位
  - GBML 按营养素均衡 (v4 核心教训: 直接求和被 Calories 支配, fat 几乎不训练)
  - 骨干/优化器与 v4 完全一致: ResNet18 + EffNet-B0, AdamW 1e-4, cosine,
    100 epochs, bs 32, WD 1e-4, no aug
  - 分位交叉: forward 内 sort (q10 ≤ q50 ≤ q90), 推理端确定性有序

输出:
  models/density_resnet18_cqr.pth / models/density_efficientnet_cqr.pth
  models/cqr_pred_cache.npz  (652 测试集上 3 分位×4 营养素, 供 CPU 实验用)

运行 (用户终端, GPU): conda activate nutri && python train_density_cqr.py
预计 RTX 3090 上 10-20 分钟。
"""
import os, warnings
import numpy as np
import torch, torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from torchvision import models
warnings.filterwarnings('ignore')

SEED = 42
np.random.seed(SEED); torch.manual_seed(SEED)
DEVICE = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
MODEL_DIR = '/home/Nutri_0811/models'
BATCH, EPOCHS, LR, WD = 32, 100, 1e-4, 1e-4
NAMES = ['Calories', 'Fat', 'Carbs', 'Protein']
TAUS = [0.1, 0.5, 0.9]


class QuantileResNet(nn.Module):
    """ResNet18 骨干, 头输出 12 维 = 4 营养素 × 3 分位, sort 保证有序。"""
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
            nn.Linear(64, 12))
    def forward(self, x):
        q = self.backbone(x).view(-1, 4, 3)
        return torch.sort(q, dim=-1).values


class QuantileEffNet(nn.Module):
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
            nn.Dropout(0.15), nn.Linear(128, 12))
    def forward(self, x):
        q = self.backbone(x).view(-1, 4, 3)
        return torch.sort(q, dim=-1).values


def train_cqr(model, loader, name):
    """pinball + GBML 按营养素均衡 (v4 配方, 仅把 MSE 换成 pinball)。"""
    model = model.to(DEVICE)
    taus = torch.tensor(TAUS, device=DEVICE).view(1, 1, 3)
    log_s = torch.zeros(4, device=DEVICE, requires_grad=True)
    opt = optim.AdamW([
        {'params': model.parameters(), 'weight_decay': WD},
        {'params': [log_s], 'weight_decay': 0.0}], lr=LR)
    sched = optim.lr_scheduler.CosineAnnealingLR(opt, EPOCHS)

    header = (f"{'Ep':>4s} | {'pin_cal':>8s} {'pin_fat':>8s} {'pin_carb':>8s} "
              f"{'pin_prot':>8s} | {'w_cal':>6s} {'w_fat':>6s} {'w_carb':>6s} "
              f"{'w_prot':>6s} | {'total':>7s} | {'best':>5s}")
    print(f"\n  [{name}] " + header)
    print("  " + "-" * len(header))

    best_loss, best_state, best_ep, inited = float('inf'), None, 0, False
    for ep in range(EPOCHS):
        model.train()
        acc = np.zeros(4); tot = 0.0; n = 0
        for x, y in loader:
            x, y = x.to(DEVICE), y.to(DEVICE)
            opt.zero_grad()
            q = model(x)                                    # (B,4,3) 已 sort
            u = y.unsqueeze(-1) - q                        # (B,4,3)
            pin = torch.maximum(taus * u, (taus - 1) * u)  # pinball
            per = pin.mean(dim=(0, 2))                     # (4,) 每营养素
            if not inited:                                  # loss-scale init (同 v4)
                with torch.no_grad():
                    log_s.data = 0.5 * torch.log(per.detach() + 1e-12)
                inited = True
            loss = (per * torch.exp(-log_s) + log_s).sum()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            with torch.no_grad():
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
        if (ep + 1) % 5 == 0 or ep == 0:
            flag = "  ★" if ep + 1 == best_ep else ""
            print(f"  {ep+1:>4d} | " + " ".join(f"{v:>8.4f}" for v in acc) + " | "
                  + " ".join(f"{v:>6.2f}" for v in wts)
                  + f" | {tot:>7.4f} |{flag}", flush=True)

    model.load_state_dict(best_state)
    print(f"  [{name}] best epoch {best_ep}, total={best_loss:.4f}")
    return model


@torch.no_grad()
def predict(model, X, bs=BATCH):
    model.eval(); preds = []
    for i in range(0, len(X), bs):
        x = torch.tensor(X[i:i+bs], dtype=torch.float32).to(DEVICE)
        preds.append(model(x).cpu().numpy())               # (b,4,3)
    return np.concatenate(preds, axis=0)


def main():
    print("=" * 70)
    print("CQR 密度模型 — pinball (τ=0.1/0.5/0.9) + GBML, log1p 空间")
    print(f"Device: {DEVICE}")
    print("=" * 70)

    data = np.load(os.path.join(MODEL_DIR, 'deep_crops_cache.npz'))
    X_train, X_test = data['X_train'], data['X_test']
    n_tr = len(X_train)
    dens = np.load(os.path.join(MODEL_DIR, 'nutrient_density.npz'))
    y_train_d, y_test_d = dens['y_train_density'], dens['y_test_density']

    # 重量预测直接复用 aec_v2 缓存 (final 权重, 免加载重量模型)
    aec = np.load(os.path.join(MODEL_DIR, 'aec_v2_pred_cache.npz'))
    w_pred_te = (aec['final_rn'][n_tr:] + aec['final_eff'][n_tr:]) / 2.0
    W = np.maximum(w_pred_te, 0).reshape(-1, 1)
    print(f"Train {n_tr}, Test {len(X_test)}")

    y_train_log = np.log1p(y_train_d)
    loader = DataLoader(TensorDataset(
        torch.tensor(X_train, dtype=torch.float32),
        torch.tensor(y_train_log, dtype=torch.float32)),
        batch_size=BATCH, shuffle=True, num_workers=2, pin_memory=True)

    print("\n[1/2] Training QuantileResNet18 (pinball+GBML)...")
    rn = train_cqr(QuantileResNet(), loader, "RN18")
    torch.save(rn.state_dict(), os.path.join(MODEL_DIR, 'density_resnet18_cqr.pth'))

    print("\n[2/2] Training QuantileEffNet (pinball+GBML)...")
    eff = train_cqr(QuantileEffNet(), loader, "Eff")
    torch.save(eff.state_dict(), os.path.join(MODEL_DIR, 'density_efficientnet_cqr.pth'))

    # ---- 集成 (log 空间均值 → expm1; 两模型各自有序 ⇒ 均值仍有序) ----
    q_rn = predict(rn, X_test)          # (652,4,3) log 空间
    q_eff = predict(eff, X_test)
    q_ens = np.expm1((q_rn + q_eff) / 2.0)
    q_ens = np.maximum(q_ens, 0)
    d_q10, d_q50, d_q90 = q_ens[..., 0], q_ens[..., 1], q_ens[..., 2]

    # ---- Sanity: 中位点估计 vs 生产点估计; 原始 80% 区间经验覆盖 ----
    conf = np.load(os.path.join(MODEL_DIR, 'nutrient_conformal_cache.npz'))
    y_nutr = conf['y_nutr']
    n_hybrid = np.maximum(conf['w_pred'], 0).reshape(-1, 1) * \
        (0.6 * conf['d_direct'] + 0.4 * conf['d_knn'])
    print("\n" + "=" * 78)
    print(f"{'Nutrient':<10s} {'PMAE hybrid':>11s} {'PMAE median':>11s} | "
          f"{'raw-cov(10-90)':>14s} {'mean-width':>11s}")
    print("-" * 78)
    for i, nm in enumerate(NAMES):
        p_hyb = np.abs(y_nutr[:, i] - n_hybrid[:, i]).mean() / y_nutr[:, i].mean() * 100
        p_med = np.abs(y_nutr[:, i] - (W * d_q50)[:, i]).mean() / y_nutr[:, i].mean() * 100
        L, U = (W * d_q10)[:, i], (W * d_q90)[:, i]
        cov = ((y_nutr[:, i] >= L) & (y_nutr[:, i] <= U)).mean() * 100
        print(f"{nm:<10s} {p_hyb:>10.1f}% {p_med:>10.1f}% | "
              f"{cov:>13.1f}% {(U - L).mean():>11.2f}")
    print("注: raw-cov 应接近 80% (10-90 分位区间); 明显偏低 ⇒ 分位模型欠拟合")

    np.savez_compressed(os.path.join(MODEL_DIR, 'cqr_pred_cache.npz'),
                        d_q10=d_q10, d_q50=d_q50, d_q90=d_q90,
                        w_pred=w_pred_te, y_nutr=y_nutr,
                        y_density=y_test_d)
    print(f"\nSaved: {MODEL_DIR}/cqr_pred_cache.npz")
    print("下一步 (沙箱可跑): python nutrient_cqr_experiment.py")
    print("Done!")


if __name__ == '__main__':
    main()
