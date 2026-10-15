#!/usr/bin/env python3
"""
DINOv2-small 现代 backbone 消融（论文任务4）
==================================================
目标：回答"容量是不是瓶颈"。用 DINOv2 ViT-S/14 冻结特征 + 深度分支 + MLP head
在 deep_crops_cache.npz 上训重量回归，与 4ch 基线（hold-out 22.0%）公平对比。

协议（与 train_deep_v2_final.py 对齐）：
  - 数据：deep_crops_cache.npz (2608 train / 652 test, 4ch RGB-D, RGB 已 ImageNet 归一化)
  - 划分：X_train 内 85/15 split (seed 42) 留 val；最终在 X_test(652) 上测 PMAE
  - 损失：log-MSE（pred 与 target 同取 log）
  - backbone：DINOv2 ViT-S/14 冻结，weights 指向本地 pth（不走官方 dl 被墙源）
  - 4ch 问题处理：DINO 吃 RGB 3ch；深度 ch3 由独立小 CNN 分支编码，拼进 MLP head
    （与 4ch ResNet/EffNet 基线可比——后者也是卷积在 4th 通道上融合）

判读：
  - 若 DINOv2 MAPE ≤ 22.0% (4ch 基线) → 容量非瓶颈，经验天花板成立
  - 若显著更好      → 需诚实报告并调整天花板叙事

用法：
  CPU 冒烟：   python train_dinov2_ablation.py --smoke
  GPU 完整：   python train_dinov2_ablation.py --bs 32 --epochs 60 --log1p
               （--log1p 与基线 v4 的 log1p-MSE 对齐；默认 log-MSE 均可）
输出：results_deep/dinov2_ablation/{preds,pmae}.npz + 控制台对照
"""
import os, sys, math, time, argparse, copy, warnings
warnings.filterwarnings('ignore')
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_percentage_error, mean_absolute_error

SEED = 42
np.random.seed(SEED); torch.manual_seed(SEED)

DEVICE = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
MODEL_DIR = '/home/Nutri_0811/models'
OUT_DIR = '/home/Nutri_0811/results_deep/dinov2_ablation'
os.makedirs(OUT_DIR, exist_ok=True)

DINO_HUB_DIR = '/root/.cache/torch/hub/facebookresearch_dinov2_main'
if not os.path.isdir(DINO_HUB_DIR):          # 可复现包回退：脚本旁 dinov2_hub/
    DINO_HUB_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'dinov2_hub')
DINO_WEIGHTS = '/root/.cache/torch/hub/checkpoints/dinov2_vits14_pretrain.pth'
if not os.path.isfile(DINO_WEIGHTS):         # 可复现包回退：models/ 内自带权重
    DINO_WEIGHTS = os.path.join(MODEL_DIR, 'dinov2_vits14_pretrain.pth')
DINO_EMB = 384          # ViT-S/14 cls token dim
DEPTH_EMB = 128
sys.path.insert(0, DINO_HUB_DIR)


# ====================== 深度通道分支 ======================
class DepthBranch(nn.Module):
    """小型 CNN 编码深度通道(ch3, [0,1]) -> 128-dim 向量。"""
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(1, 16, 7, 2, 3), nn.BatchNorm2d(16), nn.ReLU(),
            nn.Conv2d(16, 32, 5, 2, 2), nn.BatchNorm2d(32), nn.ReLU(),
            nn.Conv2d(32, 64, 3, 2, 1), nn.BatchNorm2d(64), nn.ReLU())
        self.head = nn.Sequential(nn.Linear(64, DEPTH_EMB), nn.ReLU())
    def forward(self, x):           # x: (B,1,H,W)
        x = self.net(x)
        x = x.mean(dim=[2, 3])      # GAP -> (B,64)
        return self.head(x)


# ====================== 完整模型 ======================
class Dinov2Ablation(nn.Module):
    def __init__(self, dino):
        super().__init__()
        self.dino = dino            # 冻结
        for p in self.dino.parameters():
            p.requires_grad = False
        self.depth = DepthBranch()
        self.head = nn.Sequential(
            nn.Dropout(0.3), nn.Linear(DINO_EMB + DEPTH_EMB, 512), nn.ReLU(),
            nn.Dropout(0.15), nn.Linear(512, 64), nn.ReLU(),
            nn.Linear(64, 1))       # 输出 log1p 空间（若 --log1p）或 log 空间
    def forward(self, x):           # x: (B,4,224,224)
        rgb = x[:, :3]
        d = x[:, 3:4]
        with torch.no_grad():
            feat = self.dino(rgb)
            if isinstance(feat, dict):          # 新版返回 dict
                feat = feat['x_norm_clstoken']
        z = torch.cat([feat, self.depth(d)], dim=1)
        return self.head(z).squeeze(-1).clamp(-10.0, 15.0)


# ====================== 数据增强（安全版，无 clamp bug） ======================
def augment(x):
    if np.random.random() < 0.5:
        x = torch.flip(x, dims=[-1])
    return x

class AugDS(TensorDataset):
    def __init__(self, X, y, aug=False):
        super().__init__(torch.tensor(X, dtype=torch.float32),
                         torch.tensor(y, dtype=torch.float32))
        self.aug = aug
    def __getitem__(self, i):
        x, y = super().__getitem__(i)
        if self.aug:
            x = augment(x)
        return x, y


# ====================== 损失 / 评估 ======================
def log_mse_loss(pred, target):     # pred 已是 log(weight)，直接与 log(target) 比
    return F.mse_loss(pred, torch.log(target))

def log1p_mse_loss(pred, target):   # pred 已是 log1p(weight)
    return F.mse_loss(pred, torch.log1p(target))

@torch.no_grad()
def evaluate(model, loader, log1p):
    model.eval()
    preds, trues = [], []
    for x, y in loader:
        x = x.to(DEVICE)
        out = model(x)
        preds.append((torch.expm1(out) if log1p else torch.exp(out)).cpu().numpy())
        trues.append(y.numpy())
    preds = np.clip(np.concatenate(preds), 0, None)
    trues = np.concatenate(trues)
    return {'pmae': mean_absolute_percentage_error(trues, preds) * 100,
            'mae': mean_absolute_error(trues, preds), 'preds': preds, 'trues': trues}


def calibrate(y_tr, p_tr, p_te, log1p):
    """log/log1p 线性标定（对齐基线）。"""
    lr = LinearRegression()
    if log1p:
        lr.fit(np.log1p(np.clip(p_tr, 0.1, None)).reshape(-1, 1), np.log1p(y_tr))
        cal = np.expm1(lr.predict(np.log1p(np.clip(p_te, 0.1, None)).reshape(-1, 1)))
    else:
        lr.fit(np.log(np.clip(p_tr, 0.1, None)).reshape(-1, 1), np.log(y_tr))
        cal = np.exp(lr.predict(np.log(np.clip(p_te, 0.1, None)).reshape(-1, 1)))
    return np.clip(cal, 0, None)


# ====================== 主流程 ======================
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--smoke', action='store_true', help='CPU 冒烟（小样本，自检可运行性）')
    ap.add_argument('--bs', type=int, default=32)
    ap.add_argument('--epochs', type=int, default=60)
    ap.add_argument('--lr', type=float, default=1e-3)
    ap.add_argument('--log1p', action='store_true', help='用 log1p-MSE（对齐 v4 基线）')
    a = ap.parse_args()
    log1p = a.log1p

    if a.smoke:
        print('SMOKE MODE: 小样本 CPU 自检')
        a.bs = 8; a.epochs = 3

    print(f'Device: {DEVICE}')

    # ---- 数据 ----
    data = np.load(os.path.join(MODEL_DIR, 'deep_crops_cache.npz'))
    X, Y, X_test, Y_test = data['X_train'], data['y_train'], data['X_test'], data['y_test']
    if a.smoke:
        X, Y = X[:32], Y[:32]
        X_test, Y_test = X_test[:16], Y_test[:16]

    Xtr, Xv, ytr, yv = train_test_split(X, Y, test_size=0.15, random_state=SEED)

    # ---- 加载 DINOv2-small（本地权重，手动 interpolate pos_embed 到 224）----
    print(f'Loading DINOv2 ViT-S/14 from {os.path.basename(DINO_WEIGHTS)} ...')
    from dinov2.hub.backbones import dinov2_vits14
    dino = dinov2_vits14(pretrained=False, img_size=224)   # 仅构造结构，暂不加载权重
    sd = torch.load(DINO_WEIGHTS, map_location='cpu', weights_only=True)
    if 'pos_embed' in sd and sd['pos_embed'].shape != dino.pos_embed.shape:
        pe = sd['pos_embed']                    # [1, 1370, 384] = cls + 37x37 patches (518px)
        cls_tok = pe[:, :1]                     # [1,1,384]
        patch = pe[:, 1:]                       # [1,1369,384]
        side = int(round(math.sqrt(patch.shape[1])))
        patch = patch.reshape(1, side, side, -1).permute(0, 3, 1, 2)   # [1,384,37,37]
        tgt = dino.pos_embed.shape[1] - 1        # 256 patches
        ts = int(round(math.sqrt(tgt)))
        patch = F.interpolate(patch, size=(ts, ts), mode='bicubic', align_corners=False)
        patch = patch.permute(0, 2, 3, 1).reshape(1, tgt, -1)
        sd['pos_embed'] = torch.cat([cls_tok, patch], dim=1)
    dino.load_state_dict(sd, strict=True)
    dino.eval()
    dino = dino.to(DEVICE)
    print('  DINOv2 loaded & frozen.')

    model = Dinov2Ablation(dino).to(DEVICE)

    tr_ld = DataLoader(AugDS(Xtr, ytr, aug=True), a.bs, shuffle=True, num_workers=0, pin_memory=(DEVICE.type=='cuda'))
    va_ld = DataLoader(AugDS(Xv, yv, aug=False), a.bs * 2, num_workers=0)
    te_ld = DataLoader(AugDS(X_test, Y_test, aug=False), a.bs * 2, num_workers=0)

    # ---- 只训 head + depth，不训 dino ----
    opt = torch.optim.AdamW(
        [p for n, p in model.named_parameters() if p.requires_grad],
        lr=a.lr, weight_decay=1e-4)
    sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=a.epochs, eta_min=1e-6)
    loss_fn = log1p_mse_loss if log1p else log_mse_loss

    best = {'loss': float('inf'), 'pmae': float('inf'), 'state': None}
    hdr = f"{'Ep':<5s} {'ValLoss':<10s} {'ValPMAE':<10s}"
    print(hdr); print('-' * 30)
    t0 = time.time()
    for ep in range(a.epochs):
        model.train()
        for x, y in tr_ld:
            x, y = x.to(DEVICE), y.to(DEVICE)
            opt.zero_grad()
            out = model(x)
            loss = loss_fn(out, y)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                [p for p in model.parameters() if p.requires_grad], 1.0)
            opt.step()
        sch.step()

        v = evaluate(model, va_ld, log1p)
        print(f'{ep+1:<5d} {loss.item():<10.4f} {v["pmae"]:<10.2f}')
        if loss.item() < best['loss']:
            best = {'loss': loss.item(), 'pmae': v['pmae'],
                    'state': copy.deepcopy(model.state_dict())}

    model.load_state_dict(best['state'])
    print(f'Best val: PMAE={best["pmae"]:.2f}%  ({time.time()-t0:.0f}s)')

    # ---- 测试集评估（raw + 标定）----
    te = evaluate(model, te_ld, log1p)
    # 训练集预测用于标定
    tr_pred = []
    with torch.no_grad():
        model.eval()
        for i in range(0, len(Xtr), a.bs * 2):
            x = torch.tensor(Xtr[i:i + a.bs * 2], dtype=torch.float32).to(DEVICE)
            out = model(x)
            tr_pred.append((torch.expm1(out) if log1p else torch.exp(out)).cpu().numpy())
    tr_pred = np.clip(np.concatenate(tr_pred), 0, None)
    te_cal = calibrate(ytr, tr_pred, te['preds'], log1p)
    te_cal_pmae = mean_absolute_percentage_error(Y_test, te_cal) * 100

    print('\n' + '=' * 60)
    print('TEST SET (652, hold-out) — DINOv2-small ablation')
    print('=' * 60)
    print(f'  Raw        PMAE = {te["pmae"]:.2f}%   MAE = {te["mae"]:.1f}g')
    print(f'  Calibrated PMAE = {te_cal_pmae:.2f}%')
    print(f'  4ch 基线          = 22.0% (ResNet18+EffNet-B0, hold-out)')
    verdict = ('容量非瓶颈 → 信息论天花板成立（DINOv2 未超过 CNN 基线）'
               if te_cal_pmae <= 22.0
               else 'DINOv2 更差 → 天花板依旧成立（现代强主干也打不破上限）')
    print(f'  判读: {verdict}')

    np.savez(os.path.join(OUT_DIR, 'preds.npz'),
             y_test=Y_test, p_raw=te['preds'], p_cal=te_cal,
             pmae_raw=te['pmae'], pmae_cal=te_cal_pmae, log1p=log1p)
    print(f'Saved -> {OUT_DIR}/preds.npz')


if __name__ == '__main__':
    main()