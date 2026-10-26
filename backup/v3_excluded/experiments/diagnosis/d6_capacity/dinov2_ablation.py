#!/usr/bin/env python3
"""
DINOv2-small 现代 backbone 消融（论文任务4）
==================================================
目标：回答"容量是不是瓶颈"。用 DINOv2 ViT-S/14 冻结特征 + 深度分支 + MLP head
在 deep_crops_cache.npz 上训重量回归，与 R0 的 4 通道集成（results/estimation/r0/e1_point.json）对比。

协议：
  - 数据：data/nutrition5k/cache/deep_crops_cache.npz（本次为 2791 训练 / 698 测试）
  - 划分：X_train 内 85/15 split (seed 42) 留 val；最终在 X_test 上测 MAPE
  - 损失：log-MSE（pred 与 target 同取 log）
  - backbone：DINOv2 ViT-S/14 冻结，weights 指向本地 pth（不走官方 dl 被墙源）
  - 4ch 问题处理：DINO 吃 RGB 3ch；深度 ch3 由独立小 CNN 分支编码，拼进 MLP head
    （与 4ch ResNet/EffNet 基线可比——后者也是卷积在 4th 通道上融合）

判读：和本次 4 通道集成 MAPE 比。回归头要写到
weights/dinov2/d6_dinov2_small_head.pth，没有这个文件就不算做完。

用法：
  CPU 冒烟：   python train_dinov2_ablation.py --smoke
  GPU 完整：   python train_dinov2_ablation.py --bs 32 --epochs 60 --log1p
               （--log1p 与基线 v4 的 log1p-MSE 对齐；默认 log-MSE 均可）
输出：results/diagnosis/d6_dinov2/preds.npz、summary.json，以及权重 d6_dinov2_small_head.pth
"""
import os, sys, math, time, argparse, copy, json, warnings
from pathlib import Path
warnings.filterwarnings('ignore')
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset, TensorDataset
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_percentage_error, mean_absolute_error

SEED = 42
np.random.seed(SEED); torch.manual_seed(SEED)

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

if os.environ.get("D6_FORCE_CPU") == "1":
    DEVICE = torch.device("cpu")
else:
    DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
if DEVICE.type == "cpu":
    torch.set_num_threads(int(os.environ.get("D6_THREADS", "16")))
_P = Paths.resolve()
OUT_DIR = str(_P.root / "backup" / "v3_excluded" / "results" / "diagnosis" / "d6_dinov2")
os.makedirs(OUT_DIR, exist_ok=True)

DINO_HUB_DIR = str(_P.dinov2_hub)
if not os.path.isdir(DINO_HUB_DIR):
    DINO_HUB_DIR = os.path.expanduser("~/.cache/torch/hub/facebookresearch_dinov2_main")
DINO_WEIGHTS = str(_P.expect("dino_weights"))
if not os.path.isfile(DINO_WEIGHTS):
    alt = os.path.expanduser("~/.cache/torch/hub/checkpoints/dinov2_vits14_pretrain.pth")
    if os.path.isfile(alt):
        DINO_WEIGHTS = alt
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
    def forward(self, x):           # x: (B,4,224,224), RGB stored in [0, 1]
        rgb = x[:, :3]
        d = x[:, 3:4]
        # The rebuilt cache is [0, 1]. DINOv2 was pretrained with ImageNet normalization.
        mean = rgb.new_tensor((0.485, 0.456, 0.406)).view(1, 3, 1, 1)
        std = rgb.new_tensor((0.229, 0.224, 0.225)).view(1, 3, 1, 1)
        rgb = (rgb - mean) / std
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


class TokenDS(Dataset):
    """Frozen DINOv2 tokens plus the depth map. Horizontal flip matches augment()."""

    def __init__(self, feat, feat_flip, depth, y, aug=False):
        self.feat = feat
        self.feat_flip = feat_flip
        self.depth = depth
        self.y = torch.tensor(y, dtype=torch.float32)
        self.aug = aug

    def __len__(self):
        return int(self.y.shape[0])

    def __getitem__(self, i):
        if self.aug and np.random.random() < 0.5:
            return self.feat_flip[i], torch.flip(self.depth[i], dims=[-1]), self.y[i]
        return self.feat[i], self.depth[i], self.y[i]


def _dino_feat(dino, rgb):
    mean = rgb.new_tensor((0.485, 0.456, 0.406)).view(1, 3, 1, 1)
    std = rgb.new_tensor((0.229, 0.224, 0.225)).view(1, 3, 1, 1)
    feat = dino((rgb - mean) / std)
    if isinstance(feat, dict):
        feat = feat["x_norm_clstoken"]
    return feat


@torch.no_grad()
def cache_tokens(dino, images, bs):
    """images: float NCHW in [0, 1]. Returns tokens, flipped tokens, depth."""
    dino.eval()
    feats, flips, depth = [], [], []
    t0 = time.time()
    for i in range(0, len(images), bs):
        x = torch.tensor(images[i:i + bs], dtype=torch.float32, device=DEVICE)
        feats.append(_dino_feat(dino, x[:, :3]).cpu())
        flips.append(_dino_feat(dino, torch.flip(x[:, :3], dims=[-1])).cpu())
        depth.append(x[:, 3:4].cpu())
        done = min(i + bs, len(images))
        if done == len(images) or done % (bs * 20) == 0:
            print(f"  tokens {done}/{len(images)} {time.time() - t0:.0f}s", flush=True)
    return torch.cat(feats), torch.cat(flips), torch.cat(depth)


def head_out(model, feat, depth):
    z = torch.cat([feat, model.depth(depth)], dim=1)
    return model.head(z).squeeze(-1).clamp(-10.0, 15.0)


@torch.no_grad()
def predict_cached(model, feat, depth, log1p, bs):
    model.eval()
    preds = []
    for i in range(0, feat.shape[0], bs):
        out = head_out(model, feat[i:i + bs].to(DEVICE), depth[i:i + bs].to(DEVICE))
        preds.append((torch.expm1(out) if log1p else torch.exp(out)).cpu().numpy())
    return np.clip(np.concatenate(preds), 0, None)


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

    head_path = _P.weights_dinov2 / "d6_dinov2_small_head.pth"
    summary_path = os.path.join(OUT_DIR, "summary.json")
    if (not a.smoke) and head_path.is_file() and os.path.isfile(summary_path):
        print(f"D6 weight already present, skip: {head_path}")
        return

    print(f'Device: {DEVICE}')

    # ---- 数据 ----
    data = np.load(n5k('deep_crops_cache.npz'))
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

    # The backbone has no dropout. Cache its tokens once, then train depth+head.
    token_bs = 8 if DEVICE.type == "cpu" else a.bs
    print("Caching frozen DINOv2 tokens...")
    feat_tr, flip_tr, depth_tr = cache_tokens(dino, Xtr, token_bs)
    feat_va, flip_va, depth_va = cache_tokens(dino, Xv, token_bs)
    feat_te, _flip_te, depth_te = cache_tokens(dino, X_test, token_bs)
    del dino
    if DEVICE.type == "cuda":
        torch.cuda.empty_cache()

    tr_ld = DataLoader(TokenDS(feat_tr, flip_tr, depth_tr, ytr, aug=True), a.bs, shuffle=True, num_workers=0)
    va_ld = DataLoader(TokenDS(feat_va, flip_va, depth_va, yv, aug=False), a.bs * 2, num_workers=0)

    # ---- 只训 head + depth，不训 dino ----
    opt = torch.optim.AdamW(
        [p for n, p in model.named_parameters() if p.requires_grad],
        lr=a.lr, weight_decay=1e-4)
    sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=a.epochs, eta_min=1e-6)
    loss_fn = log1p_mse_loss if log1p else log_mse_loss

    best = {'loss': float('inf'), 'pmae': float('inf'), 'depth': None, 'head': None}
    hdr = f"{'Ep':<5s} {'ValLoss':<10s} {'ValPMAE':<10s}"
    print(hdr); print('-' * 30)
    t0 = time.time()
    for ep in range(a.epochs):
        model.train()
        model.dino.eval()
        last_loss = None
        for feat, depth, y in tr_ld:
            feat, depth, y = feat.to(DEVICE), depth.to(DEVICE), y.to(DEVICE)
            opt.zero_grad()
            out = head_out(model, feat, depth)
            loss = loss_fn(out, y)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                [p for p in model.parameters() if p.requires_grad], 1.0)
            opt.step()
            last_loss = loss.item()
        sch.step()

        v_pred = predict_cached(model, feat_va, depth_va, log1p, a.bs * 2)
        v_pmae = mean_absolute_percentage_error(yv, v_pred) * 100
        print(f'{ep+1:<5d} {last_loss:<10.4f} {v_pmae:<10.2f}')
        if last_loss < best['loss']:
            best = {
                'loss': last_loss,
                'pmae': v_pmae,
                'depth': copy.deepcopy(model.depth.state_dict()),
                'head': copy.deepcopy(model.head.state_dict()),
            }

    model.depth.load_state_dict(best['depth'])
    model.head.load_state_dict(best['head'])
    print(f'Best val: PMAE={best["pmae"]:.2f}%  ({time.time()-t0:.0f}s)')
    if a.smoke:
        head_path = head_path.with_name("d6_dinov2_small_head_smoke.pth")
    head_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), head_path)
    print(f'Saved -> {head_path}')

    # ---- 测试集评估（raw + 标定）----
    te_preds = predict_cached(model, feat_te, depth_te, log1p, a.bs * 2)
    te = {
        'pmae': mean_absolute_percentage_error(Y_test, te_preds) * 100,
        'mae': mean_absolute_error(Y_test, te_preds),
        'preds': te_preds,
    }
    tr_pred = predict_cached(model, feat_tr, depth_tr, log1p, a.bs * 2)
    te_cal = calibrate(ytr, tr_pred, te['preds'], log1p)
    te_cal_pmae = mean_absolute_percentage_error(Y_test, te_cal) * 100

    baseline_4, baseline_3 = None, None
    e1_path = _P.results_estimation / "r0" / "e1_point.json"
    if e1_path.is_file():
        e1 = json.loads(e1_path.read_text(encoding="utf-8"))
        baseline_4 = float(e1["4ch_ensemble"]["mape_pct"])
        baseline_3 = float(e1["3ch_ensemble"]["mape_pct"])
    print('\n' + '=' * 60)
    print(f'TEST SET ({len(Y_test)}, hold-out) — DINOv2-small ablation')
    print('=' * 60)
    print(f'  Raw        MAPE = {te["pmae"]:.2f}%   MAE = {te["mae"]:.1f}g')
    print(f'  Calibrated MAPE = {te_cal_pmae:.2f}%')
    if baseline_4 is not None:
        print(f'  4ch ensemble MAPE = {baseline_4:.2f}%')
        print(f'  3ch ensemble MAPE = {baseline_3:.2f}%')
    if baseline_4 is None:
        verdict = "没有本次 4 通道基线，只记录 DINOv2 数字"
    elif te_cal_pmae <= baseline_4 - 0.5:
        verdict = "DINOv2 低于本次 4 通道集成，需单独看容量是否改变天花板"
    else:
        verdict = "容量非瓶颈 → DINOv2 未超过本次 4 通道集成"
    print(f'  判读: {verdict}')
    if a.smoke:
        print("smoke finished; official preds and summary were not written")
        return

    np.savez(os.path.join(OUT_DIR, 'preds.npz'),
             y_test=Y_test, p_raw=te['preds'], p_cal=te_cal,
             pmae_raw=te['pmae'], pmae_cal=te_cal_pmae, log1p=log1p)
    summary = {
        "n_test": int(len(Y_test)),
        "mape_raw_pct": float(te["pmae"]),
        "mae_raw_g": float(te["mae"]),
        "mape_calibrated_pct": float(te_cal_pmae),
        "baseline_4ch_ensemble_mape_pct": baseline_4,
        "baseline_3ch_ensemble_mape_pct": baseline_3,
        "log1p": bool(log1p),
        "epochs": int(a.epochs),
        "batch_size": int(a.bs),
        "rgb": "imagenet-normalized from cache [0,1]",
        "verdict": verdict,
    }
    summary_path = os.path.join(OUT_DIR, "summary.json")
    with open(summary_path, "w", encoding="utf-8") as fh:
        json.dump(summary, fh, indent=2)
        fh.write("\n")
    print(f'Saved -> {OUT_DIR}/preds.npz')
    print(f'Saved -> {summary_path}')


if __name__ == '__main__':
    main()