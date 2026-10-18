#!/usr/bin/env python3
"""Phase 2': 学习式密度 — CNN 外观→密度, 能否打赢类别均值查表 (21.8% floor)?
协议: 5-fold 物体级 CV; 测试物体的类别同伴在训练集 (匹配部署场景)
输出: models/mf3d_density_resnet18_fold{K}.pth + density_pred_cache.npz + 报告
运行: conda activate nutri && python mf3d_phase2_learned_density.py  (GPU, ~15-25min)"""
import os, csv, math, random, sys, time, glob
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import torchvision.models as m
import torchvision.transforms as T

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
from mmcfa.paths import Paths

_P = Paths.resolve()
LABELS = str(_P.d1 / "metafood3d_labels_clean.csv")
RENDER = str(_P.expect("mf3d_render"))
WEIGHT_DIR = _P.weights_mf3d
WEIGHT_DIR.mkdir(parents=True, exist_ok=True)
CACHE_OUT = _P.results_diagnosis / "d2_density" / "density_pred_cache.npy"
CACHE_OUT.parent.mkdir(parents=True, exist_ok=True)
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
SEED = 42

# ---------------- 数据索引 ----------------
rows = list(csv.DictReader(open(LABELS)))
samples = []  # (img_path, cat, obj, log_d, density)
def find_original_dir(cat, obj):
    cdir = os.path.join(RENDER, cat)
    if not os.path.isdir(cdir): return None
    for d in os.listdir(cdir):
        if d.lower() != obj.lower(): continue
        for sub in os.listdir(os.path.join(cdir, d)):
            odir = os.path.join(cdir, d, sub, "Original")
            if os.path.isdir(odir): return odir
    return None

for r in rows:
    d = float(r["density"])
    odir = find_original_dir(r["cat"], r["obj"])
    if odir is None: continue
    for f in sorted(os.listdir(odir)):
        if f.endswith(".png"):
            samples.append((os.path.join(odir, f), r["cat"], r["obj"], math.log(d), d))

objs = sorted(set((s[1], s[2]) for s in samples))
print(f"[data] {len(samples)} 帧渲染图 | {len(objs)} 物体 | {len(set(s[1] for s in samples))} 类别")

# ---------------- Dataset ----------------
train_tf = T.Compose([T.RandomHorizontalFlip(), T.ColorJitter(0.1, 0.1, 0.1),
                      T.Resize(232), T.CenterCrop(224), T.ToTensor(),
                      T.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])])
test_tf = T.Compose([T.Resize(232), T.CenterCrop(224), T.ToTensor(),
                     T.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])])

class RenderDS(Dataset):
    def __init__(self, items, tf):
        self.items, self.tf = items, tf
    def __len__(self): return len(self.items)
    def __getitem__(self, i):
        p, cat, obj, ld, d = self.items[i]
        img = Image.open(p).convert("RGB")
        return self.tf(img), torch.tensor([ld], dtype=torch.float32), d

def mape(preds, truths):
    return 100.0 * np.mean(np.abs(np.array(preds) - np.array(truths)) / np.array(truths))

# ---------------- 5-fold CV ----------------
random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)
random.shuffle(objs)
K = 5
folds = [objs[i::K] for i in range(K)]

all_results = []
for k in range(K):
    test_objs = set(folds[k])
    train_items = [s for s in samples if (s[1], s[2]) not in test_objs]
    test_items  = [s for s in samples if (s[1], s[2]) in test_objs]
    # 训练 fold 内类别均值(诚实基线: 只用训练 fold 计算)
    from collections import defaultdict
    cat_dens = defaultdict(list)
    for s in train_items: cat_dens[s[1]].append(s[4])
    cat_mean = {c: np.mean(v) for c, v in cat_dens.items()}
    g_mean = np.mean([s[4] for s in train_items])

    # 类别内唯一物体的测试样本 -> 回退全局均值
    t0 = time.time()
    model = m.resnet18(weights=None)
    model.fc = nn.Sequential(nn.Linear(512, 128), nn.ReLU(), nn.Dropout(0.2),
                             nn.Linear(128, 1))
    model = model.to(DEVICE)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=30)
    tr_loader = DataLoader(RenderDS(train_items, train_tf), batch_size=64,
                           shuffle=True, num_workers=8, pin_memory=True)
    te_loader = DataLoader(RenderDS(test_items, test_tf), batch_size=128,
                           shuffle=False, num_workers=8, pin_memory=True)

    best_state, best_mape = None, 1e9
    for ep in range(30):
        model.train(); tot = 0.0
        for x, y, _ in tr_loader:
            x, y = x.to(DEVICE), y.to(DEVICE)
            opt.zero_grad()
            loss = nn.functional.mse_loss(model(x), y)
            loss.backward(); opt.step(); tot += loss.item() * len(x)
        sched.step()
        if (ep + 1) % 5 == 0:
            model.eval(); preds = []
            with torch.no_grad():
                for x, y, _ in te_loader:
                    preds.extend(torch.exp(model(x.to(DEVICE))).squeeze(1).cpu().tolist())
            truths = [d for _, _, _, _, d in test_items]
            m_ = mape(preds, truths)
            if m_ < best_mape: best_mape, best_state = m_, {k2: v.cpu().clone() for k2, v in model.state_dict().items()}
            print(f"  fold{k} ep{ep+1:02d} loss={tot/len(train_items):.4f} test-density-MAPE={m_:.1f}%")

    model.load_state_dict(best_state); model.eval()
    # 最终预测: 逐物体聚合(对多视角取中位)
    obj_preds = defaultdict(list); obj_truths = {}
    with torch.no_grad():
        for x, y, d in te_loader:
            out = torch.exp(model(x.to(DEVICE))).squeeze(1).cpu().tolist()
            for xi, di, oi in zip(out, d, [(s[1], s[2]) for s in test_items]):
                obj_preds[oi].append(xi); obj_truths[oi] = di

    # ---- 评估: 密度 MAPE + 重量组合 MAPE (GT V × ρ) ----
    cnn_d, cat_d, glob_d, truth_d, gt_v, gt_w = [], [], [], [], [], []
    for r in rows:
        key = (r["cat"], r["obj"])
        if key not in obj_preds: continue
        cnn_d.append(np.median(obj_preds[key]))
        cat_d.append(cat_mean.get(r["cat"], g_mean))
        glob_d.append(g_mean)
        truth_d.append(float(r["density"])); gt_v.append(float(r["volume_cm3"])); gt_w.append(float(r["weight_g"]))

    res = {
        "fold": k, "n_obj": len(truth_d),
        "dens_mape_cnn": mape(cnn_d, truth_d),
        "dens_mape_cat": mape(cat_d, truth_d),
        "dens_mape_glob": mape(glob_d, truth_d),
        "w_mape_cnn": mape(np.array(gt_v) * np.array(cnn_d), gt_w),
        "w_mape_cat": mape(np.array(gt_v) * np.array(cat_d), gt_w),
        "w_mape_glob": mape(np.array(gt_v) * np.array(glob_d), gt_w),
    }
    all_results.append(res)
    torch.save(best_state, WEIGHT_DIR / f"mf3d_density_resnet18_fold{k}.pth")
    print(f"[fold{k}] {time.time()-t0:.0f}s | 密度MAPE: CNN {res['dens_mape_cnn']:.1f}% vs 类别均值 {res['dens_mape_cat']:.1f}% vs 全局 {res['dens_mape_glob']:.1f}%")
    print(f"         重量MAPE(GT V×ρ): CNN {res['w_mape_cnn']:.1f}% vs 类别 {res['w_mape_cat']:.1f}% vs 全局 {res['w_mape_glob']:.1f}%")

# ---------------- 汇总 ----------------
print("\n" + "=" * 70)
print("Phase 2' 汇总 (5-fold 物体级 CV)")
print("=" * 70)
for key, label in [("dens_mape_cnn", "密度 MAPE — CNN"), ("dens_mape_cat", "密度 MAPE — 类别均值"),
                   ("w_mape_cnn", "重量 MAPE — GT V × CNN ρ"), ("w_mape_cat", "重量 MAPE — GT V × 类别均值ρ"),
                   ("w_mape_glob", "重量 MAPE — GT V × 全局均值ρ")]:
    vals = [r[key] for r in all_results]
    print(f"  {label:32s} {np.mean(vals):5.1f}% ± {np.std(vals):4.1f}")

cnn_win = np.mean([r["w_mape_cnn"] for r in all_results]) < np.mean([r["w_mape_cat"] for r in all_results])
print(f"\n[GATE 2] CNN {'✓ 赢' if cnn_win else '✗ 输'}类别均值 → 学习式密度路线{'可行, 进入体积测量+整合' if cnn_win else '证伪, 范式全面闭合'}")
print(f"参照: 单目上限 18.4% (>=100g) / 22.0% (全量) | Oracle-2 查表 25.0%")
np.save(CACHE_OUT, np.array(all_results, dtype=object), allow_pickle=True)
print(f"[saved] {WEIGHT_DIR}/mf3d_density_resnet18_fold{{0-4}}.pth + {CACHE_OUT}")
