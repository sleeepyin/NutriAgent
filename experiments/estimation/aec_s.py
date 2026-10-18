#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
AEC-S v2 实验 (2026-08-20)
==========================
优化方向:
  [1] 诚实性修复 — 原版 AEC-S 的全局权重在测试集上优化 (数据泄漏, L254),
      修复为校准集优化, 并量化泄漏带来的虚高
  [2] 基模型升级 — 旧权重 (PMAE≈31%) -> v2_final
  [3] AEC-S2     — 6 候选方法 (+LogSplit, +DisagreeNorm) + Winkler 目标
      + 20k Dirichlet + 坐标细化
  [4] AEC-S2c    — 加权区间之上再做 split-conformal 重标定,
      理论上保证边际 coverage >= 1-alpha

协议: 与原版完全一致 (50/20/30 随机划分, 10 trials, alpha=0.10)
运行: python aec_s_v2_experiment.py   (CPU 可跑; 首次 ~15min 建预测缓存, 之后 ~3min)
"""
import os, sys, time, warnings
from pathlib import Path
import numpy as np
import torch, torch.nn as nn
import joblib
from torchvision import models
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_percentage_error
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
warnings.filterwarnings('ignore')

SEED = 42
ALPHA = 0.10
N_TRIALS = 10
BINS = [0, 50, 100, 200, 500, 10000]
BIN_LABELS = ['0-50g', '50-100g', '100-200g', '200-500g', '500g+']
TAU = 30                     # 经验收缩强度 (与原版 τ 一致；不是贝叶斯后验)
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

OUT_DIR = str(Paths.resolve().results_estimation / "e2_aec_s")
os.makedirs(OUT_DIR, exist_ok=True)
CACHE_PATH = n5k('aec_v2_pred_cache.npz')
DEVICE = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

PAIRS = {
    'old':   ('deep_weight_resnet18.pth', 'deep_weight_efficientnet.pth'),
    'final': ('deep_weight_resnet18_final.pth', 'deep_weight_efficientnet_final.pth'),
}

# ==================== 模型 ====================
class WeightResNet(nn.Module):
    def __init__(self, dropout=0.3):
        super().__init__()
        self.backbone = models.resnet18(weights=None)
        old = self.backbone.conv1; new = nn.Conv2d(4, 64, 7, 2, 3, bias=False)
        new.weight.data[:, :3] = old.weight.data; new.weight.data[:, 3] = old.weight.data.mean(1)
        self.backbone.conv1 = new
        f = self.backbone.fc.in_features
        self.backbone.fc = nn.Sequential(
            nn.Dropout(dropout), nn.Linear(f, 256), nn.ReLU(),
            nn.Dropout(dropout * 0.5), nn.Linear(256, 64), nn.ReLU(), nn.Linear(64, 1))
    def forward(self, x): return self.backbone(x).squeeze(-1)

class WeightEffNet(nn.Module):
    def __init__(self, dropout=0.3):
        super().__init__()
        self.backbone = models.efficientnet_b0(weights=None)
        old = self.backbone.features[0][0]; new = nn.Conv2d(4, 32, 3, 2, 1, bias=False)
        new.weight.data[:, :3] = old.weight.data; new.weight.data[:, 3] = old.weight.data.mean(1)
        self.backbone.features[0][0] = new
        f = self.backbone.classifier[1].in_features
        self.backbone.classifier = nn.Sequential(
            nn.Dropout(dropout), nn.Linear(f, 128), nn.ReLU(),
            nn.Dropout(dropout * 0.5), nn.Linear(128, 1))
    def forward(self, x): return self.backbone(x).squeeze(-1)

@torch.no_grad()
def predict(model, X, bs=64):
    model.eval(); preds = []
    for i in range(0, len(X), bs):
        x = torch.tensor(X[i:i + bs], dtype=torch.float32).to(DEVICE)
        preds.append(model(x).cpu().numpy())
    return np.clip(np.concatenate(preds), 0, None)

@torch.no_grad()
def extract_features(model, X, bs=64):
    """ResNet18 GAP 特征 (512-dim), 与原版 MLNS 一致"""
    feats = []; model.eval()
    b = model.backbone
    for i in range(0, len(X), bs):
        x = torch.tensor(X[i:i + bs], dtype=torch.float32).to(DEVICE)
        o = b.conv1(x); o = b.bn1(o); o = b.relu(o); o = b.maxpool(o)
        o = b.layer1(o); o = b.layer2(o); o = b.layer3(o); o = b.layer4(o)
        o = b.avgpool(o)
        feats.append(o.squeeze(-1).squeeze(-1).cpu().numpy())
    return np.concatenate(feats, axis=0)

# ==================== 工具 ====================
def qhat(scores, alpha=ALPHA):
    """标准 split-conformal 分位数 (与原版同公式)"""
    n = len(scores)
    return np.quantile(scores, min(1.0, np.ceil((n + 1) * (1 - alpha)) / n))

def winkler(y, l, u, alpha=ALPHA):
    w = u - l
    return float(np.mean(w + (2 / alpha) * np.maximum(0, l - y) + (2 / alpha) * np.maximum(0, y - u)))

def eval_iv(y, l, u):
    return (float(np.mean((y >= l) & (y <= u))),
            float(np.mean(u - l)),
            winkler(y, l, u))

# ==================== 候选方法 (需要 cal + test 两侧区间) ====================
def build_candidates(y_tr2, p_tr2, d_tr2, f_tr2,
                     y_cal, p_cal, d_cal, f_cal,
                     y_te, p_te, d_te, f_te):
    """返回 C[name] = (cal_l, cal_u, te_l, te_u) 和 meta"""
    eps = 1e-3
    C = {}

    # --- Split: 绝对误差 ---
    s = np.abs(y_cal - p_cal); q = qhat(s)
    C['Split'] = (np.maximum(0, p_cal - q), p_cal + q,
                  np.maximum(0, p_te - q), p_te + q)

    # --- Normalized: 相对误差 ---
    s = np.abs(y_cal - p_cal) / (p_cal + eps); q = qhat(s)
    C['Norm'] = (np.maximum(0, p_cal * (1 - q)), p_cal * (1 + q),
                 np.maximum(0, p_te * (1 - q)), p_te * (1 + q))

    # --- LogSplit: log 空间乘性区间 (新增) ---
    s = np.abs(np.log(np.maximum(y_cal, 0.1)) - np.log(np.maximum(p_cal, 0.1))); q = qhat(s)
    bc = np.maximum(p_cal, 0.1); bt = np.maximum(p_te, 0.1)
    C['LogSplit'] = (bc * np.exp(-q), bc * np.exp(q), bt * np.exp(-q), bt * np.exp(q))

    # --- DisagreeNorm: 集成分歧定标 (新增, 经典 ensemble-spread conformal) ---
    s = np.abs(y_cal - p_cal) / (d_cal + eps); q = qhat(s)
    rc = q * (d_cal + eps); rt = q * (d_te + eps)
    C['Disagree'] = (np.maximum(0, p_cal - rc), p_cal + rc,
                     np.maximum(0, p_te - rt), p_te + rt)

    # --- Mondrian: 按预测值分 bin 的相对误差 (复刻原版) ---
    MB = [(0, 75), (75, 150), (150, 500), (500, 10000)]
    def bin_idx(v):
        return np.array([next(i for i, (l, h) in enumerate(MB) if l <= x < h) for x in v])
    cb = bin_idx(p_cal); tb = bin_idx(p_te)
    gs = np.abs(y_cal - p_cal) / (p_cal + eps); gq = qhat(gs)
    cl = np.zeros_like(p_cal); cu = np.zeros_like(p_cal)
    tl = np.zeros_like(p_te); tu = np.zeros_like(p_te)
    for i, (l_, h_) in enumerate(MB):
        cmm = cb == i; tmm = tb == i
        if cmm.sum() < 30:
            q = gq * 1.15
        else:
            ss = np.abs(y_cal[cmm] - p_cal[cmm]) / (p_cal[cmm] + eps)
            q = qhat(ss) * (1 + 0.03 / np.sqrt(cmm.sum()))
        cl[cmm] = np.maximum(0, p_cal[cmm] * (1 - q)); cu[cmm] = p_cal[cmm] * (1 + q)
        tl[tmm] = np.maximum(0, p_te[tmm] * (1 - q)); tu[tmm] = p_te[tmm] * (1 + q)
    C['Mondrian'] = (cl, cu, tl, tu)

    # --- MLNS: GBR 学习的难度分数 (复刻原版) ---
    n_comp = min(20, f_tr2.shape[1])
    pca = PCA(n_components=n_comp).fit(f_tr2)
    Xp_tr, Xp_cal, Xp_te = pca.transform(f_tr2), pca.transform(f_cal), pca.transform(f_te)
    ex_tr = np.column_stack([d_tr2, p_tr2, np.zeros((len(p_tr2), 1))])
    ex_cal = np.column_stack([d_cal, p_cal, np.zeros((len(p_cal), 1))])
    ex_te = np.column_stack([d_te, p_te, np.zeros((len(p_te), 1))])
    Xs_tr = np.column_stack([Xp_tr, ex_tr])
    Xs_cal = np.column_stack([Xp_cal, ex_cal])
    Xs_te = np.column_stack([Xp_te, ex_te])
    sc = StandardScaler().fit(Xs_tr)
    Xs_tr, Xs_cal, Xs_te = sc.transform(Xs_tr), sc.transform(Xs_cal), sc.transform(Xs_te)
    gbr = GradientBoostingRegressor(n_estimators=300, max_depth=5, learning_rate=0.05,
                                    subsample=0.8, random_state=SEED)
    gbr.fit(Xs_tr, np.abs(y_tr2 - p_tr2) / (p_tr2 + eps))
    cal_raw = np.maximum(gbr.predict(Xs_cal), 1e-6)
    te_raw = np.maximum(gbr.predict(Xs_te), 1e-6)
    q = qhat(cal_raw * 1.02)
    C['MLNS'] = (np.maximum(0, p_cal * (1 - q * cal_raw / cal_raw.mean())),
                 p_cal * (1 + q * cal_raw / cal_raw.mean()),
                 np.maximum(0, p_te * (1 - q * te_raw / te_raw.mean())),
                 p_te * (1 + q * te_raw / te_raw.mean()))

    meta = {'split_q': qhat(np.abs(y_cal - p_cal)),
            'norm_q': qhat(np.abs(y_cal - p_cal) / (p_cal + eps)),
            'mond_q': gq}
    return C, meta

# ==================== AEC 权重搜索 ====================
def dirichlet_search(L, U, y, objective, target, n_iter, rng):
    """随机 Dirichlet 凸组合搜索; objective: 'width' (带 coverage 约束) 或 'winkler'"""
    n = L.shape[1]
    best_w, best = np.ones(n) / n, np.inf
    for _ in range(n_iter):
        w = rng.dirichlet(np.ones(n))
        el = L @ w; eu = U @ w
        if objective == 'width':
            if np.mean((y >= el) & (y <= eu)) < target:
                continue
            o = float(np.mean(eu - el))
        else:
            o = winkler(y, el, eu)
        if o < best:
            best, best_w = o, w
    return best_w, best

def refine(L, U, y, w, objective, target=None, deltas=(0.1, 0.05, 0.02, 0.01, 0.005)):
    """单纯形上的坐标细化: 在 (i,j) 之间转移质量 delta"""
    def obj(w):
        el = L @ w; eu = U @ w
        if objective == 'width':
            if np.mean((y >= el) & (y <= eu)) < target:
                return np.inf
            return float(np.mean(eu - el))
        return winkler(y, el, eu)
    best = obj(w)
    for _ in range(3):
        improved = False
        for i in range(len(w)):
            for j in range(len(w)):
                if i == j:
                    continue
                for dt in deltas:
                    if w[i] < dt:
                        continue
                    w2 = w.copy(); w2[i] -= dt; w2[j] += dt
                    o = obj(w2)
                    if o < best - 1e-9:
                        w, best = w2, o; improved = True
        if not improved:
            break
    return w

def global_weights(L, U, y, objective='width', target=0.9, n_iter=5000, rng=None, do_refine=False):
    w, _ = dirichlet_search(L, U, y, objective, target, n_iter, rng)
    if do_refine:
        w = refine(L, U, y, w, objective, target)
    return w

def aec_apply(Lc, Uc, p_cal, Lt, Ut, p_te, weights):
    """按预测值 bin 应用每 bin 权重到 cal 和 test 两侧"""
    cl = np.zeros_like(p_cal); cu = np.zeros_like(p_cal)
    tl = np.zeros_like(p_te); tu = np.zeros_like(p_te)
    for bi in range(len(BINS) - 1):
        lo, hi = BINS[bi], BINS[bi + 1]
        cm = (p_cal >= lo) & (p_cal < hi); tm = (p_te >= lo) & (p_te < hi)
        w = weights[bi]
        if cm.sum(): cl[cm] = Lc[cm] @ w; cu[cm] = Uc[cm] @ w
        if tm.sum(): tl[tm] = Lt[tm] @ w; tu[tm] = Ut[tm] @ w
    return cl, cu, tl, tu

def aec_stratified(Lc, Uc, y_cal, p_cal, Lt, Ut, p_te, w_global,
                   objective='width', target=0.9, n_iter=5000, rng=None, do_refine=False):
    """分层 AEC: 每 bin 局部搜索 + 经验收缩到校准集全局先验"""
    weights = []
    for bi in range(len(BINS) - 1):
        lo, hi = BINS[bi], BINS[bi + 1]
        cm = (p_cal >= lo) & (p_cal < hi)
        n = int(cm.sum())
        if n < 20:
            weights.append(w_global.copy())
            continue
        w_loc, _ = dirichlet_search(Lc[cm], Uc[cm], y_cal[cm], objective, target, n_iter, rng)
        if do_refine:
            w_loc = refine(Lc[cm], Uc[cm], y_cal[cm], w_loc, objective, target)
        a = n / (n + TAU)
        weights.append(a * w_loc + (1 - a) * w_global)
    cl, cu, tl, tu = aec_apply(Lc, Uc, p_cal, Lt, Ut, p_te, weights)
    return cl, cu, tl, tu, weights

def conformalize(cl, cu, y_cal, tl, tu):
    """加权区间之上做 split-conformal 重标定: 保证边际 coverage"""
    c_cal = (cl + cu) / 2; h_cal = np.maximum((cu - cl) / 2, 1e-6)
    s = np.abs(y_cal - c_cal) / h_cal
    q = qhat(s)
    c_te = (tl + tu) / 2; h_te = np.maximum((tu - tl) / 2, 1e-6)
    return np.maximum(0, c_te - q * h_te), c_te + q * h_te

def conformalize_mondrian(cl, cu, y_cal, p_cal, tl, tu, p_te):
    """分 bin conformal 重标定: 每 bin 独立定标, 修复条件覆盖 (bin 不足 30 用全局 q)"""
    c_cal = (cl + cu) / 2; h_cal = np.maximum((cu - cl) / 2, 1e-6)
    c_te = (tl + tu) / 2; h_te = np.maximum((tu - tl) / 2, 1e-6)
    s_all = np.abs(y_cal - c_cal) / h_cal
    q_all = qhat(s_all)
    l = np.zeros_like(p_te); u = np.zeros_like(p_te)
    for bi in range(len(BINS) - 1):
        lo, hi = BINS[bi], BINS[bi + 1]
        cm = (p_cal >= lo) & (p_cal < hi); tm = (p_te >= lo) & (p_te < hi)
        if not tm.sum():
            continue
        if cm.sum() >= 30:
            q = qhat(s_all[cm])
        else:
            q = q_all
        l[tm] = np.maximum(0, c_te[tm] - q * h_te[tm])
        u[tm] = c_te[tm] + q * h_te[tm]
    return l, u

# ==================== 预测缓存 ====================
def load_or_build_cache():
    if os.path.exists(CACHE_PATH):
        d = np.load(CACHE_PATH)
        print(f"预测缓存已存在: {CACHE_PATH}")
        return {k: d[k] for k in d.files}
    print("构建预测缓存 (一次性, ~10min CPU)...")
    data = np.load(n5k('deep_crops_cache.npz'))
    X_all = np.concatenate([data['X_train'], data['X_test']], axis=0)
    y_all = np.concatenate([data['y_train'], data['y_test']], axis=0)
    out = {'y_all': y_all}
    for pair, (rck, eck) in PAIRS.items():
        rn = WeightResNet().to(DEVICE)
        rn.load_state_dict(torch.load(n5k(rck),
                                       map_location=DEVICE, weights_only=True))
        eff = WeightEffNet().to(DEVICE)
        eff.load_state_dict(torch.load(n5k(eck),
                                        map_location=DEVICE, weights_only=True))
        t0 = time.time()
        out[f'{pair}_rn'] = predict(rn, X_all)
        out[f'{pair}_eff'] = predict(eff, X_all)
        out[f'{pair}_feat'] = extract_features(rn, X_all)
        print(f"  [{pair}] 预测+特征完成 ({time.time()-t0:.0f}s)")
    np.savez_compressed(CACHE_PATH, **out)
    print(f"缓存已保存: {CACHE_PATH}")
    return out

# ==================== 单次 trial ====================
def run_trial(seed, pair, cache):
    y_all = cache['y_all']
    n = len(y_all)
    idx = np.arange(n)
    # 与原版完全相同的划分协议
    tr_idx, te_idx = train_test_split(idx, test_size=0.3, random_state=seed)
    tr2_idx, cal_idx = train_test_split(tr_idx, test_size=0.2 / 0.7, random_state=seed)

    rn = cache[f'{pair}_rn']; eff = cache[f'{pair}_eff']; feat = cache[f'{pair}_feat']
    p_all = (rn + eff) / 2.0
    d_all = np.abs(rn - eff)

    y_tr2, y_cal, y_te = y_all[tr2_idx], y_all[cal_idx], y_all[te_idx]
    p_tr2, p_cal, p_te = p_all[tr2_idx], p_all[cal_idx], p_all[te_idx]
    d_tr2, d_cal, d_te = d_all[tr2_idx], d_all[cal_idx], d_all[te_idx]
    f_tr2, f_cal, f_te = feat[tr2_idx], feat[cal_idx], feat[te_idx]

    C, meta = build_candidates(y_tr2, p_tr2, d_tr2, f_tr2,
                               y_cal, p_cal, d_cal, f_cal,
                               y_te, p_te, d_te, f_te)

    res, ivs = {}, {}
    for m, (cl, cu, tl, tu) in C.items():
        res[m] = eval_iv(y_te, tl, tu)

    eps = 1e-3
    rng = np.random.default_rng(seed)
    FOUR = ['Split', 'Norm', 'Mondrian', 'MLNS']

    # 原版近似校准区间 (Mondrian/MLNS 用 mond_q 近似 — 复刻原版 L432-443)
    cal_l_a = {'Split': np.maximum(0, p_cal - meta['split_q']),
               'Norm': np.maximum(0, p_cal * (1 - meta['norm_q'])),
               'Mondrian': np.maximum(0, p_cal * (1 - meta['mond_q'])),
               'MLNS': np.maximum(0, p_cal * (1 - meta['mond_q']))}
    cal_u_a = {'Split': p_cal + meta['split_q'],
               'Norm': p_cal * (1 + meta['norm_q']),
               'Mondrian': p_cal * (1 + meta['mond_q']),
               'MLNS': p_cal * (1 + meta['mond_q'])}
    Lc_a = np.column_stack([cal_l_a[m] for m in FOUR])
    Uc_a = np.column_stack([cal_u_a[m] for m in FOUR])
    Lc_p = np.column_stack([C[m][0] for m in FOUR])
    Uc_p = np.column_stack([C[m][1] for m in FOUR])
    Lt4 = np.column_stack([C[m][2] for m in FOUR])
    Ut4 = np.column_stack([C[m][3] for m in FOUR])

    # --- [泄漏] 全局权重在测试集上优化 (原版做法) ---
    w_leak = global_weights(Lt4, Ut4, y_te, 'width', 1 - ALPHA, 5000, rng)
    # AEC-S(orig): 原版完整复现 (泄漏全局 + 近似校准区间)
    cl, cu, tl, tu, _ = aec_stratified(Lc_a, Uc_a, y_cal, p_cal, Lt4, Ut4, p_te,
                                        w_leak, 'width', 1 - ALPHA, 5000, rng)
    res['AEC-S(orig)'] = eval_iv(y_te, tl, tu); ivs['AEC-S(orig)'] = (tl, tu)
    # AEC-S-leak: 正确校准区间 + 泄漏全局 (隔离泄漏效应)
    cl, cu, tl, tu, _ = aec_stratified(Lc_p, Uc_p, y_cal, p_cal, Lt4, Ut4, p_te,
                                        w_leak, 'width', 1 - ALPHA, 5000, rng)
    res['AEC-S-leak'] = eval_iv(y_te, tl, tu)
    # AEC-S-honest: 全局权重也在校准集上优化 (诚实版)
    w_hon = global_weights(Lc_p, Uc_p, y_cal, 'width', 1 - ALPHA, 5000, rng)
    cl, cu, tl, tu, _ = aec_stratified(Lc_p, Uc_p, y_cal, p_cal, Lt4, Ut4, p_te,
                                        w_hon, 'width', 1 - ALPHA, 5000, rng)
    res['AEC-S-honest'] = eval_iv(y_te, tl, tu); ivs['AEC-S-honest'] = (tl, tu)

    # --- AEC-S2: 6 候选 + Winkler 目标 + 强化搜索 ---
    SIX = ['Split', 'Norm', 'LogSplit', 'Disagree', 'Mondrian', 'MLNS']
    Lc6 = np.column_stack([C[m][0] for m in SIX])
    Uc6 = np.column_stack([C[m][1] for m in SIX])
    Lt6 = np.column_stack([C[m][2] for m in SIX])
    Ut6 = np.column_stack([C[m][3] for m in SIX])
    w2 = global_weights(Lc6, Uc6, y_cal, 'winkler', None, 20000, rng, do_refine=True)
    cl6, cu6, tl6, tu6, wts = aec_stratified(Lc6, Uc6, y_cal, p_cal, Lt6, Ut6, p_te,
                                              w2, 'winkler', None, 10000, rng, do_refine=True)
    res['AEC-S2'] = eval_iv(y_te, tl6, tu6); ivs['AEC-S2'] = (tl6, tu6)
    # --- AEC-S2c: 再做 split-conformal 重标定 (保证 coverage) ---
    l_c, u_c = conformalize(cl6, cu6, y_cal, tl6, tu6)
    res['AEC-S2c'] = eval_iv(y_te, l_c, u_c); ivs['AEC-S2c'] = (l_c, u_c)
    # --- AEC-S2m: 分 bin conformal 重标定 (修复条件覆盖) ---
    l_m, u_m = conformalize_mondrian(cl6, cu6, y_cal, p_cal, tl6, tu6, p_te)
    res['AEC-S2m'] = eval_iv(y_te, l_m, u_m); ivs['AEC-S2m'] = (l_m, u_m)

    pmae = mean_absolute_percentage_error(y_te, p_te) * 100
    return res, ivs, pmae, y_te, wts

# ==================== 主流程 ====================
def main():
    print("=" * 68)
    print("AEC-S v2 实验 | 诚实性修复 + 基模型升级 + AEC-S2/AEC-S2c")
    print(f"协议: 50/20/30, {N_TRIALS} trials, alpha={ALPHA} | Device: {DEVICE}")
    print("=" * 68)

    cache = load_or_build_cache()
    y_all = cache['y_all']
    print(f"数据: {len(y_all)} 样本\n")

    all_out = {}
    # 注: md5 验证 old 与 final 权重完全相同 (08-17 仅改名备份), 只跑 final
    for pair in ['final']:
        print("=" * 68)
        print(f"基模型: {pair}")
        print("=" * 68)
        trials = []
        t0 = time.time()
        for t in range(N_TRIALS):
            res, ivs, pmae, y_te, wts = run_trial(SEED + t, pair, cache)
            trials.append((res, ivs, pmae, y_te, wts))
            print(f"  trial {t+1}/{N_TRIALS}: PMAE={pmae:.2f}% ({time.time()-t0:.0f}s)")

        names = list(trials[0][0].keys())
        print(f"\n  {'方法':<16s} {'Coverage':>14s} {'宽度':>13s} {'Winkler':>13s}")
        print("  " + "-" * 60)
        for m in names:
            cov = np.array([tr[0][m][0] for tr in trials]) * 100
            wd = np.array([tr[0][m][1] for tr in trials])
            wk = np.array([tr[0][m][2] for tr in trials])
            print(f"  {m:<16s} {cov.mean():>6.1f}±{cov.std():>4.1f}% "
                  f"{wd.mean():>7.0f}±{wd.std():>3.0f}g {wk.mean():>7.0f}±{wk.std():>3.0f}")

        # 最后一轮的 per-bin 分析
        y_te = trials[-1][3]; ivs = trials[-1][1]
        for vname in ['AEC-S-honest', 'AEC-S2c', 'AEC-S2m']:
            if vname not in ivs:
                continue
            l, u = ivs[vname]
            print(f"\n  per-bin [{vname}] (最后一轮):")
            for bi in range(len(BINS) - 1):
                lo, hi = BINS[bi], BINS[bi + 1]
                m = (y_te >= lo) & (y_te < hi)
                if m.sum():
                    cov = np.mean((y_te[m] >= l[m]) & (y_te[m] <= u[m])) * 100
                    print(f"    {BIN_LABELS[bi]:<9s} n={int(m.sum()):>3d} "
                          f"cov={cov:5.1f}% width={np.mean(u[m]-l[m]):6.0f}g")

        # AEC-S2 最后一轮的 per-bin 权重
        wts = trials[-1][4]
        SIX = ['Split', 'Norm', 'LogSplit', 'Disagree', 'Mondrian', 'MLNS']
        print(f"\n  AEC-S2 per-bin 权重 (最后一轮):")
        for bi in range(len(BINS) - 1):
            ws = ' '.join(f"{SIX[j]}={wts[bi][j]:.2f}" for j in range(len(SIX)))
            print(f"    {BIN_LABELS[bi]:<9s} {ws}")

        all_out[pair] = {
            'trials': [{'res': tr[0], 'pmae': tr[2]} for tr in trials],
            'last_ivs': {k: v for k, v in trials[-1][1].items()},
            'last_y_te': y_te,
            'last_weights': wts,
        }
        print()

    joblib.dump(all_out, os.path.join(OUT_DIR, 'aec_v2_results.pkl'))
    print(f"结果已保存: {OUT_DIR}/aec_v2_results.pkl")

if __name__ == '__main__':
    main()
