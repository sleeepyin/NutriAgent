#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Nutrient AEC-S Conformal v2 — 营养素区间预测 (2026-08-20)
==========================================================
v1 结果: AEC-S2m-N (5候选, 重量分层) nWinkler=2.42, 全营养素 cov 89.7-90.6%

v2 优化 (三个新信号):
  [1] AsymNorm  — 非对称相对分位数候选: 营养素误差右偏 (大值低估),
                  对称区间浪费下侧空间; 用符号残差的 alpha/2 双侧分位数
  [2] DisagreeD — 模型-检索分歧定标: |w·d_direct − w·d_knn| 是现成的
                  不确定性代理 (重量版 AEC-S2 的 Disagree 思路迁移)
  [3] 营养素分位分层 (strat='nutrient') — 每营养素按自身预测值三分位
                  分 bin, 替代按重量分层 (营养素不确定性更多取决于
                  营养素量级而非重量)
  [4] 两阶段校准 (v2b) — 单阶段中 Dirichlet 权重与重标定 q 共用同一段
                  cal, 权重过拟合导致 q 偏小 → eval 欠覆盖 (观测: 5W
                  89.7% > 7W 89.4% > 7S 89.0%, 容量越大欠覆盖越重)。
                  修复: cal 对半分 A/B, 权重学于 A, q 算于 B (权重冻结)

配置对比 (同一实验内公平对比, 10 trials):
  AEC-*-5W   : 原 5 候选 + 重量分层, 单阶段 (v1 获胜配置, 参考复现)
  AEC-*-5W2  : 同配置, 两阶段校准
  AEC-*-7W2  : 7 候选 + 重量分层, 两阶段
  AEC-*-7S2  : 7 候选 + 营养素三分位分层, 两阶段

协议: 严格 hold-out — 生产管线 (v2 权重 + v4 GBML 密度 + kNN α=0.60)
训练于 2608, X_test(652) 对半切 cal(326)/eval(326), 10 trials, alpha=0.10

运行: python nutrient_conformal_final.py   (缓存已建, ~2-4 min)
产物:
  results/nutrient_conformal_results.pkl     全部指标
  models/nutrient_conformal_calibrator.pkl   部署用获胜校准器
"""
import math
import os, time, warnings
import numpy as np
import torch, torch.nn as nn
import joblib
from torchvision import models
from sklearn.model_selection import train_test_split
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler
warnings.filterwarnings('ignore')

SEED = 42
ALPHA = 0.10
N_TRIALS = 10
TAU = 30                          # 经验收缩强度 (与重量版 AEC-S 的 τ 一致)
BINS = [0, 50, 100, 200, 500, np.inf]
BIN_LABELS = ['0-50g', '50-100g', '100-200g', '200-500g', '500g+']
NUTR = ['Cal', 'Fat', 'Carb', 'Prot']
CANDS5 = ['Split', 'Norm', 'LogSplit', 'Propagate', 'MondrianW']   # v1 候选集
ALL_CANDS = ['Split', 'Norm', 'LogSplit', 'AsymNorm', 'DisagreeD',
             'Propagate', 'MondrianW']                              # v2 候选集
KNN_K = 5
HYBRID_ALPHA = 0.60               # 生产管线 d_hybrid = 0.60*direct + 0.40*knn
EPS = 1e-3
MODEL_DIR = '/home/Nutri_0811/models'
OUT_DIR = '/home/Nutri_0811/results'
CACHE_PATH = os.path.join(MODEL_DIR, 'nutrient_conformal_cache.npz')
DEPLOY_PATH = os.path.join(MODEL_DIR, 'nutrient_conformal_calibrator.pkl')
DEVICE = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

# ==================== 模型定义 (与 test_100_samples_final.py 一致) ====================
class WeightResNet(nn.Module):
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
            nn.Linear(64, 1))
    def forward(self, x): return self.backbone(x).squeeze(-1)
    def features(self, x):
        x = self.backbone.conv1(x); x = self.backbone.bn1(x); x = self.backbone.relu(x)
        x = self.backbone.maxpool(x); x = self.backbone.layer1(x); x = self.backbone.layer2(x)
        x = self.backbone.layer3(x); x = self.backbone.layer4(x); x = self.backbone.avgpool(x)
        return x.squeeze(-1).squeeze(-1)

class WeightEffNet(nn.Module):
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
            nn.Dropout(0.15), nn.Linear(128, 1))
    def forward(self, x): return self.backbone(x).squeeze(-1)
    def features(self, x):
        x = self.backbone.features(x); x = self.backbone.avgpool(x)
        return x.squeeze(-1).squeeze(-1)

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

@torch.no_grad()
def _predict(model, X, bs=32):
    model.eval(); preds = []
    for i in range(0, len(X), bs):
        x = torch.tensor(X[i:i+bs], dtype=torch.float32).to(DEVICE)
        preds.append(model(x).cpu().numpy())
    return np.concatenate(preds, axis=0)

@torch.no_grad()
def _features(model, X, bs=32):
    model.eval(); feats = []
    for i in range(0, len(X), bs):
        x = torch.tensor(X[i:i+bs], dtype=torch.float32).to(DEVICE)
        feats.append(model.features(x).cpu().numpy())
    return np.concatenate(feats, axis=0)

def _load(model_cls, fname):
    m = model_cls().to(DEVICE)
    m.load_state_dict(torch.load(os.path.join(MODEL_DIR, fname),
                                  map_location=DEVICE, weights_only=True))
    return m

# ==================== 生产管线缓存 ====================
def get_test_pipeline_cache():
    """返回测试集(652)上的生产管线输出: w_pred, d_direct, d_knn, y_w, y_nutr"""
    if os.path.exists(CACHE_PATH):
        d = np.load(CACHE_PATH)
        print(f"管线缓存已存在: {CACHE_PATH}")
        return {k: d[k] for k in d.files}
    print("构建管线缓存 (一次性, CPU ~5min)...")
    data = np.load(os.path.join(MODEL_DIR, 'deep_crops_cache.npz'))
    X_train, X_test = data['X_train'], data['X_test']
    y_train_w = data['y_train']; y_test_w = data['y_test']
    n_tr = len(X_train)
    dens = np.load(os.path.join(MODEL_DIR, 'nutrient_density.npz'))
    y_train_d = dens['y_train_density']

    # 重量预测: 复用 aec_v2 预测缓存 (final 权重, X_all=[train;test] 顺序)
    aec = np.load(os.path.join(MODEL_DIR, 'aec_v2_pred_cache.npz'))
    assert np.allclose(aec['y_all'], np.concatenate([y_train_w, y_test_w])), "缓存顺序不一致"
    w_te = (aec['final_rn'][n_tr:] + aec['final_eff'][n_tr:]) / 2.0
    rn_feat_tr = aec['final_feat'][:n_tr]
    rn_feat_te = aec['final_feat'][n_tr:]

    # 密度预测 (v4 GBML, log1p 空间 ensemble)
    d_rn = _load(DensityResNet, 'density_resnet18_final.pth')
    d_eff = _load(DensityEffNet, 'density_efficientnet_final.pth')
    d_direct = np.expm1((_predict(d_rn, X_test) + _predict(d_eff, X_test)) / 2.0)
    d_direct = np.maximum(d_direct, 0)
    del d_rn, d_eff

    # k-NN 检索 (生产配方: rn512+eff1280 标准化, cosine, k=5)
    w_eff = _load(WeightEffNet, 'deep_weight_efficientnet_final.pth')
    eff_feat_tr = _features(w_eff, X_train)
    eff_feat_te = _features(w_eff, X_test)
    del w_eff
    feat_tr = np.concatenate([rn_feat_tr, eff_feat_tr], axis=1)
    feat_te = np.concatenate([rn_feat_te, eff_feat_te], axis=1)
    scaler = StandardScaler().fit(feat_tr)
    nn_idx = NearestNeighbors(n_neighbors=KNN_K, metric='cosine').fit(scaler.transform(feat_tr))
    _, ind = nn_idx.kneighbors(scaler.transform(feat_te))
    d_knn = np.array([y_train_d[ind[i]].mean(axis=0) for i in range(len(X_test))])

    out = {'w_pred': w_te, 'd_direct': d_direct, 'd_knn': d_knn,
           'y_w': y_test_w,
           'y_nutr': np.load(os.path.join(MODEL_DIR, 'nutrient_labels.npz'))['y_test_nutr']}
    np.savez_compressed(CACHE_PATH, **out)
    print(f"缓存已保存: {CACHE_PATH}")
    return out

def hybrid_nutrients(cache):
    """生产管线: nutrients = w × (0.60*direct + 0.40*knn)"""
    d_hybrid = HYBRID_ALPHA * cache['d_direct'] + (1 - HYBRID_ALPHA) * cache['d_knn']
    return np.maximum(cache['w_pred'], 0).reshape(-1, 1) * d_hybrid

def nutrient_disagreement(cache):
    """模型-检索分歧 (营养素空间): |w·d_direct − w·d_knn|, 不确定性代理"""
    w = np.maximum(cache['w_pred'], 0).reshape(-1, 1)
    return np.abs(w * cache['d_direct'] - w * cache['d_knn'])

# ==================== 共形工具 ====================
def _order_stat_at_level(scores, level):
    """k-th order statistic at level, k = ceil((n+1)*level). No interpolation."""
    s = np.asarray(scores, dtype=float).reshape(-1)
    s = s[np.isfinite(s)]
    n = int(s.size)
    if n == 0:
        return float('inf')
    k = int(math.ceil((n + 1) * float(level)))
    if k > n:
        return float('inf')
    if k <= 0:
        return float(np.min(s))
    return float(np.partition(s, k - 1)[k - 1])

def qhat(scores, alpha=ALPHA):
    return _order_stat_at_level(scores, 1.0 - alpha)

def qhat_side(s, a, side):
    """单侧顺序统计量: side='lo' 取 level=a, side='hi' 取 level=1-a。"""
    lvl = a if side == 'lo' else (1 - a)
    return _order_stat_at_level(s, lvl)

def winkler_mean(y, l, u, alpha=ALPHA):
    w = u - l
    return float(np.mean(w + (2 / alpha) * np.maximum(0, l - y)
                         + (2 / alpha) * np.maximum(0, y - u)))

def eval_iv(y, l, u):
    """(coverage, width, winkler)"""
    return (float(np.mean((y >= l) & (y <= u))),
            float(np.mean(u - l)),
            winkler_mean(y, l, u))

def _dirichlet_winkler(L, U, y, n_iter, rng, batch=512):
    """单纯形随机搜索 (批量化), Winkler 目标"""
    n = L.shape[1]
    best_w, best = np.ones(n) / n, np.inf
    done = 0
    while done < n_iter:
        B = min(batch, n_iter - done)
        W = rng.dirichlet(np.ones(n), size=B)          # (B, n)
        EL = W @ L.T                                    # (B, N)
        EU = W @ U.T
        pen = (EU - EL) + (2 / ALPHA) * np.maximum(0, EL - y) \
                          + (2 / ALPHA) * np.maximum(0, y - EU)
        o = pen.mean(axis=1)
        k = int(np.argmin(o))
        if o[k] < best:
            best, best_w = float(o[k]), W[k].copy()
        done += B
    return best_w, best

def _refine(L, U, y, w, deltas=(0.1, 0.05, 0.02, 0.01, 0.005)):
    """单纯形坐标细化"""
    def obj(w):
        el = L @ w; eu = U @ w
        return float(np.mean((eu - el) + (2 / ALPHA) * np.maximum(0, el - y)
                             + (2 / ALPHA) * np.maximum(0, y - eu)))
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
                        w, best, improved = w2, o, True
        if not improved:
            break
    return w

# ==================== AEC-S 营养素校准器 ====================
class NutrientConformalCalibrator:
    """AEC-S 营养素共形区间校准器 (可序列化)

    strat='weight'   — 按预测重量 5 bin 分层 (v1 方案)
    strat='nutrient' — 每营养素按自身预测值三分位 3 bin 分层 (v2 新增)

    fit 需要生产管线输出: (w_pred, n_pred) 及校准真值 (w_true, n_true);
    含 DisagreeD 候选时还需 n_disag (模型-检索分歧, 营养素空间)。
    """
    def __init__(self, alpha=ALPHA, tau=TAU, strat='weight', cands=None,
                 n_iter_global=10000, n_iter_bin=5000):
        self.alpha, self.tau = alpha, tau
        self.strat = strat
        self.cands = list(cands) if cands is not None else list(CANDS5)
        self.n_iter_global, self.n_iter_bin = n_iter_global, n_iter_bin
        self.fitted = False

    # ---------- 拟合 ----------
    def fit(self, w_pred, w_true, n_pred, n_true, n_disag=None, seed=SEED,
            recal=None):
        """recal 非 None 时启用两阶段校准: dict(w_pred, n_pred, n_true, n_disag),
        权重在本 fit 数据上学, 重标定 q 在 recal 数据上算 (权重冻结)。"""
        w_pred = np.asarray(w_pred, float).ravel()
        w_true = np.asarray(w_true, float).ravel()
        n_pred = np.maximum(np.asarray(n_pred, float), EPS)
        n_true = np.asarray(n_true, float)
        N, n_out = n_true.shape
        need_disag = 'DisagreeD' in self.cands
        if n_disag is None:
            n_disag = np.zeros((N, n_out))
        else:
            n_disag = np.asarray(n_disag, float)
        rng = np.random.default_rng(seed)

        # --- 候选参数 (每营养素独立; 全部计算, 便宜且避免候选间依赖) ---
        self.split_q = np.zeros(n_out)
        self.norm_q = np.zeros(n_out)
        self.log_q = np.zeros(n_out)
        self.asym_lo = np.zeros(n_out)
        self.asym_hi = np.zeros(n_out)
        self.disag_q = np.zeros(n_out)
        for i in range(n_out):
            p, y = n_pred[:, i], n_true[:, i]
            self.split_q[i] = qhat(np.abs(y - p), self.alpha)
            rel = np.abs(y - p) / (p + EPS)
            self.norm_q[i] = qhat(rel, self.alpha)
            self.log_q[i] = qhat(np.abs(np.log(np.maximum(y, 0.1))
                                        - np.log(np.maximum(p, 0.1))), self.alpha)
            sgn = (y - p) / (p + EPS)
            self.asym_lo[i] = qhat_side(sgn, self.alpha / 2, 'lo')
            self.asym_hi[i] = qhat_side(sgn, self.alpha / 2, 'hi')
            scale = n_disag[:, i] + 0.1 * p          # 10% 预测值下限防塌缩
            self.disag_q[i] = qhat(np.abs(y - p) / scale, self.alpha)
        # Propagate: 重量相对误差分位数 (单一标量, 全营养素共享)
        self.wq = qhat(np.abs(w_true - w_pred) / (w_pred + EPS), self.alpha)
        # MondrianW: 按预测重量分 bin 的相对分位数 (bin 不足 30 用全局 norm_q)
        b_w = np.digitize(w_pred, BINS[1:-1])
        self.mond_q = []
        for bi in range(len(BINS) - 1):
            m = b_w == bi
            row = []
            for i in range(n_out):
                if m.sum() >= 30:
                    row.append(qhat(np.abs(n_true[m, i] - n_pred[m, i])
                                    / (n_pred[m, i] + EPS), self.alpha))
                else:
                    row.append(self.norm_q[i])
            self.mond_q.append(np.array(row))

        # --- 分层 bin ---
        if self.strat == 'weight':
            self.n_bins = len(BINS) - 1
            self.nut_edges = None
        else:  # 'nutrient': 每营养素预测值三分位
            self.n_bins = 3
            self.nut_edges = [np.quantile(n_pred[:, i], [1/3, 2/3])
                              for i in range(n_out)]
        masks = self._strat_masks(w_pred, n_pred)

        # --- 全局权重 (每营养素) ---
        Lc, Uc = self._cand_matrices(w_pred, n_pred, n_disag)
        K = len(self.cands)
        self.w_global = np.zeros((n_out, K))
        for i in range(n_out):
            w0, _ = _dirichlet_winkler(Lc[:, i, :], Uc[:, i, :], n_true[:, i],
                                       self.n_iter_global, rng)
            self.w_global[i] = _refine(Lc[:, i, :], Uc[:, i, :], n_true[:, i], w0)

        # --- 每 bin 权重 (经验收缩 τ) ---
        self.weights = [[None] * n_out for _ in range(self.n_bins)]
        for bi in range(self.n_bins):
            for i in range(n_out):
                m = masks[i][bi]
                n_bin = int(m.sum())
                if n_bin < 20:
                    self.weights[bi][i] = self.w_global[i].copy()
                    continue
                w0, _ = _dirichlet_winkler(Lc[m, i, :], Uc[m, i, :], n_true[m, i],
                                           self.n_iter_bin, rng)
                w0 = _refine(Lc[m, i, :], Uc[m, i, :], n_true[m, i], w0)
                a = n_bin / (n_bin + self.tau)
                self.weights[bi][i] = a * w0 + (1 - a) * self.w_global[i]

        # --- 加权区间 + 重标定 q ---
        # recal 提供 (两阶段): 权重冻结, q 在独立 recal 数据上计算,
        # 消除"权重在同一 cal 上优化 → 重标定 q 偏小 → eval 欠覆盖"
        if recal is not None:
            rw = np.asarray(recal['w_pred'], float).ravel()
            rnp = np.maximum(np.asarray(recal['n_pred'], float), EPS)
            rnt = np.asarray(recal['n_true'], float)
            rnd_ = recal.get('n_disag')
            rnd = np.zeros_like(rnp) if rnd_ is None else np.asarray(rnd_, float)
            Lr, Ur = self._cand_matrices(rw, rnp, rnd)
            lwr, uwr = self._weighted(rw, Lr, Ur, rnp)
            c = (lwr + uwr) / 2
            h = np.maximum((uwr - lwr) / 2, 1e-6)
            s = np.abs(rnt - c) / h
            masks = self._strat_masks(rw, rnp)
        else:
            lw, uw = self._weighted(w_pred, Lc, Uc, n_pred)
            c = (lw + uw) / 2
            h = np.maximum((uw - lw) / 2, 1e-6)
            s = np.abs(n_true - c) / h
        self.q_glob = np.array([qhat(s[:, i], self.alpha) for i in range(n_out)])
        self.q_bin = []
        for bi in range(self.n_bins):
            row = []
            for i in range(n_out):
                m = masks[i][bi]
                row.append(qhat(s[m, i], self.alpha) if m.sum() >= 30
                           else self.q_glob[i])
            self.q_bin.append(np.array(row))

        self.fitted = True
        return self

    # ---------- 分层 mask: masks[i][bi] = 营养素 i 属于 bin bi 的布尔数组 ----------
    def _strat_masks(self, w, n_pred):
        n_out = n_pred.shape[1]
        if self.strat == 'weight':
            b = np.digitize(np.asarray(w, float).ravel(), BINS[1:-1])
            return [[b == bi for bi in range(self.n_bins)] for _ in range(n_out)]
        out = []
        for i in range(n_out):
            b = np.digitize(n_pred[:, i], self.nut_edges[i])
            out.append([b == bi for bi in range(self.n_bins)])
        return out

    # ---------- 候选区间矩阵 (N, 4, K) ----------
    def _cand_matrices(self, w, n_pred, n_disag=None):
        w = np.asarray(w, float).ravel()
        n_pred = np.maximum(np.asarray(n_pred, float), EPS)
        N, n_out = n_pred.shape
        if n_disag is None:
            n_disag = np.zeros_like(n_pred)
        b_w = np.digitize(w, BINS[1:-1])
        L = np.zeros((N, n_out, len(self.cands)))
        U = np.zeros_like(L)
        for i in range(n_out):
            p = n_pred[:, i]
            qm = np.array([self.mond_q[bi][i] for bi in b_w])
            dscale = n_disag[:, i] + 0.1 * p
            cand = {
                'Split':     (np.maximum(0, p - self.split_q[i]), p + self.split_q[i]),
                'Norm':      (np.maximum(0, p * (1 - self.norm_q[i])),
                              p * (1 + self.norm_q[i])),
                'LogSplit':  (p * np.exp(-self.log_q[i]), p * np.exp(self.log_q[i])),
                'AsymNorm':  (np.maximum(0, p * (1 + self.asym_lo[i])),
                              p * (1 + self.asym_hi[i])),
                'DisagreeD': (np.maximum(0, p - self.disag_q[i] * dscale),
                              p + self.disag_q[i] * dscale),
                'Propagate': (np.maximum(0, p * (1 - self.wq)), p * (1 + self.wq)),
                'MondrianW': (np.maximum(0, p * (1 - qm)), p * (1 + qm)),
            }
            for k, name in enumerate(self.cands):
                L[:, i, k], U[:, i, k] = cand[name]
        return L, U

    # ---------- 加权 (按分层 bin 应用集成权重) ----------
    def _weighted(self, w, L, U, n_pred):
        masks = self._strat_masks(w, n_pred)
        N, n_out, _ = L.shape
        lw = np.zeros((N, n_out)); uw = np.zeros((N, n_out))
        for i in range(n_out):
            for bi in range(self.n_bins):
                m = masks[i][bi]
                if not m.any():
                    continue
                wt = self.weights[bi][i]
                lw[m, i] = L[m, i, :] @ wt
                uw[m, i] = U[m, i, :] @ wt
        return lw, uw

    # ---------- 预测 ----------
    def predict_intervals(self, w_pred, n_pred, n_disag=None, variant='c'):
        """返回 lower, upper: (N, 4)。variant: 'none' | 'c' 全局重标定 | 'm' 分 bin 重标定"""
        assert self.fitted
        w_pred = np.asarray(w_pred, float).ravel()
        n_pred = np.maximum(np.asarray(n_pred, float), EPS)
        L, U = self._cand_matrices(w_pred, n_pred, n_disag)
        lw, uw = self._weighted(w_pred, L, U, n_pred)
        if variant == 'none':
            return np.maximum(0, lw), uw
        c = (lw + uw) / 2
        h = np.maximum((uw - lw) / 2, 1e-6)
        masks = self._strat_masks(w_pred, n_pred)
        N, n_out = c.shape
        lo = np.zeros((N, n_out)); hi = np.zeros((N, n_out))
        if variant == 'c':
            for i in range(n_out):
                lo[:, i] = np.maximum(0, c[:, i] - self.q_glob[i] * h[:, i])
                hi[:, i] = c[:, i] + self.q_glob[i] * h[:, i]
        elif variant == 'm':
            for i in range(n_out):
                for bi in range(self.n_bins):
                    m = masks[i][bi]
                    if not m.any():
                        continue
                    q = self.q_bin[bi][i]
                    lo[m, i] = np.maximum(0, c[m, i] - q * h[m, i])
                    hi[m, i] = c[m, i] + q * h[m, i]
        else:
            raise ValueError(variant)
        return lo, hi

    def candidate_intervals(self, w_pred, n_pred, n_disag=None):
        """单独输出候选方法的区间 (基线对比用), (N, 4, K)"""
        return self._cand_matrices(w_pred, n_pred, n_disag)

# ==================== 实验 ====================
CONFIGS = [
    ('5W', CANDS5, 'weight'),     # v1 获胜配置 (基线)
    ('7W', ALL_CANDS, 'weight'),  # + AsymNorm + DisagreeD
    ('7S', ALL_CANDS, 'nutrient'),# + 营养素分位分层
]
AEC_VARIANTS = [('none', 'AEC-N-%s'), ('c', 'AEC-S2c-%s'), ('m', 'AEC-S2m-%s')]

def run_experiment():
    print("=" * 70)
    print("Nutrient AEC-S Conformal v2b | 两阶段校准 + 新候选 AsymNorm/DisagreeD + 分位分层")
    print(f"协议: cal/eval = 326/326 (两阶段: cal 再对半分 A/B), {N_TRIALS} trials, alpha={ALPHA}")
    print("=" * 70)

    cache = get_test_pipeline_cache()
    n_pred_all = hybrid_nutrients(cache)
    n_disag_all = nutrient_disagreement(cache)
    y_all = cache['y_nutr']; w_true_all = cache['y_w']; w_pred_all = cache['w_pred']
    print(f"样本: {len(y_all)} | 生产管线 mean PMAE: "
          f"{np.mean([np.abs(y_all[:, i] - n_pred_all[:, i]).mean() / y_all[:, i].mean() for i in range(4)])*100:.1f}%")

    # 方法标签: 7 候选单独 + 单阶段5W参考 + 三组两阶段配置
    methods = (list(ALL_CANDS)
               + [fmt % '5W' for _, fmt in AEC_VARIANTS]
               + [(fmt % tag) + '2' for tag, _, _ in CONFIGS for _, fmt in AEC_VARIANTS])
    metrics = {m: {n: [] for n in NUTR} for m in methods}
    t0 = time.time()
    for t in range(N_TRIALS):
        idx = np.arange(len(y_all))
        cal_idx, ev_idx = train_test_split(idx, test_size=0.5, random_state=SEED + t)
        calA, calB = train_test_split(cal_idx, test_size=0.5, random_state=SEED + t)

        def pk(ix):
            return dict(w_pred=w_pred_all[ix], w_true=w_true_all[ix],
                        n_pred=n_pred_all[ix], n_true=y_all[ix],
                        n_disag=n_disag_all[ix], seed=SEED + t)

        ev_kw = dict(w_pred=w_pred_all[ev_idx], n_pred=n_pred_all[ev_idx],
                     n_disag=n_disag_all[ev_idx])

        # --- 单阶段参考: 5W 全 cal (v1 获胜配置复现) ---
        cal = NutrientConformalCalibrator(cands=CANDS5, strat='weight').fit(**pk(cal_idx))
        for var, fmt in AEC_VARIANTS:
            lo, hi = cal.predict_intervals(variant=var, **ev_kw)
            for i, nt in enumerate(NUTR):
                metrics[fmt % '5W'][nt].append(_pack(y_all[ev_idx, i], lo[:, i], hi[:, i]))

        # --- 两阶段: 权重 on calA, 重标定 q on calB ---
        for tag, cands, strat in CONFIGS:
            cal = NutrientConformalCalibrator(cands=cands, strat=strat).fit(
                **pk(calA), recal=pk(calB))
            if tag == '7W':   # 候选方法单独评估
                Lc, Uc = cal.candidate_intervals(**ev_kw)
                for k, name in enumerate(ALL_CANDS):
                    for i, nt in enumerate(NUTR):
                        metrics[name][nt].append(
                            _pack(y_all[ev_idx, i], Lc[:, i, k], Uc[:, i, k]))
            for var, fmt in AEC_VARIANTS:
                lo, hi = cal.predict_intervals(variant=var, **ev_kw)
                label = (fmt % tag) + '2'
                for i, nt in enumerate(NUTR):
                    metrics[label][nt].append(_pack(y_all[ev_idx, i], lo[:, i], hi[:, i]))
        print(f"  trial {t+1}/{N_TRIALS} 完成 ({time.time()-t0:.0f}s)")

    # ---------- 汇总 ----------
    print("\n" + "=" * 70)
    print(f"{'方法':<14s}", end='')
    for nt in NUTR:
        print(f"| {nt:^30s} ", end='')
    print()
    print(f"{'':<14s}", end='')
    for _ in NUTR:
        print(f"| {'cov%':>8s} {'relwid%':>9s} {'winkler':>9s} ", end='')
    print("\n" + "-" * 150)
    for m in methods:
        print(f"{m:<14s}", end='')
        for nt in NUTR:
            a = np.array(metrics[m][nt])
            print(f"| {a[:,0].mean()*100:>7.1f} {a[:,2].mean()*100:>8.1f} {a[:,3].mean():>9.1f} ", end='')
        print()

    # 跨营养素综合
    print("\n综合排名 (所有营养素 cov>=89.5% 才有效; 部署仅在 AEC-S2* 中选):")
    print(f"{'方法':<14s} {'min-cov%':>9s} {'mean-relwid%':>13s} {'mean-nWinkler':>14s} 有效")
    rank = []
    y_mean = {nt: y_all[:, i].mean() for i, nt in enumerate(NUTR)}
    for m in methods:
        covs = [np.array(metrics[m][nt])[:, 0].mean() for nt in NUTR]
        rw = [np.array(metrics[m][nt])[:, 2].mean() for nt in NUTR]
        nw = [np.array(metrics[m][nt])[:, 3].mean() / y_mean[nt] for nt in NUTR]
        valid = min(covs) >= 0.895
        rank.append((m, min(covs)*100, np.mean(rw)*100, np.mean(nw), valid))
        print(f"{m:<14s} {min(covs)*100:>8.1f} {np.mean(rw)*100:>12.1f} {np.mean(nw):>13.2f} "
              f"{'✓' if valid else '✗'}")
    joblib.dump({'metrics': metrics, 'rank': rank,
                 'config': {'alpha': ALPHA, 'n_trials': N_TRIALS, 'tau': TAU,
                            'hybrid_alpha': HYBRID_ALPHA, 'bins': BINS[:-1],
                            'configs': [(t, c, s) for t, c, s in CONFIGS]}},
                os.path.join(OUT_DIR, 'nutrient_conformal_results.pkl'))

    # ---------- 部署获胜者 (仅 AEC-S2* 变体, 覆盖承诺优先) ----------
    aec_labels = [r[0] for r in rank if r[0].startswith('AEC-S2') and r[4]]
    if aec_labels:
        winner = min(aec_labels, key=lambda m: next(r[3] for r in rank if r[0] == m))
        w_val = next(r for r in rank if r[0] == winner)
        name = winner.split('-')[-1]        # '5W' | '5W2' | '7W2' | '7S2'
        two_stage = name.endswith('2')
        tag = name[:-1] if two_stage else name
        cfg = {t: (c, s) for t, c, s in CONFIGS}[tag]
        var = 'c' if 'S2c' in winner else 'm'
        # pickle 兼容: 用模块版本的类构建, 其他脚本可加载
        import importlib
        _ncf = importlib.import_module('nutrient_conformal_final')
        cls = _ncf.NutrientConformalCalibrator
        if two_stage:   # 与验证协议一致: 权重 on 半份, q on 另半份
            A, B = train_test_split(np.arange(len(y_all)), test_size=0.5,
                                    random_state=SEED)
            deploy = cls(cands=cfg[0], strat=cfg[1]).fit(
                w_pred_all[A], w_true_all[A], n_pred_all[A], y_all[A],
                n_disag=n_disag_all[A], seed=SEED,
                recal=dict(w_pred=w_pred_all[B], n_pred=n_pred_all[B],
                           n_true=y_all[B], n_disag=n_disag_all[B]))
        else:
            deploy = cls(cands=cfg[0], strat=cfg[1]).fit(
                w_pred_all, w_true_all, n_pred_all, y_all,
                n_disag=n_disag_all, seed=SEED)
        joblib.dump({'calibrator': deploy, 'variant': var, 'strat': cfg[1],
                     'cands': list(cfg[0]), 'two_stage': two_stage,
                     'winner': winner, 'date': '2026-08-20'}, DEPLOY_PATH)
        print(f"\n获胜方法: {winner} (nWinkler={w_val[3]:.2f}, min-cov={w_val[1]:.1f}%)")
        print(f"部署校准器已保存: {DEPLOY_PATH} (variant={var}, strat={cfg[1]}, "
              f"two_stage={two_stage}, 全652拟合)")
    else:
        print("\n警告: 无 AEC-S2* 变体满足全覆盖约束, 保留旧部署校准器")

def _pack(y, l, u):
    cov, width, wk = eval_iv(y, l, u)
    return (cov, width, width / max(y.mean(), EPS), wk)

# ==================== 部署 API (test_100_samples_final.py 调用) ====================
def conformal_report(seed=SEED):
    """诚实验证报告: cal 半拟合 → eval 半评估 (与部署配置同协议)。返回 (文本行列表, 校准器)"""
    cache = get_test_pipeline_cache()
    n_pred = hybrid_nutrients(cache)
    n_disag = nutrient_disagreement(cache)
    y = cache['y_nutr']
    # 部署配置 (旧格式/缺失时回退默认)
    cands, strat, var, two_stage = list(CANDS5), 'weight', 'm', False
    if os.path.exists(DEPLOY_PATH):
        try:
            blob = joblib.load(DEPLOY_PATH)
            var = blob.get('variant', var)
            strat = blob.get('strat', strat)
            cands = blob.get('cands', cands)
            two_stage = blob.get('two_stage', two_stage)
        except Exception:
            pass
    idx = np.arange(len(y))
    cal_idx, ev_idx = train_test_split(idx, test_size=0.5, random_state=seed)
    cal = NutrientConformalCalibrator(cands=cands, strat=strat)
    if two_stage:   # 与部署一致的协议: cal 再对半分 A/B
        calA, calB = train_test_split(cal_idx, test_size=0.5, random_state=seed)
        cal.fit(cache['w_pred'][calA], cache['y_w'][calA],
                n_pred[calA], y[calA], n_disag=n_disag[calA], seed=seed,
                recal=dict(w_pred=cache['w_pred'][calB], n_pred=n_pred[calB],
                           n_true=y[calB], n_disag=n_disag[calB]))
    else:
        cal.fit(cache['w_pred'][cal_idx], cache['y_w'][cal_idx],
                n_pred[cal_idx], y[cal_idx], n_disag=n_disag[cal_idx], seed=seed)
    lo, hi = cal.predict_intervals(cache['w_pred'][ev_idx], n_pred[ev_idx],
                                   n_disag[ev_idx], var)
    lines = [f"Conformal intervals (AEC-S nutrient, variant={var}, strat={strat}, "
             f"two_stage={two_stage}, cal={len(cal_idx)}/eval={len(ev_idx)}, alpha={ALPHA}):"]
    for i, nt in enumerate(NUTR):
        cov, width, wk = eval_iv(y[ev_idx, i], lo[:, i], hi[:, i])
        lines.append(f"  {nt:<8s} cov={cov*100:5.1f}%  width={width:8.1f}  "
                     f"rel-width={width/y[ev_idx, i].mean()*100:5.1f}%  winkler={wk:7.1f}")
    sample = "\n".join(
        f"    [{j}] Cal {n_pred[ev_idx[j], 0]:6.0f} [{lo[j, 0]:6.0f}, {hi[j, 0]:6.0f}]  "
        f"true={y[ev_idx[j], 0]:6.0f}" for j in range(3))
    lines.append("  样例 (前3个 eval 样本, kcal):")
    lines.append(sample)
    return lines, cal

if __name__ == '__main__':
    run_experiment()
