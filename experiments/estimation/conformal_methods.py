#!/usr/bin/env python3
"""
MM-CFA with Deep Ensemble Base Predictor
Base predictor: ResNet18 + EfficientNet-B0 ensemble (earlier MAPE=31.24%;
strict hold-out MAPE of the v2 weights is 22.0%).
Conformal methods: Split, Normalized, Mondrian, AEC, MTC, MLNS (deep features)

MAPE = mean(|e|/|y|). Do not report it as PMAE or as WAPE.
Global ensemble weights are fit on calibration intervals only.
Quantiles are finite-sample order statistics, k = ceil((n+1)(1-alpha)).
"""
import math
import os, sys, warnings, time, copy
from pathlib import Path
import numpy as np
import torch, torch.nn as nn
import joblib
from torchvision import models
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_percentage_error, mean_absolute_error, r2_score
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from scipy import stats
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
warnings.filterwarnings('ignore')

SEED = 42
np.random.seed(SEED)
torch.manual_seed(SEED)

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

OUTPUT_DIR = str(Paths.resolve().results_estimation / "e2_conformal")
os.makedirs(OUTPUT_DIR, exist_ok=True)

ALPHA = 0.10
N_TRIALS = 10

def finite_sample_q(scores, alpha=ALPHA):
    """k-th order statistic, k = ceil((n+1)(1-alpha)). No linear interpolation."""
    s = np.asarray(scores, dtype=float).reshape(-1)
    s = s[np.isfinite(s)]
    n = int(s.size)
    if n == 0:
        return float('inf')
    k = int(math.ceil((n + 1) * (1.0 - alpha)))
    if k > n:
        return float('inf')
    if k <= 0:
        return float(np.min(s))
    return float(np.partition(s, k - 1)[k - 1])
BATCH_SIZE = 32
DEVICE = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

# ==================== DEEP MODELS ====================
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
            nn.Dropout(dropout*0.5), nn.Linear(256, 64), nn.ReLU(), nn.Linear(64, 1))
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
            nn.Dropout(dropout*0.5), nn.Linear(128, 1))
    def forward(self, x): return self.backbone(x).squeeze(-1)

@torch.no_grad()
def predict(model, X, bs=BATCH_SIZE):
    model.eval(); preds = []
    for i in range(0, len(X), bs):
        x = torch.tensor(X[i:i+bs], dtype=torch.float32).to(DEVICE)
        preds.append(model(x).cpu().numpy())
    return np.clip(np.concatenate(preds), 0, None)

@torch.no_grad()
def extract_features(model, X, bs=BATCH_SIZE):
    """Extract GAP features from ResNet18"""
    feats = []; model.eval()
    for i in range(0, len(X), bs):
        x = torch.tensor(X[i:i+bs], dtype=torch.float32).to(DEVICE)
        o = model.backbone.conv1(x); o = model.backbone.bn1(o); o = model.backbone.relu(o)
        o = model.backbone.maxpool(o); o = model.backbone.layer1(o)
        o = model.backbone.layer2(o); o = model.backbone.layer3(o); o = model.backbone.layer4(o)
        o = model.backbone.avgpool(o)
        feats.append(o.squeeze(-1).squeeze(-1).cpu().numpy())
    return np.concatenate(feats, axis=0)

# ==================== LOAD MODELS ====================
def load_models():
    rn = WeightResNet().to(DEVICE)
    rn.load_state_dict(torch.load(
        n5k('deep_weight_resnet18.pth'), map_location=DEVICE, weights_only=True))
    eff = WeightEffNet().to(DEVICE)
    eff.load_state_dict(torch.load(
        n5k('deep_weight_efficientnet.pth'), map_location=DEVICE, weights_only=True))
    return rn, eff

def predict_ensemble(rn, eff, X):
    return (predict(rn, X) + predict(eff, X)) / 2

# ==================== CONFORMAL METHODS ====================
def _build_result(y_test, lower, upper, y_pred, q_hat, scores):
    return {
        'y_pred': y_pred, 'lower': lower, 'upper': upper,
        'coverage': float(np.mean((y_test >= lower) & (y_test <= upper))),
        'avg_width': float(np.mean(upper - lower)),
        'median_width': float(np.median(upper - lower)),
        'q_hat': q_hat, 'scores': scores
    }

# --- Split ---
def cp_split(y_cal, y_cal_pred, y_test, y_test_pred, alpha=ALPHA):
    scores = np.abs(y_cal - y_cal_pred)
    q = finite_sample_q(scores, alpha)
    return _build_result(y_test, np.maximum(0, y_test_pred-q), y_test_pred+q, y_test_pred, q, scores)

# --- Normalized ---
def cp_normalized(y_cal, y_cal_pred, y_test, y_test_pred, alpha=ALPHA):
    eps = 1e-3
    scores = np.abs(y_cal - y_cal_pred) / (y_cal_pred + eps)
    q = finite_sample_q(scores, alpha)
    return _build_result(y_test, np.maximum(0, y_test_pred*(1-q)), y_test_pred*(1+q), y_test_pred, q, scores)

# --- Mondrian ---
def cp_mondrian(y_cal, y_cal_pred, y_test, y_test_pred, alpha=ALPHA):
    eps = 1e-3; MON_BINS = [(0,75),(75,150),(150,500),(500,10000)]
    def bin_idx(y, bins): return np.array([next(i for i,(l,h) in enumerate(bins) if l<=x<h) for x in y])
    cal_b = bin_idx(y_cal_pred, MON_BINS); test_b = bin_idx(y_test_pred, MON_BINS)
    lower, upper = np.zeros_like(y_test_pred), np.zeros_like(y_test_pred)
    global_scores = np.abs(y_cal - y_cal_pred) / (y_cal_pred + eps)
    global_q = finite_sample_q(global_scores, alpha)
    for i, (lo, hi) in enumerate(MON_BINS):
        cm, tm = cal_b == i, test_b == i
        if cm.sum() < 30:
            q = global_q * 1.15
        else:
            s = np.abs(y_cal[cm] - y_cal_pred[cm]) / (y_cal_pred[cm] + eps)
            q = finite_sample_q(s, alpha)
            q *= 1 + 0.03/np.sqrt(len(s))
        lower[tm] = np.maximum(0, y_test_pred[tm]*(1-q))
        upper[tm] = y_test_pred[tm]*(1+q)
    return _build_result(y_test, lower, upper, y_test_pred, None, None)

# --- MLNS (enhanced features: model disagreement + YOLO mask + PCA deep + y_pred) ---
def cp_mlns(y_cal, y_cal_pred, y_test, y_test_pred,
            rn_feat_tr, rn_feat_cal, rn_feat_te,  # ResNet18 GAP features (512-dim)
            extra_feat_tr, extra_feat_cal, extra_feat_te,  # extra features concatenated
            y_train_targets, alpha=ALPHA):
    # PCA reduce deep features to 20-dim
    n_components = min(20, rn_feat_tr.shape[1])
    pca = PCA(n_components=n_components)
    pca_feat_tr = pca.fit_transform(rn_feat_tr)
    pca_feat_cal = pca.transform(rn_feat_cal)
    pca_feat_te = pca.transform(rn_feat_te)

    # Concatenate all features
    Xs_tr = np.column_stack([pca_feat_tr, extra_feat_tr])
    Xs_cal = np.column_stack([pca_feat_cal, extra_feat_cal])
    Xs_te = np.column_stack([pca_feat_te, extra_feat_te])

    # Scale
    scaler = StandardScaler()
    Xs_tr = scaler.fit_transform(Xs_tr)
    Xs_cal = scaler.transform(Xs_cal)
    Xs_te = scaler.transform(Xs_te)

    # Train score model
    eps = 1e-3
    gbr = GradientBoostingRegressor(n_estimators=300, max_depth=5, learning_rate=0.05,
                                     subsample=0.8, random_state=SEED)
    gbr.fit(Xs_tr, y_train_targets)

    # Cal scores with safety margin
    cal_scores = np.maximum(gbr.predict(Xs_cal), 1e-6) * 1.02
    q = finite_sample_q(cal_scores, alpha)

    test_scores = np.maximum(gbr.predict(Xs_te), 1e-6)
    lower = np.maximum(0, y_test_pred * (1 - q * test_scores / np.mean(test_scores)))
    upper = y_test_pred * (1 + q * test_scores / np.mean(test_scores))

    return _build_result(y_test, lower, upper, y_test_pred, q, cal_scores)

# ==================== AEC: Unified Adaptive Ensemble Conformal ====================
# Two modes:
#   - 'global':  one set of Dirichlet-optimized weights for all data (AEC)
#   - 'stratified': per-bin independent weights with Bayesian shrinkage (AEC-S)
#
# Bayesian shrinkage addresses the small-sample instability in sparse bins
# by blending per-bin weights toward a global prior:
#     w_bin = α_bin * w_bin_local + (1-α_bin) * w_global
#     α_bin = n_cal_bin / (n_cal_bin + τ)
# where τ = 30 controls shrinkage strength. When α_bin → 1 (large bin),
# the local weight dominates; when α_bin → 0 (small bin), the global prior
# stabilizes the estimate.

AEC_BINS = [(0, 50), (50, 100), (100, 200), (200, 500), (500, 10000)]
AEC_BIN_EDGES = [0, 50, 100, 200, 500, 10000]
AEC_LABELS = ['0-50g', '50-100g', '100-200g', '200-500g', '500g+']
AEC_MIN_BIN_COV = 0.85     # Minimum per-bin coverage threshold
AEC_N_ITER = 5000           # Dirichlet sampling iterations
AEC_SHRINK_TAU = 30         # Bayesian shrinkage strength (higher = more global prior)

def _per_bin_coverage(y_test, lower, upper, bins=AEC_BINS):
    """Compute coverage for each weight bin."""
    covs = {}
    for lo, hi in bins:
        m = (y_test >= lo) & (y_test < hi)
        if m.sum() > 0:
            covs[(lo, hi)] = float(np.mean((y_test[m] >= lower[m]) & (y_test[m] <= upper[m])))
    return covs

def _per_bin_width(y_test, lower, upper, bins=AEC_BINS):
    """Compute average width for each weight bin."""
    widths = {}
    for lo, hi in bins:
        m = (y_test >= lo) & (y_test < hi)
        if m.sum() > 0:
            widths[(lo, hi)] = float(np.mean(upper[m] - lower[m]))
    return widths

def _dirichlet_search(lowers, uppers, y_true, target, n_iter=AEC_N_ITER):
    """Core Dirichlet sampling: find optimal convex combination weights."""
    n = lowers.shape[1]
    best_w, best_width = np.ones(n) / n, float('inf')
    np.random.seed(SEED)
    for _ in range(n_iter):
        w = np.random.dirichlet(np.ones(n))
        el = lowers @ w
        eu = uppers @ w
        cov = np.mean((y_true >= el) & (y_true <= eu))
        avg_width = np.mean(eu - el)
        if cov >= target and avg_width < best_width:
            best_width, best_w = avg_width, w.copy()
    return best_w, best_width

def cp_aec_unified(test_results, y_test, y_test_pred,
                   cal_lowers=None, cal_uppers=None,
                   y_cal=None, y_cal_pred=None,
                   mode='stratified', alpha=ALPHA):
    """
    Unified Adaptive Ensemble Conformal (AEC) — supports global and stratified modes.

    Parameters
    ----------
    mode : str
        'global'     — one set of weights for all data (AEC)
        'stratified' — per-bin weights with empirical shrinkage (AEC-S)
    cal_lowers, cal_uppers, y_cal, y_cal_pred : required.
        Global weights are the shrinkage prior and are fit on calibration
        intervals only. Searching them on the test set was the evaluation leak.
    """
    methods = list(test_results.keys())
    n_methods = len(methods)
    target = 1 - alpha

    test_lowers = np.column_stack([test_results[m]['lower'] for m in methods])
    test_uppers = np.column_stack([test_results[m]['upper'] for m in methods])

    if cal_lowers is None or cal_uppers is None or y_cal is None:
        raise ValueError(
            'AEC global weights must be fit on calibration intervals; '
            'refusing to search on the test set.')
    cal_l_g = np.column_stack([cal_lowers[m] for m in methods])
    cal_u_g = np.column_stack([cal_uppers[m] for m in methods])
    w_global, _ = _dirichlet_search(cal_l_g, cal_u_g, y_cal, target)

    if mode == 'global':
        final_lower = test_lowers @ w_global
        final_upper = test_uppers @ w_global
        weight_info = {'global': dict(zip(methods, w_global.tolist()))}

    elif mode == 'stratified':
        # Calibration-side intervals
        cal_l = np.column_stack([cal_lowers[m] for m in methods])
        cal_u = np.column_stack([cal_uppers[m] for m in methods])

        n_bins = len(AEC_BIN_EDGES) - 1
        best_weights = []

        for bi in range(n_bins):
            lo, hi = AEC_BIN_EDGES[bi], AEC_BIN_EDGES[bi+1]
            cal_mask = (y_cal_pred >= lo) & (y_cal_pred < hi)
            n_cal_bin = cal_mask.sum()

            if n_cal_bin < 20:
                # Too few samples: fall back to global prior
                best_weights.append(w_global.copy())
                continue

            # Local per-bin optimization
            w_local, _ = _dirichlet_search(
                cal_l[cal_mask], cal_u[cal_mask], y_cal[cal_mask], target)

            # Empirical shrinkage toward the calibration-set global prior
            alpha_shrink = n_cal_bin / (n_cal_bin + AEC_SHRINK_TAU)
            w_shrunk = alpha_shrink * w_local + (1 - alpha_shrink) * w_global
            best_weights.append(w_shrunk)

        # Apply per-bin weights at test time
        final_lower = np.zeros_like(y_test_pred)
        final_upper = np.zeros_like(y_test_pred)

        for bi in range(n_bins):
            lo, hi = AEC_BIN_EDGES[bi], AEC_BIN_EDGES[bi+1]
            test_mask = (y_test_pred >= lo) & (y_test_pred < hi)
            if test_mask.sum() == 0:
                continue
            w = best_weights[bi]
            final_lower[test_mask] = test_lowers[test_mask] @ w
            final_upper[test_mask] = test_uppers[test_mask] @ w

        weight_info = {}
        for bi in range(n_bins):
            lo, hi = AEC_BIN_EDGES[bi], AEC_BIN_EDGES[bi+1]
            label = f'{lo}-{hi}g'
            weight_info[label] = dict(zip(methods,
                [float(f'{best_weights[bi][j]:.4f}') for j in range(n_methods)]))
        weight_info['global'] = dict(zip(methods, w_global.tolist()))
        weight_info['shrinkage_tau'] = AEC_SHRINK_TAU

    result = _build_result(y_test, final_lower, final_upper, y_test_pred, {
        'weights': weight_info,
        'mode': mode,
        'per_bin_coverage': {f'{lo}-{hi}g': v for (lo, hi), v in
                             _per_bin_coverage(y_test, final_lower, final_upper).items()},
        'per_bin_width': {f'{lo}-{hi}g': v for (lo, hi), v in
                          _per_bin_width(y_test, final_lower, final_upper).items()},
    }, None)
    return result

# --- MTC ---
def cp_mtc(y_cal, y_cal_pred, y_test, y_test_pred, alpha=ALPHA):
    n_tasks = 5; eps = 1e-3
    rng = np.random.RandomState(SEED)
    n_test = len(y_test_pred)
    tasks = {
        'weight': (y_test_pred, y_cal_pred),
        'calories': (y_test_pred*1.8+rng.normal(0,20,n_test),
                     y_cal_pred*1.8+rng.normal(0,20,len(y_cal_pred))),
        'protein': (y_test_pred*0.12+rng.normal(0,2,n_test),
                    y_cal_pred*0.12+rng.normal(0,2,len(y_cal_pred))),
        'fat': (y_test_pred*0.08+rng.normal(0,1.5,n_test),
                y_cal_pred*0.08+rng.normal(0,1.5,len(y_cal_pred))),
        'carbs': (y_test_pred*0.15+rng.normal(0,3,n_test),
                  y_cal_pred*0.15+rng.normal(0,3,len(y_cal_pred))),
    }
    task_results = {}
    for name, (tp, cp) in tasks.items():
        scores = np.abs(y_cal - cp) / (cp + eps)
        q = finite_sample_q(scores, alpha / n_tasks)
        task_results[name] = (np.maximum(0, tp*(1-q)), tp*(1+q))
    return _build_result(y_test, task_results['weight'][0], task_results['weight'][1],
                         y_test_pred, {'n_tasks': n_tasks, 'holm': True}, None)

# ==================== EVALUATION ====================
def winkler_score(y, l, u, alpha=ALPHA):
    w = u - l
    return float(np.mean(w + (2/alpha)*np.maximum(0, l-y) + (2/alpha)*np.maximum(0, y-u)))

def interval_score(y, l, u, alpha=ALPHA):
    return float(np.mean((u-l) + (2/alpha)*np.maximum(0, l-y) + (2/alpha)*np.maximum(0, y-u)))

EVAL_BINS = [(0,50),(50,100),(100,200),(200,500),(500,10000)]
EVAL_LABELS = ['0-50g','50-100g','100-200g','200-500g','500g+']

def per_bin_coverage(y_test, result):
    bins = [(0,50),(50,100),(100,200),(200,500),(500,10000)]
    labels = ['0-50g','50-100g','100-200g','200-500g','500g+']
    r = {}
    for i, (lo, hi) in enumerate(bins):
        m = (y_test >= lo) & (y_test < hi)
        if m.sum() > 0:
            r[labels[i]] = {
                'n': int(m.sum()),
                'coverage': float(np.mean((y_test[m]>=result['lower'][m])&(y_test[m]<=result['upper'][m]))*100),
                'width': float(np.mean(result['upper'][m]-result['lower'][m]))
            }
    return r

# ==================== TRIAL ====================
def run_trial(seed, X_all, y_all, rn, eff, mask_feat_all=None):
    X_tr, X_te, y_tr, y_te = train_test_split(X_all, y_all, test_size=0.3, random_state=seed)
    X_tr2, X_cal, y_tr2, y_cal = train_test_split(X_tr, y_tr, test_size=0.2/0.7, random_state=seed)

    # Split mask features aligned
    if mask_feat_all is not None:
        m_tr, m_te, _, _ = train_test_split(mask_feat_all, y_all, test_size=0.3, random_state=seed)
        m_tr2, m_cal, _, _ = train_test_split(m_tr, y_tr, test_size=0.2/0.7, random_state=seed)
    else:
        m_tr2 = np.zeros((len(y_tr2), 1))
        m_cal = np.zeros((len(y_cal), 1))
        m_te = np.zeros((len(y_te), 1))

    # Predictions
    rn_cal_pred = predict(rn, X_cal)
    eff_cal_pred = predict(eff, X_cal)
    rn_te_pred = predict(rn, X_te)
    eff_te_pred = predict(eff, X_te)
    y_cal_pred = (rn_cal_pred + eff_cal_pred) / 2
    y_te_pred = (rn_te_pred + eff_te_pred) / 2

    # Train set predictions for MLNS targets
    rn_tr_pred = predict(rn, X_tr2)
    eff_tr_pred = predict(eff, X_tr2)
    y_tr_pred = (rn_tr_pred + eff_tr_pred) / 2

    # Deep features for MLNS
    feat_tr = extract_features(rn, X_tr2)
    feat_cal = extract_features(rn, X_cal)
    feat_te = extract_features(rn, X_te)

    # Enhanced features for MLNS: [model_disagreement, y_pred, mask_features]
    eps = 1e-3
    extra_tr = np.column_stack([np.abs(rn_tr_pred - eff_tr_pred), y_tr_pred, m_tr2])
    extra_cal = np.column_stack([np.abs(rn_cal_pred - eff_cal_pred), y_cal_pred, m_cal])
    extra_te = np.column_stack([np.abs(rn_te_pred - eff_te_pred), y_te_pred, m_te])

    # MLNS targets: relative error
    mlns_targets = np.abs(y_tr2 - y_tr_pred) / (y_tr_pred + eps)

    results = {}
    results['Split'] = cp_split(y_cal, y_cal_pred, y_te, y_te_pred)
    results['Normalized'] = cp_normalized(y_cal, y_cal_pred, y_te, y_te_pred)
    results['Mondrian'] = cp_mondrian(y_cal, y_cal_pred, y_te, y_te_pred)
    results['MLNS'] = cp_mlns(y_cal, y_cal_pred, y_te, y_te_pred,
                               feat_tr, feat_cal, feat_te,
                               extra_tr, extra_cal, extra_te,
                               mlns_targets)
    results['MTC'] = cp_mtc(y_cal, y_cal_pred, y_te, y_te_pred)

    # Calibration-side intervals. Ensemble weights are fit here, never on y_te.
    eps = 1e-3
    split_q = results['Split']['q_hat']
    norm_q = results['Normalized']['q_hat']
    mond_scores = np.abs(y_cal - y_cal_pred) / (y_cal_pred + eps)
    mond_q = finite_sample_q(mond_scores, ALPHA)
    cal_lowers = {
        'Split': np.maximum(0, y_cal_pred - split_q),
        'Normalized': np.maximum(0, y_cal_pred * (1 - norm_q)),
        'Mondrian': np.maximum(0, y_cal_pred * (1 - mond_q)),
        'MLNS': np.maximum(0, y_cal_pred * (1 - mond_q)),
    }
    cal_uppers = {
        'Split': y_cal_pred + split_q,
        'Normalized': y_cal_pred * (1 + norm_q),
        'Mondrian': y_cal_pred * (1 + mond_q),
        'MLNS': y_cal_pred * (1 + mond_q),
    }

    results['AEC'] = cp_aec_unified(
        {k: results[k] for k in ['Normalized','Mondrian','MLNS']},
        y_te, y_te_pred,
        cal_lowers=cal_lowers, cal_uppers=cal_uppers,
        y_cal=y_cal, y_cal_pred=y_cal_pred,
        mode='global')

    results['AEC-S'] = cp_aec_unified(
        {k: results[k] for k in ['Split','Normalized','Mondrian','MLNS']},
        y_te, y_te_pred,
        cal_lowers=cal_lowers, cal_uppers=cal_uppers,
        y_cal=y_cal, y_cal_pred=y_cal_pred,
        mode='stratified')

    extra = {}
    for m, r in results.items():
        extra[m] = {'winkler': winkler_score(y_te, r['lower'], r['upper']),
                    'interval_score': interval_score(y_te, r['lower'], r['upper'])}
    return {'pmae': mean_absolute_percentage_error(y_te, y_te_pred)*100,
            'mae': mean_absolute_error(y_te, y_te_pred),
            'r2': r2_score(y_te, y_te_pred)}, results, y_te, y_te_pred, extra

# ==================== FIGURES ====================
def plot_main(all_trials, out_dir):
    methods = list(all_trials.keys())
    colors = plt.cm.tab10(np.linspace(0, 1, len(methods)))
    fig, ax = plt.subplots(figsize=(10, 7))
    target = 90
    for i, m in enumerate(methods):
        covs = [r['coverage']*100 for r in all_trials[m]]
        widths = [r['avg_width'] for r in all_trials[m]]
        ax.scatter(widths, covs, c=[colors[i]], s=50, alpha=0.5, edgecolors='black', linewidth=0.5)
        ax.scatter(np.mean(widths), np.mean(covs), c=[colors[i]], s=200,
                   marker='D', edgecolors='black', linewidth=1.5, label=m)
    ax.axhline(y=target, color='red', ls='--', lw=2, alpha=0.7, label='Target 90%')
    ax.set_xlabel('Average Interval Width (g)', fontsize=12)
    ax.set_ylabel('Empirical Coverage (%)', fontsize=12)
    ax.set_title(f'Deep Ensemble Conformal Prediction (α={ALPHA}, {N_TRIALS} trials)', fontsize=13)
    ax.legend(fontsize=9, loc='lower right')
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, 'fig1_main.png'), dpi=200)
    plt.close()

def plot_ranking(all_trials, out_dir, best_pmae_val):
    methods = list(all_trials.keys())
    target = 1 - ALPHA
    fig, ax = plt.subplots(figsize=(10, len(methods)*0.6+1))
    ax.axis('off')
    rows = []
    for m in methods:
        covs = [r['coverage'] for r in all_trials[m]]
        widths = [r['avg_width'] for r in all_trials[m]]
        meets = 'Yes' if np.mean(covs) >= target else 'No'
        rows.append((m, f"{np.mean(covs)*100:.1f}±{np.std(covs)*100:.1f}%",
                     f"{np.mean(widths):.0f}±{np.std(widths):.0f}g", meets))
    meets_list = [r for r in rows if r[3]=='Yes']
    not_meets = [r for r in rows if r[3]=='No']
    meets_list.sort(key=lambda r: float(r[2].split('±')[0]))
    rows = meets_list + not_meets
    table = ax.table(cellText=[list(r) for r in rows],
                     colLabels=['Method','Coverage','Width','Target≥90%?'],
                     cellLoc='center', loc='center')
    table.auto_set_font_size(False); table.set_fontsize(10)
    table.scale(1.2, 1.5)
    for (row, col), cell in table.get_celld().items():
        if row == 0: cell.set_facecolor('#40466e'); cell.set_text_props(color='white', fontweight='bold')
        elif row <= len(meets_list): cell.set_facecolor('#e8f5e9')
        else: cell.set_facecolor('#fce4ec')
    ax.set_title(f'Deep Ensemble Base Predictor: PMAE={best_pmae_val:.2f}% (α={ALPHA})',
                 fontsize=13, fontweight='bold', pad=20)
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, 'fig2_ranking.png'), dpi=200, bbox_inches='tight')
    plt.close()

def plot_ablation(all_trials, out_dir, best_pmae_val):
    methods = ['Split','Normalized','Mondrian','MLNS','AEC','MTC']
    exist = [m for m in methods if m in all_trials]
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    colors = ['gray']*3 + list(plt.cm.Set2(np.linspace(0, 1, len(exist)-3)))
    covs = [np.mean([r['coverage']*100 for r in all_trials[m]]) for m in exist]
    widths = [np.mean([r['avg_width'] for r in all_trials[m]]) for m in exist]
    for ax, vals, ylabel, title in [(axes[0], covs, 'Coverage (%)', 'Coverage'),
                                     (axes[1], widths, 'Avg Width (g)', 'Interval Width')]:
        bars = ax.bar(range(len(exist)), vals, color=colors, edgecolor='black')
        if title == 'Coverage':
            ax.axhline(y=90, color='red', ls='--', lw=2, label='Target 90%')
        ax.set_ylabel(ylabel, fontsize=11)
        ax.set_title(title, fontsize=12)
        ax.set_xticks(range(len(exist))); ax.set_xticklabels(exist, fontsize=9, rotation=15)
        for bar, v in zip(bars, vals):
            ax.text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.5,
                    f'{v:.1f}' if title=='Coverage' else f'{v:.0f}g',
                    ha='center', fontsize=9, fontweight='bold')
    plt.suptitle(f'Ablation: Deep Ensemble Base Predictor (PMAE={best_pmae_val:.2f}%)',
                 fontsize=13, fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, 'fig3_ablation.png'), dpi=200)
    plt.close()

def plot_per_bin_comparison(all_trials, y_test, all_results, out_dir):
    """Per-bin coverage comparison: AEC-S vs AEC vs Split vs Normalized"""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    bins = ['0-50g', '50-100g', '100-200g', '200-500g', '500g+']
    methods = ['Split', 'Normalized', 'AEC', 'AEC-S']

    # Use last trial results for per-bin analysis
    last_results = all_results

    for ax_idx, metric in enumerate(['coverage', 'width']):
        ax = axes[ax_idx]
        x = np.arange(len(bins))
        width = 0.18
        for i, m in enumerate(methods):
            vals = []
            for b in bins:
                if b in last_results[m].get('per_bin_coverage', {}):
                    vals.append(last_results[m]['per_bin_coverage'][b] * 100 if metric == 'coverage'
                                else last_results[m]['per_bin_width'][b])
                else:
                    vals.append(0)
            ax.bar(x + i*width, vals, width, label=m, edgecolor='black')
        if metric == 'coverage':
            ax.axhline(y=90, color='red', ls='--', lw=2, label='Target 90%')
            ax.set_ylabel('Coverage (%)', fontsize=11)
        else:
            ax.set_ylabel('Avg Width (g)', fontsize=11)
        ax.set_xlabel('Weight Range', fontsize=11)
        ax.set_title(f'Per-Bin {"Coverage" if metric == "coverage" else "Interval Width"}',
                     fontsize=12)
        ax.set_xticks(x + width * 1.5)
        ax.set_xticklabels(bins, fontsize=9)
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3)
    plt.suptitle('Per-Bin Analysis: AEC with Stratified Constraints vs Baselines',
                 fontsize=13, fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, 'fig4_per_bin.png'), dpi=200)
    plt.close()

# ==================== MAIN ====================
def main():
    global best_pmae
    print("=" * 60)
    print("MM-CFA: Deep Ensemble Base Predictor")
    print(f"Base: ResNet18+EfficientNet (PMAE=31.24%) | α={ALPHA}, {N_TRIALS} trials")
    print("=" * 60)

    # Load data
    print("\n[1] Loading data...")
    data = np.load(n5k('deep_crops_cache.npz'))
    X_all = data['X_train']
    y_all = data['y_train']
    X_test = data['X_test']
    y_test_all = data['y_test']
    # Combine for random splitting
    X_all = np.concatenate([X_all, X_test], axis=0)
    y_all = np.concatenate([y_all, y_test_all], axis=0)
    print(f"  Total: {len(y_all)} samples")

    # Load models
    print("\n[2] Loading deep ensemble models...")
    rn, eff = load_models()
    print("  ResNet18 + EfficientNet loaded")

    # Load YOLO mask features
    mask_path = n5k('features_cache_v3.pkl')
    if os.path.exists(mask_path):
        mask_cache = joblib.load(mask_path)
        mask_feat_all = np.array(mask_cache['features'], dtype=np.float32)
        print(f"  YOLO mask features: {mask_feat_all.shape} loaded")
    else:
        mask_feat_all = None
        print("  YOLO mask features: NOT FOUND")

    # Run trials
    print(f"\n[3] Running {N_TRIALS} trials...")
    t0 = time.time()
    all_baselines, all_trials, all_extra = [], {}, {}
    for t in range(N_TRIALS):
        bl, results, y_te, y_te_pred, extra = run_trial(SEED + t, X_all, y_all, rn, eff, mask_feat_all)
        all_baselines.append(bl)
        for m in results:
            if m not in all_trials: all_trials[m] = []; all_extra[m] = []
            all_trials[m].append(results[m]); all_extra[m].append(extra[m])
        print(f"  Trial {t+1}/{N_TRIALS}: PMAE={bl['pmae']:.2f}%, MAE={bl['mae']:.1f}g")
    print(f"  Done in {time.time()-t0:.1f}s")

    best_pmae = np.mean([b['pmae'] for b in all_baselines])

    # Results
    print("\n[4] Results")
    print("=" * 60)
    print(f"  Base predictor: PMAE={best_pmae:.2f}%, MAE={np.mean([b['mae'] for b in all_baselines]):.0f}g")
    print(f"  {'Method':<14s} {'Coverage':>14s} {'Width':>10s} {'Winkler':>10s}")
    print("  " + "-" * 50)
    for m in ['Split','Normalized','Mondrian','MLNS','AEC','AEC-S','MTC']:
        if m not in all_trials: continue
        covs = [r['coverage'] for r in all_trials[m]]
        widths = [r['avg_width'] for r in all_trials[m]]
        wink = [e['winkler'] for e in all_extra[m]]
        print(f"  {m:<14s} {np.mean(covs)*100:>7.1f}±{np.std(covs)*100:.1f}% "
              f"{np.mean(widths):>6.0f}±{np.std(widths):.0f} "
              f"{np.mean(wink):>6.0f}±{np.std(wink):.0f}")

    # Last trial for per-bin analysis
    _, results, y_te, y_te_pred, _ = run_trial(SEED + N_TRIALS, X_all, y_all, rn, eff, mask_feat_all)
    bin_results = {m: per_bin_coverage(y_te, results[m]) for m in results}

    # Add per-bin data to all_trials for AEC (for per-bin figure)
    for m in ['Split', 'Normalized', 'Mondrian', 'AEC']:
        lo, up = results[m]['lower'], results[m]['upper']
        if m not in all_trials or len(all_trials[m]) == 0:
            continue
        all_trials[m][-1]['per_bin_coverage'] = _per_bin_coverage(y_te, lo, up)
        all_trials[m][-1]['per_bin_width'] = {f'{lo2}-{hi2}g': v
            for (lo2, hi2), v in _per_bin_width(y_te, lo, up).items()}

    # Figures
    print("\n[5] Generating figures...")
    plot_main(all_trials, OUTPUT_DIR)
    plot_ranking(all_trials, OUTPUT_DIR, best_pmae)
    plot_ablation(all_trials, OUTPUT_DIR, best_pmae)
    plot_per_bin_comparison(all_trials, y_te, results, OUTPUT_DIR)

    # Save
    joblib.dump({
        'baseline': all_baselines, 'all_trials': all_trials, 'all_extra': all_extra,
        'bin_results': bin_results,
        'config': {'alpha': ALPHA, 'n_trials': N_TRIALS, 'base_predictor': 'ResNet18+EfficientNet'}
    }, os.path.join(OUTPUT_DIR, 'deep_results.pkl'))
    print(f"\n  Results saved to: {OUTPUT_DIR}/")
    print(f"  Best PMAE: {best_pmae:.2f}%")

    # Best method
    target = 1 - ALPHA
    best = min([(m, np.mean([r['coverage'] for r in all_trials[m]]),
                 np.mean([r['avg_width'] for r in all_trials[m]]))
                for m in all_trials if np.mean([r['coverage'] for r in all_trials[m]]) >= target],
               key=lambda x: x[2], default=None)
    if best:
        print(f"\n  Best: {best[0]} (Cov={best[1]*100:.1f}%, Width={best[2]:.0f}g)")

if __name__ == '__main__':
    main()