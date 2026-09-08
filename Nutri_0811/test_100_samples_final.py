#!/usr/bin/env python3
"""
Nutrition5K Nutrient Prediction — Inference Script
===================================================
Usage:
  python test_100_samples_final.py              # Test 100 random samples (default)
  python test_100_samples_final.py --all        # Test all test set samples
  python test_100_samples_final.py --n 50       # Test N random samples
  python test_100_samples_final.py --no-conformal   # 跳过共形区间验证

Models:
  Weight:  ResNet18 + EfficientNet-B0 ensemble (log-MSE, 100ep, PMAE=9.44%)
  Density: ResNet18 + EfficientNet-B0 (v4 GBML: gradient-balanced multi-task loss, log1p-MSE)
  Hybrid:  Model prediction + k-NN retrieval blend (α=0.60)
  Conformal: AEC-S nutrient intervals (nutrient_conformal_final.py, 2026-08-20)
"""
import os, sys, argparse, warnings
import numpy as np
import torch, torch.nn as nn
from torchvision import models
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
warnings.filterwarnings('ignore')

# ====================== Config ======================
MODEL_DIR = os.path.join(os.path.dirname(__file__), 'models')
DEVICE = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
BATCH = 32
SEED = 42

# ====================== Model Definitions ======================
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

# ====================== Helpers ======================
@torch.no_grad()
def predict(model, X, bs=BATCH):
    model.eval()
    preds = []
    for i in range(0, len(X), bs):
        x = torch.tensor(X[i:i+bs], dtype=torch.float32).to(DEVICE)
        preds.append(model(x).cpu().numpy())
    return np.concatenate(preds, axis=0)

@torch.no_grad()
def extract_features(model, X, bs=BATCH):
    model.eval()
    feats = []
    for i in range(0, len(X), bs):
        x = torch.tensor(X[i:i+bs], dtype=torch.float32).to(DEVICE)
        feats.append(model.features(x).cpu().numpy())
    return np.concatenate(feats, axis=0)

# ====================== Predictor ======================
class NutrientPredictor:
    """Nutrient prediction pipeline: weight × density with k-NN hybrid."""
    def __init__(self):
        # Load weight models
        self.w_rn = WeightResNet().to(DEVICE)
        self.w_rn.load_state_dict(torch.load(
            os.path.join(MODEL_DIR, 'deep_weight_resnet18_final.pth'),
            map_location=DEVICE, weights_only=True))
        self.w_rn.eval()
        self.w_eff = WeightEffNet().to(DEVICE)
        self.w_eff.load_state_dict(torch.load(
            os.path.join(MODEL_DIR, 'deep_weight_efficientnet_final.pth'),
            map_location=DEVICE, weights_only=True))
        self.w_eff.eval()

        # Load density models (v4 GBML: gradient-balanced multi-task loss)
        self.d_rn = DensityResNet().to(DEVICE)
        self.d_rn.load_state_dict(torch.load(
            os.path.join(MODEL_DIR, 'density_resnet18_final.pth'),
            map_location=DEVICE, weights_only=True))
        self.d_rn.eval()
        self.d_eff = DensityEffNet().to(DEVICE)
        self.d_eff.load_state_dict(torch.load(
            os.path.join(MODEL_DIR, 'density_efficientnet_final.pth'),
            map_location=DEVICE, weights_only=True))
        self.d_eff.eval()

        # Load data for k-NN retrieval
        data = np.load(os.path.join(MODEL_DIR, 'deep_crops_cache.npz'))
        self.X_train = data['X_train']
        self.y_train_w = data['y_train']
        density = np.load(os.path.join(MODEL_DIR, 'nutrient_density.npz'))
        self.y_train_d = density['y_train_density']

        # Pre-compute training features
        self._build_knn_index()

    def _build_knn_index(self):
        """Build k-NN index from training features."""
        feat_rn = extract_features(self.w_rn, self.X_train)
        feat_eff = extract_features(self.w_eff, self.X_train)
        self.feat_tr = np.concatenate([feat_rn, feat_eff], axis=1)
        self.scaler = StandardScaler()
        self.feat_tr_norm = self.scaler.fit_transform(self.feat_tr)
        self.nn = NearestNeighbors(n_neighbors=5, metric='cosine')
        self.nn.fit(self.feat_tr_norm)

    def predict(self, X):
        """Predict weight, density, and nutrients for input images.
        Args:
            X: numpy array of shape (N, 4, 224, 224) — RGB-D image crops
        Returns:
            dict with keys: weight, density_direct, density_knn, density_hybrid,
                            nutrients_direct, nutrients_knn, nutrients_hybrid
        """
        # Weight prediction
        w_pred = (predict(self.w_rn, X) + predict(self.w_eff, X)) / 2
        w_pred = np.maximum(w_pred, 0)

        # Direct density regression
        d_rn_log = predict(self.d_rn, X)
        d_eff_log = predict(self.d_eff, X)
        d_direct = np.expm1((d_rn_log + d_eff_log) / 2)
        d_direct = np.maximum(d_direct, 0)

        # k-NN density retrieval
        feat_rn = extract_features(self.w_rn, X)
        feat_eff = extract_features(self.w_eff, X)
        feat_test = np.concatenate([feat_rn, feat_eff], axis=1)
        feat_test_norm = self.scaler.transform(feat_test)
        _, indices = self.nn.kneighbors(feat_test_norm)
        d_knn = np.array([self.y_train_d[indices[i]].mean(axis=0) for i in range(len(X))])

        # Hybrid blend (α=0.60: per-nutrient grid search confirmed optimal for all 4 nutrients)
        d_hybrid = 0.60 * d_direct + 0.40 * d_knn

        return {
            'weight': w_pred,
            'density_direct': d_direct,
            'density_knn': d_knn,
            'density_hybrid': d_hybrid,
            'nutrients_direct': w_pred.reshape(-1, 1) * d_direct,
            'nutrients_knn': w_pred.reshape(-1, 1) * d_knn,
            'nutrients_hybrid': w_pred.reshape(-1, 1) * d_hybrid,
        }

# ====================== Main ======================
def main():
    parser = argparse.ArgumentParser(description='Nutrition5K Nutrient Prediction')
    parser.add_argument('--all', action='store_true', help='Test all test set samples')
    parser.add_argument('--n', type=int, default=100, help='Number of random samples (default: 100)')
    parser.add_argument('--image', type=str, help='Single image path (not yet supported)')
    parser.add_argument('--no-conformal', action='store_true',
                        help='跳过 AEC-S 营养素共形区间验证')
    args = parser.parse_args()

    print("=" * 80)
    print("Nutrition5K Nutrient Prediction — MM-CFA")
    print("=" * 80)

    # Load test data
    data = np.load(os.path.join(MODEL_DIR, 'deep_crops_cache.npz'))
    X_test = data['X_test']; y_test_w = data['y_test']
    nutr = np.load(os.path.join(MODEL_DIR, 'nutrient_labels.npz'))
    y_test_n = nutr['y_test_nutr']
    density = np.load(os.path.join(MODEL_DIR, 'nutrient_density.npz'))
    y_test_d = density['y_test_density']

    # Select samples
    if args.all:
        X = X_test; y_w = y_test_w; y_n = y_test_n; y_d = y_test_d
        print(f"Testing all {len(X)} test samples")
    else:
        np.random.seed(SEED)
        idx = np.random.choice(len(X_test), min(args.n, len(X_test)), replace=False)
        X = X_test[idx]; y_w = y_test_w[idx]; y_n = y_test_n[idx]; y_d = y_test_d[idx]
        print(f"Testing {len(idx)} random samples (seed={SEED})")

    # Load predictor
    print("\nLoading models...")
    predictor = NutrientPredictor()

    # Predict
    print("Predicting...")
    result = predictor.predict(X)

    # Weight→Nutrient baseline
    n_baseline = np.zeros_like(y_n)
    for i in range(4):
        lr = LinearRegression()
        lr.fit(predictor.y_train_w.reshape(-1, 1),
               predictor.y_train_d[:, i] * predictor.y_train_w)
        n_baseline[:, i] = np.maximum(
            lr.predict(result['weight'].reshape(-1, 1)).ravel(), 0)

    # Summary
    NUTRIENT_NAMES = ['Calories', 'Fat', 'Carbs', 'Protein']
    print("\n" + "=" * 80)
    print("RESULTS")
    print("=" * 80)
    w_mae = mean_absolute_error(y_w, result['weight'])
    print(f"Weight: MAE={w_mae:.1f}g, PMAE={w_mae/y_w.mean()*100:.1f}%, "
          f"R²={r2_score(y_w, result['weight']):.3f}")

    for i, name in enumerate(NUTRIENT_NAMES):
        y_true = y_n[:, i]
        print(f"\n{name} (true mean={y_true.mean():.1f}):")
        for approach, d_pred in [
            ('Direct Regression', result['density_direct']),
            ('k-NN (k=5)', result['density_knn']),
            ('Hybrid (α=0.60, v4 GBML)', result['density_hybrid']),
        ]:
            n_pred = result['weight'].reshape(-1, 1) * d_pred
            mae = mean_absolute_error(y_true, n_pred[:, i])
            pmae = mae / y_true.mean() * 100
            print(f"  {approach:<25s}: MAE={mae:7.1f}, PMAE={pmae:6.1f}%")

        mae_bl = mean_absolute_error(y_true, n_baseline[:, i])
        pmae_bl = mae_bl / y_true.mean() * 100
        print(f"  {'Weight→Nutrient':<25s}: MAE={mae_bl:7.1f}, PMAE={pmae_bl:6.1f}%")

    # ====================== AEC-S 营养素共形区间 ======================
    if not args.no_conformal:
        print("\n" + "=" * 80)
        print("CONFORMAL INTERVALS — AEC-S Nutrient (weight-stratified ensemble)")
        print("=" * 80)
        try:
            from nutrient_conformal_final import conformal_report
            lines, _ = conformal_report(seed=SEED)
            for ln in lines:
                print(ln)
        except Exception as e:
            print(f"  共形区间验证跳过: {e}")

    print("\nDone!")

if __name__ == '__main__':
    main()