#!/usr/bin/env python3
"""Nutrition5K R0: retrain E1 and evaluate E2–E5 on the strict dish hold-out.

Protocol: the base models train only on the train dishes. The test dishes are
split into calibration and evaluation halves, repeated 10 times. Quantiles are
finite-sample order statistics.
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
import time
import traceback
from pathlib import Path

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "1")

import numpy as np
import torch
import torch.nn as nn
from sklearn.decomposition import PCA
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from scipy.stats import ttest_rel
from torch.utils.data import DataLoader, TensorDataset


def _boot() -> Path:
    here = Path(__file__).resolve()
    for parent in here.parents:
        if (parent / "src" / "mmcfa" / "paths.py").is_file():
            for entry in (str(parent), str(parent / "src")):
                if entry not in sys.path:
                    sys.path.insert(0, entry)
            return parent
    raise RuntimeError("找不到项目根目录")


ROOT = _boot()
from experiments.estimation.aec_s import (  # noqa: E402
    BINS,
    TAU,
    aec_apply,
    dirichlet_search,
    extract_features,
    global_weights,
)
from experiments.estimation.train_nutrients import (  # noqa: E402
    BATCH as DENSITY_BATCH,
    DensityEffNet,
    DensityResNet,
    train_gbml,
)
from experiments.estimation.train_weight_4ch import (  # noqa: E402
    BATCH,
    EPOCHS,
    SEED,
    AugDS,
    WeightNet,
    predict,
    train_model,
)
from mmcfa.conformal.finite_sample import finite_sample_quantile  # noqa: E402
from mmcfa.metrics import mae, mape, wape  # noqa: E402
from mmcfa.paths import Paths, n5k  # noqa: E402

STATUS = ROOT / "logs" / "n5k_r0_status.txt"
OUT = ROOT / "results" / "estimation" / "r0"
ALPHA = 0.10
N_TRIALS = 10
EPS = 1e-3
DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
AEC_METHODS = ["Split", "Norm", "Mondrian"]
S2_METHODS = ["Split", "Norm", "LogSplit", "Disagree", "Mondrian", "MLNS"]
BIN_LABELS = ["0-50g", "50-100g", "100-200g", "200-500g", "500g+"]
NUTR_NAMES = ["calories", "fat_g", "carb_g", "protein_g"]


def status(msg: str) -> None:
    line = time.strftime("%F %T ") + msg
    print(line, flush=True)
    STATUS.parent.mkdir(parents=True, exist_ok=True)
    with STATUS.open("a", encoding="utf-8") as fh:
        fh.write(line + "\n")


def dump_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def point_block(y, p) -> dict:
    y = np.asarray(y, dtype=float).reshape(-1)
    p = np.asarray(p, dtype=float).reshape(-1)
    return {
        "n": int(y.size),
        "mape_pct": float(100.0 * mape(y, p)),
        "wape": float(wape(y, p)),
        "mae_g": float(mae(y, p)),
    }


def with_heavy(y, p) -> dict:
    block = point_block(y, p)
    mask = np.asarray(y, dtype=float).reshape(-1) >= 100
    block["ge_100g"] = point_block(np.asarray(y).reshape(-1)[mask], np.asarray(p).reshape(-1)[mask])
    return block


class WeightCQR(nn.Module):
    def __init__(self, pretrained: bool = True):
        super().__init__()
        weights = "IMAGENET1K_V1" if pretrained else None
        from torchvision import models

        self.backbone = models.resnet18(weights=weights)
        old = self.backbone.conv1
        new = nn.Conv2d(4, 64, kernel_size=7, stride=2, padding=3, bias=False)
        if pretrained:
            new.weight.data[:, :3] = old.weight.data
            new.weight.data[:, 3] = old.weight.data.mean(dim=1)
        self.backbone.conv1 = new
        nf = self.backbone.fc.in_features
        self.backbone.fc = nn.Sequential(
            nn.Dropout(0.3), nn.Linear(nf, 256), nn.ReLU(),
            nn.Dropout(0.15), nn.Linear(256, 64), nn.ReLU(),
            nn.Linear(64, 2),
        )

    def forward(self, x):
        return torch.sort(self.backbone(x), dim=-1).values


def pinball_loss(pred, target):
    taus = pred.new_tensor([0.05, 0.95]).view(1, -1)
    err = target.unsqueeze(1) - pred
    return torch.maximum(taus * err, (taus - 1.0) * err).mean()


def train_cqr(model, train_loader, val_loader):
    model = model.to(DEVICE)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-4)
    sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=EPOCHS, eta_min=1e-6)
    best = {"loss": float("inf"), "state": None}
    for ep in range(EPOCHS):
        model.train()
        for x, y in train_loader:
            x, y = x.to(DEVICE), y.to(DEVICE)
            opt.zero_grad()
            loss = pinball_loss(model(x), torch.log(torch.clamp(y, min=0.1)))
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
        sch.step()
        if (ep + 1) % 5 == 0:
            model.eval()
            total, count = 0.0, 0
            with torch.no_grad():
                for x, y in val_loader:
                    x, y = x.to(DEVICE), y.to(DEVICE)
                    total += pinball_loss(model(x), torch.log(torch.clamp(y, min=0.1))).item() * len(x)
                    count += len(x)
            vloss = total / max(count, 1)
            print(f"  CQR ep {ep + 1:3d} val_pinball={vloss:.4f}", flush=True)
            if vloss < best["loss"]:
                best = {"loss": vloss, "state": {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}}
    model.load_state_dict(best["state"])
    return model


@torch.no_grad()
def predict_quantiles(model, x, bs=64):
    model.eval()
    out = []
    for i in range(0, len(x), bs):
        batch = torch.tensor(x[i:i + bs], dtype=torch.float32, device=DEVICE)
        out.append(torch.exp(model(batch)).cpu().numpy())
    return np.concatenate(out, axis=0)


def ensure_point_model(path: Path, backbone: str, in_channels: int, x, y) -> None:
    if path.is_file():
        status(f"skip existing {path.name}")
        return
    status(f"train {path.name} in_ch={in_channels}")
    xtr, xv, ytr, yv = train_test_split(x, y, test_size=0.15, random_state=SEED)
    train_loader = DataLoader(AugDS(xtr, ytr, aug=True), BATCH, shuffle=True, num_workers=0, pin_memory=True)
    val_loader = DataLoader(AugDS(xv, yv, aug=False), BATCH * 2, shuffle=False, num_workers=0, pin_memory=True)
    model = train_model(
        path.name,
        WeightNet(backbone, 0.3, in_channels=in_channels, pretrained=True),
        train_loader,
        val_loader,
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), path)
    del model
    torch.cuda.empty_cache()


def load_point_model(path: Path, backbone: str, in_channels: int) -> WeightNet:
    model = WeightNet(backbone, 0.3, in_channels=in_channels, pretrained=False).to(DEVICE)
    model.load_state_dict(torch.load(path, map_location=DEVICE, weights_only=True))
    model.eval()
    return model


def fit_calibrator(y_val, p_val):
    reg = LinearRegression()
    reg.fit(np.log1p(np.maximum(p_val, 0.1)).reshape(-1, 1), np.log1p(y_val))
    return reg


def apply_calibrator(reg, pred):
    return np.clip(np.expm1(reg.predict(np.log1p(np.maximum(pred, 0.1)).reshape(-1, 1))), 0, None)


def mondrian_intervals(y_cal, p_cal, p_te):
    bins = [(0, 75), (75, 150), (150, 500), (500, 10000)]

    def bin_idx(values):
        out = []
        for value in values:
            chosen = len(bins) - 1
            for i, (lo, hi) in enumerate(bins):
                if lo <= value < hi:
                    chosen = i
                    break
            out.append(chosen)
        return np.array(out)

    cal_bin = bin_idx(p_cal)
    te_bin = bin_idx(p_te)
    global_q = finite_sample_quantile(np.abs(y_cal - p_cal) / (p_cal + EPS), ALPHA)
    cl, cu = np.zeros_like(p_cal), np.zeros_like(p_cal)
    tl, tu = np.zeros_like(p_te), np.zeros_like(p_te)
    for i, _bounds in enumerate(bins):
        cm, tm = cal_bin == i, te_bin == i
        if cm.sum() < 30:
            q = global_q * 1.15
        else:
            q = finite_sample_quantile(
                np.abs(y_cal[cm] - p_cal[cm]) / (p_cal[cm] + EPS), ALPHA
            ) * (1 + 0.03 / np.sqrt(cm.sum()))
        cl[cm] = np.maximum(0, p_cal[cm] * (1 - q))
        cu[cm] = p_cal[cm] * (1 + q)
        tl[tm] = np.maximum(0, p_te[tm] * (1 - q))
        tu[tm] = p_te[tm] * (1 + q)
    return cl, cu, tl, tu


def split_intervals(y_cal, p_cal, p_te):
    q = finite_sample_quantile(np.abs(y_cal - p_cal), ALPHA)
    return np.maximum(0, p_cal - q), p_cal + q, np.maximum(0, p_te - q), p_te + q


def norm_intervals(y_cal, p_cal, p_te):
    q = finite_sample_quantile(np.abs(y_cal - p_cal) / (p_cal + EPS), ALPHA)
    return (
        np.maximum(0, p_cal * (1 - q)), p_cal * (1 + q),
        np.maximum(0, p_te * (1 - q)), p_te * (1 + q),
    )


def logsplit_intervals(y_cal, p_cal, p_te):
    score = np.abs(np.log(np.maximum(y_cal, 0.1)) - np.log(np.maximum(p_cal, 0.1)))
    q = finite_sample_quantile(score, ALPHA)
    cal = np.maximum(p_cal, 0.1)
    te = np.maximum(p_te, 0.1)
    return cal * np.exp(-q), cal * np.exp(q), te * np.exp(-q), te * np.exp(q)


def disagree_intervals(y_cal, p_cal, d_cal, p_te, d_te):
    q = finite_sample_quantile(np.abs(y_cal - p_cal) / (d_cal + EPS), ALPHA)
    rc, rt = q * (d_cal + EPS), q * (d_te + EPS)
    return np.maximum(0, p_cal - rc), p_cal + rc, np.maximum(0, p_te - rt), p_te + rt


def mlns_model(y_train, p_train, d_train, feat_train):
    pca = PCA(n_components=min(20, feat_train.shape[1])).fit(feat_train)
    extra = np.column_stack([d_train, p_train, np.zeros((len(p_train), 1))])
    raw = np.column_stack([pca.transform(feat_train), extra])
    scaler = StandardScaler().fit(raw)
    model = GradientBoostingRegressor(
        n_estimators=300, max_depth=5, learning_rate=0.05, subsample=0.8, random_state=SEED
    )
    model.fit(scaler.transform(raw), np.abs(y_train - p_train) / (p_train + EPS))
    return pca, scaler, model


def mlns_intervals(bundle, y_cal, p_cal, d_cal, feat_cal, p_te, d_te, feat_te):
    pca, scaler, model = bundle

    def design(pred, spread, feat):
        extra = np.column_stack([spread, pred, np.zeros((len(pred), 1))])
        return scaler.transform(np.column_stack([pca.transform(feat), extra]))

    cal_raw = np.maximum(model.predict(design(p_cal, d_cal, feat_cal)), 1e-6)
    te_raw = np.maximum(model.predict(design(p_te, d_te, feat_te)), 1e-6)
    q = finite_sample_quantile(cal_raw * 1.02, ALPHA)
    cal_scale = q * cal_raw / cal_raw.mean()
    te_scale = q * te_raw / te_raw.mean()
    return (
        np.maximum(0, p_cal * (1 - cal_scale)), p_cal * (1 + cal_scale),
        np.maximum(0, p_te * (1 - te_scale)), p_te * (1 + te_scale),
    )


def cqr_intervals(y_cal, lo_cal, hi_cal, lo_te, hi_te):
    score = np.maximum(lo_cal - y_cal, y_cal - hi_cal)
    q = finite_sample_quantile(score, ALPHA)
    return np.maximum(0, lo_te - q), hi_te + q


def eval_interval(y, lo, hi) -> dict:
    covered = (y >= lo) & (y <= hi)
    width = hi - lo
    winkler = width + (2 / ALPHA) * np.maximum(0, lo - y) + (2 / ALPHA) * np.maximum(0, y - hi)
    return {
        "coverage": float(np.mean(covered)),
        "width": float(np.mean(width)),
        "winkler": float(np.mean(winkler)),
    }


def stack_sides(cands, names, cal_mask_key="cal"):
    return (
        np.column_stack([cands[name][0] for name in names]),
        np.column_stack([cands[name][1] for name in names]),
        np.column_stack([cands[name][2] for name in names]),
        np.column_stack([cands[name][3] for name in names]),
    )


def stratified_weights(lc, uc, y_cal, p_cal, objective, target, n_iter, rng, shrink: bool):
    w_global = global_weights(lc, uc, y_cal, objective, target, n_iter, rng)
    weights = []
    for bi in range(len(BINS) - 1):
        lo, hi = BINS[bi], BINS[bi + 1]
        chosen = (p_cal >= lo) & (p_cal < hi)
        n = int(chosen.sum())
        if n < 20:
            weights.append(w_global.copy())
            continue
        local, _ = dirichlet_search(lc[chosen], uc[chosen], y_cal[chosen], objective, target, n_iter, rng)
        if shrink:
            mix = n / (n + TAU)
            weights.append(mix * local + (1 - mix) * w_global)
        else:
            weights.append(local)
    return w_global, weights


def recalibrate(cl, cu, y_cal, p_cal, tl, tu, p_te):
    center_cal = (cl + cu) / 2
    half_cal = np.maximum((cu - cl) / 2, 1e-6)
    scores = np.abs(y_cal - center_cal) / half_cal
    q_all = finite_sample_quantile(scores, ALPHA)
    center = (tl + tu) / 2
    half = np.maximum((tu - tl) / 2, 1e-6)
    lo, hi = np.zeros_like(p_te), np.zeros_like(p_te)
    for bi in range(len(BINS) - 1):
        left, right = BINS[bi], BINS[bi + 1]
        cm = (p_cal >= left) & (p_cal < right)
        tm = (p_te >= left) & (p_te < right)
        if not np.any(tm):
            continue
        q = finite_sample_quantile(scores[cm], ALPHA) if cm.sum() >= 30 else q_all
        lo[tm] = np.maximum(0, center[tm] - q * half[tm])
        hi[tm] = center[tm] + q * half[tm]
    return lo, hi


def per_bin_coverage(y, lo, hi) -> dict:
    edges = [0, 50, 100, 200, 500, 10000]
    out = {}
    for label, left, right in zip(BIN_LABELS, edges[:-1], edges[1:]):
        mask = (y >= left) & (y < right)
        if not np.any(mask):
            continue
        out[label] = {
            "n": int(mask.sum()),
            "coverage": float(np.mean((y[mask] >= lo[mask]) & (y[mask] <= hi[mask]))),
            "width": float(np.mean(hi[mask] - lo[mask])),
        }
    return out


def summarize(rows: list[dict], method: str) -> dict:
    cov = np.array([row["methods"][method]["coverage"] for row in rows])
    width = np.array([row["methods"][method]["width"] for row in rows])
    wink = np.array([row["methods"][method]["winkler"] for row in rows])
    return {
        "coverage_mean": float(cov.mean()),
        "coverage_std": float(cov.std(ddof=1)),
        "width_mean": float(width.mean()),
        "width_std": float(width.std(ddof=1)),
        "winkler_mean": float(wink.mean()),
        "winkler_std": float(wink.std(ddof=1)),
    }


def run_e1_e4(paths: Paths) -> None:
    data = np.load(paths.n5k_cache / "deep_crops_cache.npz")
    x_train, y_train = data["X_train"], data["y_train"].astype(np.float32)
    x_test, y_test = data["X_test"], data["y_test"].astype(np.float32)
    status(f"cache train={len(y_train)} test={len(y_test)} device={DEVICE}")
    wdir = paths.weights_n5k
    w3 = paths.weights_nvreal
    wdir.mkdir(parents=True, exist_ok=True)
    w3.mkdir(parents=True, exist_ok=True)
    specs = [
        (wdir / "deep_weight_resnet18_final.pth", "resnet18", 4, x_train, y_train),
        (wdir / "deep_weight_efficientnet_final.pth", "efficientnet_b0", 4, x_train, y_train),
        (w3 / "deep_weight_resnet18_3ch.pth", "resnet18", 3, x_train[:, :3], y_train),
        (w3 / "deep_weight_efficientnet_3ch.pth", "efficientnet_b0", 3, x_train[:, :3], y_train),
    ]
    for path, backbone, channels, x, y in specs:
        ensure_point_model(path, backbone, channels, x, y)

    cqr_path = wdir / "weight_cqr_resnet18.pth"
    if not cqr_path.is_file():
        status("train weight CQR ResNet18")
        xtr, xv, ytr, yv = train_test_split(x_train, y_train, test_size=0.15, random_state=SEED)
        train_loader = DataLoader(AugDS(xtr, ytr, aug=True), BATCH, shuffle=True, num_workers=0, pin_memory=True)
        val_loader = DataLoader(AugDS(xv, yv, aug=False), BATCH * 2, shuffle=False, num_workers=0, pin_memory=True)
        cqr = train_cqr(WeightCQR(True), train_loader, val_loader)
        torch.save(cqr.state_dict(), cqr_path)
        del cqr
        torch.cuda.empty_cache()
    else:
        status("skip existing weight_cqr_resnet18.pth")

    rn = load_point_model(wdir / "deep_weight_resnet18_final.pth", "resnet18", 4)
    eff = load_point_model(wdir / "deep_weight_efficientnet_final.pth", "efficientnet_b0", 4)
    p_rn_tr, p_eff_tr = predict(rn, x_train), predict(eff, x_train)
    p_rn_te, p_eff_te = predict(rn, x_test), predict(eff, x_test)
    p_tr = (p_rn_tr + p_eff_tr) / 2
    p_te = (p_rn_te + p_eff_te) / 2
    idx = np.arange(len(y_train))
    _tr_idx, va_idx = train_test_split(idx, test_size=0.15, random_state=SEED)
    cal_rn = fit_calibrator(y_train[va_idx], p_rn_tr[va_idx])
    cal_eff = fit_calibrator(y_train[va_idx], p_eff_tr[va_idx])
    p_te_cal = (apply_calibrator(cal_rn, p_rn_te) + apply_calibrator(cal_eff, p_eff_te)) / 2

    rn3 = load_point_model(w3 / "deep_weight_resnet18_3ch.pth", "resnet18", 3)
    eff3 = load_point_model(w3 / "deep_weight_efficientnet_3ch.pth", "efficientnet_b0", 3)
    p3 = (predict(rn3, x_test[:, :3]) + predict(eff3, x_test[:, :3])) / 2
    del rn3, eff3
    torch.cuda.empty_cache()

    e1 = {
        "protocol": "dish-disjoint hold-out stored in deep_crops_cache.npz; val is 15% of train, seed 42",
        "input": "full overhead frame resized to 224; the lost YOLO crop cache was not in the official archive",
        "n_train": int(len(y_train)),
        "n_test": int(len(y_test)),
        "4ch_resnet": with_heavy(y_test, p_rn_te),
        "4ch_efficientnet": with_heavy(y_test, p_eff_te),
        "4ch_ensemble": with_heavy(y_test, p_te),
        "4ch_ensemble_calibrated": with_heavy(y_test, p_te_cal),
        "3ch_ensemble": with_heavy(y_test, p3),
    }
    dump_json(OUT / "e1_point.json", e1)
    status(
        "E1 4ch ensemble "
        f"MAPE={e1['4ch_ensemble']['mape_pct']:.2f}% "
        f"WAPE={e1['4ch_ensemble']['wape']:.3f} "
        f"MAE={e1['4ch_ensemble']['mae_g']:.1f}g | 3ch MAPE={e1['3ch_ensemble']['mape_pct']:.2f}%"
    )

    status("extract train/test features for MLNS")
    feat_tr = extract_features(rn, x_train)
    feat_te = extract_features(rn, x_test)
    d_tr = np.abs(p_rn_tr - p_eff_tr)
    d_te = np.abs(p_rn_te - p_eff_te)
    difficulty = mlns_model(y_train, p_tr, d_tr, feat_tr)
    cqr = WeightCQR(False).to(DEVICE)
    cqr.load_state_dict(torch.load(cqr_path, map_location=DEVICE, weights_only=True))
    q_te = predict_quantiles(cqr, x_test)
    del rn, eff, cqr
    torch.cuda.empty_cache()

    np.savez(
        paths.n5k_cache / "aec_v2_pred_cache.npz",
        y_all=np.concatenate([y_train, y_test]).astype(np.float32),
        final_rn=np.concatenate([p_rn_tr, p_rn_te]).astype(np.float32),
        final_eff=np.concatenate([p_eff_tr, p_eff_te]).astype(np.float32),
        final_feat=np.concatenate([feat_tr, feat_te]).astype(np.float32),
    )

    BINS[-1] = np.inf
    trials = []
    ablation = []
    weight_rows = []
    bin_rows = []
    for trial in range(N_TRIALS):
        seed = SEED + trial
        cal_idx, ev_idx = train_test_split(np.arange(len(y_test)), test_size=0.5, random_state=seed)
        y_cal, y_ev = y_test[cal_idx], y_test[ev_idx]
        p_cal, p_ev = p_te[cal_idx], p_te[ev_idx]
        d_cal, d_ev = d_te[cal_idx], d_te[ev_idx]
        f_cal, f_ev = feat_te[cal_idx], feat_te[ev_idx]
        cands = {
            "Split": split_intervals(y_cal, p_cal, p_ev),
            "Norm": norm_intervals(y_cal, p_cal, p_ev),
            "Mondrian": mondrian_intervals(y_cal, p_cal, p_ev),
            "LogSplit": logsplit_intervals(y_cal, p_cal, p_ev),
            "Disagree": disagree_intervals(y_cal, p_cal, d_cal, p_ev, d_ev),
            "MLNS": mlns_intervals(difficulty, y_cal, p_cal, d_cal, f_cal, p_ev, d_ev, f_ev),
        }
        methods = {
            "Split": eval_interval(y_ev, *cands["Split"][2:]),
            "Normalized": eval_interval(y_ev, *cands["Norm"][2:]),
            "Mondrian": eval_interval(y_ev, *cands["Mondrian"][2:]),
            "CQR": eval_interval(y_ev, *cqr_intervals(
                y_cal, q_te[cal_idx, 0], q_te[cal_idx, 1], q_te[ev_idx, 0], q_te[ev_idx, 1]
            )),
        }
        lc, uc, lt, ut = stack_sides(cands, AEC_METHODS)
        w_global, w_nosh = stratified_weights(
            lc, uc, y_cal, p_cal, "width", 1 - ALPHA, 5000, np.random.default_rng(seed), False
        )
        _, w_sh = stratified_weights(
            lc, uc, y_cal, p_cal, "width", 1 - ALPHA, 5000, np.random.default_rng(seed), True
        )
        glo_lo, glo_hi = lt @ w_global, ut @ w_global
        _cl, _cu, nosh_lo, nosh_hi = aec_apply(lc, uc, p_cal, lt, ut, p_ev, w_nosh)
        cl, cu, sh_lo, sh_hi = aec_apply(lc, uc, p_cal, lt, ut, p_ev, w_sh)
        rec_lo, rec_hi = recalibrate(cl, cu, y_cal, p_cal, sh_lo, sh_hi, p_ev)
        methods["AEC"] = eval_interval(y_ev, glo_lo, glo_hi)
        methods["AEC-S"] = eval_interval(y_ev, sh_lo, sh_hi)

        lc6, uc6, lt6, ut6 = stack_sides(cands, S2_METHODS)
        _w2, w2_bins = stratified_weights(
            lc6, uc6, y_cal, p_cal, "winkler", None, 5000, np.random.default_rng(seed + 1000), True
        )
        c6_lo, c6_hi, t6_lo, t6_hi = aec_apply(lc6, uc6, p_cal, lt6, ut6, p_ev, w2_bins)
        s2_lo, s2_hi = recalibrate(c6_lo, c6_hi, y_cal, p_cal, t6_lo, t6_hi, p_ev)
        methods["AEC-S2m"] = eval_interval(y_ev, s2_lo, s2_hi)
        ab = {
            "global": methods["AEC"],
            "stratified": eval_interval(y_ev, nosh_lo, nosh_hi),
            "shrinkage": methods["AEC-S"],
            "recalibrated": eval_interval(y_ev, rec_lo, rec_hi),
            "s2m": methods["AEC-S2m"],
        }
        trials.append({"seed": seed, "n_cal": int(len(y_cal)), "n_eval": int(len(y_ev)), "methods": methods})
        ablation.append(ab)
        weight_rows.append(np.stack(w_sh, axis=0))
        bin_rows.append(per_bin_coverage(y_ev, sh_lo, sh_hi))
        status(
            f"E2 trial {trial + 1}/{N_TRIALS} "
            f"AEC-S cov={methods['AEC-S']['coverage']:.3f} "
            f"width={methods['AEC-S']['width']:.1f} "
            f"CQR cov={methods['CQR']['coverage']:.3f}"
        )

    method_names = ["Split", "Normalized", "Mondrian", "CQR", "AEC", "AEC-S", "AEC-S2m"]
    summary = {name: summarize(trials, name) for name in method_names}
    base_names = ["Split", "Normalized", "Mondrian", "CQR", "AEC"]
    best_base = min(base_names, key=lambda name: summary[name]["winkler_mean"])
    comparisons = {}
    aecs = np.array([row["methods"]["AEC-S"]["winkler"] for row in trials])
    for name in method_names:
        if name == "AEC-S":
            continue
        other = np.array([row["methods"][name]["winkler"] for row in trials])
        test = ttest_rel(aecs, other)
        comparisons[name] = {"t": float(test.statistic), "pvalue": float(test.pvalue)}
    dump_json(OUT / "e2_intervals.json", {
        "protocol": "test dishes split 50/50 cal/eval, 10 seeds, order-statistic quantile",
        "alpha": ALPHA,
        "summary": summary,
        "best_baseline_by_winkler": best_base,
        "aec_s_winkler_paired_ttest": comparisons,
        "trials": trials,
    })
    ab_summary = {}
    for key in ("global", "stratified", "shrinkage", "recalibrated", "s2m"):
        cov = np.array([row[key]["coverage"] for row in ablation])
        width = np.array([row[key]["width"] for row in ablation])
        wink = np.array([row[key]["winkler"] for row in ablation])
        ab_summary[key] = {
            "coverage_mean": float(cov.mean()),
            "coverage_std": float(cov.std(ddof=1)),
            "width_mean": float(width.mean()),
            "width_std": float(width.std(ddof=1)),
            "winkler_mean": float(wink.mean()),
            "winkler_std": float(wink.std(ddof=1)),
        }
    dump_json(OUT / "e3_ablation.json", {
        "steps": ["global", "stratified", "shrinkage", "recalibrated"],
        "s2m": "six candidates, winkler search, then per-bin recalibration",
        "summary": ab_summary,
    })
    bin_summary = {}
    for label in BIN_LABELS:
        covs, widths, counts = [], [], []
        for row in bin_rows:
            if label not in row:
                continue
            covs.append(row[label]["coverage"])
            widths.append(row[label]["width"])
            counts.append(row[label]["n"])
        if covs:
            bin_summary[label] = {
                "n_mean": float(np.mean(counts)),
                "coverage_mean": float(np.mean(covs)),
                "coverage_std": float(np.std(covs, ddof=1)) if len(covs) > 1 else 0.0,
                "width_mean": float(np.mean(widths)),
            }
    mean_w = np.mean(np.stack(weight_rows, axis=0), axis=0)
    dump_json(paths.root / "backup" / "v3_excluded" / "results" / "estimation" / "r0" / "e4_per_bin.json", {
        "bins": bin_summary,
        "aec_s_mean_weights": {
            BIN_LABELS[i]: {AEC_METHODS[j]: float(mean_w[i, j]) for j in range(len(AEC_METHODS))}
            for i in range(mean_w.shape[0])
        },
    })
    status(f"E2–E4 done; best baseline {best_base}; AEC-S winkler={summary['AEC-S']['winkler_mean']:.1f}")


def run_e5(paths: Paths) -> None:
    data = np.load(paths.n5k_cache / "deep_crops_cache.npz")
    x_train, x_test = data["X_train"], data["X_test"]
    dens = np.load(paths.n5k_cache / "nutrient_density.npz")
    nutr = np.load(paths.n5k_cache / "nutrient_labels.npz")
    y_train = np.nan_to_num(dens["y_train_density"].astype(np.float32), nan=0.0)
    y_test_n = nutr["y_test_nutr"].astype(np.float32)
    y_log = np.log1p(np.clip(y_train, 0, None))
    loader = DataLoader(
        TensorDataset(torch.tensor(x_train, dtype=torch.float32), torch.tensor(y_log, dtype=torch.float32)),
        batch_size=DENSITY_BATCH, shuffle=True, num_workers=0, pin_memory=True,
    )
    rn_path = paths.weights_n5k / "density_resnet18_final.pth"
    eff_path = paths.weights_n5k / "density_efficientnet_final.pth"
    if not rn_path.is_file():
        status("train density ResNet18")
        model = train_gbml(DensityResNet(), loader, "RN18")
        torch.save(model.state_dict(), rn_path)
        del model
        torch.cuda.empty_cache()
    if not eff_path.is_file():
        status("train density EfficientNet")
        model = train_gbml(DensityEffNet(), loader, "Eff")
        torch.save(model.state_dict(), eff_path)
        del model
        torch.cuda.empty_cache()

    from experiments.estimation.nutrient_intervals import DensityEffNet as EvalEff
    from experiments.estimation.nutrient_intervals import DensityResNet as EvalRn
    from experiments.estimation.nutrient_intervals import _predict

    rn = EvalRn().to(DEVICE)
    rn.load_state_dict(torch.load(rn_path, map_location=DEVICE, weights_only=True))
    eff = EvalEff().to(DEVICE)
    eff.load_state_dict(torch.load(eff_path, map_location=DEVICE, weights_only=True))
    density = np.maximum(np.expm1((_predict(rn, x_test) + _predict(eff, x_test)) / 2), 0)
    aec = np.load(paths.n5k_cache / "aec_v2_pred_cache.npz")
    n_train = len(x_train)
    weight = np.maximum((aec["final_rn"][n_train:] + aec["final_eff"][n_train:]) / 2, 0)
    pred = weight.reshape(-1, 1) * density
    nutrient_point = {}
    for i, name in enumerate(NUTR_NAMES):
        mask = y_test_n[:, i] > 0
        nutrient_point[name] = point_block(y_test_n[mask, i], pred[mask, i]) if np.any(mask) else {"n": 0}
    dump_json(OUT / "e5_point.json", {"nutrients": nutrient_point})
    status("E5 point estimates written")
    del rn, eff
    torch.cuda.empty_cache()

    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = os.environ.get("CUDA_VISIBLE_DEVICES", "1")
    env["PYTHONUNBUFFERED"] = "1"
    for cache_name in ("nutrient_conformal_cache.npz", "cqr_pred_cache.npz"):
        stale = paths.n5k_cache / cache_name
        if stale.exists():
            stale.unlink()
    commands = [
        [sys.executable, str(ROOT / "experiments/estimation/nutrient_intervals.py")],
        [sys.executable, str(ROOT / "experiments/estimation/train_density_cqr.py")],
        [sys.executable, str(ROOT / "experiments/estimation/cqr_baseline.py")],
    ]
    for cmd in commands:
        status("run " + Path(cmd[1]).name)
        proc = subprocess.run(cmd, cwd=ROOT, env=env, check=False)
        if proc.returncode != 0:
            raise RuntimeError(f"{Path(cmd[1]).name} exited {proc.returncode}")
    status("E5 interval scripts finished")


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    paths = Paths.resolve()
    if DEVICE.type != "cuda":
        raise SystemExit("R0 需要 GPU，当前 torch 看不到 CUDA")
    status(f"R0 start device={DEVICE} torch={torch.__version__}")
    run_e1_e4(paths)
    try:
        run_e5(paths)
    except Exception:
        status("E5 failed; E1–E4 results are kept")
        traceback.print_exc()
        raise
    status("R0 complete")


if __name__ == "__main__":
    main()
