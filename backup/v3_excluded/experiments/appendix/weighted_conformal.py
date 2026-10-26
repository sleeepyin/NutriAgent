#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Weighted Conformal (Tibshirani et al. 2019) vs AEC-S2m-N 营养素区间对比
========================================================================
动机: 审稿人会问"为什么只比 Split/Norm/Mondrian/CQR"。补一个现代协变量偏移
加权共形基线。协议与 nutrient_cqr_experiment.py 完全一致 — 严格 hold-out,
652 测试集对半切 cal(326)/eval(326), 10 trials (seed 42..51, 与 v2/CQR 同批
split, 直接可比), α=0.10。有效性: 全营养素 cov ≥ 89.5%。

Weighted Conformal (WC) 实现要点:
  - 中心: 生产 hybrid 营养素点估计 p = w·(0.6·d_direct + 0.4·d_knn) (与 AEC 同中心)
  - 分数: 相对分数 s = |y − p| / (p + ε) (尺度无关, 与 Norm 候选同族)
  - 权重: 密度比 w_i = p̂(test|x_i)/p̂(cal|x_i), 用逻辑回归在特征
    [w_pred, n_pred(4)] 上区分 cal/eval 估计 (Tibshirani 标准做法)。
    域内 cal/eval 同分布 ⇒ 分类器无判别力 ⇒ w_i ≈ 1 ⇒ WC 退化为 Split。
    这正是预期结论: 无可利用的协变量偏移时, 加权共形不带来增益。
  - 加权分位: q̂ = 加权 (1−α) 分位 (含 +∞ 点权重 1 的有限样本修正)。
  - 区间: [max(0, p − q̂·(p+ε)), p + q̂·(p+ε)]。

对照: AEC-S2m-5W (部署获胜者, nWinkler=2.42), 同 split 重跑保证公平。

运行 (CPU 沙箱可跑, ~1-2 min): python nutrient_weighted_conformal_experiment.py
产物: results/nutrient_weighted_conformal_results.pkl
"""
import os, sys, time, joblib
from pathlib import Path
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler


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
from experiments.estimation.nutrient_intervals import (
    NutrientConformalCalibrator, CANDS5, eval_iv, NUTR,
    SEED, ALPHA, get_test_pipeline_cache, hybrid_nutrients, nutrient_disagreement)

N_TRIALS = 10
EPS = 1e-3
OUT = str(Paths.resolve().results_appendix / "weighted_conformal_results.pkl")
VARIANTS = ['WC', 'AEC-S2m-5W']


def _pack(y, l, u):
    cov, width, wk = eval_iv(y, l, u)
    return (cov, width, width / max(y.mean(), EPS), wk)


def weighted_quantile(scores, weights, alpha):
    """Tibshirani 2019 加权共形分位: 含 +∞ 点 (权重 1) 的 (1−α) 加权分位。"""
    order = np.argsort(scores)
    s_sorted = scores[order]
    w_sorted = weights[order]
    total = w_sorted.sum() + 1.0                      # +∞ 点权重 1
    cum = np.cumsum(w_sorted) / total
    idx = np.searchsorted(cum, 1.0 - alpha)
    if idx >= len(s_sorted):
        return float(np.inf)
    return float(s_sorted[idx])


def density_ratio_weights(feat_cal, feat_eval):
    """逻辑回归区分 cal(0)/eval(1) ⇒ cal 样本的密度比权重 w = p̂(eval)/p̂(cal)。"""
    X = np.concatenate([feat_cal, feat_eval], axis=0)
    lab = np.concatenate([np.zeros(len(feat_cal)), np.ones(len(feat_eval))])
    sc = StandardScaler().fit(X)
    clf = LogisticRegression(max_iter=2000).fit(sc.transform(X), lab)
    p_eval = clf.predict_proba(sc.transform(feat_cal))[:, 1]
    p_eval = np.clip(p_eval, 1e-3, 1 - 1e-3)
    w = p_eval / (1 - p_eval)                          # 密度比
    return w / max(w.mean(), EPS)                      # 归一化均值 1 (分位对尺度不敏感)


def main():
    print("=" * 70)
    print("Weighted Conformal (Tibshirani 2019) vs AEC-S2m-N | 同协议 10-trial")
    print("=" * 70)

    cache = get_test_pipeline_cache()
    n_hybrid = hybrid_nutrients(cache)
    n_disag = nutrient_disagreement(cache)
    y = cache['y_nutr']; w = cache['w_pred']; y_w = cache['y_w']
    n_pred = np.maximum(n_hybrid, EPS)
    # 密度比特征: 预测重量 + 4 营养素预测
    feat = np.concatenate([w.reshape(-1, 1), n_pred], axis=1)
    print(f"样本 {len(y)} | 特征维度 {feat.shape[1]}")

    metrics = {m: {n: [] for n in NUTR} for m in VARIANTS}
    wstats = []
    t0 = time.time()
    for t in range(N_TRIALS):
        idx = np.arange(len(y))
        cal_idx, ev_idx = train_test_split(idx, test_size=0.5, random_state=SEED + t)
        cc, ec = cal_idx, ev_idx
        yc = y[cc]

        # --- WC: 密度比权重 + 加权分位 (每营养素独立) ---
        wgt = density_ratio_weights(feat[cc], feat[ec])
        wstats.append((float(wgt.mean()), float(wgt.std()),
                       float(wgt.min()), float(wgt.max())))
        for i, nt in enumerate(NUTR):
            s_cal = np.abs(yc[:, i] - n_pred[cc, i]) / (n_pred[cc, i] + EPS)
            q = weighted_quantile(s_cal, wgt, ALPHA)
            p_e = n_pred[ec, i]
            lo = np.maximum(0, p_e - q * (p_e + EPS))
            hi = p_e + q * (p_e + EPS)
            metrics['WC'][nt].append(_pack(y[ec, i], lo, hi))

        # --- 基线 AEC-S2m-5W (同 split) ---
        cal = NutrientConformalCalibrator(cands=CANDS5, strat='weight').fit(
            w[cc], y_w[cc], n_hybrid[cc], y[cc], n_disag=n_disag[cc], seed=SEED + t)
        lo, hi = cal.predict_intervals(w[ec], n_hybrid[ec], n_disag[ec], 'm')
        for i, nt in enumerate(NUTR):
            metrics['AEC-S2m-5W'][nt].append(_pack(y[ec, i], lo[:, i], hi[:, i]))
        print(f"  trial {t+1}/{N_TRIALS} ({time.time()-t0:.0f}s)")

    # ---------- 汇总 ----------
    y_mean = {nt: y[:, i].mean() for i, nt in enumerate(NUTR)}
    ws = np.array(wstats)
    print(f"\n密度比权重统计 (10 trials): mean={ws[:,0].mean():.3f} "
          f"std={ws[:,1].mean():.3f} min={ws[:,2].mean():.3f} max={ws[:,3].mean():.3f}")
    print("(std≈0 ⇒ cal/eval 无可判别偏移 ⇒ WC 退化为 Split, 符合预期)")

    print("\n综合排名 (有效 = 全营养素 cov≥89.5%):")
    print(f"{'方法':<14s} {'min-cov%':>9s} {'mean-relwid%':>13s} {'mean-nWinkler':>14s} 有效")
    rank = []
    for m in VARIANTS:
        covs = [np.array(metrics[m][nt])[:, 0].mean() for nt in NUTR]
        rw = [np.array(metrics[m][nt])[:, 2].mean() for nt in NUTR]
        nw = [np.array(metrics[m][nt])[:, 3].mean() / y_mean[nt] for nt in NUTR]
        valid = min(covs) >= 0.895
        rank.append((m, min(covs) * 100, np.mean(rw) * 100, np.mean(nw), valid))
        print(f"{m:<14s} {min(covs)*100:>8.1f} {np.mean(rw)*100:>12.1f} "
              f"{np.mean(nw):>13.2f} {'✓' if valid else '✗'}")

    base = next(r for r in rank if r[0] == 'AEC-S2m-5W')
    wc = next(r for r in rank if r[0] == 'WC')
    verdict = (f"WC nWinkler={wc[3]:.3f} (cov={wc[1]:.1f}%) vs AEC-S2m-5W "
               f"{base[3]:.3f} → {'WC 获胜 ✓' if (wc[4] and wc[3] < base[3]) else 'AEC 保持获胜 ✗'}")
    print(f"\n结论: {verdict}")
    joblib.dump({'metrics': metrics, 'rank': rank, 'verdict': verdict,
                 'weight_stats': {'mean': float(ws[:,0].mean()), 'std': float(ws[:,1].mean()),
                                  'min': float(ws[:,2].mean()), 'max': float(ws[:,3].mean())},
                 'config': {'alpha': ALPHA, 'n_trials': N_TRIALS, 'date': '2026-08-24'}}, OUT)
    print(f"Saved: {OUT}")


if __name__ == '__main__':
    main()
