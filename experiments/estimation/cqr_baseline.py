#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
CQR vs AEC-S2m-N 营养素区间对比 (2026-08-20)
==============================================
前置: train_density_cqr.py 已在 GPU 上跑完 (models/cqr_pred_cache.npz 存在)。
协议: 与 nutrient_conformal_final.py 完全一致 — 严格 hold-out, 652 测试集对半
切 cal(326)/eval(326), 10 trials (seed 42..51 ⇒ 与 v2 实验同一批 split, 直接可比),
α=0.10。有效性: 全营养素 cov ≥ 89.5%。

CQR 变体 (区间均作用在营养素空间 L=w·q10, U=w·q90, p=w·q50):
  CQR-raw  : 原始 10-90 分位区间, 不共形化 (参考)
  CQR-abs  : 标准 CQR — s = max(L−y, y−U), 区间 [L−q̂, U+q̂]
  CQR-rel  : 相对分数 — s = max(L−y, y−U)/p, 区间 [L−q̂p, U+q̂p]
  CQR-bin  : CQR-abs + 按预测重量分 bin 的 q̂ (bin<30 回退全局, 同 MondrianW 规则)
  CQR-hyb  : 中心=生产 hybrid 点估计, 半宽 h=(U−L)/2 来自分位模型 (难度自适应),
             s=|y−c|/h → c±q̂h
  CQR-hyb-b: CQR-hyb + 分 bin q̂ (对应 AEC 获胜配方 S2m 的分层重标定)

基线: AEC-S2m-5W (部署获胜者, nWinkler=2.42), 同 split 重跑保证公平。

运行 (CPU 沙箱可跑, ~3-5 min): python nutrient_cqr_experiment.py
产物: results/nutrient_cqr_results.pkl
"""
import os, sys, time, joblib
from pathlib import Path
import numpy as np
from sklearn.model_selection import train_test_split


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
from experiments.estimation.nutrient_intervals import (
    NutrientConformalCalibrator, CANDS5, qhat, eval_iv, BINS, NUTR,
    SEED, ALPHA, get_test_pipeline_cache, hybrid_nutrients, nutrient_disagreement)

N_TRIALS = 10
EPS = 1e-3
CQR_CACHE = n5k('cqr_pred_cache.npz')
OUT = str(Paths.resolve().root / "backup" / "v3_excluded" / "results" / "estimation" / "e5" / "nutrient_cqr_results.pkl")
VARIANTS = ['CQR-raw', 'CQR-abs', 'CQR-rel', 'CQR-bin', 'CQR-hyb', 'CQR-hyb-b',
            'AEC-S2m-5W']


def _pack(y, l, u):
    cov, width, wk = eval_iv(y, l, u)
    return (cov, width, width / max(y.mean(), EPS), wk)


def qhat_bin(s_cal, w_cal, w_eval, alpha=ALPHA):
    """按重量 bin 的分位 q̂; bin<30 样本回退全局。返回 eval 上每营养素的 q̂。"""
    q_glob = np.array([qhat(s_cal[:, i], alpha) for i in range(s_cal.shape[1])])
    b_cal = np.digitize(w_cal, BINS[1:-1])
    b_eval = np.digitize(w_eval, BINS[1:-1])
    n_out = s_cal.shape[1]
    q_mat = np.zeros((len(w_eval), n_out))
    for bi in range(len(BINS) - 1):
        mc, me = b_cal == bi, b_eval == bi
        if not me.any():
            continue
        for i in range(n_out):
            if mc.sum() >= 30:
                q_mat[me, i] = qhat(s_cal[mc, i], alpha)
            else:
                q_mat[me, i] = q_glob[i]
    return q_mat


def main():
    print("=" * 70)
    print("CQR vs AEC-S2m-N | 同协议 10-trial, cal/eval=326/326, α=0.10")
    print("=" * 70)
    assert os.path.exists(CQR_CACHE), "先在 GPU 终端运行 train_density_cqr.py"

    cache = get_test_pipeline_cache()
    n_hybrid = hybrid_nutrients(cache)
    n_disag = nutrient_disagreement(cache)
    y = cache['y_nutr']; w = cache['w_pred']; y_w = cache['y_w']

    cqr = np.load(CQR_CACHE)
    assert np.allclose(cqr['w_pred'], w), "重量预测不一致, 检查缓存"
    assert np.allclose(cqr['y_nutr'], y), "营养素标签不一致, 检查缓存"
    L = np.maximum(w, 0).reshape(-1, 1) * cqr['d_q10']
    U = np.maximum(w, 0).reshape(-1, 1) * cqr['d_q90']
    P = np.maximum(w, 0).reshape(-1, 1) * cqr['d_q50']
    h_half = np.maximum((U - L) / 2, 1e-6)
    print(f"样本 {len(y)} | 中位点估计 PMAE: "
          f"{np.mean([np.abs(y[:, i]-P[:, i]).mean()/y[:, i].mean() for i in range(4)])*100:.1f}%"
          f" | 原始区间 cov: "
          f"{np.mean([( (y[:, i]>=L[:, i]) & (y[:, i]<=U[:, i]) ).mean() for i in range(4)])*100:.1f}%")

    metrics = {m: {n: [] for n in NUTR} for m in VARIANTS}
    t0 = time.time()
    for t in range(N_TRIALS):
        idx = np.arange(len(y))
        cal_idx, ev_idx = train_test_split(idx, test_size=0.5, random_state=SEED + t)
        cc, ec = cal_idx, ev_idx
        yc = y[cc]

        # --- CQR 分数 (cal 上拟合 q̂, eval 上应用) ---
        s_abs = np.maximum(L[cc] - yc, yc - U[cc])                 # (n,4)
        p_safe = np.maximum(P[cc], 0.1)
        s_rel = s_abs / p_safe
        s_hyb = np.abs(yc - n_hybrid[cc]) / h_half[cc]

        # raw
        for i, nt in enumerate(NUTR):
            metrics['CQR-raw'][nt].append(_pack(y[ec, i], L[ec, i], U[ec, i]))
        # abs
        qa = np.array([qhat(s_abs[:, i]) for i in range(4)])
        for i, nt in enumerate(NUTR):
            metrics['CQR-abs'][nt].append(
                _pack(y[ec, i], L[ec, i] - qa[i], U[ec, i] + qa[i]))
        # rel
        qr = np.array([qhat(s_rel[:, i]) for i in range(4)])
        for i, nt in enumerate(NUTR):
            metrics['CQR-rel'][nt].append(
                _pack(y[ec, i], L[ec, i] - qr[i] * np.maximum(P[ec, i], 0.1),
                      U[ec, i] + qr[i] * np.maximum(P[ec, i], 0.1)))
        # bin (abs 分数, 分 bin q̂)
        qb = qhat_bin(s_abs, w[cc], w[ec])                         # (n_eval, 4)
        for i, nt in enumerate(NUTR):
            metrics['CQR-bin'][nt].append(
                _pack(y[ec, i], L[ec, i] - qb[:, i], U[ec, i] + qb[:, i]))
        # hyb: 中心=hybrid 点, 半宽来自 CQR
        qh = np.array([qhat(s_hyb[:, i]) for i in range(4)])
        for i, nt in enumerate(NUTR):
            c_e = n_hybrid[ec, i]
            metrics['CQR-hyb'][nt].append(
                _pack(y[ec, i], np.maximum(0, c_e - qh[i] * h_half[ec, i]),
                      c_e + qh[i] * h_half[ec, i]))
        # hyb-b: hyb + 分 bin q̂ (hyb 分数)
        qhb = qhat_bin(s_hyb, w[cc], w[ec])
        for i, nt in enumerate(NUTR):
            c_e = n_hybrid[ec, i]
            metrics['CQR-hyb-b'][nt].append(
                _pack(y[ec, i], np.maximum(0, c_e - qhb[:, i] * h_half[ec, i]),
                      c_e + qhb[:, i] * h_half[ec, i]))

        # --- 基线 AEC-S2m-5W (同 split) ---
        cal = NutrientConformalCalibrator(cands=CANDS5, strat='weight').fit(
            w[cc], y_w[cc], n_hybrid[cc], y[cc], n_disag=n_disag[cc], seed=SEED + t)
        lo, hi = cal.predict_intervals(w[ec], n_hybrid[ec], n_disag[ec], 'm')
        for i, nt in enumerate(NUTR):
            metrics['AEC-S2m-5W'][nt].append(_pack(y[ec, i], lo[:, i], hi[:, i]))
        print(f"  trial {t+1}/{N_TRIALS} ({time.time()-t0:.0f}s)")

    # ---------- 汇总 ----------
    y_mean = {nt: y[:, i].mean() for i, nt in enumerate(NUTR)}
    print("\n" + "=" * 150)
    print(f"{'方法':<12s}", end='')
    for nt in NUTR:
        print(f"| {nt:^30s} ", end='')
    print()
    print(f"{'':<12s}", end='')
    for _ in NUTR:
        print(f"| {'cov%':>8s} {'relwid%':>9s} {'winkler':>9s} ", end='')
    print("\n" + "-" * 150)
    for m in VARIANTS:
        print(f"{m:<12s}", end='')
        for nt in NUTR:
            a = np.array(metrics[m][nt])
            print(f"| {a[:,0].mean()*100:>7.1f} {a[:,2].mean()*100:>8.1f} "
                  f"{a[:,3].mean():>9.1f} ", end='')
        print()

    print("\n综合排名 (有效 = 全营养素 cov≥89.5%):")
    print(f"{'方法':<12s} {'min-cov%':>9s} {'mean-relwid%':>13s} {'mean-nWinkler':>14s} 有效")
    rank = []
    for m in VARIANTS:
        covs = [np.array(metrics[m][nt])[:, 0].mean() for nt in NUTR]
        rw = [np.array(metrics[m][nt])[:, 2].mean() for nt in NUTR]
        nw = [np.array(metrics[m][nt])[:, 3].mean() / y_mean[nt] for nt in NUTR]
        valid = min(covs) >= 0.895
        rank.append((m, min(covs) * 100, np.mean(rw) * 100, np.mean(nw), valid))
        print(f"{m:<12s} {min(covs)*100:>8.1f} {np.mean(rw)*100:>12.1f} "
              f"{np.mean(nw):>13.2f} {'✓' if valid else '✗'}")

    base = next(r for r in rank if r[0] == 'AEC-S2m-5W')
    valid_cqr = [r for r in rank if r[0] != 'AEC-S2m-5W' and r[4]]
    if valid_cqr:
        best = min(valid_cqr, key=lambda r: r[3])
        verdict = (f"CQR 最优变体 {best[0]} nWinkler={best[3]:.3f} vs AEC-S2m-5W "
                   f"{base[3]:.3f} → {'CQR 获胜 ✓' if best[3] < base[3] else 'AEC 保持获胜 ✗'}")
    else:
        verdict = "所有 CQR 变体均未满足覆盖约束 → AEC-S2m-5W 保持获胜 ✗"
    print(f"\n结论: {verdict}")
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    joblib.dump({'metrics': metrics, 'rank': rank, 'verdict': verdict,
                 'config': {'alpha': ALPHA, 'n_trials': N_TRIALS, 'taus': [0.1, 0.5, 0.9],
                            'date': '2026-08-20'}}, OUT)
    print(f"Saved: {OUT}")


if __name__ == '__main__':
    main()
