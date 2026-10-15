#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
论文图表生成 (任务 5, 2026-08-24)
=================================
生成 5 张学术图到 /home/Nutri_0811/figures/ (300dpi, 英文标签, 色盲友好):
  fig_error_budget.png      §6.2 误差预算分解 (log W = log V + log ρ 类内方差分解)
  fig_blueberry_paradox.png §6.2 蓝莓悖论 (散装 vs 装碗密度分布)
  fig_coverage_collapse.png §5.8 共形覆盖崩溃分箱 (N5K vs NV-Real)
  fig_joint_training.png    §5.8 联合训练 3 配置 × 2 测试集
  fig_sam3d_scale_prior.png §6.3 SAM 3D 尺度先验散点 (可选)

数据源:
  /home/metafood3d/mf3d_phase1_results.csv
  /home/metafood3d/mf3d_oracle6_results.csv
  /home/nvreal/phase3_conformal_results.json

运行 (CPU, ~10s): python make_paper_figures.py
"""
import os, json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

OUT = '/home/Nutri_0811/figures'
os.makedirs(OUT, exist_ok=True)

# Okabe-Ito 色盲友好调色板
BLUE, ORANGE, GREEN, SKY, VERM, PURPLE, GRAY = (
    '#0072B2', '#E69F00', '#009E73', '#56B4E9', '#D55E00', '#CC79A7', '#999999')

plt.rcParams.update({
    'font.size': 10, 'axes.spines.top': False, 'axes.spines.right': False,
    'figure.dpi': 300, 'savefig.dpi': 300, 'savefig.bbox': 'tight',
})

PHASE1 = '/home/metafood3d/mf3d_phase1_results.csv'
ORACLE6 = '/home/metafood3d/mf3d_oracle6_results.csv'
PHASE3 = '/home/nvreal/phase3_conformal_results.json'


# ==================== 图 1: 误差预算分解 ====================
def fig_error_budget():
    df = pd.read_csv(PHASE1)
    df = df[(df['w_true'] > 0) & (df['v_gt'] > 0) & (df['d_gt'] > 0)
            & (df['d_loo_mean'] > 0)].copy()

    # 类别内密度查表固有误差 (论文口径): |log(ρ / ρ_LOO)| 均值
    disp = float(np.abs(np.log(df['d_gt'] / df['d_loo_mean'])).mean() * 100)
    # Oracle-2 (GT 体积 × LOO 类别均值密度) MAPE
    mape_clean = float((np.abs(df['w_oracle2'] - df['w_true']) / df['w_true']).mean() * 100)
    # ≥100g 子集 (与论文 §6.2 叙述一致: 23.4% + 2.6pp)
    m100 = df['w_true'] >= 100
    mape100 = float((np.abs(df.loc[m100, 'w_oracle2'] - df.loc[m100, 'w_true'])
                     / df.loc[m100, 'w_true']).mean() * 100)
    # ±10% 体积误差的增量 (与原脚本 mf3d_phase1_oracle.py [C] 完全同协议:
    # seed 42, eps 依次 0/0.05/0.10 各消耗一轮随机数, 取 eps=0.10 的预测)
    import random
    v, dl, wt = df['v_gt'].values, df['d_loo_mean'].values, df['w_true'].values
    random.seed(42)
    preds = None
    for eps in [0, 0.05, 0.10]:
        preds = np.array([v[i] * random.choice([1 + eps, 1 - eps]) * dl[i]
                          for i in range(len(df))])
    mape_noisy100 = float((np.abs(preds[m100] - wt[m100]) / wt[m100]).mean() * 100)
    vol_cost = mape_noisy100 - mape100
    mono = 22.0                                   # N5K 单目基线 (§5.1)
    mono100 = 18.4                                # N5K ≥100g 物理上限 (§6.2)
    print(f"[fig1] 密度查表固有误差 {disp:.1f}% | Oracle-2(≥100g) {mape100:.1f}% | "
          f"±10%体积代价 +{vol_cost:.1f}pp | 单目 {mono}% / ≥100g {mono100}%")

    fig, axes = plt.subplots(1, 2, figsize=(9.6, 3.8))

    # (a) 密度误差 ≈ 整个问题难度
    ax = axes[0]
    labels = ['Within-category\ndensity dispersion\n|log(ρ/ρ_LOO)|',
              'Monocular RGB-D\nproblem difficulty\n(§4, 4ch hold-out MAPE)']
    vals = [disp, mono]
    bars = ax.bar(labels, vals, color=[ORANGE, GREEN], width=0.55,
                  edgecolor='black', linewidth=0.5)
    for b, v in zip(bars, vals):
        ax.text(b.get_x() + b.get_width() / 2, v + 0.4, f'{v:.1f}%',
                ha='center', fontsize=11, fontweight='bold')
    ax.set_ylabel('Magnitude (%)')
    ax.set_title('(a) Density dispersion alone matches\nthe whole problem difficulty', fontsize=10)
    ax.set_ylim(0, 27)
    ax.tick_params(axis='x', labelsize=9)

    # (b) MAPE 口径的误差预算 (完美几何下, ≥100g 子集)
    ax = axes[1]
    cats = ['Perfect volume +\ncategory-mean density\n(Oracle-2, ≥100g)',
            'Additional cost of\n±10% volume error',
            'Monocular RGB-D\nbaseline, ≥100g\n(§5.1)']
    vals2 = [mape100, vol_cost, mono100]
    bars = ax.bar(cats, vals2, color=[ORANGE, SKY, GREEN], width=0.55,
                  edgecolor='black', linewidth=0.5)
    for b, v in zip(bars, vals2):
        ax.text(b.get_x() + b.get_width() / 2, v + 0.4, f'{v:.1f}%',
                ha='center', fontsize=11, fontweight='bold')
    ax.set_ylabel('Mass MAPE (%)')
    ax.set_title('(b) Error budget with perfect geometry\n(MetaFood3D, GT volume, ≥100g)', fontsize=10)
    ax.set_ylim(0, 30)
    ax.tick_params(axis='x', labelsize=9)

    fig.tight_layout()
    fig.savefig(f'{OUT}/fig_error_budget.png')
    plt.close(fig)
    print(f"[fig1] saved {OUT}/fig_error_budget.png")


# ==================== 图 2: 蓝莓悖论 ====================
def fig_blueberry():
    df = pd.read_csv(PHASE1)
    loose = df[df['cat'] == 'Blueberries']['d_gt'].values
    bowl = df[df['cat'] == 'Blueberry(bowl)']['d_gt'].values
    disp = lambda d: np.abs(np.log(d / np.median(d))).mean() * 100   # 中位数口径
    print(f"[fig2] 散装 n={len(loose)} disp={disp(loose):.1f}% | "
          f"装碗 n={len(bowl)} disp={disp(bowl):.1f}%")

    fig, ax = plt.subplots(figsize=(4.6, 3.6))
    parts = ax.violinplot([loose, bowl], positions=[0, 1], widths=0.75,
                          showmeans=False, showextrema=False)
    for body, c in zip(parts['bodies'], [VERM, BLUE]):
        body.set_facecolor(c); body.set_alpha(0.35); body.set_edgecolor(c)
    # 叠加原始数据点
    rng = np.random.default_rng(0)
    ax.scatter(rng.normal(0, 0.05, len(loose)), loose, color=VERM, s=28,
               zorder=3, edgecolor='black', linewidth=0.4, label='Loose (n=8)')
    ax.scatter(rng.normal(1, 0.05, len(bowl)), bowl, color=BLUE, s=28,
               zorder=3, edgecolor='black', linewidth=0.4, label='Bowl-packed (n=5)')
    for x, m, c in [(0, loose.mean(), VERM), (1, bowl.mean(), BLUE)]:
        ax.hlines(m, x - 0.28, x + 0.28, color=c, lw=2, zorder=4)
    ax.set_xticks([0, 1], ['Loose\nblueberries', 'Bowl-packed\nblueberries'])
    ax.set_ylabel('Effective density (g/cm³)')
    ax.set_title('The blueberry paradox: same category,\n3× density range from packing alone')
    ax.legend(frameon=False, fontsize=9)
    fig.tight_layout()
    fig.savefig(f'{OUT}/fig_blueberry_paradox.png')
    plt.close(fig)
    print(f"[fig2] saved {OUT}/fig_blueberry_paradox.png")


# ==================== 图 3: 覆盖崩溃分箱 ====================
def fig_coverage():
    d = json.load(open(PHASE3))
    bins = d['bins']
    labels = [b['bin'] for b in bins]
    n5k = [b['n5k_split_cov'] for b in bins]
    nv_split = [b['nv_split_cov'] for b in bins]
    nv_mond = [b['nv_mond_cov'] for b in bins]
    n_nv = [b['n_nv'] for b in bins]

    x = np.arange(len(labels))
    wbar = 0.26
    fig, ax = plt.subplots(figsize=(6.2, 3.8))
    ax.bar(x - wbar, n5k, wbar, color=GREEN, edgecolor='black', linewidth=0.5,
           label='N5K in-domain (Split)')
    ax.bar(x, nv_split, wbar, color=VERM, edgecolor='black', linewidth=0.5,
           label='NV-Real zero-shot (Split)')
    ax.bar(x + wbar, nv_mond, wbar, color=SKY, edgecolor='black', linewidth=0.5,
           label='NV-Real zero-shot (MondrianW)')
    ax.axhline(90, color='black', ls='--', lw=1)
    ax.text(0.08, 92, 'Nominal 90%', ha='left', va='bottom', fontsize=8)
    for xi, (a, b, c) in enumerate(zip(n5k, nv_split, nv_mond)):
        ax.text(xi - wbar, a + 1.5, f'{a:.0f}', ha='center', fontsize=8)
        ax.text(xi, b + 1.5, f'{b:.0f}', ha='center', fontsize=8)
        ax.text(xi + wbar, c + 1.5, f'{c:.0f}', ha='center', fontsize=8)
    ax.set_xticks(x, [f'{l}\n(n={n})' for l, n in zip(labels, n_nv)])
    ax.set_xlabel('Dish weight bin (NV-Real sample size)')
    ax.set_ylabel('Coverage (%)')
    ax.set_ylim(0, 108)
    ax.set_title('Conformal coverage collapse under domain shift,\nby weight bin (frozen N5K calibrator)')
    ax.legend(frameon=False, fontsize=8, loc='upper right', ncol=1)
    fig.tight_layout()
    fig.savefig(f'{OUT}/fig_coverage_collapse.png')
    plt.close(fig)
    print(f"[fig3] saved {OUT}/fig_coverage_collapse.png")


# ==================== 图 4: 联合训练 ====================
def fig_joint():
    cfgs = ['N5K only\n(zero-shot NV)', 'NV-Real only', 'Joint\n(N5K + NV-Real)']
    n5k_test = [31.4, 124.7, 31.5]
    nv_test = [77.4, 25.6, 23.9]
    x = np.arange(len(cfgs))
    wbar = 0.36
    fig, ax = plt.subplots(figsize=(5.6, 3.8))
    b1 = ax.bar(x - wbar / 2, n5k_test, wbar, color=BLUE, edgecolor='black',
                linewidth=0.5, label='N5K test (652)')
    b2 = ax.bar(x + wbar / 2, nv_test, wbar, color=ORANGE, edgecolor='black',
                linewidth=0.5, label='NV-Real test (177)')
    for bars in (b1, b2):
        for b in bars:
            ax.text(b.get_x() + b.get_width() / 2, b.get_height() + 2,
                    f'{b.get_height():.1f}', ha='center', fontsize=9)
    ax.axhline(31.4, color=BLUE, ls=':', lw=1)
    ax.set_xticks(x, cfgs)
    ax.set_ylabel('Dish-level MAPE (%)')
    ax.set_title('Multi-domain joint training: win-win\n(zero source-domain cost)')
    ax.legend(frameon=False, fontsize=9)
    ax.set_ylim(0, 140)
    fig.tight_layout()
    fig.savefig(f'{OUT}/fig_joint_training.png')
    plt.close(fig)
    print(f"[fig4] saved {OUT}/fig_joint_training.png")


# ==================== 图 5: SAM 3D 尺度先验 ====================
def fig_sam3d():
    df = pd.read_csv(ORACLE6)
    # 与原脚本同过滤: s_l2c>0 且 v_est_cam>0
    df = df[(df['s_l2c'] > 0) & (df['v_est_cam'] > 0) & (df['v_gt_cm3'] > 0)].copy()
    lg, le = np.log10(df['v_gt_cm3']), np.log10(df['v_est_cam'])
    # 类别均值 (论文: log-log r = 0.92; 原脚本口径: 自然空间均值再取 log)
    gm = df.groupby('cat').apply(
        lambda g: pd.Series({'lg': np.log10(g['v_gt_cm3'].mean()),
                             'le': np.log10(g['v_est_cam'].mean())}),
        include_groups=False)
    r = np.corrcoef(gm['lg'], gm['le'])[0, 1]
    print(f"[fig5] n_obj={len(df)} n_cat={len(gm)} 类别均值 log-log r={r:.2f}")

    fig, ax = plt.subplots(figsize=(5.0, 4.4))
    ax.scatter(lg, le, s=12, color=GRAY, alpha=0.45, edgecolor='none',
               label='Objects (n=%d)' % len(df))
    ax.scatter(gm['lg'], gm['le'], s=34, color=BLUE, edgecolor='black',
               linewidth=0.4, zorder=3, label='Category means (n=%d)' % len(gm))
    k, b_ = np.polyfit(gm['lg'], gm['le'], 1)
    xs = np.linspace(gm['lg'].min(), gm['lg'].max(), 50)
    ax.plot(xs, k * xs + b_, color=VERM, lw=1.5,
            label=f'OLS on category means (r={r:.2f})')
    lims = [min(lg.min(), le.min()) - 0.3, max(lg.max(), le.max()) + 0.3]
    ax.plot(lims, lims, color='black', ls='--', lw=0.8, label='y = x (metric scale)')
    ax.set_xlabel('log₁₀ ground-truth volume (cm³)')
    ax.set_ylabel('log₁₀ SAM 3D estimated volume (camera space, cm³)')
    ax.set_title('SAM 3D scale prior: strong between-category\nranking, no universal metric scale')
    ax.legend(frameon=False, fontsize=8)
    fig.tight_layout()
    fig.savefig(f'{OUT}/fig_sam3d_scale_prior.png')
    plt.close(fig)
    print(f"[fig5] saved {OUT}/fig_sam3d_scale_prior.png")


if __name__ == '__main__':
    fig_error_budget()
    fig_blueberry()
    fig_coverage()
    fig_joint()
    fig_sam3d()
    print("\n全部完成 →", OUT)
