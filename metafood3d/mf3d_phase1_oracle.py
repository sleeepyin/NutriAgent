#!/usr/bin/env python3
"""Phase 1: Oracle 误差预算实验 — 回答"体积完美时，密度查表还剩多少误差？"
范式: log W = log V + log ρ (误差在 log 空间可加)
Oracle-1: GT V × GT ρ          sanity (~0%)
Oracle-2: GT V × LOO类别均值ρ   范式上界(核心)
Oracle-2b: GT V × LOO类别中位ρ  稳健版
Oracle-3: GT V × 全局均值ρ      无类别信息下界
Oracle-5: 灵敏度扫描 体积×(1±ε) × LOOρ  ε∈{5,10,15,20}%
输出: mf3d_phase1_results.csv + 报告"""
import csv, math, os, statistics as st

DIR = "/home/metafood3d"
rows = []
with open(os.path.join(DIR, "metafood3d_labels_clean.csv")) as f:
    for r in csv.DictReader(f):
        rows.append({"cat": r["cat"], "obj": r["obj"], "w": float(r["weight_g"]),
                     "v": float(r["volume_cm3"]), "d": float(r["density"])})

def mape(preds, truths):
    return 100 * sum(abs(p-t)/t for p, t in zip(preds, truths)) / len(truths)

def report(name, preds, truths, subset=None):
    """subset: 可选过滤函数"""
    if subset:
        pp = [p for p, t in zip(preds, truths) if subset(t)]
        tt = [t for t in truths if subset(t)]
    else:
        pp, tt = preds, truths
    m = mape(pp, tt)
    # log-space bias (系统性偏大/偏小)
    bias = 100 * sum(math.log(p/t) for p, t in zip(pp, tt)) / len(tt)
    print(f"  {name:42s} MAPE={m:6.2f}%  log-bias={bias:+6.2f}%  n={len(tt)}")
    return m

# 类别分组
from collections import defaultdict
by_cat = defaultdict(list)
for r in rows: by_cat[r["cat"]].append(r)

# LOO 密度(均值/中位)
for r in rows:
    others = [o for o in by_cat[r["cat"]] if o is not r]
    if others:
        r["d_loo_mean"] = sum(o["d"] for o in others)/len(others)
        r["d_loo_med"] = st.median([o["d"] for o in others])
    else:
        r["d_loo_mean"] = r["d_loo_med"] = None
solo = [r for r in rows if r["d_loo_mean"] is None]  # 类别内唯一物体
if solo: print(f"[*] {len(solo)} 个物体类别内无同伴, LOO 用全局均值替代")
g_mean = sum(r["d"] for r in rows)/len(rows)
g_med = st.median([r["d"] for r in rows])
for r in rows:
    if r["d_loo_mean"] is None: r["d_loo_mean"], r["d_loo_med"] = g_mean, g_med

W = [r["w"] for r in rows]; V = [r["v"] for r in rows]
ge100 = lambda t: t >= 100

print("=" * 78)
print("Phase 1 Oracle 误差预算 (MetaFood3D, n=%d, 清洗后)" % len(rows))
print("=" * 78)

print("\n[A] 全量物体 (对照: Nutrition5K 单目全 hold-out PMAE=22.0%)")
report("Oracle-1: GT V × GT ρ (sanity)", [r["v"]*r["d"] for r in rows], W)
report("Oracle-2: GT V × LOO类别均值ρ", [r["v"]*r["d_loo_mean"] for r in rows], W)
report("Oracle-2b: GT V × LOO类别中位ρ", [r["v"]*r["d_loo_med"] for r in rows], W)
report("Oracle-3: GT V × 全局均值ρ", [r["v"]*g_mean for r in rows], W)
report("Oracle-3b: GT V × 全局中位ρ", [r["v"]*g_med for r in rows], W)

print("\n[B] >=100g 子集 (对照: Nutrition5K 单目 18.4% 物理上限)")
report("Oracle-2: GT V × LOO类别均值ρ", [r["v"]*r["d_loo_mean"] for r in rows], W, ge100)
report("Oracle-2b: GT V × LOO类别中位ρ", [r["v"]*r["d_loo_med"] for r in rows], W, ge100)
report("Oracle-3: GT V × 全局均值ρ", [r["v"]*g_mean for r in rows], W, ge100)

print("\n[C] Oracle-5 灵敏度: 体积误差 ε 对重量 MAPE 的影响 (LOO均值ρ)")
import random
random.seed(42)
for eps in [0, 0.05, 0.10, 0.15, 0.20]:
    preds = []
    for r in rows:
        noise = random.choice([1+eps, 1-eps])  # 确定性交替,避免随机性
        preds.append(r["v"] * noise * r["d_loo_mean"])
    report(f"ε={eps*100:.0f}% (交替±)  全量", preds, W)
    report(f"ε={eps*100:.0f}% (交替±)  >=100g", preds, W, ge100)

print("\n[D] 密度本身的离散度 (类别内 vs 全局, log 空间)")
intra = []
for cat, rs in by_cat.items():
    if len(rs) >= 2:
        for r in rs:
            intra.append(abs(math.log(r["d"]/r["d_loo_mean"])))
print(f"  类别内 |log(ρ/ρ_LOO)| 均值: {100*sum(intra)/len(intra):.2f}%  (密度查表的固有误差)")
print(f"  全局 |log(ρ/ρ_med)| 均值: {100*sum(abs(math.log(r['d']/g_med)) for r in rows)/len(rows):.2f}%")

# 保存逐物体结果
with open(os.path.join(DIR, "mf3d_phase1_results.csv"), "w", newline="") as f:
    wtr = csv.writer(f)
    wtr.writerow(["cat", "obj", "w_true", "v_gt", "d_gt", "d_loo_mean", "d_loo_med",
                  "w_oracle2", "w_oracle3"])
    for r in rows:
        wtr.writerow([r["cat"], r["obj"], r["w"], r["v"], r["d"], r["d_loo_mean"], r["d_loo_med"],
                      r["v"]*r["d_loo_mean"], r["v"]*g_mean])
print("\n[saved] mf3d_phase1_results.csv")
