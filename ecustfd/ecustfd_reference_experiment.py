# -*- coding: utf-8 -*-
"""
ECUSTFD 参照物实验: 硬币定标消除尺度歧义后, 重量估计还剩多少误差?

论证链定位:
  - Nutrition5K 单目上限 18.4% (尺度歧义, R2=0.39)
  - MetaFood3D: 密度离散 21.8% 是主导 (蓝莓悖论), 但刚体类 <15%
  - 本实验: 刚体类(ECUSTFD 全是) + 硬币参照物(25.0mm 一元硬币)
           → 参照物路线的实际上界测试

实验设计 (全部 LOO/诚实协议):
  E1 (尺度歧义量化):   像素面积 → 硬币定标 → mm; GT 体积 vs "外观面积推断体积"
      对比: 有硬币定标 vs 无定标(假设固定拍摄距离) 的体积误差
  E2 (密度查表 + LOO): GT 体积 × LOO 类别均值密度 → 重量 MAPE (MetaFood3D 同款协议)
  E3 (全链路上界):     硬币定标体积估计(简单几何) × LOO 密度 → 端到端重量误差
      对比单目 18.4% → 参照物路线值不值得, 一锤定音
"""
import glob, os, re, sys
import numpy as np
import pandas as pd
import xml.etree.ElementTree as ET
from PIL import Image

ROOT = "/home/ecustfd"
COIN_MM = 25.0          # 一元硬币直径 mm
CSV = os.path.join(ROOT, "ecustfd_reference_results.csv")

# ---------- 1. 载入密度表 (每实例 体积ml/重量g) ----------
xl = pd.ExcelFile(os.path.join(ROOT, "density.xls"))
recs = []
for s in xl.sheet_names:
    df = xl.parse(s)
    if "volume(mm^3)" not in df.columns:
        continue
    for _, r in df.iterrows():
        recs.append(dict(inst=str(r["id"]), cat=str(r["type"]),
                         vol_ml=float(r["volume(mm^3)"]), wt_g=float(r["weight(g)"])))
GT = pd.DataFrame(recs)
GT["rho"] = GT["wt_g"] / GT["vol_ml"]      # g/ml
print(f"[data] 实例: {len(GT)} | 类别: {GT['cat'].nunique()} | 密度范围 {GT['rho'].min():.2f}-{GT['rho'].max():.2f} g/ml")

# ---------- 2. 解析 XML: 食物/硬币 bbox, 匹配实例 id ----------
def parse_xml(f):
    r = ET.parse(f).getroot()
    objs = {}
    for o in r.iter("object"):
        n = o.find("name").text
        b = o.find("bndbox")
        objs[n] = (float(b.find("xmin").text), float(b.find("ymin").text),
                   float(b.find("xmax").text), float(b.find("ymax").text))
    return objs, (int(r.find("size/width").text), int(r.find("size/height").text))

samples = []   # (inst, cat, view, food_px_area, coin_px_diam, img_size)
for xf in sorted(glob.glob(os.path.join(ROOT, "Annotations", "*.xml"))):
    base = os.path.basename(xf)
    m = re.match(r"([a-z_]+?\d{3})([ST])", base)
    if not m:
        continue
    inst, view = m.group(1), m.group(2)
    if inst not in set(GT["inst"]):
        continue
    objs, _ = parse_xml(xf)
    if "coin" not in objs:
        continue
    food_cat = [k for k in objs if k != "coin"]
    if len(food_cat) != 1:
        continue
    cat = food_cat[0]
    fx0, fy0, fx1, fy1 = objs[cat]
    cx0, cy0, cx1, cy1 = objs["coin"]
    food_area = (fx1 - fx0) * (fy1 - fy0)
    coin_diam = max(cx1 - cx0, cy1 - cy0)
    if food_area <= 0 or coin_diam <= 0:
        continue
    g = GT[GT["inst"] == inst].iloc[0]
    samples.append(dict(inst=inst, cat=cat, view=view,
                        food_px=food_area, coin_px=coin_diam,
                        vol_ml=g["vol_ml"], wt_g=g["wt_g"]))

S = pd.DataFrame(samples)
print(f"[data] 图像样本: {len(S)} (每实例 T/S 多组) | 覆盖实例 {S['inst'].nunique()}")

# ---------- E1: 尺度歧义量化 ----------
# 有定标: mm_per_px = COIN_MM / coin_px; 无定标: 全局中位 mm_per_px (模拟"固定距离拍摄")
S["mmpp"] = COIN_MM / S["coin_px"]
global_mmpp = S["mmpp"].median()
S["food_mm2_cal"] = S["food_px"] * S["mmpp"] ** 2      # 定标后面积 mm²
S["food_mm2_fix"] = S["food_px"] * global_mmpp ** 2    # 固定距离假设
# 面积→体积: 用 顶视面积 × 高度估计; 这里先用 顶视面积与 GT 体积的 log-log r 表征"面积携带多少体积信息"
T = S[S["view"] == "T"].groupby("inst").first()
r_cal = np.corrcoef(np.log(T["food_mm2_cal"]), np.log(T["vol_ml"]))[0, 1]
r_fix = np.corrcoef(np.log(T["food_mm2_fix"]), np.log(T["vol_ml"]))[0, 1]
# 定标的实际价值: mmpp 的变异幅度 = 尺度歧义的大小
print(f"\n[E1 尺度歧义] mm/px 范围: {S['mmpp'].min():.4f}-{S['mmpp'].max():.4f} "
      f"(变异 {100*(S['mmpp'].max()/S['mmpp'].min()-1):.0f}% → 没有定标时的尺度歧义)")
print(f"[E1] 顶视面积-体积 log-log r: 有定标 {r_cal:.3f} | 固定距离假设 {r_fix:.3f}")

# ---------- E2: LOO 密度查表 (MetaFood3D 同款协议) ----------
errs, errs_global = [], []
for _, g in GT.iterrows():
    same = GT[GT["cat"] == g["cat"]]
    loo = same[same["inst"] != g["inst"]]
    rho_hat = loo["rho"].mean() if len(loo) else GT["rho"].mean()
    errs.append(abs(g["vol_ml"] * rho_hat - g["wt_g"]) / g["wt_g"])
    errs_global.append(abs(g["vol_ml"] * GT["rho"].mean() - g["wt_g"]) / g["wt_g"])
e2 = 100 * np.mean(errs)
e2g = 100 * np.mean(errs_global)
print(f"\n[E2 GT体积×LOO类别密度] 重量 MAPE: {e2:.1f}%  (全局密度: {e2g:.1f}%)")
print(f"    对照: MetaFood3D 同协议 25.0% | 单目上限 18.4%")

# ---------- E3: 端到端 (定标面积 → 体积 → LOO 密度) ----------
# 体积估计: 顶视定标面积 × 类别 LOO 高度先验 (height = vol / area 的类内均值)
# 即 "面积已知 + 高度假设类内恒定" —— 最简单的几何基线, 上界由 E2 界定
rows = []
for inst, t in T.iterrows():
    g = GT[GT["inst"] == inst].iloc[0]
    same = GT[GT["cat"] == g["cat"]]
    # LOO 面积→体积比例 (用同类其他实例的 顶视面积/体积)
    peers = [i for i in T.index if i != inst and GT.set_index("inst").loc[i, "cat"] == g["cat"]]
    if len(peers) < 1:
        continue
    k_hat = np.median([T.loc[p, "food_mm2_cal"] / GT.set_index("inst").loc[p, "vol_ml"] for p in peers])
    vol_hat = t["food_mm2_cal"] / k_hat
    loo_rho = same[same["inst"] != inst]["rho"].mean()
    w_hat = vol_hat * loo_rho
    rows.append(dict(inst=inst, cat=g["cat"], wt_gt=g["wt_g"], wt_hat=w_hat,
                     vol_hat=vol_hat, vol_gt=g["vol_ml"]))
E3 = pd.DataFrame(rows)
E3["err"] = abs(E3["wt_hat"] - E3["wt_gt"]) / E3["wt_gt"]
e3 = 100 * E3["err"].mean()
vol_mape = 100 * (abs(E3["vol_hat"] - E3["vol_gt"]) / E3["vol_gt"]).mean()
print(f"\n[E3 端到端: 硬币定标面积×LOO几何×LOO密度]")
print(f"    体积 MAPE: {vol_mape:.1f}%")
print(f"    重量 MAPE: {e3:.1f}%  (n={len(E3)})")
print(f"\n{'='*60}")
print(f"  [结论判定]")
print(f"  E2 (GT体积+LOO密度)  : {e2:.1f}%   ← 参照物路线的下界(体积完美)")
print(f"  E3 (定标几何+LOO密度): {e3:.1f}%   ← 端到端实际水平")
print(f"  单目上限 (Nutrition5K): 18.4%")
if e2 < 18.4:
    print(f"  ✓ E2 < 18.4%: 参照物路线在刚体域成立, 值得写进 future work 证据")
else:
    print(f"  ✗ E2 >= 18.4%: 即使刚体+参照物, 密度查表仍不够")
print(f"  注: ECUSTFD 全为刚体/天然食物 (MetaFood3D 分析中密度离散<15% 的子域)")

# ---------- 保存 ----------
S.to_csv(CSV, index=False)
print(f"\n[save] {CSV} ({len(S)} 行)")
