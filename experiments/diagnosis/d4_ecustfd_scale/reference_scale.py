# -*- coding: utf-8 -*-
"""
ECUSTFD 参照物实验: 硬币定标消除尺度歧义后, 重量估计还剩多少误差?

论证链定位:
  - Nutrition5K 对照用 results/estimation/r0/e1_point.json，不用重跑前的数字
  - MetaFood3D: 密度离散 21.8% 是主导 (蓝莓悖论), 但刚体类 <15%
  - 本实验: 刚体类(ECUSTFD 全是) + 硬币参照物(25.0mm 一元硬币)
           → 参照物路线的实际上界测试

实验设计 (全部 LOO/诚实协议):
  E1 (尺度歧义量化):   像素面积 → 硬币定标 → mm; GT 体积 vs "外观面积推断体积"
      对比: 有硬币定标 vs 无定标(假设固定拍摄距离) 的体积误差
  E2 (密度查表 + LOO): GT 体积 × LOO 类别均值密度 → 重量 MAPE (MetaFood3D 同款协议)
  E3 (全链路上界):     硬币定标体积估计(简单几何) × LOO 密度 → 端到端重量误差
      和 R0 的 4 通道集成比，看参照物路线是否低于本次单目点估计
"""
import glob, json, os, re, sys
from pathlib import Path
import numpy as np
import pandas as pd
import xml.etree.ElementTree as ET
from PIL import Image


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

_P = Paths.resolve()
ROOT = str(_P.data_ecustfd)
COIN_MM = 25.0          # 一元硬币直径 mm
CSV = str(_P.d4_csv)

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

def point_errors(pred, truth):
    pred = np.asarray(pred, dtype=float)
    truth = np.asarray(truth, dtype=float)
    abs_err = np.abs(pred - truth)
    return {
        "mape": float(100 * np.mean(abs_err / truth)),
        "wape": float(100 * np.sum(abs_err) / np.sum(np.abs(truth))),
        "mae": float(np.mean(abs_err)),
        "n": int(len(truth)),
    }

# ---------- E2: LOO 密度查表 (MetaFood3D 同款协议) ----------
e2_pred, e2g_pred = [], []
for _, g in GT.iterrows():
    same = GT[GT["cat"] == g["cat"]]
    loo = same[same["inst"] != g["inst"]]
    rho_hat = loo["rho"].mean() if len(loo) else GT["rho"].mean()
    e2_pred.append(g["vol_ml"] * rho_hat)
    e2g_pred.append(g["vol_ml"] * GT["rho"].mean())
e2_block = point_errors(e2_pred, GT["wt_g"])
e2g_block = point_errors(e2g_pred, GT["wt_g"])
e2 = e2_block["mape"]
print(f"\n[E2 GT体积×LOO类别密度] 重量 MAPE: {e2_block['mape']:.1f}%  "
      f"WAPE: {e2_block['wape']:.1f}%  MAE: {e2_block['mae']:.1f}g  (n={e2_block['n']})")
print(f"    全局密度 MAPE: {e2g_block['mape']:.1f}%  WAPE: {e2g_block['wape']:.1f}%")
print("    对照: D1 Oracle-2 见 phase1_results.csv | Nutrition5K 见 results/estimation/r0/e1_point.json")

# ---------- E3: 端到端 (面积 → 体积 → LOO 密度) ----------
# 体积估计: 顶视面积 / k，k 为同类其他实例「面积/体积」的中位数
# 有硬币用定标面积；无硬币用全局中位 mm/px，同一套留一协议
def end_to_end(area_col):
    rows = []
    gt_by_inst = GT.set_index("inst")
    for inst, t in T.iterrows():
        g = gt_by_inst.loc[inst]
        peers = [i for i in T.index if i != inst and gt_by_inst.loc[i, "cat"] == g["cat"]]
        if len(peers) < 1:
            continue
        k_hat = np.median([T.loc[p, area_col] / gt_by_inst.loc[p, "vol_ml"] for p in peers])
        vol_hat = t[area_col] / k_hat
        loo_rho = GT[(GT["cat"] == g["cat"]) & (GT["inst"] != inst)]["rho"].mean()
        rows.append(dict(inst=inst, cat=g["cat"], wt_gt=g["wt_g"], wt_hat=vol_hat * loo_rho,
                         vol_hat=vol_hat, vol_gt=g["vol_ml"]))
    return pd.DataFrame(rows)

E3 = end_to_end("food_mm2_cal")
E3_fix = end_to_end("food_mm2_fix")
e3_block = point_errors(E3["wt_hat"], E3["wt_gt"])
e3_fix_block = point_errors(E3_fix["wt_hat"], E3_fix["wt_gt"])
vol_mape = 100 * (abs(E3["vol_hat"] - E3["vol_gt"]) / E3["vol_gt"]).mean()
vol_mape_fix = 100 * (abs(E3_fix["vol_hat"] - E3_fix["vol_gt"]) / E3_fix["vol_gt"]).mean()
e3 = e3_block["mape"]
print(f"\n[E3 端到端: 硬币定标面积×LOO几何×LOO密度]")
print(f"    体积 MAPE: {vol_mape:.1f}%")
print(f"    重量 MAPE: {e3_block['mape']:.1f}%  WAPE: {e3_block['wape']:.1f}%  MAE: {e3_block['mae']:.1f}g  (n={e3_block['n']})")
print(f"[E3 无硬币: 固定距离面积×同一 LOO 协议]")
print(f"    体积 MAPE: {vol_mape_fix:.1f}%")
print(f"    重量 MAPE: {e3_fix_block['mape']:.1f}%  WAPE: {e3_fix_block['wape']:.1f}%  MAE: {e3_fix_block['mae']:.1f}g  (n={e3_fix_block['n']})")
print(f"\n{'='*60}")
print(f"  [结论判定]")
print(f"  E2 (GT体积+LOO密度)  : {e2:.1f}%   ← 参照物路线的下界(体积完美)")
print(f"  E3 (定标几何+LOO密度): {e3:.1f}%   ← 端到端实际水平")
print(f"  E3 无硬币: {e3_fix_block['mape']:.1f}%")
e1_path = _P.results_estimation / "r0" / "e1_point.json"
if e1_path.is_file():
    e1 = json.loads(e1_path.read_text(encoding="utf-8"))
    mono = float(e1["4ch_ensemble"]["mape_pct"])
    mono100 = float(e1["4ch_ensemble"]["ge_100g"]["mape_pct"])
    print(f"  R0 4通道集成 MAPE: {mono:.1f}% | ≥100g {mono100:.1f}%")
    if e2 < mono100:
        print(f"  E2 {e2:.1f}% < R0 ≥100g {mono100:.1f}%: 真值体积上的密度查表低于本次单目点估计")
    else:
        print(f"  E2 {e2:.1f}% >= R0 ≥100g {mono100:.1f}%")
else:
    print("  没有 R0 点估计文件，不做跨数据集门槛比较")
print(f"  注: ECUSTFD 全为刚体/天然食物 (MetaFood3D 分析中密度离散<15% 的子域)")

# ---------- 保存 ----------
S.to_csv(CSV, index=False)
summary = {
    "e1_mmpp_min": float(S["mmpp"].min()),
    "e1_mmpp_max": float(S["mmpp"].max()),
    "e1_loglog_r_cal": float(r_cal),
    "e1_loglog_r_fix": float(r_fix),
    "e2_loo": e2_block,
    "e2_global": e2g_block,
    "e3_coin": {**e3_block, "volume_mape": float(vol_mape)},
    "e3_no_coin": {**e3_fix_block, "volume_mape": float(vol_mape_fix)},
    "n_images": int(len(S)),
    "n_instances_top": int(T.shape[0]),
}
out = _P.results_diagnosis / "d4_ecustfd" / "summary.json"
out.parent.mkdir(parents=True, exist_ok=True)
out.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
print(f"\n[save] {CSV} ({len(S)} 行)")
print(f"[save] {out}")
