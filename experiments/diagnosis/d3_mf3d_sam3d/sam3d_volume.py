#!/usr/bin/env python3
"""Oracle-6: SAM 3D Objects 零样本单图体积估计 vs GT Volume
输入: Blender 渲染 (622 物体, RGBA 自带 mask, 取中间帧=正面视角)
输出: 体积 MAPE + log-bias + 全局尺度校正后 MAPE (诊断) + 分层报告
运行: conda activate sam3d && python mf3d_oracle6_sam3d.py [--limit N] [--thresh 0.5]
断点续跑: 结果逐行追加到 mf3d_oracle6_results.csv, 重跑自动跳过已完成物体"""
import os, sys, csv, math, time, argparse, glob
import numpy as np
from PIL import Image
import torch

# ---- SAM 3D 环境 ----
def _boot():
    from pathlib import Path
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
_SAM = _P.sam3d_code
sys.path.insert(0, str(_SAM))
sys.path.insert(0, str(_SAM / "notebook"))
os.environ.setdefault("CUDA_HOME", os.environ.get("CONDA_PREFIX", "/opt/miniconda3/envs/sam3d"))
os.environ["LIDRA_SKIP_INIT"] = "true"
os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")  # huggingface.co 被墙, MoGe 等运行时模型走镜像
if not _SAM.is_dir():
    raise SystemExit(f"SAM 3D 代码不在 {_SAM}。把代码放到该目录，或设置 SAM3D_CODE。")
os.chdir(_SAM)

from inference import Inference  # noqa: E402

RENDER = str(_P.expect("mf3d_render"))
RESULTS = str(_P.d3 / "oracle6_results.csv")
LABELS = str(_P.d1 / "metafood3d_labels_clean.csv")

ap = argparse.ArgumentParser()
ap.add_argument("--limit", type=int, default=0, help="只跑前 N 个物体 (pilot)")
ap.add_argument("--sample", type=int, default=0, help="分层抽样 N 个物体 (按类别配额, 0=全部)")
ap.add_argument("--thresh", type=float, default=0.5, help="高斯密度场阈值")
ap.add_argument("--seed", type=int, default=42)
args = ap.parse_args()

# ---------------- gaussian splat → 体积 ----------------
@torch.no_grad()
def gaussian_volume(gs, thresh=0.5, max_voxels=4_000_000):
    """密度场体素化: 对每个高斯在其 3σ 盒内贡献 o·N(μ,Σ), 阈值计数.
    返回 (体积_原生单位, 单位体素边长_估计)"""
    xyz = gs.get_xyz.float()           # 属性而非方法 (官方 notebook L222 同款)
    sc = gs.get_scaling.float().clamp(min=1e-6)  # (N,3) σ
    op = gs.get_opacity.float().squeeze(-1).clamp(0, 1)  # (N,)
    lo = (xyz - 3 * sc).min(0).values
    hi = (xyz + 3 * sc).max(0).values
    diag = (hi - lo).max().item()
    # 体素边长: 控制总格点数 <= max_voxels
    n_target = int(round(max_voxels ** (1 / 3)))
    voxel = max(diag / n_target, sc.median().item() * 0.5)
    n = torch.ceil((hi - lo) / voxel).long() + 1
    n = torch.clamp(n, max=n_target)
    field = torch.zeros(int(n[0]), int(n[1]), int(n[2]), device=xyz.device)
    # 逐高斯 splat (分块)
    CH = 4096
    for i in range(0, xyz.shape[0], CH):
        mu, s, o = xyz[i:i+CH], sc[i:i+CH], op[i:i+CH]
        # 每个高斯覆盖的体素盒: ±2σ
        r = 2.0
        ilo = torch.floor((mu - r * s - lo) / voxel).long().clamp(min=0)
        ihi = torch.ceil((mu + r * s - lo) / voxel).long()
        ihi[:, 0] = ihi[:, 0].clamp(max=n[0]-1); ihi[:, 1] = ihi[:, 1].clamp(max=n[1]-1); ihi[:, 2] = ihi[:, 2].clamp(max=n[2]-1)
        for bi in range(mu.shape[0]):
            x0, y0, z0 = ilo[bi].tolist(); x1, y1, z1 = ihi[bi].tolist()
            if x1 < x0 or y1 < y0 or z1 < z0: continue
            xs = torch.arange(x0, x1+1, device=xyz.device).view(-1,1,1)
            ys = torch.arange(y0, y1+1, device=xyz.device).view(1,-1,1)
            zs = torch.arange(z0, z1+1, device=xyz.device).view(1,1,-1)
            vx = lo[0] + xs*voxel - mu[bi,0]; vy = lo[1] + ys*voxel - mu[bi,1]; vz = lo[2] + zs*voxel - mu[bi,2]
            d2 = (vx/s[bi,0])**2 + (vy/s[bi,1])**2 + (vz/s[bi,2])**2
            field[x0:x1+1, y0:y1+1, z0:z1+1] += o[bi] * torch.exp(-0.5 * d2)
    count = (field > thresh).sum().item()
    return count * voxel**3, voxel

# ---------------- 数据索引 ----------------
rows = list(csv.DictReader(open(LABELS)))

def find_frame(cat, obj):
    cdir = os.path.join(RENDER, cat)
    if not os.path.isdir(cdir): return None
    for d in os.listdir(cdir):
        if d.lower() != obj.lower(): continue
        for sub in os.listdir(os.path.join(cdir, d)):
            odir = os.path.join(cdir, d, sub, "Original")
            if os.path.isdir(odir):
                fs = sorted(f for f in os.listdir(odir) if f.endswith(".png"))
                if fs:
                    # 中间帧 ≈ 正面视角
                    return os.path.join(odir, fs[len(fs)//2])
    return None

todo = []
done_keys = set()
if os.path.exists(RESULTS):
    for r in csv.DictReader(open(RESULTS)):
        done_keys.add((r["cat"], r["obj"]))

for r in rows:
    key = (r["cat"], r["obj"])
    if key in done_keys: continue
    f = find_frame(r["cat"], r["obj"])
    if f: todo.append((r, f))

if args.limit > 0: todo = todo[:args.limit]

# 分层抽样: 选实例数最多的 top 类别, 每类抽 k 个 —— 保证类间 AND 类内两个分析都有样本
# (核心判据是"类间 r 高 / 类内 r≈0", 每类必须有多实例)
if args.sample > 0 and len(todo) > args.sample:
    import random as _rnd
    _rnd.seed(args.seed)
    by_cat = {}
    for item in todo:
        by_cat.setdefault(item[0]["cat"], []).append(item)
    # 类别按实例数降序, 只留实例 >=3 的类别 (类内分析最低要求)
    big_cats = sorted([c for c in by_cat if len(by_cat[c]) >= 3],
                      key=lambda c: -len(by_cat[c]))
    k = 4                                  # 每类抽 4 个实例
    n_cats = min(len(big_cats), max(1, args.sample // k))
    picked = []
    for cat in big_cats[:n_cats]:
        picked.extend(_rnd.sample(by_cat[cat], min(k, len(by_cat[cat]))))
    todo = sorted(picked, key=lambda x: (x[0]["cat"], x[0]["obj"]))
    print(f"[oracle6] 分层抽样 {len(todo)} 物体 / {n_cats} 类 x {k} 实例 (seed={args.seed})")

print(f"[oracle6] 待跑 {len(todo)} 物体 | 阈值 {args.thresh} | 已完成 {len(done_keys)}")

# ---------------- 推理 ----------------
t_start = time.time()
if not os.path.exists(RESULTS):
    with open(RESULTS, "w", newline="") as f:
        csv.writer(f).writerow(["cat", "obj", "v_gt_cm3", "v_est_native", "s_l2c", "v_est_cam",
                                "n_gauss", "voxel", "sec"])

model = Inference("checkpoints/hf/pipeline.yaml", compile=False)

for idx, (r, frame) in enumerate(todo):
    t0 = time.time()
    try:
        rgba = Image.open(frame).convert("RGBA")
        a = np.array(rgba)
        # Inference.__call__ 期望 numpy: image HxWx3 uint8, mask bool(0/1) — 官方 load_mask 同款, ×255 在内部做
        mask_np = a[..., 3] > 127
        rgb_np = a[..., :3].astype(np.uint8)
        out = model(rgb_np, mask_np, seed=args.seed)
        gs = out["gs"]
        v_est, voxel = gaussian_volume(gs, thresh=args.thresh)
        # 潜在空间→相机空间的每物体尺度 (官方 notebook L267/L278: scale_l2c=output["scale"], 高斯×s)
        # refine_scale 保证三轴相等 (各向同性) ⇒ 体积 ×s³
        try:
            s_l2c = float(torch.as_tensor(out["scale"]).reshape(-1)[0].item())
        except Exception:
            s_l2c = float("nan")
        v_cam = v_est * s_l2c ** 3 if s_l2c == s_l2c else float("nan")
        v_gt = float(r["volume_cm3"])
        sec = time.time() - t0
        with open(RESULTS, "a", newline="") as f:
            csv.writer(f).writerow([r["cat"], r["obj"], v_gt, f"{v_est:.6g}",
                                    f"{s_l2c:.6g}", f"{v_cam:.6g}",
                                    int(gs.get_xyz.shape[0]), f"{voxel:.6g}", f"{sec:.1f}"])
        if (idx+1) % 10 == 0 or idx == len(todo)-1:
            el = time.time() - t_start
            print(f"  [{idx+1}/{len(todo)}] {r['cat']}/{r['obj']} v_lat={v_est:.4g} s={s_l2c:.4g} "
                  f"v_cam={v_cam:.4g} v_gt={v_gt:.4g} ratio_cam={v_cam/v_gt:.3f} "
                  f"({sec:.0f}s, 平均 {el/(idx+1):.0f}s/obj)")
        # 显存监控: 峰值(PyTorch 记账) / 已分配 / 预留(含缓存块), 单物体间对比即可验证稳定性
        print(f"    [vram] peak={torch.cuda.max_memory_allocated()/1e9:.2f}GB "
              f"alloc={torch.cuda.memory_allocated()/1e9:.2f}GB "
              f"reserved={torch.cuda.memory_reserved()/1e9:.2f}GB")
        torch.cuda.reset_peak_memory_stats()
        del out, gs
        torch.cuda.empty_cache()
    except torch.cuda.OutOfMemoryError:
        print(f"  [{idx+1}/{len(todo)}] OOM! {r['cat']}/{r['obj']} — 跳过, 清显存")
        torch.cuda.empty_cache()
        time.sleep(2)
    except Exception as e:
        print(f"  [{idx+1}/{len(todo)}] FAIL {r['cat']}/{r['obj']}: {type(e).__name__}: {str(e)[:120]}")

# ---------------- 分析 ----------------
if os.path.exists(RESULTS):
    data = list(csv.DictReader(open(RESULTS)))
    data = [d for d in data if d.get("v_est_native") and float(d["v_est_native"]) > 0]
    if len(data) >= 5:
        v_gt = np.array([float(d["v_gt_cm3"]) for d in data])
        v_est = np.array([float(d["v_est_native"]) for d in data])
        ratio = v_est / v_gt
        mape = 100 * np.mean(np.abs(ratio - 1))
        bias = 100 * np.mean(np.log(ratio))
        # 全局尺度校正 (诊断: 形状质量 vs 尺度先验质量)
        s = np.exp(np.mean(np.log(ratio)))
        mape_scaled = 100 * np.mean(np.abs(ratio / s - 1))
        print("\n" + "=" * 70)
        print(f"[Oracle-6 报告] n={len(data)} (阈值={args.thresh})")
        print(f"  [潜在空间] raw MAPE : {mape:.1f}%  (规范盒输出, 无绝对尺度, 仅作基准)")
        print(f"  [潜在空间] 校正后  : {mape_scaled:.1f}%  | 中位 ratio {np.median(ratio):.4g}")
        # ---- 相机空间: v_cam = v_latent × s_l2c³ (官方 scale_l2c) ----
        cam = [d for d in data if d.get("s_l2c") and d.get("v_est_cam")
               and float(d["s_l2c"]) > 0 and float(d["v_est_cam"]) > 0]
        if len(cam) >= 5:
            vg = np.array([float(d["v_gt_cm3"]) for d in cam])
            vc = np.array([float(d["v_est_cam"]) for d in cam])
            rc = vc / vg
            mape_c = 100 * np.mean(np.abs(rc - 1))
            s_fit = np.exp(np.mean(np.log(rc)))
            mape_cs = 100 * np.mean(np.abs(rc / s_fit - 1))
            r_ll = float(np.corrcoef(np.log(vc), np.log(vg))[0, 1])
            print(f"  [相机空间] metric MAPE : {mape_c:.1f}%   ← 主指标 (s_l2c³ 修正后)")
            print(f"  [相机空间] 中位 ratio  : {np.median(rc):.4g}  (≈1e-6 ⇒ 相机单位是米)")
            print(f"  [相机空间] log-log r   : {r_ll:.3f}   ← 尺度信息是否单调存在")
            print(f"  [相机空间] 全局校准后  : {mape_cs:.1f}%   ← 相对尺度质量 (单位无关)")
            # ---- 核心判据: 类间 vs 类内分解 (尺度先验是"类别级"还是"实例级") ----
            # 类内 r: 每类内部 log v_cam vs log v_gt 的相关性 (类体积跨度不足时记 None)
            # 类间 r: 类均值 log v_cam vs 类均值 log v_gt 的相关性
            byc = {}
            for d in cam:
                byc.setdefault(d["cat"], []).append(d)
            within_rs = []
            for c, ds in byc.items():
                if len(ds) < 3: continue
                g = np.array([float(x["v_gt_cm3"]) for x in ds])
                e = np.array([float(x["v_est_cam"]) for x in ds])
                if g.min() > 0 and g.max()/g.min() > 1.3:   # 类内体积跨度足够才有意义
                    if np.std(np.log(e)) > 0 and np.std(np.log(g)) > 0:
                        within_rs.append(np.corrcoef(np.log(e), np.log(g))[0, 1])
            if len(byc) >= 5:
                mg = np.array([np.mean([float(x["v_gt_cm3"]) for x in ds]) for ds in byc.values()])
                me = np.array([np.mean([float(x["v_est_cam"]) for x in ds]) for ds in byc.values()])
                r_between = float(np.corrcoef(np.log(mg), np.log(me))[0, 1])
                print(f"\n  [尺度先验粒度分解] (核心判据)")
                print(f"    类间 r (类均值)   : {r_between:.3f}   ← '杏仁小碗大'的类别级知识")
                if within_rs:
                    print(f"    类内 r (实例级,n={len(within_rs)}类): {np.mean(within_rs):+.3f} (±{np.std(within_rs):.3f})   ← 能否区分类内实例大小")
                else:
                    print(f"    类内 r : 无满足条件的类别 (体积跨度不足)")
            print(f"  参照: TSDF 多视角 ~11% | Oracle-2 查表 25% | 单目上限 18.4%")
