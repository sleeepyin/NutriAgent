#!/usr/bin/env python3
"""Build Nutrition5K overhead RGB-D cache for the E1–E5 rerun.

The lost YOLO crop cache is not in the official archive. This reads
realsense_overhead rgb.png + depth_raw.png from the local tar, resizes the
full overhead frame to 224, and writes a dish-disjoint 80/20 split.
"""
from __future__ import annotations

import io
import json
import shutil
import sys
import tarfile
import time
from pathlib import Path

import numpy as np
from PIL import Image

try:
    _RESAMPLE = Image.Resampling.BILINEAR
except AttributeError:
    _RESAMPLE = Image.BILINEAR
from sklearn.model_selection import train_test_split

ROOT = Path("/home/sdnu/nutriagent")
ARCHIVE = Path("/var/tmp/nutri-downloads/nutrition5k/nutrition5k_dataset.tar.gz")
META_DIR = ROOT / "data" / "nutrition5k" / "raw" / "metadata"
OUT_DIR = ROOT / "data" / "nutrition5k" / "cache"
CSV_PATH = ROOT / "data" / "nutrition5k" / "raw" / "yolo_crops" / "dish_weights_full.csv"
STATUS = ROOT / "logs" / "n5k_r0_status.txt"
WORK = Path("/var/tmp/n5k_build")
MARK = OUT_DIR / "BUILD_COMPLETE.json"
SIZE = 224
MAX_DISHES = 8000
SEED = 42
OVERHEAD = "/imagery/realsense_overhead/"


def status(msg: str) -> None:
    line = time.strftime("%F %T ") + msg
    print(line, flush=True)
    STATUS.parent.mkdir(parents=True, exist_ok=True)
    with STATUS.open("a", encoding="utf-8") as fh:
        fh.write(line + "\n")


def _ingredient_totals(row: list[str]) -> tuple[float, float, float, float] | None:
    rest = row[6:]
    calories = fat = carb = protein = 0.0
    i = 0
    ok = False
    while i + 6 < len(rest):
        try:
            calories += float(rest[i + 3])
            fat += float(rest[i + 4])
            carb += float(rest[i + 5])
            protein += float(rest[i + 6])
            ok = True
        except ValueError:
            break
        i += 7
    if not ok:
        return None
    return calories, fat, carb, protein


def load_metadata() -> dict[str, dict]:
    dishes: dict[str, dict] = {}
    recovered = 0
    for name in ("dish_metadata_cafe1.csv", "dish_metadata_cafe2.csv"):
        path = META_DIR / name
        with path.open(encoding="utf-8", errors="replace") as fh:
            for line in fh:
                row = line.rstrip("\n").split(",")
                if not row or not row[0].startswith("dish_"):
                    continue
                try:
                    calories, mass, fat, carb, protein = (float(v) for v in row[1:6])
                except ValueError:
                    continue
                if mass <= 0:
                    continue
                if calories == 0 and fat == 0 and carb == 0 and protein == 0:
                    summed = _ingredient_totals(row)
                    if summed is not None and any(v > 0 for v in summed):
                        calories, fat, carb, protein = summed
                        recovered += 1
                dishes[row[0]] = {
                    "mass": mass,
                    "nutr": (calories, fat, carb, protein),
                }
    status(f"metadata dishes with mass={len(dishes)} ingredient_fallback={recovered}")
    return dishes


def parse_member(name: str) -> tuple[str, str] | None:
    if OVERHEAD not in name or not name.endswith(".png"):
        return None
    parts = name.split("/")
    if len(parts) < 2:
        return None
    dish, fname = parts[-2], parts[-1]
    if dish.startswith("dish_") and fname in ("rgb.png", "depth_raw.png"):
        return dish, fname
    return None


def discard(handle) -> None:
    if handle is None:
        return
    while handle.read(16 * 1024 * 1024):
        pass


def resize_rgb(blob: bytes) -> np.ndarray:
    image = Image.open(io.BytesIO(blob)).convert("RGB")
    image = image.resize((SIZE, SIZE), _RESAMPLE)
    arr = np.asarray(image, dtype=np.float32) / 255.0
    return np.transpose(arr, (2, 0, 1))


def depth_meters(blob: bytes) -> np.ndarray:
    image = Image.open(io.BytesIO(blob))
    raw = np.array(image)
    values = raw.astype(np.float32)
    # Official RealSense raw depth is uint16 in 1e-4 metre units.
    if values.max() > 255:
        values = values * 1e-4
    else:
        values = values / 255.0
    image = Image.fromarray(values, mode="F").resize((SIZE, SIZE), _RESAMPLE)
    return np.array(image, dtype=np.float32)


def normalize_depth(meters: np.ndarray, lo: float, hi: float) -> np.ndarray:
    span = max(hi - lo, 1e-6)
    out = np.clip((meters - lo) / span, 0.0, 1.0)
    out[meters <= 0] = 0.0
    return out.astype(np.float32)


def lock_scale(buffer: list[dict]) -> tuple[float, float]:
    valid = []
    for item in buffer:
        depth = item["depth_m"]
        kept = depth[depth > 0]
        if kept.size:
            valid.append(kept.reshape(-1))
    if not valid:
        raise RuntimeError("前 32 张深度图没有有效像素")
    stacked = np.concatenate(valid)
    lo, hi = np.percentile(stacked, [1, 99])
    if not np.isfinite(lo) or not np.isfinite(hi) or hi <= lo:
        lo, hi = float(np.min(stacked)), float(np.max(stacked) + 1e-3)
    status(
        f"depth scale locked lo={float(lo):.4f} hi={float(hi):.4f} "
        f"median={float(np.median(stacked)):.4f} n_pixels={stacked.size}"
    )
    return float(lo), float(hi)


def pack(rgb: np.ndarray, depth_m: np.ndarray, lo: float, hi: float) -> np.ndarray:
    depth = normalize_depth(depth_m, lo, hi)
    return np.concatenate([rgb, depth[None, ...]], axis=0).astype(np.float32)


def already_built() -> bool:
    npz = OUT_DIR / "deep_crops_cache.npz"
    if not (MARK.is_file() and npz.is_file()):
        return False
    with np.load(npz) as data:
        ok = {"X_train", "y_train", "X_test", "y_test", "dish_id_train", "dish_id_test"} <= set(data.files)
        ok = ok and data["X_train"].ndim == 4 and data["X_train"].shape[1:] == (4, SIZE, SIZE)
    if ok:
        status(f"cache exists, skip rebuild: {npz}")
    return ok


def write_outputs(meta: list[dict], store: np.memmap, lo: float, hi: float) -> None:
    n = len(meta)
    order = sorted(range(n), key=lambda i: meta[i]["dish_id"])
    ids = np.array([meta[i]["dish_id"] for i in order])
    train_ids, test_ids = train_test_split(ids, test_size=0.2, random_state=SEED)
    train_set, test_set = set(train_ids.tolist()), set(test_ids.tolist())
    train_rows = [i for i in order if meta[i]["dish_id"] in train_set]
    test_rows = [i for i in order if meta[i]["dish_id"] in test_set]
    status(f"split train={len(train_rows)} test={len(test_rows)} seed={SEED}")

    def take(rows: list[int]) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        x = np.stack([np.array(store[i], dtype=np.float32) for i in rows], axis=0)
        y = np.array([meta[i]["mass"] for i in rows], dtype=np.float32)
        nutr = np.array([meta[i]["nutr"] for i in rows], dtype=np.float32)
        density = nutr / y[:, None]
        did = np.array([meta[i]["dish_id"] for i in rows])
        return x, y, nutr, density, did

    x_train, y_train, n_train, d_train, id_train = take(train_rows)
    x_test, y_test, n_test, d_test, id_test = take(test_rows)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    CSV_PATH.parent.mkdir(parents=True, exist_ok=True)
    tmp = WORK / "deep_crops_cache.npz"
    np.savez(
        tmp,
        X_train=x_train,
        y_train=y_train,
        X_test=x_test,
        y_test=y_test,
        dish_id_train=id_train,
        dish_id_test=id_test,
    )
    dest = OUT_DIR / "deep_crops_cache.npz"
    if dest.exists():
        dest.unlink()
    shutil.move(str(tmp), dest)
    np.savez(OUT_DIR / "nutrient_labels.npz", y_train_nutr=n_train, y_test_nutr=n_test)
    np.savez(OUT_DIR / "nutrient_density.npz", y_train_density=d_train, y_test_density=d_test)
    with CSV_PATH.open("w", encoding="utf-8") as fh:
        fh.write("dish_id,total_mass_g\n")
        for i in order:
            fh.write(f"{meta[i]['dish_id']},{meta[i]['mass']:.6f}\n")
    info = {
        "n_total": n,
        "n_train": len(train_rows),
        "n_test": len(test_rows),
        "seed": SEED,
        "test_size": 0.2,
        "image": "full realsense_overhead frame resized to 224",
        "rgb": "[0,1]",
        "depth_unit": "uint16 * 1e-4 m, then global percentile [1,99] -> [0,1]",
        "depth_lo_m": lo,
        "depth_hi_m": hi,
        "mass_g_train_mean": float(y_train.mean()),
        "mass_g_test_mean": float(y_test.mean()),
        "archive": str(ARCHIVE),
    }
    MARK.write_text(json.dumps(info, indent=2) + "\n", encoding="utf-8")
    status(f"wrote {dest} train={info['n_train']} test={info['n_test']}")


def main() -> None:
    if "--metadata-only" in sys.argv:
        load_metadata()
        return
    if already_built():
        return
    if not ARCHIVE.is_file():
        raise SystemExit(f"缺少压缩包: {ARCHIVE}")
    meta_table = load_metadata()
    WORK.mkdir(parents=True, exist_ok=True)
    mmap_path = WORK / "X.mmap"
    if mmap_path.exists():
        mmap_path.unlink()
    store = np.lib.format.open_memmap(
        mmap_path, mode="w+", dtype=np.float32, shape=(MAX_DISHES, 4, SIZE, SIZE)
    )
    pending: dict[str, dict[str, bytes]] = {}
    buffer: list[dict] = []
    meta: list[dict] = []
    lo = hi = None
    seen_bytes = 0
    last_report = 0
    skipped = {"no_mass": 0, "bad_image": 0, "empty_depth": 0}
    t0 = time.time()
    status(f"scanning {ARCHIVE}")
    with tarfile.open(ARCHIVE, "r|gz") as tar:
        for member in tar:
            seen_bytes += member.size
            if seen_bytes - last_report >= 5 * 1024 ** 3:
                last_report = seen_bytes
                status(
                    f"scan {seen_bytes / 1024 ** 3:.1f} GB uncompressed, "
                    f"dishes={len(meta) + len(buffer)} elapsed={time.time() - t0:.0f}s"
                )
            parsed = parse_member(member.name)
            extracted = tar.extractfile(member) if member.isfile() else None
            if parsed is None:
                discard(extracted)
                continue
            dish, kind = parsed
            blob = extracted.read() if extracted is not None else b""
            slot = pending.setdefault(dish, {})
            slot[kind] = blob
            if "rgb.png" not in slot or "depth_raw.png" not in slot:
                continue
            rgb_blob = slot.pop("rgb.png")
            depth_blob = slot.pop("depth_raw.png")
            pending.pop(dish, None)
            info = meta_table.get(dish)
            if info is None:
                skipped["no_mass"] += 1
                continue
            try:
                rgb = resize_rgb(rgb_blob)
                depth_m = depth_meters(depth_blob)
            except Exception as exc:
                skipped["bad_image"] += 1
                status(f"skip {dish}: {type(exc).__name__}: {exc}")
                continue
            if int((depth_m > 0).sum()) < 100:
                skipped["empty_depth"] += 1
                continue
            item = {
                "dish_id": dish,
                "mass": float(info["mass"]),
                "nutr": tuple(float(v) for v in info["nutr"]),
                "rgb": rgb,
                "depth_m": depth_m,
            }
            if lo is None:
                buffer.append(item)
                if len(buffer) >= 32:
                    lo, hi = lock_scale(buffer)
                    for saved in buffer:
                        store[len(meta)] = pack(saved["rgb"], saved["depth_m"], lo, hi)
                        meta.append(
                            {"dish_id": saved["dish_id"], "mass": saved["mass"], "nutr": saved["nutr"]}
                        )
                    buffer.clear()
                    status(f"first dishes stored n={len(meta)}")
                continue
            if len(meta) >= MAX_DISHES:
                raise RuntimeError(f"菜品数超过 {MAX_DISHES}")
            store[len(meta)] = pack(rgb, depth_m, lo, hi)
            meta.append({"dish_id": dish, "mass": item["mass"], "nutr": item["nutr"]})
            if len(meta) % 200 == 0:
                status(f"stored dishes={len(meta)}")
    if lo is None:
        if len(buffer) < 8:
            raise RuntimeError(f"有效俯拍菜品太少: {len(buffer)}")
        lo, hi = lock_scale(buffer)
        for saved in buffer:
            store[len(meta)] = pack(saved["rgb"], saved["depth_m"], lo, hi)
            meta.append({"dish_id": saved["dish_id"], "mass": saved["mass"], "nutr": saved["nutr"]})
    store.flush()
    status(
        f"scan done dishes={len(meta)} pending={len(pending)} "
        f"skipped={skipped} elapsed={time.time() - t0:.0f}s"
    )
    if len(meta) < 500:
        raise RuntimeError(f"有效菜品只有 {len(meta)}，停止写缓存")
    ids = [item["dish_id"] for item in meta]
    if len(ids) != len(set(ids)):
        raise RuntimeError("dish_id 重复")
    write_outputs(meta, store, lo, hi)
    del store
    if mmap_path.exists():
        mmap_path.unlink()


if __name__ == "__main__":
    main()
