"""Ask the local Qwen3.8-27B once per hold-out image and write cache/vlm/.

The process is resumable: a finished JSON is not requested again. Closing the
terminal is safe only when this file is started from the detached runner.
"""
from __future__ import annotations

import argparse
import io
import sys
import tarfile
import threading
import time
from pathlib import Path

from PIL import Image, ImageOps

ROOT = Path(__file__).resolve().parents[2]
for entry in (str(ROOT), str(ROOT / "src")):
    if entry not in sys.path:
        sys.path.insert(0, entry)

from experiments.vlm.holdout import build_holdout
from experiments.vlm.parse import parse_prediction, self_check as parse_check, usable
from experiments.vlm.prompt import PROMPT, PROMPT_SHA256
from experiments.vlm.store import (
    attempts,
    error_file,
    is_done,
    write_error,
    write_manifest,
    write_record,
)
from mmcfa.llm.qwen_local import chat_vision_json
from mmcfa.paths import Paths
from mmcfa.runtime import Heartbeat, atomic_write_json, log, read_json

ARCHIVE = Path("/var/tmp/nutri-downloads/nutrition5k/nutrition5k_dataset.tar.gz")
IMAGE_DIR = Path("/var/tmp/nutri-vlm-images/n5k")
MAX_CONTENT_ATTEMPTS = 3
MAX_TRANSPORT_ATTEMPTS = 40


def read_json_error_kind(dataset: str, sample_id: str) -> str:
    obj = read_json(error_file(dataset, sample_id), {})
    if not isinstance(obj, dict) or obj.get("prompt_sha256") != PROMPT_SHA256:
        return ""
    return str(obj.get("kind") or "")


def _transport(message: str) -> bool:
    text = message.lower()
    keys = ("timed out", "timeout", "connection", "http 5", "http 429", "temporarily", "unreachable")
    return any(key in text for key in keys)


def _permanent(dataset: str, sample_id: str) -> bool:
    kind = read_json_error_kind(dataset, sample_id)
    n = attempts(dataset, sample_id)
    if kind == "transport":
        return n >= MAX_TRANSPORT_ATTEMPTS
    if kind == "content":
        return n >= MAX_CONTENT_ATTEMPTS
    return False


def jpeg_bytes(path: Path, max_side: int = 1024) -> tuple[bytes, int, int]:
    image = ImageOps.exif_transpose(Image.open(path)).convert("RGB")
    width, height = image.size
    scale = min(1.0, max_side / max(width, height))
    sent = image
    if scale < 1:
        sent = image.resize((max(1, int(width * scale)), max(1, int(height * scale))), Image.BILINEAR)
    buf = io.BytesIO()
    sent.save(buf, format="JPEG", quality=85)
    return buf.getvalue(), width, height


def _discard(handle) -> None:
    if handle is None:
        return
    while handle.read(16 * 1024 * 1024):
        pass


def extract_n5k(dish_ids: set[str]) -> dict[str, Path]:
    IMAGE_DIR.mkdir(parents=True, exist_ok=True)
    def usable(path: Path) -> bool:
        if not path.is_file() or path.stat().st_size < 8000:
            return False
        try:
            with Image.open(path) as image:
                return max(image.size) >= 320
        except Exception:
            return False

    found = {}
    for dish in dish_ids:
        path = IMAGE_DIR / f"{dish}.jpg"
        if usable(path):
            found[dish] = path
    need = set(dish_ids) - set(found)
    if not need:
        log(f"N5K images already on disk: {len(found)}")
        return found
    if not ARCHIVE.is_file():
        log(f"N5K archive missing: {ARCHIVE}")
        return found
    log(f"N5K extract start need={len(need)} archive={ARCHIVE}")
    examples = []
    t0 = time.time()
    seen = 0
    with tarfile.open(ARCHIVE, "r|gz") as tar:
        for member in tar:
            if not member.isfile():
                continue
            handle = tar.extractfile(member)
            name = member.name
            if "rgb" in name.lower() and len(examples) < 8:
                examples.append(name)
            dish = name.split("/")[-2] if "/" in name else ""
            wanted = (
                "realsense_overhead" in name
                and name.endswith("rgb.png")
                and dish in need
            )
            if not wanted:
                _discard(handle)
                seen += 1
                if seen % 5000 == 0:
                    log(f"N5K scan members={seen} found={len(found)} elapsed={time.time() - t0:.0f}s")
                continue
            blob = handle.read() if handle is not None else b""
            try:
                image = Image.open(io.BytesIO(blob)).convert("RGB")
                dest = IMAGE_DIR / f"{dish}.jpg"
                image.save(dest, format="JPEG", quality=90)
                found[dish] = dest
                need.discard(dish)
            except Exception as exc:  # noqa: BLE001
                log(f"N5K decode failed {dish}: {exc}")
            if not need:
                break
            if len(found) % 25 == 0:
                log(f"N5K extracted {len(found)}/{len(dish_ids)} elapsed={time.time() - t0:.0f}s")
    log(f"N5K extract done found={len(found)} missing={len(need)} examples={examples[:4]}")
    return found


def export_cache224(missing: set[str], index_of: dict[str, int]) -> dict[str, Path]:
    import numpy as np

    if not missing:
        return {}
    crops = np.load(Paths.resolve().expect("deep_crops"))
    images = crops["X_test"]
    out = {}
    IMAGE_DIR.mkdir(parents=True, exist_ok=True)
    for dish in missing:
        index = index_of.get(dish)
        if index is None:
            continue
        rgb = np.transpose(images[index, :3], (1, 2, 0))
        frame = Image.fromarray((rgb * 255).clip(0, 255).astype("uint8"))
        dest = IMAGE_DIR / f"{dish}.jpg"
        frame.save(dest, format="JPEG", quality=90)
        out[dish] = dest
    log(f"N5K 224 fallback wrote {len(out)}")
    return out


def _query(row: dict) -> str:
    dataset, sample_id = row["dataset"], row["sample_id"]
    if is_done(dataset, sample_id):
        return "skip"
    if _permanent(dataset, sample_id):
        return "giveup"
    image = Path(row.get("image") or "")
    if not image.is_file():
        write_error(dataset, sample_id, "missing image", kind="content")
        return "missing"
    blob, width, height = jpeg_bytes(image)
    last = "unknown"
    for try_i in range(2):
        try:
            raw = chat_vision_json(PROMPT, blob, timeout=300, max_tokens=800 if try_i == 0 else 1200)
            finish = raw.pop("_finish_reason", None)
            parsed = parse_prediction(raw, width, height)
            if not usable(parsed):
                raise RuntimeError(f"incomplete JSON finish={finish}")
            parsed["meta"] = {
                "dataset": dataset,
                "sample_id": sample_id,
                "image": str(image),
                "image_source": row.get("image_source", ""),
                "width": width,
                "height": height,
                "model": "qwen3.8-27b",
                "prompt_sha256": PROMPT_SHA256,
                "finish_reason": finish,
            }
            write_record(dataset, sample_id, parsed)
            return "ok"
        except Exception as exc:  # noqa: BLE001
            last = f"{type(exc).__name__}: {exc}"
            time.sleep(3 * (try_i + 1))
    write_error(dataset, sample_id, last, kind="transport" if _transport(last) else "content")
    return "fail"


def _run_rows(rows: list[dict], heartbeat: Heartbeat, done_base: int, total: int) -> dict:
    counts = {"ok": 0, "skip": 0, "fail": 0, "giveup": 0, "missing": 0}
    t0 = time.time()
    for i, row in enumerate(rows, start=1):
        status = _query(row)
        counts[status] = counts.get(status, 0) + 1
        finished = done_base + i
        if i == 1 or i % 10 == 0 or i == len(rows):
            elapsed = time.time() - t0
            rate = i / max(elapsed, 1e-6)
            eta = (len(rows) - i) / max(rate, 1e-6)
            log(
                f"VLM {finished}/{total} last={status} {row['dataset']}:{row['sample_id']} "
                f"ok={counts['ok']} skip={counts['skip']} fail={counts['fail']} eta={eta/60:.1f}min"
            )
            heartbeat.touch("vlm", f"{finished}/{total}")
            atomic_write_json(
                Paths.resolve().vlm_cache / "progress.json",
                {"done_passes": finished, "total": total, "counts": counts, "eta_min": round(eta / 60, 1)},
            )
    return counts


def main(argv=None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--dataset", default="")
    args = parser.parse_args(argv)
    parse_check()
    paths = Paths.resolve()
    paths.vlm_cache.mkdir(parents=True, exist_ok=True)
    rows = build_holdout(paths)
    counts_by = {}
    for row in rows:
        counts_by[row["dataset"]] = counts_by.get(row["dataset"], 0) + 1
    log(f"hold-out counts {counts_by}")
    if not args.limit:
        for dataset in ("ecustfd", "mf3d", "nvreal"):
            chosen = [row for row in rows if row["dataset"] == dataset]
            missing = sum(1 for row in chosen if not row.get("image"))
            if chosen and missing / len(chosen) > 0.05:
                log(f"{dataset} missing images {missing}/{len(chosen)}")
                return 1
    n5k_rows = [row for row in rows if row["dataset"] == "n5k"]
    dish_ids = {row["sample_id"] for row in n5k_rows}
    index_of = {row["sample_id"]: int(row["n5k_index"]) for row in n5k_rows}
    others = [row for row in rows if row["dataset"] != "n5k"]
    if args.dataset:
        others = [row for row in others if row["dataset"] == args.dataset]
        n5k_rows = n5k_rows if args.dataset in ("", "n5k") else []
    if args.limit:
        others = others[: args.limit]
        if args.dataset == "n5k":
            n5k_rows = n5k_rows[: args.limit]
        elif args.dataset == "":
            n5k_rows = []
    holder: dict = {}

    def _extract() -> None:
        try:
            holder["found"] = extract_n5k(dish_ids)
        except Exception as exc:  # noqa: BLE001
            holder["error"] = str(exc)
            log(f"N5K extract failed: {exc}")

    thread = None
    if n5k_rows:
        thread = threading.Thread(target=_extract, name="n5k-extract")
        thread.start()
    heartbeat = Heartbeat(paths.vlm_cache / "heartbeat.json", interval=30)
    heartbeat.start()
    total = len(others) + len(n5k_rows)
    write_manifest(rows)
    part = _run_rows(others, heartbeat, 0, total) if others else {}
    if thread is not None:
        thread.join()
        found = holder.get("found") or {}
        missing = {row["sample_id"] for row in n5k_rows} - set(found)
        if missing:
            fallback = export_cache224(missing, index_of)
            for dish, path in fallback.items():
                found[dish] = path
            for row in rows:
                if row["dataset"] == "n5k" and row["sample_id"] in fallback:
                    row["image_source"] = "cache224"
        for row in rows:
            if row["dataset"] != "n5k" or row["sample_id"] not in found:
                continue
            row["image"] = str(found[row["sample_id"]])
            if row.get("image_source") != "cache224":
                row["image_source"] = "overhead"
    write_manifest(rows)
    part2 = _run_rows(n5k_rows, heartbeat, len(others), total) if n5k_rows else {}
    heartbeat.stop()
    if args.limit or args.dataset:
        log(f"partial run others={part} n5k={part2}")
        return 0
    retryable = 0
    permanent = 0
    for row in rows:
        if is_done(row["dataset"], row["sample_id"]):
            continue
        if _permanent(row["dataset"], row["sample_id"]):
            permanent += 1
        else:
            retryable += 1
    log(f"VLM pass retryable={retryable} permanent={permanent}")
    if retryable:
        return 2
    missing_images = sum(1 for row in rows if not row.get("image"))
    if missing_images > max(5, int(0.05 * len(rows))):
        log(f"too many images missing: {missing_images}")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
