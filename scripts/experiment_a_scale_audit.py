#!/usr/bin/env python3
"""Count official Nutrition5k dishes that have both a top-view RGB and a side-view RGB.

The count is read from the local official archive
``/var/tmp/nutri-downloads/nutrition5k/nutrition5k_dataset.tar.gz``.
It does not take a pair count from GitHub issue #12, and it does not treat
the paper's "about 3.5k / 5,006" overhead RGB-D figure as a top/side RGB
pair count.

A dish enters the usable set only when all of the following hold:

1. ``dish_id`` is listed in the archive's ``dish_ids_all.txt``.
2. The archive has ``imagery/realsense_overhead/<dish_id>/rgb.png`` of at
   least 10,000 bytes.
3. The archive has at least one ``camera_{A,B,C,D}.h264`` of at least
   100,000 bytes under ``imagery/side_angles/<dish_id>/``.
4. That side video decodes to at least one RGB frame.
5. Official cafe metadata has a finite mass strictly above 0 g.

The side camera is the first of A, B, C, D that clears the size floor.
The kept frame is the middle decoded frame (or the middle of the first 300
frames when the video is longer). Depth files are counted in an appendix
and are not part of the usable RGB pair rule.
"""

from __future__ import annotations

import hashlib
import io
import json
import multiprocessing
import sys
import tarfile
import time
from collections import defaultdict, deque
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from experiment_a_common import (  # noqa: E402
    ARCHIVE_PATH,
    CAMERA_PRIORITY,
    DEPTH_MIN_BYTES,
    FRAME_CAP,
    JPEG_QUALITY,
    LOCAL_N5K,
    RESULT_DIR,
    RGB_MIN_BYTES,
    SIDE_LONG_SIDE,
    SIDE_MIN_BYTES,
    id_digest,
    load_id_list,
    parse_dish_metadata_line,
    sha256_bytes,
    sha256_file,
    write_json,
)

import av  # noqa: E402
from PIL import Image  # noqa: E402


def _log(message: str, log_path: Path) -> None:
    stamp = time.strftime("%H:%M:%S")
    line = f"[{stamp}] {message}"
    print(line, flush=True)
    with log_path.open("a", encoding="utf-8") as handle:
        handle.write(line + "\n")


def _resize_long_side(image: Image.Image, long_side: int) -> Image.Image:
    width, height = image.size
    longest = max(width, height)
    if longest <= long_side:
        return image
    scale = long_side / float(longest)
    resized = image.resize((max(1, round(width * scale)), max(1, round(height * scale))), Image.Resampling.LANCZOS)
    return resized


def decode_and_save(payload: bytes, dest: str, camera: str) -> dict:
    """Worker entry. Decode one side video and write the middle frame."""
    image, meta = _middle_frame(payload)
    path = Path(dest)
    partial = path.with_suffix(".jpg.partial")
    image.save(partial, format="JPEG", quality=JPEG_QUALITY)
    partial.replace(path)
    meta.update({"camera": camera, "video_bytes": len(payload), "path": _rel(path), "dish_id": path.stem})
    return meta


def _open_h264(payload: bytes):
    container = av.open(io.BytesIO(payload), format="h264")
    stream = container.streams.video[0]
    stream.thread_count = 1
    return container, stream


def _middle_frame(payload: bytes) -> tuple[Image.Image, dict]:
    """Keep the middle frame without resizing every decoded frame."""
    container, stream = _open_h264(payload)
    n_frames = 0
    truncated = False
    try:
        for _frame in container.decode(stream):
            n_frames += 1
            if n_frames >= FRAME_CAP:
                truncated = True
                break
    finally:
        container.close()
    if n_frames == 0:
        raise RuntimeError("side video decoded zero frames")
    kept_index = (n_frames - 1) // 2
    container, stream = _open_h264(payload)
    kept = None
    try:
        for index, frame in enumerate(container.decode(stream)):
            if index == kept_index:
                kept = _resize_long_side(frame.to_image(), SIDE_LONG_SIDE)
                break
    finally:
        container.close()
    if kept is None:
        raise RuntimeError("middle side frame was not recovered")
    meta = {
        "frames_decoded": n_frames,
        "frame_index_kept": kept_index,
        "frame_cap_hit": truncated,
        "saved_width": kept.size[0],
        "saved_height": kept.size[1],
    }
    return kept, meta


def _rel(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(ROOT))
    except ValueError:
        return str(path)


def _primary_reason(dish_id: str, tables: dict) -> str | None:
    if dish_id not in tables["positive_mass"]:
        if dish_id not in tables["known_mass"]:
            return "metadata_mass_missing"
        return "metadata_mass_not_positive"
    if dish_id not in tables["overhead_ok"]:
        if dish_id in tables["overhead_too_small"]:
            return "overhead_rgb_too_small"
        return "overhead_rgb_missing"
    if dish_id not in tables["side_ok"]:
        if dish_id in tables["side_too_small_only"]:
            return "side_video_below_size_floor"
        return "side_video_missing"
    if dish_id not in tables["decoded"]:
        return "side_video_decode_failed"
    return None


def _write_report(summary: dict, path: Path) -> None:
    counts = summary["counts"]
    lines = [
        "# 实验 A 规模核对",
        "",
        "这次核对只回答一个问题：官方 Nutrition5k 归档里，有多少盘同时有顶视 RGB 和侧视 RGB，并且能按 dish_id 对齐。",
        "侧视不是度量双相机。归档没有公布侧视位姿，深度不进入可用配对规则。",
        "",
        "## 用了哪些官方文件",
        "",
        f"- 归档：`{summary['archive']['path']}`",
        f"- 归档大小：{summary['archive']['size_bytes']} 字节",
        f"- 盘号清单：归档内 `nutrition5k_dataset/dish_ids/dish_ids_all.txt`（sha256 `{summary['official_files']['dish_ids_all.txt']['sha256']}`）",
        "- 质量标签：归档内 `metadata/dish_metadata_cafe1.csv` 与 `metadata/dish_metadata_cafe2.csv`",
        "- 顶视 RGB：归档内 `imagery/realsense_overhead/<dish_id>/rgb.png`",
        "- 侧视 RGB：归档内 `imagery/side_angles/<dish_id>/camera_{A,B,C,D}.h264`，再解出一帧",
        "- 官方 RGB 划分只用于后面的对照实验，不用于定义配对规模：`splits/rgb_train_ids.txt`、`splits/rgb_test_ids.txt`",
        "",
        "本地 `data/nutrition5k/` 里的清单和元数据与归档逐字节比对。顶视 png 若与归档 sha256 一致，就沿用本地文件；不一致或不存在时，从归档抽出一份放到本实验结果目录，不覆盖本地原文件。",
        "",
        "## 纳入规则",
        "",
        f"1. dish_id 在官方 `dish_ids_all.txt` 中。官方盘数是 {counts['official_dish_ids']}。",
        f"2. 顶视 `rgb.png` 存在，且大小至少 {RGB_MIN_BYTES} 字节。",
        f"3. 至少一路侧视 `camera_A/B/C/D.h264` 大小至少 {SIDE_MIN_BYTES} 字节。摄像头按 A、B、C、D 的顺序取第一路达标的，不按画面质量挑选。",
        f"4. 该侧视视频能解出至少一帧。保留解码序列的中间帧；超过 {FRAME_CAP} 帧就在上限处停下，保留上限内的中间帧。原始码流没有可用时间戳。",
        "5. 官方元数据里的总质量是有限数值，并且严格大于 0 克。",
        "",
        "深度文件只在附录里计数。GitHub issue #12 的约 3490 不是输入，也不是本计数的目标。论文里约 3.5k / 5,006 盘含顶视 RGB-D，是论文约数，不是俯视 RGB 与侧视 RGB 的可用配对数。",
        "",
        "## 计数",
        "",
        f"- 官方盘号：{counts['official_dish_ids']}",
        f"- 元数据里有质量字段的盘：{counts['metadata_with_mass_field']}",
        f"- 质量大于 0 克：{counts['metadata_mass_positive']}",
        f"- 达标顶视 RGB（官方盘号内）：{counts['official_overhead_rgb']}",
        f"- 达标侧视视频（官方盘号内，至少一路摄像头）：{counts['official_side_video']}",
        f"- 归档文件级同时具备两路、且质量大于 0（尚未要求解码成功）：{counts['archive_pairs_positive_mass']}",
        f"- 侧视解码失败：{counts['side_decode_failed']}",
        f"- **可用配对（两路 RGB 对齐、解码成功、质量大于 0）：{counts['usable_pairs']}**",
        "",
        "可用配对再按官方 RGB 划分切开，供五项对照使用：",
        "",
        f"- 官方 rgb_train：{counts['usable_rgb_train']}",
        f"- 官方 rgb_test：{counts['usable_rgb_test']}",
        f"- 不在官方 RGB 划分里（不进入五项对照，也不拿来调方法）：{counts['usable_outside_rgb_split']}",
        "",
        "互斥的未纳入原因（只统计官方盘号，每盘一个主原因）：",
        "",
    ]
    for reason, count in summary["primary_exclusions"].items():
        lines.append(f"- {reason}：{count}")
    lines.extend(
        [
            "",
            "侧视摄像头选择（可用配对）：",
            "",
        ]
    )
    for camera, count in summary["chosen_camera"].items():
        lines.append(f"- camera_{camera}：{count}")
    depth = summary["depth_appendix"]
    lines.extend(
        [
            "",
            "## 附录：深度（不进入主表）",
            "",
            f"可用 RGB 配对里，同时有达标深度文件的盘：{depth['usable_pairs_with_depth']}。",
            f"其中 `depth_raw.png` ≥ {DEPTH_MIN_BYTES} 字节：{depth['usable_with_depth_raw']}。",
            f"`depth_color.png` ≥ {DEPTH_MIN_BYTES} 字节：{depth['usable_with_depth_color']}。",
            "这些深度没有读进实验，也没有参与配对定义。",
            "",
            "## 本地顶视文件核对",
            "",
            f"- 本地 `realsense_overhead/*/rgb.png`：{summary['local_overhead']['files']}",
            f"- 与归档 sha256 一致并被沿用：{summary['local_overhead']['hash_match']}",
            f"- 因缺失或哈希不一致而从归档另存：{summary['local_overhead']['extracted_from_archive']}",
            "",
            "官方文本文件与本地副本是否逐字节一致：",
            "",
        ]
    )
    for name, info in summary["official_files"].items():
        lines.append(f"- {name}：{'一致' if info['local_sha256_match'] else '不一致'}")
    lines.extend(
        [
            "",
            "非官方盘号（出现在影像路径里，但不在 `dish_ids_all.txt`）不计入可用配对：",
            f"- 仅顶视：{counts['overhead_not_in_official_ids']}",
            f"- 仅侧视：{counts['side_not_in_official_ids']}",
            "",
        ]
    )
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> int:
    if not ARCHIVE_PATH.is_file():
        raise SystemExit(f"official archive is missing: {ARCHIVE_PATH}")
    RESULT_DIR.mkdir(parents=True, exist_ok=True)
    frame_dir = RESULT_DIR / "side_frames"
    overhead_cache = RESULT_DIR / "overhead_rgb_from_archive"
    frame_dir.mkdir(parents=True, exist_ok=True)
    log_path = RESULT_DIR / "scale_audit.log"
    log_path.write_text("", encoding="utf-8")
    started = time.time()
    _log(f"scanning {ARCHIVE_PATH}", log_path)

    official_ids: list[str] = []
    rgb_train: list[str] = []
    rgb_test: list[str] = []
    official_files: dict[str, dict] = {}
    mass_by_id: dict[str, float] = {}
    ingredients_by_id: dict[str, list[dict]] = {}
    duplicate_mass = 0
    metadata_parse_skip = 0
    trailing_field_rows = 0

    side_sizes: dict[str, dict[str, int]] = defaultdict(dict)
    side_small: dict[str, dict[str, int]] = defaultdict(dict)
    other_side_files = 0
    overhead_member: dict[str, tuple[int, str, str]] = {}
    overhead_too_small: dict[str, int] = {}
    overhead_other: dict[str, dict[str, int]] = defaultdict(dict)

    current_dish: str | None = None
    held_priority = 99
    held_camera: str | None = None
    held_bytes: bytes | None = None
    decoded: dict[str, dict] = {}
    decode_fail: dict[str, str] = {}
    frames_written = 0
    official_set_live: set[str] = set()
    finished_side: set[str] = set()
    noncontiguous_side = 0
    inflight: deque = deque()
    context = multiprocessing.get_context("spawn")
    executor = ProcessPoolExecutor(max_workers=32, mp_context=context)

    def collect(future) -> None:
        nonlocal frames_written
        dish_id = future.dish_id
        try:
            meta = future.result()
        except Exception as exc:  # noqa: BLE001 — one bad video must not stop the census
            decode_fail[dish_id] = f"{type(exc).__name__}: {exc}"
            dest = frame_dir / f"{dish_id}.jpg"
            if dest.exists():
                dest.unlink()
            return
        decoded[meta["dish_id"]] = {key: value for key, value in meta.items() if key != "dish_id"}
        frames_written += 1
        if frames_written % 100 == 0:
            _log(f"decoded side frames {frames_written}", log_path)

    def drain(block_all: bool = False, limit: int = 40) -> None:
        while inflight and (block_all or len(inflight) >= limit):
            collect(inflight.popleft())

    def flush_side() -> None:
        nonlocal current_dish, held_priority, held_camera, held_bytes
        dish_id = current_dish
        payload = held_bytes
        camera = held_camera
        current_dish = None
        held_priority = 99
        held_camera = None
        held_bytes = None
        if dish_id is None:
            return
        finished_side.add(dish_id)
        if payload is None or camera is None:
            return
        if dish_id not in official_set_live or not (mass_by_id.get(dish_id, -1.0) > 0):
            return
        dest = frame_dir / f"{dish_id}.jpg"
        future = executor.submit(decode_and_save, payload, str(dest), camera)
        future.dish_id = dish_id
        inflight.append(future)
        drain(False, limit=40)

    def note_text_member(key: str, member: tarfile.TarInfo, payload: bytes, local_path: Path) -> str:
        digest = sha256_bytes(payload)
        local_match = local_path.is_file() and sha256_file(local_path) == digest
        official_files[key] = {
            "archive_member": member.name,
            "size_bytes": len(payload),
            "sha256": digest,
            "local_path": _rel(local_path),
            "local_sha256_match": local_match,
        }
        return payload.decode("utf-8")

    member_count = 0
    with tarfile.open(ARCHIVE_PATH, "r:gz") as archive:
        for member in archive:
            member_count += 1
            if member_count % 2000 == 0:
                _log(
                    f"members {member_count} side_dishes {len(side_sizes)} overhead {len(overhead_member)} decoded {frames_written}",
                    log_path,
                )
            name = member.name
            if not member.isfile():
                continue
            base = name.split("nutrition5k_dataset/", 1)[-1]
            if base == "dish_ids/dish_ids_all.txt":
                text = note_text_member(
                    "dish_ids_all.txt",
                    member,
                    archive.extractfile(member).read(),
                    LOCAL_N5K / "dish_ids" / "dish_ids_all.txt",
                )
                official_ids = load_id_list(text)
                official_set_live.update(official_ids)
                continue
            if base == "dish_ids/splits/rgb_train_ids.txt":
                text = note_text_member(
                    "rgb_train_ids.txt",
                    member,
                    archive.extractfile(member).read(),
                    LOCAL_N5K / "dish_ids" / "splits" / "rgb_train_ids.txt",
                )
                rgb_train = load_id_list(text)
                continue
            if base == "dish_ids/splits/rgb_test_ids.txt":
                text = note_text_member(
                    "rgb_test_ids.txt",
                    member,
                    archive.extractfile(member).read(),
                    LOCAL_N5K / "dish_ids" / "splits" / "rgb_test_ids.txt",
                )
                rgb_test = load_id_list(text)
                continue
            if base in {"metadata/dish_metadata_cafe1.csv", "metadata/dish_metadata_cafe2.csv"}:
                payload = archive.extractfile(member).read()
                text = note_text_member(base, member, payload, LOCAL_N5K / "metadata" / Path(base).name)
                for line in text.splitlines():
                    parsed = parse_dish_metadata_line(line)
                    if parsed is None:
                        if line.strip():
                            metadata_parse_skip += 1
                        continue
                    if parsed["trailing_fields"]:
                        trailing_field_rows += 1
                    dish_id = parsed["dish_id"]
                    if dish_id in mass_by_id:
                        duplicate_mass += 1
                        continue
                    mass_by_id[dish_id] = float(parsed["mass_g"])
                    ingredients_by_id[dish_id] = parsed["ingredients"]
                continue
            if "/imagery/side_angles/" in name and name.endswith(".h264"):
                if not official_set_live or not mass_by_id:
                    raise SystemExit("side imagery appeared before official ids and metadata")
                parent = Path(name).parent.name
                camera = Path(name).stem.replace("camera_", "")
                if camera not in CAMERA_PRIORITY or not parent.startswith("dish_"):
                    other_side_files += 1
                    continue
                if parent != current_dish:
                    if parent in finished_side:
                        noncontiguous_side += 1
                    flush_side()
                    current_dish = parent
                if member.size < SIDE_MIN_BYTES:
                    side_small[parent][camera] = int(member.size)
                    continue
                side_sizes[parent][camera] = int(member.size)
                priority = CAMERA_PRIORITY.index(camera)
                if priority < held_priority:
                    want_frame = parent in official_set_live and mass_by_id.get(parent, -1.0) > 0
                    if want_frame:
                        extracted = archive.extractfile(member)
                        held_bytes = b"" if extracted is None else extracted.read()
                    else:
                        held_bytes = None
                    held_camera = camera
                    held_priority = priority
                continue
            if "/imagery/realsense_overhead/" in name:
                if not official_set_live:
                    raise SystemExit("overhead imagery appeared before official ids")
                flush_side()
                dish_id = Path(name).parent.name
                filename = Path(name).name
                if not dish_id.startswith("dish_"):
                    continue
                if filename == "rgb.png":
                    if member.size < RGB_MIN_BYTES:
                        overhead_too_small[dish_id] = int(member.size)
                        continue
                    payload = archive.extractfile(member).read()
                    digest = sha256_bytes(payload)
                    local_path = LOCAL_N5K / "imagery" / "realsense_overhead" / dish_id / "rgb.png"
                    if local_path.is_file() and sha256_file(local_path) == digest:
                        stored = _rel(local_path)
                    else:
                        dest = overhead_cache / f"{dish_id}.png"
                        dest.parent.mkdir(parents=True, exist_ok=True)
                        partial = dest.with_suffix(".png.partial")
                        partial.write_bytes(payload)
                        partial.replace(dest)
                        stored = _rel(dest)
                    overhead_member[dish_id] = (len(payload), digest, stored)
                    continue
                overhead_other[dish_id][filename] = int(member.size)
    flush_side()
    drain(True)
    executor.shutdown(wait=True)
    if noncontiguous_side:
        raise SystemExit(f"side-view directories were not contiguous: {noncontiguous_side}")
    _log(f"scan done members={member_count} elapsed_s={time.time() - started:.1f}", log_path)

    official_set = set(official_ids)
    if len(official_set) != len(official_ids):
        raise SystemExit("official dish_ids_all.txt contains duplicate ids")
    train_set = set(rgb_train)
    test_set = set(rgb_test)
    if train_set & test_set:
        raise SystemExit("official rgb_train and rgb_test overlap")

    positive_mass = {dish_id for dish_id, mass in mass_by_id.items() if mass > 0}
    overhead_ok = set(overhead_member)
    side_ok = set(side_sizes)
    side_too_small_only = {dish_id for dish_id in side_small if dish_id not in side_ok}
    tables = {
        "positive_mass": positive_mass,
        "known_mass": set(mass_by_id),
        "overhead_ok": overhead_ok,
        "overhead_too_small": set(overhead_too_small),
        "side_ok": side_ok,
        "side_too_small_only": side_too_small_only,
        "decoded": set(decoded),
    }

    usable: list[str] = []
    exclusions: list[dict] = []
    for dish_id in official_ids:
        reason = _primary_reason(dish_id, tables)
        if reason is None:
            usable.append(dish_id)
        else:
            exclusions.append({"dish_id": dish_id, "reason": reason})

    usable_set = set(usable)
    # Drop side frames that failed a later rule (no overhead, non-official, bad mass).
    removed_frames = 0
    for path in frame_dir.glob("*.jpg"):
        if path.stem not in usable_set:
            path.unlink()
            removed_frames += 1

    def split_of(dish_id: str) -> str:
        if dish_id in test_set:
            return "rgb_test"
        if dish_id in train_set:
            return "rgb_train"
        return "outside_rgb_split"

    manifests = {"rgb_train": [], "rgb_test": [], "outside_rgb_split": []}
    chosen_camera: dict[str, int] = defaultdict(int)
    depth_raw = 0
    depth_color = 0
    with_depth = 0
    local_match = 0
    extracted = 0
    for dish_id in usable:
        info = decoded[dish_id]
        chosen_camera[info["camera"]] += 1
        size_bytes, _digest, stored = overhead_member[dish_id]
        if stored.startswith("results/experiment_a_n5k_dual_view/overhead_rgb_from_archive/"):
            extracted += 1
        else:
            local_match += 1
        other = overhead_other.get(dish_id, {})
        has_raw = other.get("depth_raw.png", 0) >= DEPTH_MIN_BYTES
        has_color = other.get("depth_color.png", 0) >= DEPTH_MIN_BYTES
        if has_raw:
            depth_raw += 1
        if has_color:
            depth_color += 1
        if has_raw or has_color:
            with_depth += 1
        row = {
            "dish_id": dish_id,
            "mass_g": mass_by_id[dish_id],
            "overhead_rgb": stored,
            "overhead_rgb_bytes": size_bytes,
            "side_frame": info["path"],
            "side_camera": info["camera"],
            "side_video_bytes": info["video_bytes"],
            "side_frames_decoded": info["frames_decoded"],
            "side_frame_index_kept": info["frame_index_kept"],
            "side_frame_cap_hit": info["frame_cap_hit"],
            "side_saved_width": info["saved_width"],
            "side_saved_height": info["saved_height"],
        }
        group = split_of(dish_id)
        if group == "rgb_train":
            row["ingredients"] = ingredients_by_id.get(dish_id, [])
        manifests[group].append(row)

    for group, rows in manifests.items():
        out = RESULT_DIR / f"manifest_{group}.jsonl"
        with out.open("w", encoding="utf-8") as handle:
            for row in rows:
                handle.write(json.dumps(row, ensure_ascii=False) + "\n")

    exclusion_counts: dict[str, int] = defaultdict(int)
    for row in exclusions:
        exclusion_counts[row["reason"]] += 1
    exclusion_path = RESULT_DIR / "scale_exclusions.jsonl"
    with exclusion_path.open("w", encoding="utf-8") as handle:
        for row in exclusions:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")

    archive_pairs = [
        dish_id
        for dish_id in official_ids
        if dish_id in positive_mass and dish_id in overhead_ok and dish_id in side_ok
    ]
    local_overhead_files = list((LOCAL_N5K / "imagery" / "realsense_overhead").glob("*/rgb.png"))
    summary = {
        "archive": {
            "path": str(ARCHIVE_PATH),
            "size_bytes": ARCHIVE_PATH.stat().st_size,
            "members_visited": member_count,
        },
        "rules": {
            "rgb_min_bytes": RGB_MIN_BYTES,
            "side_min_bytes": SIDE_MIN_BYTES,
            "depth_min_bytes": DEPTH_MIN_BYTES,
            "camera_priority": list(CAMERA_PRIORITY),
            "frame_rule": "middle decoded frame; stop at FRAME_CAP and keep the middle of that prefix",
            "frame_cap": FRAME_CAP,
            "side_long_side": SIDE_LONG_SIDE,
            "jpeg_quality": JPEG_QUALITY,
            "mass_rule": "finite metadata mass strictly greater than 0 g",
            "alignment": "exact dish_id directory name",
            "depth_in_pair_rule": False,
            "github_issue_12_not_used": True,
            "paper_3_5k_not_used_as_pair_count": True,
        },
        "official_files": official_files,
        "counts": {
            "official_dish_ids": len(official_ids),
            "official_id_digest": id_digest(official_ids),
            "rgb_train_ids": len(rgb_train),
            "rgb_test_ids": len(rgb_test),
            "metadata_with_mass_field": len(mass_by_id),
            "metadata_mass_positive": len(positive_mass & official_set),
            "metadata_duplicate_ids_skipped": duplicate_mass,
            "metadata_rows_skipped": metadata_parse_skip,
            "metadata_rows_with_trailing_fields": trailing_field_rows,
            "official_overhead_rgb": len(overhead_ok & official_set),
            "overhead_not_in_official_ids": len(overhead_ok - official_set),
            "official_side_video": len(side_ok & official_set),
            "side_not_in_official_ids": len(side_ok - official_set),
            "other_side_files_ignored": other_side_files,
            "archive_pairs_positive_mass": len(archive_pairs),
            "side_decode_failed": len(decode_fail),
            "side_frames_removed_after_rules": removed_frames,
            "usable_pairs": len(usable),
            "usable_id_digest": id_digest(usable),
            "usable_rgb_train": len(manifests["rgb_train"]),
            "usable_rgb_test": len(manifests["rgb_test"]),
            "usable_outside_rgb_split": len(manifests["outside_rgb_split"]),
        },
        "primary_exclusions": dict(exclusion_counts),
        "chosen_camera": dict(chosen_camera),
        "depth_appendix": {
            "usable_pairs_with_depth": with_depth,
            "usable_with_depth_raw": depth_raw,
            "usable_with_depth_color": depth_color,
            "note": "Depth is not an input to the five-way comparison.",
        },
        "local_overhead": {
            "files": len(local_overhead_files),
            "hash_match": local_match,
            "extracted_from_archive": extracted,
        },
        "elapsed_s": round(time.time() - started, 1),
    }
    if len(decode_fail) <= 30:
        summary["decode_failures"] = decode_fail
    else:
        failure_path = RESULT_DIR / "side_decode_failures.json"
        write_json(failure_path, decode_fail)
        summary["decode_failures_path"] = _rel(failure_path)

    write_json(RESULT_DIR / "scale_audit.json", summary)
    _write_report(summary, RESULT_DIR / "scale_audit.md")
    _log(
        "usable_pairs "
        f"{summary['counts']['usable_pairs']} "
        f"train {summary['counts']['usable_rgb_train']} "
        f"test {summary['counts']['usable_rgb_test']}",
        log_path,
    )
    print(json.dumps(summary["counts"], ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
