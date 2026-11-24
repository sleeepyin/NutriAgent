"""Train the Phase 1 teachers and four students, then read the holdout once."""

from __future__ import annotations

import gc
import logging
import os
import subprocess
import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from digi_nutri.datasets import PrivilegedTrainDataset, collate_privileged
from digi_nutri.engine import freeze_module, seed_all, train_student, train_teacher
from digi_nutri.evaluate import run_official_holdout
from digi_nutri.fingerprint import code_fingerprint
from digi_nutri.holdout_guard import HoldoutLocked
from digi_nutri.index_data import (
    build_records,
    index_depth_files,
    split_hashes,
    train_target_stats,
    train_vocab,
)
from digi_nutri.jsonio import read_json, write_json
from digi_nutri.losses import LossWeights
from digi_nutri.metadata import load_metadata
from digi_nutri.models.geometry_teacher import build_geometry_teacher
from digi_nutri.models.ingredient_teacher import build_ingredient_teacher
from digi_nutri.models.student import build_student
from digi_nutri.protocol import (
    DEPTH_ROOT,
    FROZEN_CONFIG,
    HOLDOUT_METRICS,
    LIVE_METADATA,
    LIVE_RGB_ROOT,
    LIVE_SPLITS,
    OUTPUTS,
    RESULTS,
    WORK_ROOT,
    Phase1Config,
    config_dict,
    needs_aux,
    needs_geo,
    needs_sem,
)
from digi_nutri.report import render_not_trained
from digi_nutri.text import build_text_cache, load_text_cache, save_text_cache

LOGGER = logging.getLogger("digi_nutri")


def gpus_look_free() -> tuple[bool, str]:
    try:
        text = subprocess.check_output(
            [
                "nvidia-smi",
                "--query-gpu=index,memory.used,utilization.gpu",
                "--format=csv,noheader",
            ],
            text=True,
        ).strip()
    except (subprocess.CalledProcessError, FileNotFoundError) as exc:
        return False, str(exc)
    free = True
    for line in text.splitlines():
        parts = [part.strip() for part in line.split(",")]
        used = int(parts[1].split()[0])
        if used > 500:
            free = False
    return free, text


def depth_roots() -> list[Path]:
    roots = [DEPTH_ROOT]
    pointer = DEPTH_ROOT / "DEPTH_ROOT.txt"
    if pointer.is_file():
        roots.append(Path(pointer.read_text(encoding="utf-8").strip()))
    roots.append(LIVE_RGB_ROOT)
    return roots


def wait_for_depth(timeout_s: int = 6 * 3600) -> None:
    done = DEPTH_ROOT / "EXTRACT_DONE"
    reuse = DEPTH_ROOT / "DEPTH_ROOT.txt"
    deadline = time.time() + timeout_s
    while time.time() < deadline:
        if done.is_file() or reuse.is_file():
            LOGGER.info("depth marker ready done=%s reuse=%s", done.is_file(), reuse.is_file())
            return
        LOGGER.info("waiting for official depth_raw.png extraction")
        time.sleep(30)
    raise SystemExit("official depth was not ready before the wait expired")


def assert_official_split(train_report: dict, test_report: dict, overlap: set) -> None:
    """Train ids with no local RGB or depth are excluded. They are not fatal.

    A missing test dish, an overlapping id, or a test count other than 507 stops
    both training and the one-shot eval.
    """

    if overlap or test_report["missing"]:
        sample = list(test_report["missing"])[:12]
        raise SystemExit(
            "official test split is incomplete or overlapping: "
            f"{sample} overlap={sorted(overlap)[:8]}"
        )
    if int(test_report["n_kept"]) != 507:
        raise SystemExit(f"kept {test_report['n_kept']} test dishes, expected 507")


def exclusion_record(dish_id: str, index_reason: str) -> dict:
    """Verify one excluded id on disk. Do not infer the reason from the count alone."""

    rgb_path = LIVE_RGB_ROOT / dish_id / "rgb.png"
    rgb_present = rgb_path.is_file()
    candidates = (
        DEPTH_ROOT / "nutrition5k_dataset" / "imagery" / "realsense_overhead" / dish_id / "depth_raw.png",
        DEPTH_ROOT / dish_id / "depth_raw.png",
        LIVE_RGB_ROOT / dish_id / "depth_raw.png",
    )
    depth_files = []
    for path in candidates:
        if path.exists():
            depth_files.append({"path": str(path), "bytes": path.stat().st_size})
    nonempty = [row for row in depth_files if row["bytes"] > 0]
    if not rgb_present:
        reason = "no_rgb"
    elif depth_files and not nonempty:
        reason = "zero_byte_depth"
    elif not depth_files:
        reason = "no_depth"
    else:
        reason = index_reason
    return {
        "dish_id": dish_id,
        "reason": reason,
        "index_reason": index_reason,
        "rgb_present": rgb_present,
        "depth_files": depth_files,
    }


def _loader(dataset, batch_size: int, seed: int, workers: int, device: torch.device):
    generator = torch.Generator()
    generator.manual_seed(seed)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        generator=generator,
        num_workers=workers,
        pin_memory=device.type == "cuda",
        collate_fn=collate_privileged,
    )


def _probe_batch_size(device: torch.device, image_size: int, hidden: int) -> tuple[int, str]:
    if device.type != "cuda":
        return 8, "cpu run uses batch 8"
    for size in (32, 16, 8):
        model = None
        optimizer = None
        try:
            torch.cuda.empty_cache()
            model = build_geometry_teacher(
                pretrained=True, use_checkpoint=True, hidden=hidden
            ).to(device)
            rgb = torch.randn(size, 3, image_size, image_size, device=device)
            depth = torch.randn(size, 1, image_size, image_size, device=device)
            optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
            loss = model(rgb, depth)["nutrients"].float().square().mean()
            loss.backward()
            optimizer.step()
            note = f"geometry teacher probe fit batch {size}"
            LOGGER.info(note)
            return size, note
        except RuntimeError as exc:
            if "out of memory" not in str(exc).lower():
                raise
            LOGGER.info("geometry teacher probe OOM at batch %s", size)
        finally:
            del optimizer, model
            gc.collect()
            torch.cuda.empty_cache()
    raise SystemExit("geometry teacher did not fit batch 8")


def _write_not_trained(reason: str) -> None:
    RESULTS.mkdir(parents=True, exist_ok=True)
    (RESULTS / "REPORT.md").write_text(render_not_trained(reason), encoding="utf-8")
    LOGGER.error(reason)


def _complete(path: Path, fingerprint: str, epochs: int) -> bool:
    side = path.with_suffix(".json")
    if not path.is_file() or not side.is_file():
        return False
    meta = read_json(side)
    return bool(
        meta.get("complete")
        and meta.get("fingerprint") == fingerprint
        and int(meta.get("epochs", -1)) == int(epochs)
    )


def _save(model, path: Path, fingerprint: str, epochs: int) -> None:
    temporary = path.with_suffix(".pt.tmp")
    torch.save(model.state_dict(), temporary)
    os.replace(temporary, path)
    write_json(
        path.with_suffix(".json"),
        {"complete": True, "fingerprint": fingerprint, "epochs": epochs},
    )


def _discard_checkpoints() -> None:
    OUTPUTS.mkdir(parents=True, exist_ok=True)
    for path in OUTPUTS.glob("*"):
        if path.is_file():
            path.unlink()


def _curves(name: str, history: list[dict], params: int) -> None:
    path = RESULTS / "train_curves.json"
    payload = read_json(path) if path.is_file() else {}
    payload[name] = {"trainable_params": params, "history": history}
    write_json(path, payload)


def _params(module) -> int:
    return sum(parameter.numel() for parameter in module.parameters() if parameter.requires_grad)


def run_train(cpu: bool = False, epochs_teachers: int | None = None, epochs_students: int | None = None, batch_size: int | None = None, workers: int | None = None) -> None:
    if WORK_ROOT.resolve() != Path("/home/sdnu/nutriagent-digi").resolve():
        raise SystemExit("refusing to train outside the digi worktree")
    if HOLDOUT_METRICS.exists():
        raise HoldoutLocked("holdout metrics already exist; refusing to train or evaluate again")
    RESULTS.mkdir(parents=True, exist_ok=True)
    OUTPUTS.mkdir(parents=True, exist_ok=True)
    free, gpu_text = gpus_look_free()
    LOGGER.info("nvidia-smi\n%s", gpu_text)
    if not cpu and not free:
        _write_not_trained(
            "Training has not started. nvidia-smi showed a GPU above 500 MiB, so no Phase 1 job was launched.\n"
            + gpu_text
        )
        return
    device = torch.device("cuda" if torch.cuda.is_available() and not cpu else "cpu")
    if device.type == "cuda":
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        torch.backends.cudnn.benchmark = True
        torch.set_float32_matmul_precision("high")
    wait_for_depth()
    free, gpu_text = gpus_look_free()
    LOGGER.info("nvidia-smi after depth\n%s", gpu_text)
    if not cpu and not free:
        _write_not_trained(
            "Training has not started. nvidia-smi showed a GPU above 500 MiB after depth was ready, so no Phase 1 job was launched.\n"
            + gpu_text
        )
        return
    metadata = load_metadata(LIVE_METADATA)
    roots = depth_roots()
    depth_index = index_depth_files(roots)
    reshoot = WORK_ROOT / "data" / "reshoot_groups.tsv"
    reshoot = reshoot if reshoot.is_file() else None
    train_records, train_report = build_records(
        LIVE_SPLITS, "train", LIVE_METADATA, LIVE_RGB_ROOT, roots,
        metadata=metadata, depth_index=depth_index, reshoot_tsv=reshoot,
    )
    test_records, test_report = build_records(
        LIVE_SPLITS, "test", LIVE_METADATA, LIVE_RGB_ROOT, roots,
        metadata=metadata, depth_index=depth_index, reshoot_tsv=reshoot,
    )
    overlap = {row["dish_id"] for row in train_records} & {row["dish_id"] for row in test_records}
    index_report = {
        "train": {key: train_report[key] for key in ("n_ids", "n_kept", "missing_counts")},
        "test": {key: test_report[key] for key in ("n_ids", "n_kept", "missing_counts")},
        "n_depth_files": len(depth_index),
        "depth_roots": [str(path) for path in roots],
        "reshoot_tsv": str(reshoot) if reshoot else None,
        "plate_groups_test": len({row["plate_group"] for row in test_records}),
        "overlap": sorted(overlap),
        "train_absent_ids": [item["dish_id"] for item in train_report["missing"]],
        "test_absent_ids": [item["dish_id"] for item in test_report["missing"]],
        "excluded": [
            exclusion_record(item["dish_id"], item["reason"])
            for item in train_report["missing"] + test_report["missing"]
        ],
    }
    write_json(RESULTS / "data_index_report.json", index_report)
    assert_official_split(train_report, test_report, overlap)
    if train_report["missing"]:
        LOGGER.info(
            "train ids with no local official rgb and depth are excluded: %s",
            index_report["train_absent_ids"],
        )
    stats = train_target_stats(train_records)
    vocab = train_vocab(train_records)
    hashes = split_hashes(LIVE_SPLITS)
    fingerprint = code_fingerprint()
    if FROZEN_CONFIG.exists():
        frozen = read_json(FROZEN_CONFIG)
        if frozen.get("code_fingerprint") != fingerprint:
            LOGGER.info("training code changed before the holdout; resetting checkpoints")
            _discard_checkpoints()
            frozen = None
        elif list(frozen["vocab"]) != vocab or not np.allclose(
            frozen["target_mean"], stats["mean"], rtol=0, atol=1e-6
        ):
            raise SystemExit("frozen config does not match the recomputed train statistics")
    else:
        frozen = None
    if frozen is None:
        cfg = Phase1Config()
        if epochs_teachers is not None:
            cfg.epochs_teachers = epochs_teachers
        if epochs_students is not None:
            cfg.epochs_students = epochs_students
        if workers is not None:
            cfg.num_workers = workers
        if cfg.gram_weight:
            raise SystemExit("the Phase 1 ingredient teacher is names-only")
        if device.type == "cuda" and batch_size is None:
            chosen, note = _probe_batch_size(device, cfg.image_size, cfg.hidden)
        else:
            chosen = batch_size or (8 if device.type == "cpu" else cfg.batch_size)
            note = f"batch {chosen} selected without a new probe"
        cfg.batch_size = chosen
        cfg.batch_size_note = note
        cfg.target_mean = stats["mean"]
        cfg.target_std = stats["std"]
        cfg.vocab = vocab
        cfg.split_train_sha256 = hashes["train"]
        cfg.split_test_sha256 = hashes["test"]
        cfg.code_fingerprint = fingerprint
        cfg.n_train = len(train_records)
        cfg.n_test = len(test_records)
        frozen = config_dict(cfg)
        write_json(FROZEN_CONFIG, frozen)
    text_path = OUTPUTS / "text_cache.pt"
    names = [name for record in train_records + test_records for name in record["names"]]
    if not text_path.is_file():
        text_device = "cuda" if device.type == "cuda" else "cpu"
        cache = build_text_cache(names, frozen["text_model"], text_device)
        save_text_cache(text_path, cache)
        gc.collect()
    cache = load_text_cache(text_path)
    text_dim = int(next(iter(cache.values())).shape[-1])
    if text_dim != 512:
        raise SystemExit(f"expected CLIP width 512, got {text_dim}")
    mean = torch.tensor(frozen["target_mean"], dtype=torch.float32)
    std = torch.tensor(frozen["target_std"], dtype=torch.float32)
    weights = LossWeights(frozen["lambda_geo"], frozen["lambda_sem"], frozen["lambda_aux"])
    vocab_index = {name: index for index, name in enumerate(frozen["vocab"])}
    workers_n = int(frozen["num_workers"])
    batch = int(frozen["batch_size"])
    seed = int(frozen["seed"])

    def train_rows(include_depth: bool, include_ingredients: bool):
        dataset = PrivilegedTrainDataset(
            train_records,
            int(frozen["image_size"]),
            depth_fill=frozen["depth_fill"],
            include_depth=include_depth,
            include_ingredients=include_ingredients,
            text_cache=cache if include_ingredients else None,
            vocab_index=vocab_index if include_ingredients else None,
        )
        return _loader(dataset, batch, seed, workers_n, device)

    geo_path = OUTPUTS / "geometry_teacher.pt"
    if not _complete(geo_path, fingerprint, frozen["epochs_teachers"]):
        seed_all(seed)
        model = build_geometry_teacher(True, True, int(frozen["hidden"])).to(device)
        history = train_teacher(
            model, train_rows(True, False), device, int(frozen["epochs_teachers"]),
            float(frozen["lr"]), float(frozen["weight_decay"]), float(frozen["eta_min"]),
            mean, std, "geometry", float(frozen["pmae_eps"]),
        )
        _curves("geometry_teacher", history, _params(model))
        _save(model, geo_path, fingerprint, int(frozen["epochs_teachers"]))
        del model
        gc.collect()
        if device.type == "cuda":
            torch.cuda.empty_cache()
    ing_path = OUTPUTS / "ingredient_teacher.pt"
    if not _complete(ing_path, fingerprint, frozen["epochs_teachers"]):
        seed_all(seed)
        model = build_ingredient_teacher(True, True, int(frozen["hidden"]), text_dim).to(device)
        history = train_teacher(
            model, train_rows(False, True), device, int(frozen["epochs_teachers"]),
            float(frozen["lr"]), float(frozen["weight_decay"]), float(frozen["eta_min"]),
            mean, std, "ingredient", float(frozen["pmae_eps"]),
        )
        _curves("ingredient_teacher", history, _params(model))
        _save(model, ing_path, fingerprint, int(frozen["epochs_teachers"]))
        del model
        gc.collect()
        if device.type == "cuda":
            torch.cuda.empty_cache()

    def load_geo():
        model = build_geometry_teacher(False, False, int(frozen["hidden"]))
        model.load_state_dict(torch.load(geo_path, map_location="cpu", weights_only=True))
        model.to(device)
        freeze_module(model)
        return model

    def load_ing():
        model = build_ingredient_teacher(False, False, int(frozen["hidden"]), text_dim)
        model.load_state_dict(torch.load(ing_path, map_location="cpu", weights_only=True))
        model.to(device)
        freeze_module(model)
        return model

    for variant in ("b1", "b2a", "b2b", "b3"):
        path = OUTPUTS / f"{variant}.pt"
        if _complete(path, fingerprint, frozen["epochs_students"]):
            LOGGER.info("reusing complete checkpoint %s", path.name)
            continue
        seed_all(seed)
        n_labels = len(vocab_index) if needs_aux(variant) else 0
        model = build_student(
            n_labels=n_labels,
            pretrained=True,
            use_checkpoint=True,
            hidden=int(frozen["hidden"]),
            fusion=frozen["fusion"],
        ).to(device)
        geo = load_geo() if needs_geo(variant) else None
        ing = load_ing() if needs_sem(variant) else None
        history = train_student(
            model,
            train_rows(needs_geo(variant), needs_sem(variant)),
            device,
            int(frozen["epochs_students"]),
            float(frozen["lr"]),
            float(frozen["weight_decay"]),
            float(frozen["eta_min"]),
            mean,
            std,
            variant,
            weights,
            int(frozen["warmup_epochs"]),
            int(frozen["ramp_epochs"]),
            float(frozen["pmae_eps"]),
            geo_teacher=geo,
            ing_teacher=ing,
        )
        _curves(variant, history, _params(model))
        _save(model, path, fingerprint, int(frozen["epochs_students"]))
        del model, geo, ing
        gc.collect()
        if device.type == "cuda":
            torch.cuda.empty_cache()
    run_official_holdout(test_records, frozen, OUTPUTS, cache, device)


def run_eval(cpu: bool = False) -> None:
    """Score the official holdout once from finished checkpoints."""

    if HOLDOUT_METRICS.exists():
        raise HoldoutLocked(
            f"refusing a second official holdout eval; metrics already exist at {HOLDOUT_METRICS}"
        )
    if not FROZEN_CONFIG.is_file():
        raise SystemExit("frozen config is missing; train before eval")
    frozen = read_json(FROZEN_CONFIG)
    device = torch.device("cuda" if torch.cuda.is_available() and not cpu else "cpu")
    metadata = load_metadata(LIVE_METADATA)
    roots = depth_roots()
    depth_index = index_depth_files(roots)
    reshoot = WORK_ROOT / "data" / "reshoot_groups.tsv"
    reshoot = reshoot if reshoot.is_file() else None
    _train_records, train_report = build_records(
        LIVE_SPLITS, "train", LIVE_METADATA, LIVE_RGB_ROOT, roots,
        metadata=metadata, depth_index=depth_index, reshoot_tsv=reshoot,
    )
    test_records, test_report = build_records(
        LIVE_SPLITS, "test", LIVE_METADATA, LIVE_RGB_ROOT, roots,
        metadata=metadata, depth_index=depth_index, reshoot_tsv=reshoot,
    )
    overlap = {row["dish_id"] for row in _train_records} & {row["dish_id"] for row in test_records}
    assert_official_split(train_report, test_report, overlap)
    if train_report["missing"]:
        LOGGER.info(
            "eval keeps the train exclusions: %s",
            [item["dish_id"] for item in train_report["missing"]],
        )
    cache = load_text_cache(OUTPUTS / "text_cache.pt")
    run_official_holdout(test_records, frozen, OUTPUTS, cache, device)
