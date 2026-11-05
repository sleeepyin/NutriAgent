#!/usr/bin/env python3
"""Train the P0b RGB Mass Estimator and score the rgb_train hold-out once.

Backbone choice is a fixed fit-internal probe (sha256 residue 1). The winner
is then trained on the whole rgb_train fit pool. Hold-out labels are read
only after the checkpoint is written. rgb_test labels are not read. Depth is
not read. No RL.

    CUDA_VISIBLE_DEVICES=1 /var/tmp/nutri-venv/bin/python -u scripts/mass_estimator_train.py
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np  # noqa: E402
from PIL import Image, ImageFile  # noqa: E402

from vdr.tools.mass_estimator import (  # noqa: E402
    BACKBONE_ORDER,
    DEFAULT_CHECKPOINT,
    PROTOCOL_ID,
    TARGET_NAMES,
    MassEstimatorError,
    build_mass_net,
    clear_model_cache,
    estimate_mass,
    eval_transform,
    iter_rgb_train_dishes,
    laplace_nll,
    partition_rgb_train,
    predict_original,
    save_checkpoint,
    select_backbone,
    target_mean_std,
    train_transform,
)

ImageFile.LOAD_TRUNCATED_IMAGES = True

# Fixed before any probe. Not searched on hold-out or rgb_test.
PROBE_EPOCHS = 3
TRAIN_EPOCHS = 12
BATCH_SIZE = 16
LR_BACKBONE = 3e-5
LR_HEAD = 1e-3
WEIGHT_DECAY = 1e-2
SEED = 0
GRAD_CLIP = 1.0
# Fit-set sanity only. A broken head is far above this. Not a hold-out rule.
FIT_MASS_MAE_FACTOR = 2.0

FROZEN = (
    ROOT / "results" / "n5k_rgb_test_overhead_b0_b1.json",
    ROOT / "results" / "n5k_rgb_test_overhead_b0_b1.jsonl",
    ROOT / "results" / "n5k_subset_b0_b1.json",
    ROOT / "results" / "n5k_subset_b0_b1.jsonl",
    ROOT / "results" / "n5k_subset_b0_b1_n50_seed42.json",
    ROOT / "results" / "n5k_subset_b0_b1_n50_seed42.jsonl",
    ROOT / "results" / "weak_steps" / "rgb_train_fit_all.jsonl",
    ROOT / "results" / "weak_steps" / "rgb_train_holdout_all.jsonl",
    ROOT / "results" / "weak_steps" / "rgb_train_fit_all.summary.json",
    ROOT / "results" / "weak_steps" / "rgb_train_holdout_all.summary.json",
)
FROZEN_PREFIXES = {
    "n5k_rgb_test_overhead_b0_b1.jsonl": "a4dfdd82e3a595f4",
    "rgb_train_fit_all.jsonl": "ed47f9239bfecb3c",
}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _frozen_digests() -> dict[str, str]:
    digests = {}
    for path in FROZEN:
        if not path.is_file():
            raise MassEstimatorError(f"missing frozen file {path}")
        digest = _sha256(path)
        expected = FROZEN_PREFIXES.get(path.name)
        if expected is not None and not digest.startswith(expected):
            raise MassEstimatorError(f"{path.name} sha256 {digest[:16]} != {expected}")
        digests[path.name] = digest
    return digests


def _set_seed(seed: int) -> None:
    import torch

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _device():
    import torch

    if not torch.cuda.is_available():
        raise MassEstimatorError("training expects CUDA; refusing a silent CPU run")
    return torch.device("cuda")


def _loader(rows: list, mean: np.ndarray, std: np.ndarray, *, train: bool):
    import torch
    from torch.utils.data import DataLoader, Dataset

    transform = train_transform() if train else eval_transform()

    class DishSet(Dataset):
        def __len__(self) -> int:
            return len(rows)

        def __getitem__(self, index: int):
            row = rows[index]
            with Image.open(row.image) as handle:
                image = handle.convert("RGB")
            features = transform(image)
            target = (row.targets() - mean) / std
            return features, torch.from_numpy(target.astype(np.float32))

    generator = torch.Generator()
    generator.manual_seed(SEED)
    return DataLoader(
        DishSet(),
        batch_size=BATCH_SIZE,
        shuffle=train,
        num_workers=0,
        pin_memory=True,
        generator=generator if train else None,
    )


def _optimizer(model):
    import torch

    head = list(model.head.parameters())
    backbone = [param for name, param in model.named_parameters() if not name.startswith("head.")]
    return torch.optim.AdamW(
        [
            {"params": backbone, "lr": LR_BACKBONE},
            {"params": head, "lr": LR_HEAD},
        ],
        weight_decay=WEIGHT_DECAY,
    )


def _train_one(
    backbone: str,
    train_rows: list,
    mean: np.ndarray,
    std: np.ndarray,
    epochs: int,
    device,
) -> tuple[object, list[float]]:
    import torch

    _set_seed(SEED)
    model = build_mass_net(backbone, pretrained=True).to(device)
    optimizer = _optimizer(model)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    scaler = torch.amp.GradScaler("cuda")
    loader = _loader(train_rows, mean, std, train=True)
    losses: list[float] = []
    for epoch in range(epochs):
        model.train()
        total = 0.0
        seen = 0
        for features, target in loader:
            features = features.to(device, non_blocking=True)
            target = target.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast("cuda"):
                pred_mean, pred_scale = model(features)
            loss = laplace_nll(pred_mean.float(), pred_scale.float(), target.float())
            if not torch.isfinite(loss):
                raise MassEstimatorError(f"{backbone} epoch {epoch} produced a non-finite loss")
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
            scaler.step(optimizer)
            scaler.update()
            total += float(loss.detach()) * int(features.shape[0])
            seen += int(features.shape[0])
        scheduler.step()
        mean_loss = total / max(seen, 1)
        losses.append(mean_loss)
        print(
            f"{backbone} epoch {epoch + 1}/{epochs} train_nll={mean_loss:.4f}",
            flush=True,
        )
    model.eval()
    return model, losses


def _predict_split(model, rows: list, mean: np.ndarray, std: np.ndarray, device) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    import torch

    loader = _loader(rows, mean, std, train=False)
    means: list[np.ndarray] = []
    scales: list[np.ndarray] = []
    model.eval()
    for features, _target in loader:
        pred_mean, pred_scale = predict_original(model, features, mean, std)
        means.append(pred_mean)
        scales.append(pred_scale)
    target = np.stack([row.targets() for row in rows], axis=0)
    if not means:
        raise MassEstimatorError("empty prediction split")
    # DataLoader does not shuffle eval, so row order matches the dataset order.
    del torch
    return np.concatenate(means, axis=0), np.concatenate(scales, axis=0), target


def _mae_block(pred: np.ndarray, target: np.ndarray) -> dict[str, dict[str, float]]:
    point = np.maximum(pred, 0.0)
    error = np.abs(point - target)
    block = {}
    for index, name in enumerate(TARGET_NAMES):
        block[name] = {
            "mae": round(float(error[:, index].mean()), 4),
            "median_ae": round(float(np.median(error[:, index])), 4),
        }
    return block


def _coverage(mean: np.ndarray, scale: np.ndarray, target: np.ndarray) -> dict[str, float]:
    from vdr.tools.mass_estimator import LAPLACE_Z95

    width = LAPLACE_Z95 * np.maximum(scale, 0.0)
    low = np.maximum(mean - width, 0.0)
    high = np.maximum(mean + width, low)
    inside = (target >= low) & (target <= high)
    return {name: round(float(inside[:, index].mean()), 4) for index, name in enumerate(TARGET_NAMES)}


def _write(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def _report(path: Path, lines: list[str]) -> None:
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _hyper_meta(backbone: str) -> dict:
    return {
        "protocol": PROTOCOL_ID,
        "task": "P0b Mass Estimator",
        "rgb_only": True,
        "depth_used": False,
        "uses_rgb_test_gt": False,
        "holdout_used_for_selection": False,
        "rl": False,
        "legacy_weights_loaded": False,
        "volume_tool": False,
        "backbone": backbone,
        "backbone_candidates": list(BACKBONE_ORDER),
        "pretrained": "ImageNet-1K V1",
        "targets": list(TARGET_NAMES),
        "loss": "laplace_nll on fit-standardized targets",
        "sigma": "laplace_scale in original units",
        "interval": "central 95% Laplace, low clipped at 0",
        "confidence": "1 / (1 + sigma_mass_g / 100)",
        "probe_epochs": PROBE_EPOCHS,
        "train_epochs": TRAIN_EPOCHS,
        "batch_size": BATCH_SIZE,
        "lr_backbone": LR_BACKBONE,
        "lr_head": LR_HEAD,
        "weight_decay": WEIGHT_DECAY,
        "seed": SEED,
        "grad_clip": GRAD_CLIP,
        "image_size": 224,
        "eval_resize": 256,
        "train_pool": "rgb_train fit (sha256 % 10 != 0) with overhead rgb.png",
        "probe_val": "fit residue 1 only",
        "eval_pool": "rgb_train hold-out (sha256 % 10 == 0) with overhead rgb.png",
        "checkpoint_rule": "last train epoch; not chosen on hold-out",
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Train the RGB-only P0b Mass Estimator")
    parser.add_argument("--self-check", action="store_true")
    args = parser.parse_args()
    if args.self_check:
        from vdr.tools.mass_estimator import run_self_check

        run_self_check()
        print("mass estimator self-check ok")
        return 0

    import torch

    torch.backends.cudnn.benchmark = True
    device = _device()
    out = DEFAULT_CHECKPOINT
    started = time.perf_counter()
    frozen_before = _frozen_digests()
    rows = iter_rgb_train_dishes()
    probe_train, probe_val, fit, holdout = partition_rgb_train(rows)
    print(
        f"rgb_only fit={len(fit)} probe_train={len(probe_train)} "
        f"probe_val={len(probe_val)} holdout={len(holdout)} device={torch.cuda.get_device_name(device)}",
        flush=True,
    )

    probe_rows = []
    for backbone in BACKBONE_ORDER:
        mean, std = target_mean_std(probe_train)
        model, losses = _train_one(backbone, probe_train, mean, std, PROBE_EPOCHS, device)
        pred, _scale, target = _predict_split(model, probe_val, mean, std, device)
        table = _mae_block(pred, target)
        probe_rows.append(
            {
                "backbone": backbone,
                "mass_mae": table["mass_g"]["mae"],
                "energy_mae": table["energy_kcal"]["mae"],
                "protein_mae": table["protein_g"]["mae"],
                "carb_mae": table["carb_g"]["mae"],
                "fat_mae": table["fat_g"]["mae"],
                "n_probe_train": len(probe_train),
                "n_probe_val": len(probe_val),
                "epochs": PROBE_EPOCHS,
                "train_nll": [round(loss, 4) for loss in losses],
            }
        )
        print(
            f"probe {backbone} mass_mae={table['mass_g']['mae']:.4f} "
            f"energy_mae={table['energy_kcal']['mae']:.4f}",
            flush=True,
        )
        del model
        torch.cuda.empty_cache()

    winner = select_backbone(probe_rows)
    probe_payload = {
        **_hyper_meta(winner),
        "selection_rule": "lower fit-internal mass MAE, then energy MAE, then vit_b_16 before resnet101",
        "selected_backbone": winner,
        "rows": probe_rows,
        "holdout_scored": False,
        "rgb_test_scored": False,
    }
    _write(out / "backbone_probe.json", probe_payload)
    print(f"selected {winner} on fit-internal probe", flush=True)

    mean, std = target_mean_std(fit)
    model, losses = _train_one(winner, fit, mean, std, TRAIN_EPOCHS, device)
    fit_pred, _fit_scale, fit_target = _predict_split(model, fit, mean, std, device)
    fit_table = _mae_block(fit_pred, fit_target)
    baseline = _mae_block(np.repeat(mean.reshape(1, -1), len(fit), axis=0), fit_target)
    if fit_table["mass_g"]["mae"] > FIT_MASS_MAE_FACTOR * baseline["mass_g"]["mae"]:
        raise MassEstimatorError(
            "fit mass MAE is far above the fit-mean baseline; "
            f"model={fit_table['mass_g']['mae']} baseline={baseline['mass_g']['mae']}. "
            "Hold-out was not scored."
        )
    meta = _hyper_meta(winner)
    meta.update(
        {
            "n_fit": len(fit),
            "n_probe_train": len(probe_train),
            "n_probe_val": len(probe_val),
            "n_holdout_unscored_at_save": len(holdout),
            "train_nll": [round(loss, 4) for loss in losses],
            "fit_mae_in_sample": fit_table,
            "fit_mean_baseline_mae": baseline,
            "selected_from_probe": winner,
        }
    )
    save_checkpoint(out, model, mean=mean, std=std, meta=meta)
    clear_model_cache()

    hold_pred, hold_scale, hold_target = _predict_split(model, holdout, mean, std, device)
    hold_table = _mae_block(hold_pred, hold_target)
    hold_baseline = _mae_block(np.repeat(mean.reshape(1, -1), len(holdout), axis=0), hold_target)
    coverage = _coverage(hold_pred, hold_scale, hold_target)
    mean_sigma = {
        name: round(float(np.maximum(hold_scale, 0.0)[:, index].mean()), 4)
        for index, name in enumerate(TARGET_NAMES)
    }
    del model
    torch.cuda.empty_cache()

    def _missing_code(code: str, catalog_path: str | None = None) -> dict[str, float]:
        del catalog_path
        raise KeyError(code)

    tool = estimate_mass(
        fit[0].image,
        [{"code": "tool_smoke", "score": 1.0, "name": "smoke"}],
        checkpoint=out,
        lookup_fn=_missing_code,
    )
    if tool["rgb_only"] is not True or tool["depth_used"] is not False or tool["portion_src"] != "mass_estimator":
        raise MassEstimatorError("estimate_mass did not stay RGB-only after training")
    if not np.isfinite(tool["mass_g"]):
        raise MassEstimatorError("estimate_mass returned a non-finite mass")

    frozen_after = _frozen_digests()
    if frozen_after != frozen_before:
        raise MassEstimatorError("a frozen n5k or weak_steps file changed")
    elapsed = round(time.perf_counter() - started, 1)
    summary = {
        **_hyper_meta(winner),
        "definition": (
            "Dish-level MAE of the RGB Mass Estimator on the existing rgb_train hold-out "
            "(sha256 residue 0) among dishes with overhead rgb.png. "
            "Point estimates are clipped at 0. The backbone, epoch, and learning rates "
            "were fixed before this file was written. rgb_test was not scored."
        ),
        "n_holdout": len(holdout),
        "n_fit": len(fit),
        "mae": hold_table,
        "mass_mae_g": hold_table["mass_g"]["mae"],
        "energy_mae_kcal": hold_table["energy_kcal"]["mae"],
        "fit_mean_baseline_mae": hold_baseline,
        "interval_coverage_0.95": coverage,
        "mean_sigma": mean_sigma,
        "fit_mae_in_sample": fit_table,
        "tool_smoke": {
            "dish_id": fit[0].dish_id,
            "split": "fit",
            "mass_g": tool["mass_g"],
            "confidence": tool["confidence"],
            "portion_g": tool["portion_g"],
            "rgb_only": True,
            "compared_to_gt": False,
        },
        "frozen_sha256": {name: digest[:16] for name, digest in sorted(frozen_after.items())},
        "elapsed_seconds": elapsed,
        "status_file": "docs/STATUS.md was not modified.",
    }
    _write(out / "eval_holdout.json", summary)
    _report(
        out / "report_holdout.txt",
        [
            "P0b Mass Estimator — RGB-only",
            f"protocol: {PROTOCOL_ID}",
            "modality: RGB only. Depth was not read. rgb_test labels were not read.",
            "Main table for this specialist is RGB-only, same image channel as rgb_test overhead rgb.png.",
            f"backbone: {winner} (ImageNet-1K V1), chosen on fit-internal residue 1, not on hold-out.",
            f"checkpoint: last of {TRAIN_EPOCHS} fit epochs. No RL. Legacy weights were not loaded.",
            f"train n={len(fit)}  probe_train n={len(probe_train)}  probe_val n={len(probe_val)}  hold-out n={len(holdout)}",
            "Probe (fit residue 1; selection input):",
            *[
                f"  {row['backbone']}: mass MAE {row['mass_mae']:.4f} g, energy MAE {row['energy_mae']:.4f} kcal"
                for row in probe_rows
            ],
            f"Selected: {winner}",
            "Hold-out (scored once, after the checkpoint was saved):",
            f"  mass MAE {hold_table['mass_g']['mae']:.4f} g (median AE {hold_table['mass_g']['median_ae']:.4f})",
            f"  energy MAE {hold_table['energy_kcal']['mae']:.4f} kcal (median AE {hold_table['energy_kcal']['median_ae']:.4f})",
            f"  protein MAE {hold_table['protein_g']['mae']:.4f} g",
            f"  carb MAE {hold_table['carb_g']['mae']:.4f} g",
            f"  fat MAE {hold_table['fat_g']['mae']:.4f} g",
            "Fit-mean baseline on the same hold-out dishes (constant = fit mean, not a tuned model):",
            f"  mass MAE {hold_baseline['mass_g']['mae']:.4f} g",
            f"  energy MAE {hold_baseline['energy_kcal']['mae']:.4f} kcal",
            "95% Laplace interval coverage on hold-out (descriptive; the width was not adjusted):",
            *[f"  {name} {coverage[name]:.4f}" for name in TARGET_NAMES],
            "In-sample fit MAE is in eval_holdout.json under fit_mae_in_sample and is not the reported number.",
            f"elapsed_seconds {elapsed}",
            "docs/STATUS.md was not modified.",
        ],
    )
    print(
        f"hold-out n={len(holdout)} mass_mae={hold_table['mass_g']['mae']:.4f} "
        f"energy_mae={hold_table['energy_kcal']['mae']:.4f} backbone={winner}",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
