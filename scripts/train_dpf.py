"""Train the DPF fusion module once, or stop if Food2K init cannot be verified."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from repro.data import FusionPlates, predict_fusion_depth, read_rgb
from repro.losses import geometric_l1
from repro.metadata import load_dishes
from repro.models import DPFNutrition, DepthModule
from repro.protocol import ROOT as REPO
from repro.protocol import load_config
from repro.runtime import load_checkpoint, output_dir, require_gpu, save_checkpoint, seed_all
from repro.splits import audit_split
from repro.weights_check import inspect_food2k


def stop(reason: str) -> None:
    dest = REPO / "results" / "repro" / "dpf_stop.json"
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_text(json.dumps({"status": "stopped", "reason": reason}, indent=2) + "\n")
    print(reason, flush=True)


def cache_depth(config: dict, dish_ids: list[str], device: torch.device, cache: Path) -> None:
    cache.mkdir(parents=True, exist_ok=True)
    missing = [dish_id for dish_id in dish_ids if not (cache / f"{dish_id}.npy").exists()]
    if not missing:
        print("depth cache complete", flush=True)
        return
    if not (REPO / "outputs" / "dpt" / "DONE").exists():
        raise SystemExit("DPT training is not complete, so fusion depth cannot be cached")
    model = DepthModule()
    model.load_state_dict(load_checkpoint(REPO / "outputs" / "dpt" / "last.pt")["model"])
    model.to(device)
    model.eval()
    rgb_root = Path(config["paths"]["rgb_root"])
    with torch.no_grad():
        for index, dish_id in enumerate(missing, start=1):
            depth = predict_fusion_depth(model, read_rgb(rgb_root / dish_id / "rgb.png"), config)
            np.save(cache / f"{dish_id}.npy", depth.astype(np.float32))
            if index % 200 == 0 or index == len(missing):
                print(f"cached depth {index}/{len(missing)}", flush=True)


def main() -> None:
    device = require_gpu()
    config = load_config()
    seed_all(int(config["seed"]))
    out = output_dir("dpf")
    if (out / "DONE").exists():
        print("dpf already complete")
        return
    weight_path = REPO / config["weights"]["food2k_resnet101"]["path"]
    minimum = int(config["weights"]["food2k_resnet101"]["min_bytes"])
    if not weight_path.exists() or weight_path.stat().st_size < minimum:
        stop("Food2K classification file is missing. Fusion was not trained, and ImageNet was not substituted.")
        return
    try:
        inspected = inspect_food2k(
            weight_path,
            expected_sha256=config["weights"]["food2k_resnet101"]["sha256"],
        )
    except Exception as exc:
        stop(f"Food2K file failed the classification-backbone check: {exc}. Fusion was not trained.")
        return
    audit = audit_split(config)
    dishes = load_dishes(config["paths"]["cafe1"], config["paths"]["cafe2"])
    cache = REPO / "outputs" / "cache" / "dpt_depth"
    cache_depth(config, audit.train_ids, device, cache)
    spec = config["dpf"]["fusion_module"]
    model = DPFNutrition()
    model.load_food2k(inspected["state"])
    print(
        f"food2k fc {inspected['fc_shape']} conv1 {inspected['conv1_shape']} tensors {inspected['tensors']}",
        flush=True,
    )
    model.to(device)
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=float(spec["lr"]),
        weight_decay=float(spec["weight_decay"]),
        betas=tuple(spec["betas"]),
    )
    start = 0
    ckpt_path = out / "last.pt"
    if ckpt_path.exists():
        ckpt = load_checkpoint(ckpt_path)
        model.load_state_dict(ckpt["model"])
        optimizer.load_state_dict(ckpt["optimizer"])
        start = int(ckpt["epoch"])
        print(f"resume dpf at epoch {start}", flush=True)
    dataset = FusionPlates(audit.train_ids, dishes, config, train=True, cache_dir=cache)
    loader = DataLoader(
        dataset,
        batch_size=int(spec["batch_size"]),
        shuffle=True,
        num_workers=2,
        pin_memory=True,
    )
    epochs = int(spec["epochs"])
    for epoch in range(start, epochs):
        dataset.set_epoch(epoch)
        lr = float(spec["lr"]) * (float(spec["lr_decay"]) ** epoch)
        for group in optimizer.param_groups:
            group["lr"] = lr
        model.train()
        total = 0.0
        seen = 0
        for image, depth, target, _dish_ids in loader:
            image = image.to(device, non_blocking=True)
            depth = depth.to(device, non_blocking=True)
            target = target.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            loss = geometric_l1(model(image, depth), target)
            loss.backward()
            optimizer.step()
            total += float(loss.detach()) * image.shape[0]
            seen += image.shape[0]
        save_checkpoint(
            ckpt_path,
            {"epoch": epoch + 1, "model": model.state_dict(), "optimizer": optimizer.state_dict()},
        )
        print(f"dpf epoch {epoch + 1}/{epochs} loss {total / seen:.6f} lr {lr:.8g}", flush=True)
    (out / "DONE").write_text("last epoch\n")
    print("dpf done", flush=True)


if __name__ == "__main__":
    main()
