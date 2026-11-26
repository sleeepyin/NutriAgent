"""Train Swin-Nutrition once on the 2754 kept plates."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from repro.data import SwinPlates
from repro.losses import UncertaintyLoss
from repro.metadata import label_vector, load_dishes
from repro.models import SwinNutrition
from repro.protocol import ROOT as REPO
from repro.protocol import load_config
from repro.runtime import load_checkpoint, output_dir, require_gpu, save_checkpoint, seed_all
from repro.splits import audit_split


def main() -> None:
    device = require_gpu()
    config = load_config()
    seed_all(int(config["seed"]))
    audit = audit_split(config)
    dishes = load_dishes(config["paths"]["cafe1"], config["paths"]["cafe2"])
    out = output_dir("swin")
    if (out / "DONE").exists():
        print("swin already complete")
        return
    spec = config["swin"]
    model = SwinNutrition(drop_path_rate=float(spec["drop_path_rate"]), ffm_channels=int(spec["ffm_channels"]))
    matched = model.load_imagenet(REPO / config["weights"]["swin_imagenet1k"]["path"])
    print(f"swin imagenet tensors {matched}", flush=True)
    order = tuple(config["label_order"])
    means = np.mean([label_vector(dishes[dish_id], order) for dish_id in audit.train_ids], axis=0)
    criterion = UncertaintyLoss(torch.tensor(means, dtype=torch.float32))
    model.to(device)
    criterion.to(device)
    optimizer = torch.optim.Adam(
        list(model.parameters()) + list(criterion.parameters()),
        lr=float(spec["lr"]),
        weight_decay=float(spec["weight_decay"]),
        betas=tuple(spec["betas"]),
    )
    start = 0
    ckpt_path = out / "last.pt"
    if ckpt_path.exists():
        ckpt = load_checkpoint(ckpt_path)
        model.load_state_dict(ckpt["model"])
        criterion.load_state_dict(ckpt["criterion"])
        optimizer.load_state_dict(ckpt["optimizer"])
        start = int(ckpt["epoch"])
        print(f"resume swin at epoch {start}", flush=True)
    dataset = SwinPlates(audit.train_ids, dishes, config, train=True)
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
        for image, target, _dish_ids in loader:
            image = image.to(device, non_blocking=True)
            target = target.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            loss = criterion(model(image), target)
            loss.backward()
            optimizer.step()
            total += float(loss.detach()) * image.shape[0]
            seen += image.shape[0]
        save_checkpoint(
            ckpt_path,
            {
                "epoch": epoch + 1,
                "model": model.state_dict(),
                "criterion": criterion.state_dict(),
                "optimizer": optimizer.state_dict(),
                "train_mean": means.tolist(),
            },
        )
        print(f"swin epoch {epoch + 1}/{epochs} loss {total / seen:.6f} lr {lr:.8g}", flush=True)
    (out / "DONE").write_text("last epoch\n")
    print("swin done", flush=True)


if __name__ == "__main__":
    main()
