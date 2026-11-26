"""Train the DPT depth module once on the 2754 kept plates."""

from __future__ import annotations

import sys
from pathlib import Path

import torch
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from repro.data import DepthPlates
from repro.losses import masked_l1
from repro.models import DepthModule
from repro.protocol import ROOT as REPO
from repro.protocol import load_config
from repro.runtime import load_checkpoint, output_dir, require_gpu, save_checkpoint, seed_all
from repro.splits import audit_split


def main() -> None:
    device = require_gpu()
    config = load_config()
    seed_all(int(config["seed"]))
    audit = audit_split(config)
    out = output_dir("dpt")
    if (out / "DONE").exists():
        print("dpt already complete")
        return
    spec = config["dpf"]["depth_module"]
    encoder_path = REPO / config["weights"]["dpt_hybrid_encoder"]["path"]
    minimum = int(config["weights"]["dpt_hybrid_encoder"]["min_bytes"])
    size = encoder_path.stat().st_size if encoder_path.exists() else 0
    if size < minimum:
        raise SystemExit(f"encoder file is incomplete ({size} bytes, need {minimum})")
    model = DepthModule()
    matched = model.load_encoder(encoder_path)
    print(f"dpt encoder tensors {matched}", flush=True)
    model.to(device)
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=float(spec["lr"]),
        weight_decay=float(spec["weight_decay"]),
        betas=tuple(spec["betas"]),
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=int(spec["epochs"]), eta_min=float(spec["eta_min"])
    )
    start = 0
    ckpt_path = out / "last.pt"
    if ckpt_path.exists():
        ckpt = load_checkpoint(ckpt_path)
        model.load_state_dict(ckpt["model"])
        optimizer.load_state_dict(ckpt["optimizer"])
        scheduler.load_state_dict(ckpt["scheduler"])
        start = int(ckpt["epoch"])
        print(f"resume dpt at epoch {start}", flush=True)
    dataset = DepthPlates(audit.train_ids, config, train=True)
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
        model.train()
        total = 0.0
        seen = 0
        for image, depth, valid, _dish_ids in loader:
            image = image.to(device, non_blocking=True)
            depth = depth.to(device, non_blocking=True)
            valid = valid.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            loss = masked_l1(model(image), depth, valid)
            loss.backward()
            optimizer.step()
            total += float(loss.detach()) * image.shape[0]
            seen += image.shape[0]
        scheduler.step()
        save_checkpoint(
            ckpt_path,
            {
                "epoch": epoch + 1,
                "model": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "scheduler": scheduler.state_dict(),
            },
        )
        lr = optimizer.param_groups[0]["lr"]
        print(f"dpt epoch {epoch + 1}/{epochs} loss {total / seen:.6f} lr {lr:.8g}", flush=True)
    (out / "DONE").write_text("last epoch\n")
    print("dpt done", flush=True)


if __name__ == "__main__":
    main()
