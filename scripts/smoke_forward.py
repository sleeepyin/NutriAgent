"""One forward/backward step on GPU 1. Not a training run."""

from __future__ import annotations

import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
os.environ["CUDA_VISIBLE_DEVICES"] = "1"

import torch

from repro.losses import UncertaintyLoss, geometric_l1, masked_l1
from repro.models import DPFNutrition, DepthModule, SwinNutrition
from repro.runtime import require_gpu


def main() -> None:
    device = require_gpu()
    depth = DepthModule().to(device).train()
    rgb = torch.rand(1, 3, 384, 384, device=device)
    target = torch.rand(1, 384, 384, device=device)
    valid = torch.ones(1, 384, 384, device=device, dtype=torch.bool)
    pred = depth(rgb)
    loss = masked_l1(pred, target, valid)
    loss.backward()
    print("dpt", tuple(pred.shape), float(loss))

    dpf = DPFNutrition().to(device).train()
    image = torch.rand(2, 3, 336, 448, device=device)
    depth_image = torch.rand(2, 3, 336, 448, device=device)
    nutrients = torch.rand(2, 5, device=device)
    out = dpf(image, depth_image)
    loss = geometric_l1(out, nutrients)
    loss.backward()
    print("dpf", tuple(out.shape), float(loss))

    swin = SwinNutrition().to(device).train()
    image = torch.rand(2, 3, 224, 224, device=device)
    out = swin(image)
    criterion = UncertaintyLoss(torch.ones(5, device=device))
    loss = criterion(out, nutrients)
    loss.backward()
    print("swin", tuple(out.shape), float(loss))

    ruler = torch.rand(1, 3, 448, 448, device=device)
    with torch.no_grad():
        swin.eval()
        depth.eval()
        dpf.eval()
        print("swin448", tuple(swin(ruler).shape))
        print("dpt448", tuple(depth(ruler).shape))
        print("dpf448", tuple(dpf(ruler, ruler).shape))
    del rgb, target, valid, pred, image, depth_image, nutrients, ruler
    del depth, dpf, swin
    torch.cuda.empty_cache()
    swin = SwinNutrition().to(device).train()
    image = torch.rand(64, 3, 224, 224, device=device)
    loss = swin(image).sum()
    loss.backward()
    print("swin64_mib", torch.cuda.max_memory_allocated() / (1024**2))
    del swin, image, loss
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    depth = DepthModule().to(device).train()
    image = torch.rand(8, 3, 384, 384, device=device)
    loss = depth(image).sum()
    loss.backward()
    print("dpt8_mib", torch.cuda.max_memory_allocated() / (1024**2))
    del depth, image, loss
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    dpf = DPFNutrition().to(device).train()
    image = torch.rand(8, 3, 336, 448, device=device)
    depth_image = torch.rand(8, 3, 336, 448, device=device)
    loss = dpf(image, depth_image).sum()
    loss.backward()
    print("dpf8_mib", torch.cuda.max_memory_allocated() / (1024**2))
    print("smoke_ok")


if __name__ == "__main__":
    main()
