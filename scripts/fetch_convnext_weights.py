"""Download the official timm convnext_small.fb_in1k checkpoint and load it."""

from __future__ import annotations

import hashlib
import sys
import urllib.request
from pathlib import Path

import torch
import timm
from timm.models.convnext import checkpoint_filter_fn

ROOT = Path(__file__).resolve().parents[1]
URL = "https://dl.fbaipublicfiles.com/convnext/convnext_small_1k_224_ema.pth"
DEST = ROOT / "outputs" / "weights" / "convnext_small_1k_224_ema.pth"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def download(url: str, dest: Path) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    temporary = dest.with_suffix(".pth.partial")
    request = urllib.request.Request(url, headers={"User-Agent": "nutriagent-v3-step1"})
    with urllib.request.urlopen(request, timeout=60) as response, temporary.open("wb") as handle:
        while True:
            chunk = response.read(1024 * 1024)
            if not chunk:
                break
            handle.write(chunk)
    temporary.replace(dest)


def main() -> None:
    if not DEST.exists() or DEST.stat().st_size < 1_000_000:
        print(f"download {URL}", flush=True)
        download(URL, DEST)
    digest = sha256_file(DEST)
    print(f"sha256 {digest}", flush=True)
    print(f"bytes {DEST.stat().st_size}", flush=True)
    net = timm.create_model(
        "convnext_small",
        pretrained=False,
        num_classes=0,
        global_pool="",
        drop_path_rate=0.0,
    )
    before = sum(parameter.numel() for parameter in net.parameters())
    raw = torch.load(DEST, map_location="cpu", weights_only=False)
    mapped = checkpoint_filter_fn(raw, net)
    incompatible = net.load_state_dict(mapped, strict=False)
    missing = [key for key in incompatible.missing_keys if not key.startswith("head.fc.")]
    unexpected = [key for key in incompatible.unexpected_keys if not key.startswith("head.fc.")]
    if missing or unexpected:
        raise SystemExit(f"load failed missing={missing[:12]} unexpected={unexpected[:12]}")
    after = sum(parameter.numel() for parameter in net.parameters())
    if after != before:
        raise SystemExit(f"parameter count changed {before} -> {after}")
    stem = net.stem[0].weight.detach()
    if not torch.isfinite(stem).all() or float(stem.abs().mean()) <= 0:
        raise SystemExit("loaded stem weights are empty")
    fresh = timm.create_model(
        "convnext_small",
        pretrained=False,
        num_classes=0,
        global_pool="",
        drop_path_rate=0.0,
    )
    if torch.allclose(stem, fresh.stem[0].weight):
        raise SystemExit("checkpoint did not change the stem")
    print(f"loaded convnext_small params {after} stem_mean {float(stem.abs().mean()):.6f}", flush=True)


if __name__ == "__main__":
    sys.exit(main())
