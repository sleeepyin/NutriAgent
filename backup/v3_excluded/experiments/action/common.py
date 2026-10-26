"""Shared helpers for the action-layer experiments."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
for entry in (str(ROOT), str(ROOT / "src")):
    if entry not in sys.path:
        sys.path.insert(0, entry)

from mmcfa.paths import Paths
from mmcfa.runtime import atomic_write_json, log, utc_now


def add_common_args(p: argparse.ArgumentParser) -> argparse.ArgumentParser:
    p.add_argument("--out", default="", help="override output dir")
    p.add_argument("--seed", type=int, default=42)
    return p


def out_dir(name: str, args=None) -> Path:
    paths = Paths.resolve()
    d = Path(args.out) if args and getattr(args, "out", "") else paths.results / name
    d.mkdir(parents=True, exist_ok=True)
    return d


def save_status(d: Path, ok: bool, **extra) -> None:
    payload = {"ok": ok, "ts": utc_now(), **extra}
    atomic_write_json(d / "status.json", payload)
    log(f"status {d.name} ok={ok}")
