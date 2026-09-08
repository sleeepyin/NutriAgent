"""Shared experiment helpers."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from agent_extension.src.paths import Paths
from agent_extension.src.runtime import Heartbeat, atomic_write_json, log, utc_now


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


def load_rows(paths: Paths | None = None) -> list[dict]:
    paths = paths or Paths.resolve()
    p = paths.results / "e02_action_table" / "episodes.jsonl"
    if not p.exists():
        return []
    return [json.loads(x) for x in p.read_text().splitlines() if x.strip()]


def load_json(path: Path, default=None):
    if not path.exists():
        return default
    return json.loads(path.read_text())


def np_save(path: Path, **kw) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, **kw)
