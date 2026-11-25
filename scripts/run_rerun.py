"""Train B1, T_geo, and T_ing once on cuda:1, then evaluate once."""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
os.chdir(ROOT)

import torch

from digi_nutri.engine import gpu_memory_mib, require_cuda1, train_model
from digi_nutri.evaluate import evaluate_holdout, evaluate_train, load_config
from digi_nutri.metadata import load_dish_table
from digi_nutri.protocol import CONFIG_PATH, HOLDOUT_METRICS, META_DIR, MODELS
from digi_nutri.reshoot import plate_groups
from digi_nutri.evaluate import _cohort, _records


def _pid_alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
    except OSError:
        return False
    return True


def _lock(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        pid = int(path.read_text().strip())
        if _pid_alive(pid) and pid != os.getpid():
            raise SystemExit(f"rerun already running as pid {pid}")
    path.write_text(str(os.getpid()))


def _unlock(path: Path) -> None:
    if path.exists() and path.read_text().strip() == str(os.getpid()):
        path.unlink()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="cuda:1")
    parser.add_argument("--config", type=Path, default=Path(CONFIG_PATH))
    args = parser.parse_args()
    device = torch.device(args.device)
    require_cuda1(device)
    holdout = ROOT / HOLDOUT_METRICS
    if holdout.exists():
        raise SystemExit("holdout metrics already exist; refusing another official evaluation")
    memory = gpu_memory_mib()
    print(f"gpu memory MiB before training: {memory}", flush=True)
    if memory.get(1, 1) > 500:
        raise SystemExit(f"GPU 1 is not idle ({memory.get(1)} MiB); not starting")
    lock = ROOT / "outputs" / "rerun.pid"
    _lock(lock)
    try:
        config = load_config(args.config)
        train_ids, _test_ids = _cohort(config)
        dishes = load_dish_table(
            Path(META_DIR) / "dish_metadata_cafe1.csv",
            Path(META_DIR) / "dish_metadata_cafe2.csv",
        )
        for kind in MODELS:
            groups = plate_groups(train_ids, dishes)
            records = _records(kind, train_ids, dishes, config, ROOT)
            if [row["group_id"] for row in records] != [groups[dish_id] for dish_id in train_ids]:
                raise RuntimeError("train group ids drifted")
            print(f"start {kind} on {device}", flush=True)
            train_model(kind, config, records, device, args.config, gpu0_baseline=memory[0])
            torch.cuda.empty_cache()
            assert_now = gpu_memory_mib()
            print(f"gpu memory MiB after {kind}: {assert_now}", flush=True)
            if assert_now[0] > memory[0] + 800:
                raise SystemExit("GPU 0 memory rose; leaving that process untouched")
        evaluate_train(config, device, ROOT, args.config)
        evaluate_holdout(config, device, ROOT, args.config)
    finally:
        _unlock(lock)


if __name__ == "__main__":
    main()
