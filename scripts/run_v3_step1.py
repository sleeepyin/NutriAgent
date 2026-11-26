"""Train B1, then T_multi, then read the official holdout once."""

from __future__ import annotations

import os

os.environ["CUDA_VISIBLE_DEVICES"] = "1"

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
os.chdir(ROOT)

import torch

from digi_nutri.engine import (
    assert_gpu0_not_grown,
    assert_gpu1_idle,
    assert_no_holdout,
    require_physical_gpu1,
    train_model,
)
from digi_nutri.evaluate import cohort, evaluate_holdout, evaluate_train, load_config, records_for
from digi_nutri.metadata import load_dish_table
from digi_nutri.protocol import CONFIG_PATH, HOLDOUT_METRICS, META_DIR, TRAIN_METRICS
from digi_nutri.reshoot import plate_groups


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
            raise SystemExit(f"v3 step 1 already running as pid {pid}")
    path.write_text(str(os.getpid()))


def _unlock(path: Path) -> None:
    if path.exists() and path.read_text().strip() == str(os.getpid()):
        path.unlink()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=Path(CONFIG_PATH))
    parser.add_argument("--smoke-steps", type=int, default=0)
    parser.add_argument("--only", choices=("b1", "t_multi", "both"), default="both")
    args = parser.parse_args()
    if not args.config.exists():
        raise SystemExit("frozen config is missing; run scripts/prepare_config.py first")
    if (ROOT / HOLDOUT_METRICS).exists():
        raise SystemExit("holdout metrics already exist; refusing another official evaluation")
    assert_no_holdout()
    device = require_physical_gpu1()
    memory = assert_gpu1_idle()
    print(f"gpu memory MiB before training: {memory}", flush=True)
    lock = ROOT / "outputs" / "v3_step1.pid"
    _lock(lock)
    try:
        config = load_config(args.config)
        train_ids, _test_ids = cohort(config)
        dishes = load_dish_table(
            Path(META_DIR) / "dish_metadata_cafe1.csv",
            Path(META_DIR) / "dish_metadata_cafe2.csv",
        )
        groups = plate_groups(train_ids, dishes)
        records = records_for(train_ids, dishes, config, ROOT)
        if [row["group_id"] for row in records] != [groups[dish_id] for dish_id in train_ids]:
            raise RuntimeError("train group ids drifted")
        kinds = ("b1", "t_multi") if args.only == "both" else (args.only,)
        for kind in kinds:
            print(f"start {kind} on physical GPU 1 ({device})", flush=True)
            train_model(
                kind,
                config,
                records,
                device,
                args.config,
                gpu0_baseline=memory[0],
                smoke_steps=int(args.smoke_steps),
            )
            assert_gpu0_not_grown(memory[0])
        if int(args.smoke_steps) > 0:
            print("smoke finished without a holdout", flush=True)
            return
        torch.cuda.empty_cache()
        assert_gpu0_not_grown(memory[0])
        if not (ROOT / TRAIN_METRICS).exists():
            evaluate_train(config, device, ROOT, args.config)
        evaluate_holdout(config, device, ROOT, args.config)
    finally:
        _unlock(lock)


if __name__ == "__main__":
    main()
