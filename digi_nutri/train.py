"""Train Plan A T_geo only."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import torch

from digi_nutri.audit_rule import use_zero_exclusion
from digi_nutri.cohort import load_cohort
from digi_nutri.data import make_records
from digi_nutri.engine import train_model
from digi_nutri.freeze import assert_config_matches_protocol
from digi_nutri.metadata import load_dish_table
from digi_nutri.protocol import AUDIT_PATH, META_DIR
from digi_nutri.reshoot import plate_groups


def assert_plan_a_only(kind: str) -> None:
    if kind != "t_geo":
        raise RuntimeError("Plan A retrains T_geo only")


def _pid_alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
    except OSError:
        return False
    return True


def load_config(path: Path) -> dict:
    config = json.loads(path.read_text())
    assert_config_matches_protocol(config)
    audit = json.loads(Path(AUDIT_PATH).read_text())
    if bool(config["apply_d_table_zero_exclusion"]) != use_zero_exclusion(audit):
        raise RuntimeError("config and the D_table audit disagree")
    return config


def records_for(config: dict) -> list[dict]:
    assert_plan_a_only("t_geo")
    train_ids, test_ids = load_cohort(config)
    if set(train_ids) & set(test_ids):
        raise RuntimeError("train ids overlap the official test ids")
    meta = Path(META_DIR)
    dishes = load_dish_table(meta / "dish_metadata_cafe1.csv", meta / "dish_metadata_cafe2.csv")
    groups = plate_groups(train_ids, dishes)
    return make_records("t_geo", train_ids, dishes, groups)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=Path("configs/plan_a_frozen.json"))
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()
    config = load_config(args.config)
    output = Path(config["output_dir"])
    output.mkdir(parents=True, exist_ok=True)
    pid_path = output / "t_geo.pid"
    if pid_path.exists() and _pid_alive(int(pid_path.read_text().strip())):
        raise SystemExit(f"t_geo is already running as pid {pid_path.read_text().strip()}")
    pid_path.write_text(str(os.getpid()))
    try:
        if not torch.cuda.is_available():
            raise RuntimeError("Plan A training requires CUDA")
        records = records_for(config)
        train_model(config, records, torch.device(args.device), args.config)
    finally:
        if pid_path.exists() and pid_path.read_text().strip() == str(os.getpid()):
            pid_path.unlink()


if __name__ == "__main__":
    main()
