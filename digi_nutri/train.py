"""Train one of B1, T_geo, or T_ing from the frozen config."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import torch

from digi_nutri.data import make_records
from digi_nutri.engine import train_model
from digi_nutri.ingredients import build_vocab
from digi_nutri.metadata import load_dish_table
from digi_nutri.prepare import assert_config_matches_protocol
from digi_nutri.protocol import META_DIR
from digi_nutri.reshoot import plate_groups
from digi_nutri.usda import load_usda


def _pid_alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
    except OSError:
        return False
    return True


def load_config(path: Path) -> dict:
    config = json.loads(path.read_text())
    assert_config_matches_protocol(config)
    return config


def records_for(kind: str, config: dict) -> list[dict]:
    meta = Path(META_DIR)
    dishes = load_dish_table(meta / "dish_metadata_cafe1.csv", meta / "dish_metadata_cafe2.csv")
    hygiene = json.loads(Path("results/v2_q0/split_hygiene.json").read_text())
    excluded = {row["dish_id"] for row in hygiene["train_exclusions"]}
    from digi_nutri.splits import read_id_file
    from digi_nutri.protocol import SPLIT_DIR

    train_ids = [
        dish_id
        for dish_id in read_id_file(Path(SPLIT_DIR) / "depth_train_ids.txt")
        if dish_id not in excluded
    ]
    if len(train_ids) != int(config["n_train"]):
        raise RuntimeError("train count does not match the frozen config")
    groups = plate_groups(train_ids, dishes)
    usda = None
    vocab = None
    if kind == "t_ing":
        usda = load_usda(meta / "ingredients_metadata.csv")
        vocab = json.loads(Path("configs/ingredient_vocab.json").read_text())
        built = build_vocab(usda)
        if built["tokens"] != vocab["tokens"]:
            raise RuntimeError("vocabulary does not match the frozen ingredient table")
    return make_records(kind, train_ids, dishes, groups, usda=usda, vocab=vocab)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=Path("configs/q0_frozen.json"))
    parser.add_argument("--model", required=True, choices=("b1", "t_geo", "t_ing"))
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()
    config = load_config(args.config)
    output = Path(config["output_dir"])
    output.mkdir(parents=True, exist_ok=True)
    pid_path = output / f"{args.model}.pid"
    if pid_path.exists() and _pid_alive(int(pid_path.read_text().strip())):
        raise SystemExit(f"{args.model} is already running as pid {pid_path.read_text().strip()}")
    pid_path.write_text(str(os.getpid()))
    try:
        if not torch.cuda.is_available():
            raise RuntimeError("Q0 training requires CUDA")
        device = torch.device(args.device)
        records = records_for(args.model, config)
        train_model(args.model, config, records, device, args.config)
    finally:
        if pid_path.exists() and pid_path.read_text().strip() == str(os.getpid()):
            pid_path.unlink()


if __name__ == "__main__":
    main()
