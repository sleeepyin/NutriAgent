"""Load the frozen reproduction config."""

from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = ROOT / "configs" / "repro_frozen.json"


def load_config() -> dict:
    return json.loads(CONFIG_PATH.read_text())


def label_order(config: dict | None = None) -> tuple[str, ...]:
    config = load_config() if config is None else config
    return tuple(config["label_order"])
