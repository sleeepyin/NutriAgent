"""Shared FNDDS catalog loader.

Priority:
1. Explicit catalog_path argument
2. Env VDR_FNDDS_PATH (file or directory containing foods.json)
3. data/fndds/2019-2020/foods.json (real catalog, if built)
4. data/fndds/toy/foods.json (smoke fallback)
"""

from __future__ import annotations

import json
import os
from functools import lru_cache
from pathlib import Path
from typing import Any

_REPO_ROOT = Path(__file__).resolve().parents[2]
_REAL = _REPO_ROOT / "data" / "fndds" / "2019-2020" / "foods.json"
_TOY = _REPO_ROOT / "data" / "fndds" / "toy" / "foods.json"


def resolve_catalog_path(catalog_path: str | Path | None = None) -> Path:
    if catalog_path is not None:
        p = Path(catalog_path)
    else:
        env = os.environ.get("VDR_FNDDS_PATH")
        if env:
            p = Path(env)
        elif _REAL.is_file():
            p = _REAL
        else:
            p = _TOY
    if p.is_dir():
        p = p / "foods.json"
    return p


@lru_cache(maxsize=4)
def _load_raw(path_str: str) -> dict[str, Any]:
    path = Path(path_str)
    if not path.is_file():
        raise FileNotFoundError(
            f"FNDDS catalog not found: {path}. "
            "Run scripts/build_fndds_catalog.py or use data/fndds/toy/foods.json; "
            "or set VDR_FNDDS_PATH."
        )
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    if "foods" not in data or not isinstance(data["foods"], list):
        raise ValueError(f"Invalid catalog schema at {path}: missing foods[]")
    return data


def load_catalog(catalog_path: str | Path | None = None) -> dict[str, Any]:
    path = resolve_catalog_path(catalog_path)
    return _load_raw(str(path.resolve()))


def clear_catalog_cache() -> None:
    _load_raw.cache_clear()


def foods_by_code(catalog_path: str | Path | None = None) -> dict[str, dict[str, Any]]:
    data = load_catalog(catalog_path)
    return {str(f["code"]): f for f in data["foods"]}


def iter_foods(catalog_path: str | Path | None = None) -> list[dict[str, Any]]:
    return list(load_catalog(catalog_path)["foods"])


def catalog_stats(catalog_path: str | Path | None = None) -> dict[str, Any]:
    path = resolve_catalog_path(catalog_path)
    data = load_catalog(path)
    return {
        "path": str(path),
        "n_foods": len(data["foods"]),
        "version": data.get("version"),
        "note": data.get("note"),
    }
