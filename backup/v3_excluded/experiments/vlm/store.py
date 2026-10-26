"""Cache paths. One JSON per hold-out image. Labels stay out of these files."""
from __future__ import annotations

import json
from pathlib import Path

from mmcfa.paths import Paths
from mmcfa.runtime import atomic_write_json, read_json

from .prompt import PROMPT_SHA256


def cache_file(dataset: str, sample_id: str, root: Path | None = None) -> Path:
    base = root or Paths.resolve().vlm_cache
    safe = str(sample_id).replace("/", "_")
    return base / dataset / f"{safe}.json"


def error_file(dataset: str, sample_id: str, root: Path | None = None) -> Path:
    return cache_file(dataset, sample_id, root).with_suffix(".error.json")


def is_done(dataset: str, sample_id: str, root: Path | None = None) -> bool:
    obj = read_json(cache_file(dataset, sample_id, root), None)
    if not isinstance(obj, dict):
        return False
    meta = obj.get("meta") or {}
    if meta.get("prompt_sha256") != PROMPT_SHA256:
        return False
    weight = obj.get("weight_guess_g")
    return bool(obj.get("food_name")) and isinstance(weight, (int, float)) and weight > 0


def attempts(dataset: str, sample_id: str, root: Path | None = None) -> int:
    obj = read_json(error_file(dataset, sample_id, root), {})
    if not isinstance(obj, dict) or obj.get("prompt_sha256") != PROMPT_SHA256:
        return 0
    return int(obj.get("attempts") or 0)


def write_record(dataset: str, sample_id: str, record: dict, root: Path | None = None) -> None:
    atomic_write_json(cache_file(dataset, sample_id, root), record)
    err = error_file(dataset, sample_id, root)
    if err.exists():
        err.unlink()


def write_error(
    dataset: str,
    sample_id: str,
    message: str,
    root: Path | None = None,
    kind: str = "content",
) -> None:
    prev = attempts(dataset, sample_id, root)
    atomic_write_json(
        error_file(dataset, sample_id, root),
        {
            "prompt_sha256": PROMPT_SHA256,
            "attempts": prev + 1,
            "kind": kind,
            "error": message[:500],
        },
    )


def load_record(dataset: str, sample_id: str, root: Path | None = None) -> dict | None:
    if not is_done(dataset, sample_id, root):
        return None
    return json.loads(cache_file(dataset, sample_id, root).read_text(encoding="utf-8"))


def manifest_path(root: Path | None = None) -> Path:
    return (root or Paths.resolve().vlm_cache) / "manifest.jsonl"


def write_manifest(rows: list[dict], root: Path | None = None) -> None:
    path = manifest_path(root)
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [json.dumps(row, ensure_ascii=False, default=str) for row in rows]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def read_manifest(root: Path | None = None) -> list[dict]:
    path = manifest_path(root)
    if not path.is_file():
        return []
    rows = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            rows.append(json.loads(line))
    return rows
