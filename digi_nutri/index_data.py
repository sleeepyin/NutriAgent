"""Build the official RGB-D dish index. Incomplete splits are not silently dropped."""

from __future__ import annotations

from collections import Counter
from pathlib import Path

import numpy as np

from digi_nutri.metadata import load_metadata
from digi_nutri.plate_groups import assign_plate_groups
from digi_nutri.splits import load_official_split, sha256_file


def index_depth_files(roots: list[Path]) -> dict[str, str]:
    found: dict[str, str] = {}
    for root in roots:
        root = Path(root)
        if not root.exists():
            continue
        direct = list(root.glob("dish_*/depth_raw.png"))
        paths = direct if direct else list(root.rglob("depth_raw.png"))
        for path in paths:
            if path.stat().st_size <= 0:
                continue
            found.setdefault(path.parent.name, str(path))
    return found


def build_records_from_ids(
    dish_ids: list[str],
    metadata: dict,
    rgb_root: Path,
    depth_index: dict[str, str],
    groups: dict[str, str],
) -> tuple[list[dict], dict]:
    records = []
    missing = []
    for dish_id in dish_ids:
        rgb_path = Path(rgb_root) / dish_id / "rgb.png"
        depth_path = depth_index.get(dish_id)
        if dish_id not in metadata:
            missing.append({"dish_id": dish_id, "reason": "metadata"})
            continue
        if not rgb_path.is_file():
            missing.append({"dish_id": dish_id, "reason": "rgb"})
            continue
        if depth_path is None:
            missing.append({"dish_id": dish_id, "reason": "depth"})
            continue
        source = metadata[dish_id]
        records.append(
            {
                "dish_id": dish_id,
                "rgb": str(rgb_path),
                "depth": depth_path,
                "y": list(source["y"]),
                "names": list(source["names"]),
                "plate_group": groups[dish_id],
            }
        )
    report = {
        "n_ids": len(dish_ids),
        "n_kept": len(records),
        "missing": missing,
        "missing_counts": dict(Counter(item["reason"] for item in missing)),
    }
    return records, report


def build_records(
    splits_dir: Path,
    which: str,
    metadata_paths: list[Path] | tuple[Path, ...],
    rgb_root: Path,
    depth_roots: list[Path],
    metadata: dict | None = None,
    depth_index: dict[str, str] | None = None,
    reshoot_tsv: Path | None = None,
) -> tuple[list[dict], dict]:
    dish_ids = load_official_split(splits_dir, which)
    if metadata is None:
        metadata = load_metadata(metadata_paths)
    if depth_index is None:
        depth_index = index_depth_files(depth_roots)
    groups = assign_plate_groups(dish_ids, reshoot_tsv)
    records, report = build_records_from_ids(
        dish_ids, metadata, rgb_root, depth_index, groups
    )
    report["split"] = which
    return records, report


def train_target_stats(records: list[dict]) -> dict:
    targets = np.asarray([record["y"] for record in records], dtype=np.float64)
    if targets.ndim != 2 or targets.shape[1] != 5:
        raise ValueError("targets must be N,5")
    mean = targets.mean(axis=0)
    std = np.maximum(targets.std(axis=0), 1e-3)
    return {"mean": mean.tolist(), "std": std.tolist()}


def train_vocab(records: list[dict]) -> list[str]:
    return sorted({name for record in records for name in record["names"]})


def split_hashes(splits_dir: Path) -> dict[str, str]:
    return {
        "train": sha256_file(Path(splits_dir) / "depth_train_ids.txt"),
        "test": sha256_file(Path(splits_dir) / "depth_test_ids.txt"),
    }
