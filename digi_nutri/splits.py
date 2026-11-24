"""Official Nutrition5k depth train/test ids. No other split is accepted."""

from __future__ import annotations

import hashlib
from pathlib import Path

from digi_nutri.protocol import OFFICIAL_SPLIT_COUNTS, OFFICIAL_SPLIT_FILES


def load_official_split(splits_dir: Path, which: str) -> list[str]:
    if which not in OFFICIAL_SPLIT_FILES:
        raise ValueError(
            "only the official Nutrition5k depth train/test split is allowed"
        )
    path = Path(splits_dir) / OFFICIAL_SPLIT_FILES[which]
    if path.name not in OFFICIAL_SPLIT_FILES.values():
        raise ValueError(f"refusing split file {path.name}")
    ids: list[str] = []
    seen: set[str] = set()
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            dish_id = line.strip()
            if not dish_id:
                continue
            if dish_id in seen:
                raise ValueError(f"duplicate id {dish_id} in {path.name}")
            seen.add(dish_id)
            ids.append(dish_id)
    expected = OFFICIAL_SPLIT_COUNTS[which]
    if len(ids) != expected:
        raise ValueError(f"{path.name} has {len(ids)} ids, expected {expected}")
    return ids


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()
