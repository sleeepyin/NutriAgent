"""Official depth-split guard. Test ids stay intact."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


class SplitError(RuntimeError):
    pass


@dataclass
class SplitAudit:
    train_ids: list[str]
    test_ids: list[str]
    exclusions: list[dict]


def read_ids(path: str | Path) -> list[str]:
    return [line.strip() for line in Path(path).read_text().splitlines() if line.strip()]


def file_reasons(dish_id: str, rgb_root: Path, depth_root: Path) -> list[str]:
    reasons = []
    rgb = rgb_root / dish_id / "rgb.png"
    depth = depth_root / dish_id / "depth_raw.png"
    if not rgb.exists():
        reasons.append("rgb_missing")
    elif rgb.stat().st_size == 0:
        reasons.append("rgb_zero")
    if not depth.exists():
        reasons.append("depth_missing")
    elif depth.stat().st_size == 0:
        reasons.append("depth_zero")
    return reasons


def audit_split(config: dict) -> SplitAudit:
    paths = config["paths"]
    expected = config["split"]
    split_dir = Path(paths["split_dir"])
    rgb_root = Path(paths["rgb_root"])
    depth_root = Path(paths["depth_root"])
    train_ids = read_ids(split_dir / "depth_train_ids.txt")
    test_ids = read_ids(split_dir / "depth_test_ids.txt")
    if len(train_ids) != expected["train_file_count"]:
        raise SplitError(f"depth_train_ids has {len(train_ids)} rows")
    if len(test_ids) != expected["test_count"]:
        raise SplitError(f"depth_test_ids has {len(test_ids)} rows")
    if len(train_ids) != len(set(train_ids)) or len(test_ids) != len(set(test_ids)):
        raise SplitError("split id list contains duplicates")
    overlap = sorted(set(train_ids) & set(test_ids))
    if overlap:
        raise SplitError(f"train/test overlap ({len(overlap)})")

    test_bad = []
    for dish_id in test_ids:
        reasons = file_reasons(dish_id, rgb_root, depth_root)
        if reasons:
            test_bad.append({"dish_id": dish_id, "reasons": reasons})
    if test_bad:
        raise SplitError(f"test split has missing or zero-byte files: {test_bad[:8]}")

    kept = []
    exclusions = []
    for dish_id in train_ids:
        reasons = file_reasons(dish_id, rgb_root, depth_root)
        if reasons:
            exclusions.append({"dish_id": dish_id, "split": "train", "reasons": reasons})
        else:
            kept.append(dish_id)
    found = sorted(item["dish_id"] for item in exclusions)
    required = sorted(expected["train_only_exclusions"])
    if found != required:
        raise SplitError(f"train exclusions {found} != {required}")
    if len(kept) != expected["train_kept"]:
        raise SplitError(f"kept train count is {len(kept)}")
    return SplitAudit(train_ids=kept, test_ids=list(test_ids), exclusions=exclusions)
