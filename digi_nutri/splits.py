"""Official Nutrition5k depth-split guards."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from digi_nutri.protocol import EXPECTED_TEST_N


class SplitError(RuntimeError):
    pass


@dataclass
class SplitAudit:
    train_ids: list[str]
    test_ids: list[str]
    exclusions: list[dict]


def read_id_file(path: str | Path) -> list[str]:
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


def audit_split(
    train_ids: list[str],
    test_ids: list[str],
    rgb_root: str | Path,
    depth_root: str | Path,
    expected_test_n: int = EXPECTED_TEST_N,
) -> SplitAudit:
    """Keep the official test ids intact. Drop bad train files only."""
    train_ids = [dish_id.strip() for dish_id in train_ids if dish_id.strip()]
    test_ids = [dish_id.strip() for dish_id in test_ids if dish_id.strip()]
    overlap = sorted(set(train_ids) & set(test_ids))
    if overlap:
        raise SplitError(f"train/test overlap ({len(overlap)}): {overlap[:8]}")
    if len(test_ids) != expected_test_n:
        raise SplitError(f"official test list has {len(test_ids)} ids, expected {expected_test_n}")
    if len(train_ids) != len(set(train_ids)) or len(test_ids) != len(set(test_ids)):
        raise SplitError("split id list contains duplicates")

    rgb_path = Path(rgb_root)
    depth_path = Path(depth_root)
    test_bad = []
    for dish_id in test_ids:
        reasons = file_reasons(dish_id, rgb_path, depth_path)
        if reasons:
            test_bad.append({"dish_id": dish_id, "reasons": reasons})
    if test_bad:
        raise SplitError(f"test split has missing or zero-byte files: {test_bad[:8]}")

    kept = []
    exclusions = []
    for dish_id in train_ids:
        reasons = file_reasons(dish_id, rgb_path, depth_path)
        if reasons:
            exclusions.append({"dish_id": dish_id, "split": "train", "reasons": reasons})
        else:
            kept.append(dish_id)
    if len(test_ids) != expected_test_n:
        raise SplitError(f"n_kept test is {len(test_ids)}, expected {expected_test_n}")
    return SplitAudit(train_ids=kept, test_ids=list(test_ids), exclusions=exclusions)
