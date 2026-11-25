"""Official split guards used by Plan A. Test files are checked, not scored."""

from __future__ import annotations

from pathlib import Path

from digi_nutri.protocol import SPLIT_DIR
from digi_nutri.splits import audit_split, read_id_file


def load_cohort(config: dict) -> tuple[list[str], list[str]]:
    split = Path(SPLIT_DIR)
    train_file = read_id_file(split / "depth_train_ids.txt")
    test_file = read_id_file(split / "depth_test_ids.txt")
    audit = audit_split(train_file, test_file, config["rgb_root"], config["depth_root"])
    if len(audit.test_ids) != int(config["n_test"]):
        raise RuntimeError(f"n_kept test is {len(audit.test_ids)}, config says {config['n_test']}")
    if len(audit.train_ids) != int(config["n_train"]):
        raise RuntimeError(f"n_kept train is {len(audit.train_ids)}, config says {config['n_train']}")
    found = {row["dish_id"] for row in audit.exclusions}
    expected = {row["dish_id"] for row in config["train_exclusions"]}
    if found != expected:
        raise RuntimeError(f"train exclusions changed: {sorted(found)}")
    return audit.train_ids, audit.test_ids
