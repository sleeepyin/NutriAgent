"""Cache resized RGB and relative-height maps outside the public data tree."""

from __future__ import annotations

import json
from pathlib import Path

from digi_nutri.data import ensure_cache
from digi_nutri.protocol import DEPTH_ROOT, RGB_ROOT, SPLIT_DIR
from digi_nutri.splits import read_id_file


def main() -> None:
    config = json.loads(Path("configs/q0_frozen.json").read_text())
    excluded = {row["dish_id"] for row in config["train_exclusions"]}
    train_ids = [
        dish_id
        for dish_id in read_id_file(Path(SPLIT_DIR) / "depth_train_ids.txt")
        if dish_id not in excluded
    ]
    test_ids = read_id_file(Path(SPLIT_DIR) / "depth_test_ids.txt")
    ensure_cache(train_ids + test_ids, RGB_ROOT, DEPTH_ROOT, config["cache_dir"])
    print(f"cached {len(train_ids) + len(test_ids)}", flush=True)


if __name__ == "__main__":
    main()
