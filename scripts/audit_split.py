"""Write the 2754/507 split audit. Does not train."""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from repro.groups import cross_split_pairs, plate_groups
from repro.metadata import load_dishes
from repro.protocol import ROOT as REPO
from repro.protocol import load_config
from repro.splits import audit_split


def main() -> None:
    config = load_config()
    audit = audit_split(config)
    dishes = load_dishes(config["paths"]["cafe1"], config["paths"]["cafe2"])
    missing = [dish_id for dish_id in audit.train_ids + audit.test_ids if dish_id not in dishes]
    if missing:
        raise SystemExit(f"metadata missing {missing[:8]}")
    groups = plate_groups(audit.test_ids, dishes, config)
    n_groups = len(set(groups.values()))
    if n_groups != config["split"]["test_groups"]:
        raise SystemExit(f"test groups {n_groups}")
    overlap = len(set(audit.train_ids) & set(audit.test_ids))
    payload = {
        "n_train_file": config["split"]["train_file_count"],
        "n_train": len(audit.train_ids),
        "n_test": len(audit.test_ids),
        "n_test_groups": n_groups,
        "train_holdout_overlap": overlap,
        "train_exclusions": audit.exclusions,
        "cross_split_reshoots": cross_split_pairs(audit.train_ids, audit.test_ids, dishes, config),
        "status_commit": config["status_commit"],
    }
    dest = REPO / "results" / "repro" / "split_hygiene.json"
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_text(json.dumps(payload, indent=2) + "\n")
    print(json.dumps({key: payload[key] for key in ("n_train", "n_test", "n_test_groups", "train_holdout_overlap")}))


if __name__ == "__main__":
    main()
