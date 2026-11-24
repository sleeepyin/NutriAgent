"""Hash of the training code. Evaluation prose is not part of the hash."""

from __future__ import annotations

import hashlib
from pathlib import Path

TRAIN_FILES = (
    "protocol.py",
    "depth.py",
    "metadata.py",
    "splits.py",
    "plate_groups.py",
    "semantics.py",
    "losses.py",
    "schedule.py",
    "text.py",
    "datasets.py",
    "index_data.py",
    "engine.py",
    "models/backbone.py",
    "models/fusion.py",
    "models/heads.py",
    "models/geometry_teacher.py",
    "models/ingredient_teacher.py",
    "models/student.py",
)


def code_fingerprint(package_root: Path | None = None) -> str:
    root = package_root or Path(__file__).resolve().parent
    digest = hashlib.sha256()
    for relative in TRAIN_FILES:
        path = root / relative
        digest.update(relative.encode())
        digest.update(path.read_bytes())
    return digest.hexdigest()
