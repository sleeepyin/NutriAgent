"""The official holdout file is written once."""

from __future__ import annotations

import json
import os
from pathlib import Path

from digi_nutri.jsonio import to_jsonable


class HoldoutLocked(RuntimeError):
    pass


def assert_holdout_available(path: Path) -> None:
    if Path(path).exists():
        raise HoldoutLocked(
            f"refusing a second official holdout eval; metrics already exist at {path}"
        )


def publish_holdout(path: Path, payload: dict) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        raise HoldoutLocked(
            f"refusing a second official holdout eval; metrics already exist at {path}"
        )
    text = json.dumps(to_jsonable(payload), indent=2, sort_keys=True) + "\n"
    flags = os.O_CREAT | os.O_EXCL | os.O_WRONLY
    try:
        fd = os.open(path, flags, 0o644)
    except FileExistsError as exc:
        raise HoldoutLocked(
            f"refusing a second official holdout eval; metrics already exist at {path}"
        ) from exc
    with os.fdopen(fd, "w", encoding="utf-8") as handle:
        handle.write(text)
        handle.flush()
        os.fsync(handle.fileno())
