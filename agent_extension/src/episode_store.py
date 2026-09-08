"""Episode store: load/save parquet-like JSONL + numpy sidecar."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable

from .runtime import atomic_write_text


class EpisodeStore:
    def __init__(self, path: str | Path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def write_all(self, rows: Iterable[dict]) -> int:
        lines = [json.dumps(r, ensure_ascii=False, default=str) for r in rows]
        atomic_write_text(self.path, "\n".join(lines) + ("\n" if lines else ""))
        return len(lines)

    def read_all(self) -> list[dict]:
        if not self.path.exists():
            return []
        out = []
        for line in self.path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                out.append(json.loads(line))
        return out

    def by_split(self, split: str) -> list[dict]:
        return [r for r in self.read_all() if r.get("split") == split]
