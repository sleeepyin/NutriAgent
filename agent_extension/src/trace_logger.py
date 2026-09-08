"""Structured episode traces. No faces, EXIF, or health identifiers."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from .runtime import atomic_write_text, sha256_obj, utc_now


class TraceLogger:
    def __init__(self, path: str | Path, policy_version: str = "caia-mm-v1"):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.policy_version = policy_version
        self._buf: list[dict[str, Any]] = []

    def add(self, trace: dict[str, Any]) -> None:
        trace = dict(trace)
        trace.setdefault("policy_version", self.policy_version)
        trace.setdefault("logged_at", utc_now())
        self._buf.append(trace)

    def flush(self) -> str:
        text = "\n".join(json.dumps(t, ensure_ascii=False, default=str) for t in self._buf)
        if text:
            text += "\n"
        atomic_write_text(self.path, text)
        return sha256_obj({"n": len(self._buf), "head": self._buf[:3] if self._buf else []})
