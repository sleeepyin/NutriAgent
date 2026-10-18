"""Shared runtime: logging, heartbeat, atomic IO, hashing, timeouts."""
from __future__ import annotations

import hashlib
import json
import os
import sys
import threading
import time
import traceback
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Iterator


def utc_now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def sha256_file(path: str | Path, chunk: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while True:
            b = f.read(chunk)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def sha256_obj(obj: Any) -> str:
    blob = json.dumps(obj, sort_keys=True, default=str, ensure_ascii=False)
    return sha256_text(blob)


def atomic_write_text(path: str | Path, text: str) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + f".tmp.{os.getpid()}")
    tmp.write_text(text, encoding="utf-8")
    os.replace(tmp, path)


def atomic_write_json(path: str | Path, obj: Any, indent: int = 2) -> None:
    atomic_write_text(path, json.dumps(obj, indent=indent, ensure_ascii=False, default=str) + "\n")


def read_json(path: str | Path, default: Any = None) -> Any:
    path = Path(path)
    if not path.exists():
        return default
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return default


class Heartbeat:
    """Background heartbeat so the watchdog can detect hangs."""

    def __init__(self, path: str | Path, interval: float = 20.0):
        self.path = Path(path)
        self.interval = interval
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self.step = "boot"
        self.note = ""

    def start(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()
        self.touch()

    def touch(self, step: str | None = None, note: str = "") -> None:
        if step:
            self.step = step
        if note:
            self.note = note
        atomic_write_json(
            self.path,
            {
                "ts": utc_now(),
                "unix": time.time(),
                "pid": os.getpid(),
                "step": self.step,
                "note": self.note,
            },
        )

    def _loop(self) -> None:
        while not self._stop.wait(self.interval):
            try:
                self.touch()
            except Exception:
                pass

    def stop(self) -> None:
        self._stop.set()


class TeeLogger:
    def __init__(self, log_path: str | Path):
        self.log_path = Path(log_path)
        self.log_path.parent.mkdir(parents=True, exist_ok=True)
        self._fh = open(self.log_path, "a", encoding="utf-8", buffering=1)
        self._stdout = sys.stdout
        self._stderr = sys.stderr

    def write(self, s: str) -> int:
        self._stdout.write(s)
        self._fh.write(s)
        return len(s)

    def flush(self) -> None:
        self._stdout.flush()
        self._fh.flush()

    def close(self) -> None:
        try:
            self._fh.close()
        except Exception:
            pass

    def install(self) -> None:
        sys.stdout = self  # type: ignore[assignment]
        sys.stderr = self  # type: ignore[assignment]

    def restore(self) -> None:
        sys.stdout = self._stdout
        sys.stderr = self._stderr
        self.close()


def log(msg: str, level: str = "INFO") -> None:
    print(f"[{utc_now()}] [{level}] {msg}", flush=True)


@contextmanager
def step_guard(name: str) -> Iterator[None]:
    log(f"BEGIN {name}")
    t0 = time.time()
    try:
        yield
        log(f"END {name} ok dt={time.time() - t0:.1f}s")
    except Exception as e:
        log(f"END {name} FAIL {type(e).__name__}: {e} dt={time.time() - t0:.1f}s", "ERROR")
        traceback.print_exc()
        raise


def retry_call(
    fn: Callable,
    *,
    retries: int = 3,
    backoff: float = 2.0,
    label: str = "call",
    retry_exceptions: tuple = (Exception,),
) -> Any:
    last = None
    for i in range(1, retries + 1):
        try:
            return fn()
        except retry_exceptions as e:
            last = e
            log(f"{label} attempt {i}/{retries} failed: {type(e).__name__}: {e}", "WARN")
            if i < retries:
                time.sleep(backoff * i)
    raise last
