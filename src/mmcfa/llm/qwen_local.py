"""Local Qwen3.8-27B client. The API key stays outside this repository."""
from __future__ import annotations

import json
import os
import urllib.error
import urllib.request
from pathlib import Path


def _key() -> str:
    env = os.environ.get("QWEN_API_KEY", "").strip()
    if env:
        return env
    path = os.environ.get("QWEN_API_KEY_FILE", "/home/sdnu/qwen38-vlm-api/api.key")
    file = Path(path)
    if file.is_file():
        return file.read_text(encoding="utf-8").strip()
    raise RuntimeError("Set QWEN_API_KEY or QWEN_API_KEY_FILE. Do not commit the key.")


def chat_json(prompt: str, timeout: int = 120) -> dict:
    payload = {
        "model": os.environ.get("QWEN_MODEL", "qwen3.8-27b"),
        "temperature": 0,
        "max_tokens": 160,
        "chat_template_kwargs": {"enable_thinking": False},
        "response_format": {"type": "json_object"},
        "messages": [{"role": "user", "content": prompt}],
    }
    base = os.environ.get("QWEN_BASE", "http://127.0.0.1:18080").rstrip("/")
    req = urllib.request.Request(
        base + "/v1/chat/completions",
        data=json.dumps(payload).encode(),
        method="POST",
    )
    req.add_header("Content-Type", "application/json")
    req.add_header("Authorization", "Bearer " + _key())
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            body = json.loads(resp.read().decode())
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode(errors="replace")[:300]
        raise RuntimeError(f"Qwen HTTP {exc.code}: {detail}") from exc
    content = body["choices"][0]["message"]["content"]
    return json.loads(content)
