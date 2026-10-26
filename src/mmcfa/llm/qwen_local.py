"""Local Qwen3.8-27B client. The API key stays outside this repository."""
from __future__ import annotations

import base64
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


def _post(payload: dict, timeout: int) -> dict:
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
            return json.loads(resp.read().decode())
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode(errors="replace")[:300]
        raise RuntimeError(f"Qwen HTTP {exc.code}: {detail}") from exc


def chat_json(prompt: str, timeout: int = 120) -> dict:
    payload = {
        "model": os.environ.get("QWEN_MODEL", "qwen3.8-27b"),
        "temperature": 0,
        "max_tokens": 160,
        "chat_template_kwargs": {"enable_thinking": False},
        "response_format": {"type": "json_object"},
        "messages": [{"role": "user", "content": prompt}],
    }
    body = _post(payload, timeout)
    content = body["choices"][0]["message"]["content"]
    return json.loads(content)


def chat_vision_json(prompt: str, image_jpeg: bytes, timeout: int = 300, max_tokens: int = 800) -> dict:
    """One image plus text. The image is a JPEG byte string, not a path that might carry a label."""
    encoded = base64.b64encode(image_jpeg).decode("ascii")
    payload = {
        "model": os.environ.get("QWEN_MODEL", "qwen3.8-27b"),
        "temperature": 0,
        "max_tokens": max_tokens,
        "chat_template_kwargs": {"enable_thinking": False},
        "response_format": {"type": "json_object"},
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt},
                    {
                        "type": "image_url",
                        "image_url": {"url": "data:image/jpeg;base64," + encoded},
                    },
                ],
            }
        ],
    }
    body = _post(payload, timeout)
    choice = body["choices"][0]
    content = choice["message"].get("content") or ""
    finish = choice.get("finish_reason")
    try:
        parsed = json.loads(content)
    except json.JSONDecodeError:
        start = content.find("{")
        end = content.rfind("}")
        if start < 0 or end <= start:
            raise RuntimeError(f"Qwen vision did not return JSON (finish={finish}): {content[:240]}")
        parsed = json.loads(content[start : end + 1])
    if not isinstance(parsed, dict):
        raise RuntimeError("Qwen vision JSON was not an object")
    parsed["_finish_reason"] = finish
    return parsed
