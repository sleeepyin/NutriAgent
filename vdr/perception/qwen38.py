"""Local Qwen3.8-27B vision client for on-policy perception.

The weights and server live on this machine, outside the git repo:

- weights: ``/host/var/tmp/sdnu-qwen38-vlm/Qwen3.8-27B-UD-Q5_K_M.gguf``
- mmproj: ``/host/var/tmp/sdnu-qwen38-vlm/mmproj-BF16.gguf``
- server: llama.cpp ``llama-server`` (build b11581) on ``127.0.0.1:18080``,
  GPU 0, OpenAI-compatible ``/v1/chat/completions``
- alias: ``qwen3.8-27b``

The API key is read from ``QWEN_API_KEY`` or from the server config file.
It is never written into results. This client is the closed-loop perception
backbone. It is not a faithful DietAI24 or FoodAgent call.

No cafe label and no Nutrition5K ingredient row is read here.
"""

from __future__ import annotations

import base64
import json
import os
import re
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any

DEFAULT_BASE = "http://127.0.0.1:18080"
DEFAULT_MODEL = "qwen3.8-27b"
DEFAULT_CONFIG = Path("/home/sdnu/qwen38-vlm-api/config.env")
WEIGHTS_FILE = "Qwen3.8-27B-UD-Q5_K_M.gguf"
MMPROJ_FILE = "mmproj-BF16.gguf"
SERVING = "llama.cpp llama-server b11581 OpenAI /v1/chat/completions GPU0:18080"

_HYP_PROMPT = (
    "This is one overhead photo of a plate of food. "
    "List the visible edible components. "
    'Reply with JSON only: {"components": [{"name": "short English food name", '
    '"portion_g": number}]}. '
    "portion_g is estimated edible grams. "
    "Do not include the plate, utensils, or nutrition totals. No markdown."
)

_THINK_RE = re.compile(r"<think>.*?</think>", flags=re.DOTALL | re.IGNORECASE)


class Qwen38Error(RuntimeError):
    """The local vision server did not return usable hypotheses."""


def api_base() -> str:
    return (os.environ.get("QWEN_BASE") or DEFAULT_BASE).rstrip("/")


def model_alias() -> str:
    return os.environ.get("QWEN_MODEL") or DEFAULT_MODEL


def _api_key() -> str:
    env = os.environ.get("QWEN_API_KEY")
    if env and env.strip():
        return env.strip()
    path = Path(os.environ.get("QWEN_CONFIG") or DEFAULT_CONFIG)
    if path.is_file():
        for line in path.read_text(encoding="utf-8").splitlines():
            if line.startswith("API_KEY="):
                value = line.split("=", 1)[1].strip()
                if value:
                    return value
    raise Qwen38Error(
        "QWEN_API_KEY is unset and the local llama-server config has no API_KEY. "
        "Refusing to invent hypotheses."
    )


def _request(path: str, payload: dict[str, Any] | None, timeout: float) -> Any:
    data = None if payload is None else json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(
        api_base() + path,
        data=data,
        method="GET" if data is None else "POST",
    )
    if data is not None:
        req.add_header("Content-Type", "application/json")
    if path != "/health":
        req.add_header("Authorization", "Bearer " + _api_key())
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            body = resp.read().decode("utf-8")
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="replace")[:240]
        raise Qwen38Error(f"Qwen HTTP {exc.code} on {path}: {detail}") from exc
    except urllib.error.URLError as exc:
        raise Qwen38Error(f"Qwen server unreachable at {api_base()}{path}: {exc.reason}") from exc
    if not body:
        return {}
    try:
        return json.loads(body)
    except json.JSONDecodeError as exc:
        raise Qwen38Error(f"Qwen returned non-JSON from {path}: {body[:180]!r}") from exc


def server_ready() -> dict[str, Any]:
    """Confirm /health and that the alias is one of /v1/models. Does not generate."""
    health = _request("/health", None, timeout=5)
    models = _request("/v1/models", None, timeout=20)
    ids: list[str] = []
    for row in models.get("data") or []:
        if isinstance(row, dict) and row.get("id"):
            ids.append(str(row["id"]))
    alias = model_alias()
    if alias not in ids:
        raise Qwen38Error(f"model alias {alias!r} not in {ids or models!r}")
    return {
        "base_url": api_base(),
        "model_alias": alias,
        "weights_file": WEIGHTS_FILE,
        "mmproj_file": MMPROJ_FILE,
        "serving": SERVING,
        "health": health if isinstance(health, (dict, str)) else str(health),
        "model_ids": ids,
    }


def _strip_think(text: str) -> str:
    return _THINK_RE.sub("", text).strip()


def _extract_json(text: str) -> Any:
    fenced = re.search(r"```(?:json)?\s*(.*?)```", text, flags=re.DOTALL)
    body = fenced.group(1).strip() if fenced else text.strip()
    try:
        return json.loads(body)
    except json.JSONDecodeError:
        pass
    starts = [i for i in (body.find("["), body.find("{")) if i >= 0]
    if not starts:
        raise Qwen38Error(f"no JSON in VLM output: {text[:240]!r}")
    decoder = json.JSONDecoder()
    value, _ = decoder.raw_decode(body[min(starts) :])
    return value


def _portion(value: Any) -> float | None:
    if value is None or value == "":
        return None
    if isinstance(value, str):
        match = re.search(r"[-+]?\d+(?:\.\d+)?", value)
        if not match:
            return None
        value = match.group(0)
    try:
        portion = float(value)
    except (TypeError, ValueError):
        return None
    if portion <= 0:
        return None
    return portion


def _message_text(payload: dict[str, Any]) -> str:
    choices = payload.get("choices") or []
    if not choices or not isinstance(choices[0], dict):
        raise Qwen38Error(f"Qwen completion has no choices: {str(payload)[:240]}")
    message = choices[0].get("message") or {}
    content = message.get("content")
    if isinstance(content, list):
        parts: list[str] = []
        for item in content:
            if isinstance(item, str):
                parts.append(item)
            elif isinstance(item, dict):
                parts.append(str(item.get("text") or ""))
        content = "".join(parts)
    if not isinstance(content, str) or not content.strip():
        raise Qwen38Error(f"Qwen completion content is empty: {str(payload)[:240]}")
    return _strip_think(content)


def _parse_hypotheses(text: str) -> list[dict[str, Any]]:
    parsed = _extract_json(text)
    if isinstance(parsed, dict):
        parsed = parsed.get("components") or parsed.get("hypotheses") or parsed.get("foods") or []
    if not isinstance(parsed, list):
        raise Qwen38Error(f"VLM hypotheses were not a list: {text[:240]!r}")
    hyps: list[dict[str, Any]] = []
    for item in parsed:
        if isinstance(item, str):
            name, portion = item, None
        elif isinstance(item, dict):
            name = item.get("name") or item.get("food") or item.get("hyp")
            portion = _portion(item.get("portion_g", item.get("grams")))
        else:
            continue
        if not name or not str(name).strip():
            continue
        hyps.append({"name": str(name).strip(), "portion_g": portion})
    return hyps


def _complete(image_path: Path, *, json_object: bool) -> str:
    encoded = base64.standard_b64encode(image_path.read_bytes()).decode("ascii")
    payload: dict[str, Any] = {
        "model": model_alias(),
        "temperature": 0,
        "max_tokens": 384,
        "chat_template_kwargs": {"enable_thinking": False},
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": _HYP_PROMPT},
                    {
                        "type": "image_url",
                        "image_url": {"url": "data:image/png;base64," + encoded},
                    },
                ],
            }
        ],
    }
    if json_object:
        payload["response_format"] = {"type": "json_object"}
    try:
        body = _request("/v1/chat/completions", payload, timeout=180)
    except Qwen38Error:
        if not json_object:
            raise
        payload.pop("response_format", None)
        body = _request("/v1/chat/completions", payload, timeout=180)
    if not isinstance(body, dict):
        raise Qwen38Error(f"Qwen completion was not an object: {str(body)[:180]}")
    return _message_text(body)


def ingredient_hypotheses(image_path: str | Path) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """One overhead image → component hypotheses. Does not read dish labels."""
    path = Path(image_path)
    if not path.is_file():
        raise Qwen38Error(f"image not found: {path}")
    last_error: Exception | None = None
    hyps: list[dict[str, Any]] = []
    for attempt in range(2):
        try:
            text = _complete(path, json_object=(attempt == 0))
            hyps = _parse_hypotheses(text)
            last_error = None
            break
        except Qwen38Error as exc:
            last_error = exc
    if last_error is not None:
        raise Qwen38Error(str(last_error)) from last_error
    meta = {
        "model_alias": model_alias(),
        "base_url": api_base(),
        "weights_file": WEIGHTS_FILE,
        "mmproj_file": MMPROJ_FILE,
        "serving": SERVING,
        "n_hypotheses_raw": len(hyps),
        "temperature": 0,
        "not_faithful_dietai24": True,
        "not_faithful_foodagent": True,
    }
    return hyps, meta
