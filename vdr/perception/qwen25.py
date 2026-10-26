"""Qwen2.5-VL-7B perception for B0 and B1.

Weights are not in the repo. Set VDR_VLM_PATH, or use the default local directory.
The loader does not touch legacy nutrition-regression checkpoints.
"""

from __future__ import annotations

import json
import os
import re
from pathlib import Path
from typing import Any

DEFAULT_VLM_PATH = "/var/tmp/vdr-models/Qwen2.5-VL-7B-Instruct"

_HYP_PROMPT = (
    "This is one overhead photo of a plate of food. "
    "List the visible edible components. "
    "Reply with JSON only: a list of objects, each with "
    '"name" (short English food name) and "portion_g" (estimated edible grams, a number). '
    "Do not include the plate, utensils, or nutrition totals. No markdown."
)

_DIRECT_PROMPT = (
    "This is one overhead photo of a plate of food. "
    "Estimate the edible food on the plate. "
    "Reply with JSON only, one object with numeric keys "
    "energy_kcal, protein_g, carb_g, fat_g, mass_g. No markdown."
)

_model = None
_processor = None
_loaded_path: str | None = None


def vlm_path() -> Path:
    return Path(os.environ.get("VDR_VLM_PATH") or DEFAULT_VLM_PATH)


def _load():
    global _model, _processor, _loaded_path
    path = str(vlm_path())
    if _model is not None and _loaded_path == path:
        return _model, _processor
    import torch
    from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration

    if not Path(path).is_dir():
        raise FileNotFoundError(
            f"Qwen2.5-VL weights not found at {path}. Set VDR_VLM_PATH."
        )
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        path,
        dtype=torch.bfloat16,
        device_map="cuda:0",
    )
    model.eval()
    processor = AutoProcessor.from_pretrained(path)
    _model, _processor, _loaded_path = model, processor, path
    return model, processor


def generate(image_path: str | Path, prompt: str, max_new_tokens: int = 256) -> str:
    """Run one image-text turn and return the decoded completion."""
    import torch
    from qwen_vl_utils import process_vision_info

    model, processor = _load()
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": str(image_path)},
                {"type": "text", "text": prompt},
            ],
        }
    ]
    text = processor.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    image_inputs, video_inputs = process_vision_info(messages)
    inputs = processor(
        text=[text],
        images=image_inputs,
        videos=video_inputs,
        padding=True,
        return_tensors="pt",
    )
    inputs = inputs.to(model.device)
    with torch.inference_mode():
        generated = model.generate(**inputs, max_new_tokens=max_new_tokens)
    trimmed = generated[:, inputs["input_ids"].shape[1] :]
    return processor.batch_decode(trimmed, skip_special_tokens=True)[0].strip()


def _extract_json(text: str) -> Any:
    fenced = re.search(r"```(?:json)?\s*(.*?)```", text, flags=re.DOTALL)
    body = fenced.group(1).strip() if fenced else text.strip()
    try:
        return json.loads(body)
    except json.JSONDecodeError:
        pass
    starts = [i for i in (body.find("["), body.find("{")) if i >= 0]
    if not starts:
        raise ValueError(f"no JSON in VLM output: {text[:240]!r}")
    start = min(starts)
    decoder = json.JSONDecoder()
    value, _ = decoder.raw_decode(body[start:])
    return value


def _portion(value: Any) -> float | None:
    if value is None or value == "":
        return None
    if isinstance(value, str):
        match = re.search(r"[-+]?\d+(?:\.\d+)?", value)
        if not match:
            return None
        value = match.group(0)
    portion = float(value)
    if portion <= 0:
        return None
    return portion


def ingredient_hypotheses(image_path: str | Path) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Return FNDDS-ready component hypotheses and a small VLM trace."""
    raw = generate(image_path, _HYP_PROMPT)
    parsed = _extract_json(raw)
    if isinstance(parsed, dict):
        parsed = parsed.get("components") or parsed.get("hypotheses") or parsed.get("foods") or []
    if not isinstance(parsed, list):
        raise ValueError(f"VLM hypotheses were not a list: {raw[:240]!r}")
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
        if len(hyps) >= 8:
            break
    meta = {"model_path": str(vlm_path()), "raw": raw, "n_hypotheses": len(hyps)}
    return hyps, meta


def direct_nutrition(image_path: str | Path) -> tuple[dict[str, float | None], dict[str, Any]]:
    """B0: one-shot nutrition numbers, no tools."""
    raw = generate(image_path, _DIRECT_PROMPT)
    parsed = _extract_json(raw)
    if not isinstance(parsed, dict):
        raise ValueError(f"VLM nutrition output was not an object: {raw[:240]!r}")
    keys = ("energy_kcal", "protein_g", "carb_g", "fat_g", "mass_g")
    out: dict[str, float | None] = {}
    for key in keys:
        value = parsed.get(key)
        out[key] = None if value is None else float(value)
    meta = {"model_path": str(vlm_path()), "raw": raw}
    return out, meta
