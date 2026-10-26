"""B0 Direct-VLM baseline.

Single-shot Qwen2.5-VL nutrition prediction; no tools, no evidence graph.
Answers: is table lookup necessary?
"""

from __future__ import annotations

from pathlib import Path
from typing import Any


def _dish_id(image_path: str, explicit: str | None) -> str:
    if explicit:
        return explicit
    parent = Path(image_path).parent.name
    if parent.startswith("dish_"):
        return parent
    return Path(image_path).stem


def predict(image_path: str, **kwargs: Any) -> dict[str, Any]:
    """Single-shot VLM nutrition. No tools and no evidence path, so G stays empty."""
    from vdr.perception.qwen25 import direct_nutrition

    nutrition, vlm_meta = direct_nutrition(image_path)
    dish_id = _dish_id(image_path, kwargs.get("dish_id"))
    return {
        "dish_id": dish_id,
        "n": nutrition,
        "G": {
            "nodes": [{"id": "dish", "type": "dish", "label": dish_id}],
            "edges": [],
        },
        "s": "pass",
        "tau": [
            {
                "t": 0,
                "action": "STOP",
                "obs": {"baseline": "b0_direct_vlm"},
                "state_summary": "direct VLM nutrition, no tools",
            }
        ],
        "abstain_reason": None,
        "meta": {"baseline": "b0_direct_vlm", "vlm": vlm_meta},
    }
