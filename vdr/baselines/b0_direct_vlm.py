"""B0 Direct-VLM baseline (stub).

Single-shot VLM nutrition prediction; no tools, no evidence graph.
Answers: is table lookup necessary?
"""

from __future__ import annotations

from typing import Any


def predict(image_path: str, **kwargs: Any) -> dict[str, Any]:
    """Return a VDR-shaped dict with n filled, G empty/minimal, s='pass', short tau.

    Raises:
        NotImplementedError: until a 7B-8B VLM runtime is configured (2x4090D).
    """
    raise NotImplementedError(
        "b0_direct_vlm.predict: needs VLM runtime on 4090D. See docs/needs_human.md."
    )
