"""Volume estimation tool — STUB, DEFAULT OFF.

Per scheme / Tanabe 2025: monocular volume prompting can catastrophically
fail on open-source VLMs. Do not call unless ablation flag
`VDR_ENABLE_VOLUME=1` is set. This module must not be imported by default
tool routers.
"""

from __future__ import annotations

from typing import Any


ENABLED_BY_DEFAULT = False


def estimate_volume(*args: Any, **kwargs: Any) -> dict[str, Any]:
    raise NotImplementedError(
        "volume_estimate: intentionally unimplemented and DEFAULT OFF. "
        "Ablation only; see schemas/actions.json optional VOLUME_ESTIMATE."
    )
