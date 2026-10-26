"""State feature builder h for the controller (stub).

Expected features (scheme): perception summary, Top-k score gap,
verify_codes one-hot, step budget, error proxy (rule confidence when no GT).
"""

from __future__ import annotations

from typing import Any

import numpy as np


def build_features(state: dict[str, Any], **kwargs: Any) -> np.ndarray:
    raise NotImplementedError(
        "controller.features.build_features: wire in W3-W4 after IR + rule_verify."
    )
