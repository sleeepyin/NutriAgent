"""Behavior-cloning policy pi(a|h) stub.

Stage A: cross-entropy BC on rule-teacher (+ partial human) trajectories.
Do NOT backprop into 7B VLM; small MLP/Transformer only.
"""

from __future__ import annotations

from typing import Any


class PolicyBC:
    def __init__(self, n_actions: int = 5, **kwargs: Any) -> None:
        self.n_actions = n_actions
        self._model = None  # torch module reserved

    def predict_action(self, features: Any) -> str:
        raise NotImplementedError("PolicyBC.predict_action: train after weak labels (W4+).")

    def fit(self, X: Any, y: Any, **kwargs: Any) -> "PolicyBC":
        raise NotImplementedError("PolicyBC.fit: not yet implemented.")
