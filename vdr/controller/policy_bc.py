"""Behavior-cloning policy pi(a|h).

Stage A in the plan is a small controller on top of a frozen perception
model. This module is the CPU toy: multinomial logistic regression in
numpy. It does not load torch, does not touch the 7B VLM, and does not
read ``obs_t``.

``LOGISTIC_EPOCHS``, ``LOGISTIC_LR``, and ``LOGISTIC_L2`` are fixed. They
were not selected on the train hold-out or on rgb_test. The linear head stays
the baseline. The small MLP sibling is ``vdr.controller.policy_mlp.PolicyMLP``
and uses the same ``build_features`` vector.
"""

from __future__ import annotations

from typing import Any, Sequence

import numpy as np

from vdr.controller.features import FEATURE_DIM
from vdr.schemas.loader import load_actions

# Full-batch gradient steps. Convex multinomial logistic, zero init.
LOGISTIC_EPOCHS = 80
LOGISTIC_LR = 2.0
LOGISTIC_L2 = 1e-4
_STD_EPS = 1e-8


def bc_action_names() -> tuple[str, ...]:
    """Legal BC actions: ``actions.json`` enum, volume tool removed."""
    spec = load_actions()
    names = tuple(spec["enums"]["action"])
    banned = {"VOLUME_ESTIMATE"}
    kept = tuple(name for name in names if name not in banned)
    if not kept or any(name in banned for name in kept):
        raise RuntimeError(f"BC action set is empty or still contains a volume action: {kept}")
    return kept


BC_ACTIONS = bc_action_names()


class PolicyBC:
    """Linear softmax policy. ``fit`` trains; ``predict_action`` reads one h."""

    def __init__(self, n_actions: int | None = None, **kwargs: Any) -> None:
        actions = kwargs.pop("actions", None)
        # kwargs kept so older call sites that passed unused options still construct.
        del kwargs
        if actions is None:
            self.actions = BC_ACTIONS
        else:
            self.actions = tuple(str(name) for name in actions)
            if "VOLUME_ESTIMATE" in self.actions:
                raise ValueError("VOLUME_ESTIMATE is not a BC action")
            if len(set(self.actions)) != len(self.actions):
                raise ValueError(f"duplicate BC actions: {self.actions}")
        if n_actions is not None and int(n_actions) != len(self.actions):
            raise ValueError(
                f"n_actions={n_actions} does not match the BC action set "
                f"({len(self.actions)}): {self.actions}"
            )
        self.n_actions = len(self.actions)
        self._index = {name: i for i, name in enumerate(self.actions)}
        self._model: str | None = None
        self._mean: np.ndarray | None = None
        self._std: np.ndarray | None = None
        self._W: np.ndarray | None = None
        self._b: np.ndarray | None = None

    def fit(self, X: Any, y: Any, **kwargs: Any) -> "PolicyBC":
        """Fit the linear softmax on pre-action features and action labels.

        ``X`` is ``(n, FEATURE_DIM)`` from ``build_features``. ``y`` is a
        sequence of action names. Extra keyword arguments are rejected so a
        caller cannot pass a hold-out matrix in by accident.
        """
        if kwargs:
            unexpected = ", ".join(sorted(kwargs))
            raise TypeError(f"PolicyBC.fit does not take {unexpected}")
        matrix = np.asarray(X, dtype=np.float64)
        if matrix.ndim != 2 or matrix.shape[1] != FEATURE_DIM:
            raise ValueError(
                f"X must have shape (n, {FEATURE_DIM}), got {getattr(matrix, 'shape', None)}"
            )
        if not np.isfinite(matrix).all():
            raise ValueError("X contains a non-finite feature")
        labels = _as_labels(y)
        if len(labels) != matrix.shape[0]:
            raise ValueError(f"X has {matrix.shape[0]} rows but y has {len(labels)} labels")
        if not labels:
            raise ValueError("PolicyBC.fit needs at least one row")
        unknown = sorted({label for label in labels if label not in self._index})
        if unknown:
            raise ValueError(f"labels outside the BC action set: {unknown}")
        if "VOLUME_ESTIMATE" in labels:
            raise ValueError("VOLUME_ESTIMATE cannot be a BC label")

        target = np.zeros((len(labels), self.n_actions), dtype=np.float64)
        for row, label in enumerate(labels):
            target[row, self._index[label]] = 1.0

        mean = matrix.mean(axis=0)
        std = matrix.std(axis=0)
        std = np.where(std < _STD_EPS, 1.0, std)
        centered = (matrix - mean) / std

        weights = np.zeros((FEATURE_DIM, self.n_actions), dtype=np.float64)
        bias = np.zeros(self.n_actions, dtype=np.float64)
        n_rows = centered.shape[0]
        for _epoch in range(LOGISTIC_EPOCHS):
            logits = centered @ weights + bias
            probs = _softmax(logits)
            grad = (probs - target) / n_rows
            weights -= LOGISTIC_LR * (centered.T @ grad + LOGISTIC_L2 * weights)
            bias -= LOGISTIC_LR * grad.sum(axis=0)

        self._mean = mean
        self._std = std
        self._W = weights
        self._b = bias
        self._model = "numpy_multinomial_logistic"
        return self

    def predict_action(self, features: Any) -> str:
        """Return one action name for a single feature vector."""
        self._require_fit()
        vector = np.asarray(features, dtype=np.float64)
        if vector.ndim == 1:
            vector = vector.reshape(1, -1)
        if vector.shape != (1, FEATURE_DIM):
            raise ValueError(
                f"features must have shape ({FEATURE_DIM},) or (1, {FEATURE_DIM}), got {vector.shape}"
            )
        return self.predict_actions(vector)[0]

    def predict_actions(self, X: Any) -> list[str]:
        """Argmax action for each row of ``X``."""
        self._require_fit()
        matrix = np.asarray(X, dtype=np.float64)
        if matrix.ndim != 2 or matrix.shape[1] != FEATURE_DIM:
            raise ValueError(
                f"X must have shape (n, {FEATURE_DIM}), got {getattr(matrix, 'shape', None)}"
            )
        assert self._mean is not None and self._std is not None
        assert self._W is not None and self._b is not None
        centered = (matrix - self._mean) / self._std
        scores = centered @ self._W + self._b
        chosen = np.argmax(scores, axis=1)
        return [self.actions[int(index)] for index in chosen]

    def _require_fit(self) -> None:
        if self._model is None or self._W is None:
            raise RuntimeError("PolicyBC.predict_action: call fit before predict.")


def _as_labels(y: Any) -> list[str]:
    if isinstance(y, np.ndarray):
        return [str(item) for item in y.tolist()]
    if isinstance(y, Sequence) and not isinstance(y, (str, bytes)):
        return [str(item) for item in y]
    raise TypeError(f"y must be a sequence of action names, got {type(y).__name__}")


def _softmax(logits: np.ndarray) -> np.ndarray:
    shifted = logits - logits.max(axis=1, keepdims=True)
    exp = np.exp(shifted)
    return exp / exp.sum(axis=1, keepdims=True)
