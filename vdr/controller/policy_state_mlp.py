"""Closed-loop MLP on ``build_state_features`` (STATE_FEATURE_DIM).

Architecture matches open-loop B5 capacity: input → 64 → 64 → n_actions.
Training uses fit-pool labels only; hold-out is for evaluation elsewhere.
"""

from __future__ import annotations

from typing import Any, Sequence

import numpy as np

from vdr.controller.policy_bc import BC_ACTIONS, _as_labels, _softmax
from vdr.controller.policy_mlp import (
    MLP_ADAM_B1,
    MLP_ADAM_B2,
    MLP_ADAM_EPS,
    MLP_BATCH_SIZE,
    MLP_DROPOUT,
    MLP_EPOCHS,
    MLP_HIDDEN_SIZES,
    MLP_L2,
    MLP_LR,
    MLP_SEED,
    _Adam,
    inverse_sqrt_class_weights,
)
from vdr.controller.state_features import STATE_FEATURE_DIM

_STD_EPS = 1e-8


class PolicyStateMLP:
    """Two-layer ReLU MLP on closed-loop state vectors."""

    def __init__(self, n_actions: int | None = None, **kwargs: Any) -> None:
        actions = kwargs.pop("actions", None)
        del kwargs
        if MLP_DROPOUT != 0.0:
            raise RuntimeError(f"MLP_DROPOUT is fixed at 0 (got {MLP_DROPOUT})")
        if actions is None:
            self.actions = BC_ACTIONS
        else:
            self.actions = tuple(str(name) for name in actions)
            if "VOLUME_ESTIMATE" in self.actions:
                raise ValueError("VOLUME_ESTIMATE is not a BC action")
        if n_actions is not None and int(n_actions) != len(self.actions):
            raise ValueError(f"n_actions={n_actions} != {len(self.actions)}")
        self.n_actions = len(self.actions)
        self._index = {name: i for i, name in enumerate(self.actions)}
        self._model: str | None = None
        self._mean: np.ndarray | None = None
        self._std: np.ndarray | None = None
        self._W1: np.ndarray | None = None
        self._b1: np.ndarray | None = None
        self._W2: np.ndarray | None = None
        self._b2: np.ndarray | None = None
        self._W3: np.ndarray | None = None
        self._b3: np.ndarray | None = None
        self._class_weight: dict[str, float] | None = None
        self._emit: np.ndarray | None = None

    def fit(self, X: Any, y: Any, **kwargs: Any) -> "PolicyStateMLP":
        if kwargs:
            unexpected = ", ".join(sorted(kwargs))
            raise TypeError(f"PolicyStateMLP.fit does not take {unexpected}")
        matrix = np.asarray(X, dtype=np.float64)
        if matrix.ndim != 2 or matrix.shape[1] != STATE_FEATURE_DIM:
            raise ValueError(
                f"X must have shape (n, {STATE_FEATURE_DIM}), got {getattr(matrix, 'shape', None)}"
            )
        if not np.isfinite(matrix).all():
            raise ValueError("X contains a non-finite feature")
        labels = _as_labels(y)
        if len(labels) != matrix.shape[0]:
            raise ValueError(f"X has {matrix.shape[0]} rows but y has {len(labels)} labels")
        if not labels:
            raise ValueError("PolicyStateMLP.fit needs at least one row")
        unknown = sorted({label for label in labels if label not in self._index})
        if unknown:
            raise ValueError(f"labels outside the BC action set: {unknown}")

        target = np.zeros((len(labels), self.n_actions), dtype=np.float64)
        y_index = np.empty(len(labels), dtype=np.int64)
        for row, label in enumerate(labels):
            index = self._index[label]
            target[row, index] = 1.0
            y_index[row] = index

        weights = inverse_sqrt_class_weights(labels, self.actions)
        w_vec = np.array([weights[name] for name in self.actions], dtype=np.float64)
        row_w = w_vec[y_index]
        emit = w_vec > 0.0

        mean = matrix.mean(axis=0)
        std = matrix.std(axis=0)
        std = np.where(std < _STD_EPS, 1.0, std)
        centered = (matrix - mean) / std

        rng = np.random.default_rng(MLP_SEED)
        hidden1, hidden2 = MLP_HIDDEN_SIZES
        in_dim = STATE_FEATURE_DIM
        W1 = rng.normal(0.0, np.sqrt(2.0 / in_dim), size=(in_dim, hidden1)).astype(np.float64)
        b1 = np.zeros(hidden1, dtype=np.float64)
        W2 = rng.normal(0.0, np.sqrt(2.0 / hidden1), size=(hidden1, hidden2)).astype(np.float64)
        b2 = np.zeros(hidden2, dtype=np.float64)
        W3 = rng.normal(0.0, np.sqrt(1.0 / hidden2), size=(hidden2, self.n_actions)).astype(np.float64)
        b3 = np.zeros(self.n_actions, dtype=np.float64)
        adam = _Adam(
            [W1.shape, b1.shape, W2.shape, b2.shape, W3.shape, b3.shape],
            lr=MLP_LR,
            b1=MLP_ADAM_B1,
            b2=MLP_ADAM_B2,
            eps=MLP_ADAM_EPS,
        )

        n_rows = centered.shape[0]
        for _epoch in range(MLP_EPOCHS):
            order = rng.permutation(n_rows)
            for start in range(0, n_rows, MLP_BATCH_SIZE):
                batch = order[start : start + MLP_BATCH_SIZE]
                xb = centered[batch]
                tb = target[batch]
                wb = row_w[batch]
                z1 = xb @ W1 + b1
                h1 = np.maximum(z1, 0.0)
                z2 = h1 @ W2 + b2
                h2 = np.maximum(z2, 0.0)
                logits = h2 @ W3 + b3
                probs = _softmax(logits)
                residual = (probs - tb) * wb[:, None]
                residual /= wb.sum()
                d_h2 = residual @ W3.T
                d_z2 = d_h2 * (z2 > 0.0)
                d_h1 = d_z2 @ W2.T
                d_z1 = d_h1 * (z1 > 0.0)
                grads = (
                    xb.T @ d_z1 + MLP_L2 * W1,
                    d_z1.sum(axis=0),
                    h1.T @ d_z2 + MLP_L2 * W2,
                    d_z2.sum(axis=0),
                    h2.T @ residual + MLP_L2 * W3,
                    residual.sum(axis=0),
                )
                adam.step((W1, b1, W2, b2, W3, b3), grads)

        self._mean = mean
        self._std = std
        self._W1 = W1
        self._b1 = b1
        self._W2 = W2
        self._b2 = b2
        self._W3 = W3
        self._b3 = b3
        self._class_weight = weights
        self._emit = emit
        self._model = f"numpy_state_mlp_{hidden1}_{hidden2}"
        return self

    def predict_actions(self, X: Any) -> list[str]:
        self._require_fit()
        matrix = np.asarray(X, dtype=np.float64)
        if matrix.ndim != 2 or matrix.shape[1] != STATE_FEATURE_DIM:
            raise ValueError(
                f"X must have shape (n, {STATE_FEATURE_DIM}), got {getattr(matrix, 'shape', None)}"
            )
        assert self._mean is not None and self._std is not None
        assert self._W1 is not None and self._b1 is not None
        assert self._W2 is not None and self._b2 is not None
        assert self._W3 is not None and self._b3 is not None
        assert self._emit is not None
        centered = (matrix - self._mean) / self._std
        h1 = np.maximum(centered @ self._W1 + self._b1, 0.0)
        h2 = np.maximum(h1 @ self._W2 + self._b2, 0.0)
        scores = h2 @ self._W3 + self._b3
        scores = scores.copy()
        scores[:, ~self._emit] = -np.inf
        chosen = np.argmax(scores, axis=1)
        return [self.actions[int(index)] for index in chosen]

    def export_numpy(self) -> dict[str, Any]:
        self._require_fit()
        return {
            "model": self._model,
            "actions": self.actions,
            "mean": self._mean,
            "std": self._std,
            "W1": self._W1,
            "b1": self._b1,
            "W2": self._W2,
            "b2": self._b2,
            "W3": self._W3,
            "b3": self._b3,
            "class_weight": self._class_weight,
            "input_dim": STATE_FEATURE_DIM,
        }

    def _require_fit(self) -> None:
        if self._model is None or self._W1 is None:
            raise RuntimeError("PolicyStateMLP.predict_actions: call fit before predict.")
