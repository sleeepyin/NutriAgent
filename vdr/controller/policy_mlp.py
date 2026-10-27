"""Small MLP behavior-cloning policy.

Stage A in the plan is a small controller on frozen perception. ``PolicyBC``
is the linear baseline. This module is the fixed MLP sibling:

    31 → 64 → 64 → n_actions

ReLU between the hidden layers. ``MLP_DROPOUT`` is fixed at 0: dropout is not
applied, and a non-zero rate is rejected so it cannot be flipped on without
a new implementation. A run is deterministic given ``MLP_SEED``. L2 is applied
to the three weight matrices (not the biases). Features are standardized with
the fit-set mean and standard deviation. Class weights are the inverse
square-root frequency of the fit labels, scaled so the average row weight is
1. A class with no fit rows (the teacher never emits ``RE_DECOMPOSE``) has
weight 0 and is masked at prediction.

The constants below were not selected on the train hold-out or on rgb_test.
``fit`` reads only the matrix it is given. It does not open a jsonl, does not
read ``obs_t``, and does not load torch.
"""

from __future__ import annotations

from typing import Any, Sequence

import numpy as np

from vdr.controller.features import FEATURE_DIM
from vdr.controller.policy_bc import BC_ACTIONS, _as_labels, _softmax

# Two hidden layers of width 64. Fixed. Not a hold-out search.
MLP_HIDDEN_SIZES = (64, 64)
MLP_EPOCHS = 80
MLP_BATCH_SIZE = 512
MLP_LR = 1e-2
MLP_L2 = 1e-4
# Dropout is not used. A non-zero value is rejected in ``PolicyMLP.__init__``.
MLP_DROPOUT = 0.0
MLP_SEED = 0
MLP_ADAM_B1 = 0.9
MLP_ADAM_B2 = 0.999
MLP_ADAM_EPS = 1e-8
_STD_EPS = 1e-8


def inverse_sqrt_class_weights(
    labels: Sequence[str],
    actions: Sequence[str],
) -> dict[str, float]:
    """Inverse square-root frequency, scaled so the mean row weight is 1.

    Counts come from ``labels`` only. A class with zero rows gets weight 0.
    """
    actions = tuple(actions)
    counts = {name: 0 for name in actions}
    for label in labels:
        if label not in counts:
            raise ValueError(f"label {label!r} is outside the action set")
        counts[label] += 1
    n_rows = sum(counts.values())
    if n_rows < 1:
        raise ValueError("inverse_sqrt_class_weights needs at least one label")
    raw = {
        name: (1.0 / np.sqrt(count) if count > 0 else 0.0)
        for name, count in counts.items()
    }
    mass = sum(counts[name] * raw[name] for name in actions)
    if mass <= 0.0:
        raise ValueError("class-weight mass is zero")
    scale = n_rows / mass
    return {name: raw[name] * scale for name in actions}


class PolicyMLP:
    """Two-layer ReLU MLP. ``fit`` trains; ``predict_action`` reads one h."""

    def __init__(self, n_actions: int | None = None, **kwargs: Any) -> None:
        actions = kwargs.pop("actions", None)
        del kwargs
        if MLP_DROPOUT != 0.0:
            raise RuntimeError(
                f"MLP_DROPOUT is fixed at 0 (got {MLP_DROPOUT}); "
                "dropout is not applied"
            )
        if len(MLP_HIDDEN_SIZES) != 2 or any(width < 1 for width in MLP_HIDDEN_SIZES):
            raise RuntimeError(f"MLP_HIDDEN_SIZES must be two positive widths, got {MLP_HIDDEN_SIZES}")
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
        self._W1: np.ndarray | None = None
        self._b1: np.ndarray | None = None
        self._W2: np.ndarray | None = None
        self._b2: np.ndarray | None = None
        self._W3: np.ndarray | None = None
        self._b3: np.ndarray | None = None
        self._class_weight: dict[str, float] | None = None
        self._emit: np.ndarray | None = None

    def fit(self, X: Any, y: Any, **kwargs: Any) -> "PolicyMLP":
        """Fit on pre-action features and action labels.

        ``X`` is ``(n, FEATURE_DIM)`` from ``build_features``. ``y`` is a
        sequence of action names. Extra keyword arguments are rejected so a
        caller cannot pass a hold-out matrix in by accident.
        """
        if kwargs:
            unexpected = ", ".join(sorted(kwargs))
            raise TypeError(f"PolicyMLP.fit does not take {unexpected}")
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
            raise ValueError("PolicyMLP.fit needs at least one row")
        unknown = sorted({label for label in labels if label not in self._index})
        if unknown:
            raise ValueError(f"labels outside the BC action set: {unknown}")
        if "VOLUME_ESTIMATE" in labels:
            raise ValueError("VOLUME_ESTIMATE cannot be a BC label")

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
        W1 = rng.normal(0.0, np.sqrt(2.0 / FEATURE_DIM), size=(FEATURE_DIM, hidden1)).astype(np.float64)
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
                # Mean of row-weighted cross-entropy. Weight sum is over the batch.
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
        self._model = "numpy_mlp_64_64"
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
        """Argmax action for each row of ``X``. Zero-count fit classes are masked."""
        self._require_fit()
        matrix = np.asarray(X, dtype=np.float64)
        if matrix.ndim != 2 or matrix.shape[1] != FEATURE_DIM:
            raise ValueError(
                f"X must have shape (n, {FEATURE_DIM}), got {getattr(matrix, 'shape', None)}"
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

    def _require_fit(self) -> None:
        if self._model is None or self._W1 is None:
            raise RuntimeError("PolicyMLP.predict_action: call fit before predict.")


class _Adam:
    """In-place Adam on a fixed list of arrays. Biases share the same state."""

    def __init__(
        self,
        shapes: Sequence[tuple[int, ...]],
        *,
        lr: float,
        b1: float,
        b2: float,
        eps: float,
    ) -> None:
        self.m = [np.zeros(shape, dtype=np.float64) for shape in shapes]
        self.v = [np.zeros(shape, dtype=np.float64) for shape in shapes]
        self.t = 0
        self.lr = lr
        self.b1 = b1
        self.b2 = b2
        self.eps = eps

    def step(self, params: Sequence[np.ndarray], grads: Sequence[np.ndarray]) -> None:
        self.t += 1
        b1_hat = 1.0 - self.b1**self.t
        b2_hat = 1.0 - self.b2**self.t
        for moment, second, param, grad in zip(self.m, self.v, params, grads):
            moment *= self.b1
            moment += (1.0 - self.b1) * grad
            second *= self.b2
            second += (1.0 - self.b2) * (grad * grad)
            param -= self.lr * (moment / b1_hat) / (np.sqrt(second / b2_hat) + self.eps)
