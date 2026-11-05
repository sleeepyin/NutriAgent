"""Supervised ``Q_φ(z, a) ≈ E[ΔL | z, a]``. Not reinforcement learning.

Each logged row supplies one action and the nutrition-loss change of that
action. The loss is squared error on that coordinate. Every other action on
the same row is anchored at 0, because the row did not measure it. There is
no bootstrap, no target network, no ``max_a Q(z', a)``, and no policy gradient.
Actions that never appear in the fit set are masked at selection time
because they have no logged ``ΔL``; that mask is a count, not a penalty.

Selection is ``a* = argmax_a Q_φ(z, a) / Cost(a)`` with the costs in
``vdr.controller.utility``. Width, step size, epoch count, and seed match
the state-aware BC MLP and were not searched on hold-out or on rgb_test.
"""

from __future__ import annotations

from typing import Any, Mapping, Sequence

import numpy as np

from vdr.controller.policy_bc import BC_ACTIONS
from vdr.controller.policy_mlp import (
    MLP_ADAM_B1,
    MLP_ADAM_B2,
    MLP_ADAM_EPS,
    MLP_BATCH_SIZE,
    MLP_EPOCHS,
    MLP_HIDDEN_SIZES,
    MLP_L2,
    MLP_LR,
    MLP_SEED,
    _Adam,
)
from vdr.controller.q_state import Z_DIM
from vdr.controller.utility import argmax_value_over_cost

_STD_EPS = 1e-8
# Same constants as PolicyStateMLP. Not a second search.
Q_EPOCHS = MLP_EPOCHS
Q_BATCH_SIZE = MLP_BATCH_SIZE
Q_LR = MLP_LR
Q_L2 = MLP_L2
Q_SEED = MLP_SEED
# Unobserved actions on a logged row are anchored at 0 gain. This row did not
# measure them. The anchor is 1, the same weight as the observed residual, and
# it was not searched on hold-out or on rgb_test. It is not a bootstrap.
Q_UNOBSERVED_ANCHOR = 1.0


class QPhi:
    """Two-layer ReLU regression head. One scalar per action."""

    def __init__(self) -> None:
        self.actions = BC_ACTIONS
        self.n_actions = len(self.actions)
        self._index = {name: index for index, name in enumerate(self.actions)}
        self._model: str | None = None
        self._mean: np.ndarray | None = None
        self._std: np.ndarray | None = None
        self._W1: np.ndarray | None = None
        self._b1: np.ndarray | None = None
        self._W2: np.ndarray | None = None
        self._b2: np.ndarray | None = None
        self._W3: np.ndarray | None = None
        self._b3: np.ndarray | None = None
        self._emit: np.ndarray | None = None
        self.action_counts: dict[str, int] | None = None
        self.train_mse_: float | None = None

    def fit(self, Z: Any, actions: Sequence[str], delta_l: Sequence[float]) -> "QPhi":
        """Regress logged ``ΔL``. ``Z`` is the error state. No next-state backup."""
        matrix = np.asarray(Z, dtype=np.float64)
        if matrix.ndim != 2 or matrix.shape[1] != Z_DIM:
            raise ValueError(f"Z must have shape (n, {Z_DIM}), got {getattr(matrix, 'shape', None)}")
        if not np.isfinite(matrix).all():
            raise ValueError("Z contains a non-finite feature")
        labels = [str(action) for action in actions]
        target = np.asarray(delta_l, dtype=np.float64)
        if len(labels) != matrix.shape[0] or target.shape != (matrix.shape[0],):
            raise ValueError("Z, actions, and delta_l must have the same length")
        if not np.isfinite(target).all():
            raise ValueError("delta_l contains a non-finite value")
        if not labels:
            raise ValueError("QPhi.fit needs at least one row")
        unknown = sorted({label for label in labels if label not in self._index})
        if unknown:
            raise ValueError(f"actions outside the controller set: {unknown}")

        counts = {name: 0 for name in self.actions}
        y_index = np.empty(len(labels), dtype=np.int64)
        for row, label in enumerate(labels):
            y_index[row] = self._index[label]
            counts[label] += 1
        emit = np.array([counts[name] > 0 for name in self.actions], dtype=bool)
        if not bool(emit.any()):
            raise ValueError("every action count is zero")

        mean = matrix.mean(axis=0)
        std = matrix.std(axis=0)
        std = np.where(std < _STD_EPS, 1.0, std)
        centered = (matrix - mean) / std

        rng = np.random.default_rng(Q_SEED)
        hidden1, hidden2 = MLP_HIDDEN_SIZES
        W1 = rng.normal(0.0, np.sqrt(2.0 / Z_DIM), size=(Z_DIM, hidden1)).astype(np.float64)
        b1 = np.zeros(hidden1, dtype=np.float64)
        W2 = rng.normal(0.0, np.sqrt(2.0 / hidden1), size=(hidden1, hidden2)).astype(np.float64)
        b2 = np.zeros(hidden2, dtype=np.float64)
        W3 = rng.normal(0.0, np.sqrt(1.0 / hidden2), size=(hidden2, self.n_actions)).astype(np.float64)
        b3 = np.zeros(self.n_actions, dtype=np.float64)
        adam = _Adam(
            [W1.shape, b1.shape, W2.shape, b2.shape, W3.shape, b3.shape],
            lr=Q_LR,
            b1=MLP_ADAM_B1,
            b2=MLP_ADAM_B2,
            eps=MLP_ADAM_EPS,
        )

        n_rows = centered.shape[0]
        for _epoch in range(Q_EPOCHS):
            order = rng.permutation(n_rows)
            for start in range(0, n_rows, Q_BATCH_SIZE):
                batch = order[start : start + Q_BATCH_SIZE]
                xb = centered[batch]
                yb = y_index[batch]
                tb = target[batch]
                n_batch = xb.shape[0]
                h1_pre = xb @ W1 + b1
                h1 = np.maximum(h1_pre, 0.0)
                h2_pre = h1 @ W2 + b2
                h2 = np.maximum(h2_pre, 0.0)
                pred = h2 @ W3 + b3
                chosen = pred[np.arange(n_batch), yb]
                residual = chosen - tb
                grad_pred = Q_UNOBSERVED_ANCHOR * pred
                # Observed coordinate: d/dQ of 0.5 (Q - target)^2, averaged over the batch.
                # Other coordinates: d/dQ of 0.5 * anchor * Q^2, the zero-gain anchor.
                grad_pred[np.arange(n_batch), yb] = residual
                grad_pred /= n_batch
                d_h2 = grad_pred @ W3.T
                d_h2_pre = d_h2 * (h2_pre > 0.0)
                d_h1 = d_h2_pre @ W2.T
                d_h1_pre = d_h1 * (h1_pre > 0.0)
                grads = (
                    xb.T @ d_h1_pre + Q_L2 * W1,
                    d_h1_pre.sum(axis=0),
                    h1.T @ d_h2_pre + Q_L2 * W2,
                    d_h2_pre.sum(axis=0),
                    h2.T @ grad_pred + Q_L2 * W3,
                    grad_pred.sum(axis=0),
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
        self._emit = emit
        self.action_counts = counts
        self._model = f"numpy_q_phi_{hidden1}_{hidden2}"
        fitted = self.predict_q(matrix, mask_unseen=False)
        logged = fitted[np.arange(n_rows), y_index]
        self.train_mse_ = float(np.mean((logged - target) ** 2))
        return self

    def _require_fit(self) -> None:
        if self._W1 is None or self._emit is None or self._mean is None or self._std is None:
            raise RuntimeError("QPhi.predict_q: call fit or from_npz before predict.")

    def _forward(self, Z: Any) -> np.ndarray:
        self._require_fit()
        assert self._mean is not None and self._std is not None
        assert self._W1 is not None and self._b1 is not None
        assert self._W2 is not None and self._b2 is not None
        assert self._W3 is not None and self._b3 is not None
        matrix = np.asarray(Z, dtype=np.float64)
        if matrix.ndim != 2 or matrix.shape[1] != Z_DIM:
            raise ValueError(f"Z must have shape (n, {Z_DIM}), got {getattr(matrix, 'shape', None)}")
        if not np.isfinite(matrix).all():
            raise ValueError("Z contains a non-finite feature")
        centered = (matrix - self._mean) / self._std
        h1 = np.maximum(centered @ self._W1 + self._b1, 0.0)
        h2 = np.maximum(h1 @ self._W2 + self._b2, 0.0)
        return np.array(h2 @ self._W3 + self._b3, dtype=np.float64, copy=True)

    def predict_q(self, Z: Any, *, mask_unseen: bool = True) -> np.ndarray:
        """``Q_φ(z, a)`` with columns in ``BC_ACTIONS`` order.

        Unseen fit actions are ``-inf`` when ``mask_unseen`` is set, so they
        cannot win ``argmax Q/Cost``.
        """
        scores = self._forward(Z)
        if mask_unseen:
            assert self._emit is not None
            scores[:, ~self._emit] = -np.inf
        return scores

    def predict_actions(self, Z: Any) -> list[str]:
        """One ``argmax Q/Cost`` action per row."""
        values = self.predict_q(Z)
        chosen: list[str] = []
        for row in values:
            payload = {action: float(row[index]) for index, action in enumerate(self.actions)}
            chosen.append(argmax_value_over_cost(payload))
        return chosen

    def export_numpy(self) -> dict[str, Any]:
        self._require_fit()
        assert self._emit is not None
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
            "emit": self._emit.astype(np.float64),
            "input_dim": Z_DIM,
            "train_mse": self.train_mse_,
            "action_counts": self.action_counts,
        }

    @classmethod
    def from_npz(cls, path: Any, *, emit: Sequence[float] | None = None) -> "QPhi":
        """Reload a fit. Does not train and does not read labels."""
        model = cls()
        required = ("mean", "std", "W1", "b1", "W2", "b2", "W3", "b3", "emit")
        with np.load(path, allow_pickle=False) as blob:
            arrays = {key: np.asarray(blob[key], dtype=np.float64).copy() for key in required}
        hidden1, hidden2 = MLP_HIDDEN_SIZES
        shapes = {
            "mean": (Z_DIM,),
            "std": (Z_DIM,),
            "W1": (Z_DIM, hidden1),
            "b1": (hidden1,),
            "W2": (hidden1, hidden2),
            "b2": (hidden2,),
            "W3": (hidden2, model.n_actions),
            "b3": (model.n_actions,),
            "emit": (model.n_actions,),
        }
        for key, shape in shapes.items():
            if arrays[key].shape != shape:
                raise ValueError(f"{key} shape {arrays[key].shape} != {shape}")
            if not np.isfinite(arrays[key]).all():
                raise ValueError(f"{key} contains a non-finite value")
        loaded = arrays["emit"] > 0.5
        if emit is not None:
            override = np.asarray(emit, dtype=np.float64) > 0.5
            if override.shape != loaded.shape or not np.array_equal(override, loaded):
                raise ValueError("emit mask does not match the checkpoint")
        if not bool(loaded.any()):
            raise ValueError("emit mask drops every action")
        model._mean = arrays["mean"]
        model._std = arrays["std"]
        model._W1 = arrays["W1"]
        model._b1 = arrays["b1"]
        model._W2 = arrays["W2"]
        model._b2 = arrays["b2"]
        model._W3 = arrays["W3"]
        model._b3 = arrays["b3"]
        model._emit = loaded
        model._model = f"numpy_q_phi_{hidden1}_{hidden2}_loaded"
        model.action_counts = None
        return model


def run_self_check() -> None:
    """A tiny supervised fit ranks the action whose logged ΔL was higher.

    The targets are synthetic. This does not read a jsonl or a cafe label.
    """
    rng = np.random.default_rng(0)
    n_each = 64
    rows: list[np.ndarray] = []
    actions: list[str] = []
    deltas: list[float] = []

    def _blank() -> np.ndarray:
        return np.zeros(Z_DIM, dtype=np.float64)

    from vdr.controller.q_state import Z_FEATURE_NAMES

    mass_i = Z_FEATURE_NAMES.index("U_mass")
    id_i = Z_FEATURE_NAMES.index("U_id")
    code_i = Z_FEATURE_NAMES.index("gate_code_present")
    for _ in range(n_each):
        row = _blank()
        row[mass_i] = 1.0
        row[code_i] = 1.0
        row += rng.normal(0.0, 0.01, size=Z_DIM)
        rows.append(row)
        actions.append("RE_PORTION")
        deltas.append(0.45)
        other = _blank()
        other[id_i] = 1.0
        other += rng.normal(0.0, 0.01, size=Z_DIM)
        rows.append(other)
        actions.append("RETRIEVE")
        deltas.append(0.15)
        stop = _blank()
        stop += rng.normal(0.0, 0.01, size=Z_DIM)
        rows.append(stop)
        actions.append("STOP")
        deltas.append(0.0)

    model = QPhi()
    model.fit(np.vstack(rows), actions, deltas)
    if model.train_mse_ is None or model.train_mse_ > 0.02:
        raise AssertionError(f"supervised fit did not track ΔL (mse={model.train_mse_})")
    if model.action_counts is None or model.action_counts["RE_DECOMPOSE"] != 0:
        raise AssertionError(model.action_counts)
    probe = _blank()
    probe[mass_i] = 1.0
    probe[code_i] = 1.0
    chosen = model.predict_actions(probe.reshape(1, -1))[0]
    if chosen != "RE_PORTION":
        q = model.predict_q(probe.reshape(1, -1))[0]
        raise AssertionError((chosen, {name: float(q[i]) for i, name in enumerate(BC_ACTIONS)}))
    q_row = model.predict_q(probe.reshape(1, -1))[0]
    if np.isfinite(q_row[BC_ACTIONS.index("RE_DECOMPOSE")]):
        raise AssertionError("an action with no logged ΔL was left scorable")
    # Selection itself is Q/Cost, not a greedy copy of the logged name.
    if model._model is None or "q_phi" not in model._model:
        raise AssertionError(model._model)
