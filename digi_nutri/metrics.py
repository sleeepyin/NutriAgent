"""Headline MAE and the two PMAE definitions."""

from __future__ import annotations

import numpy as np

from digi_nutri.protocol import NUTRIENTS


def mae(pred: np.ndarray, target: np.ndarray) -> float:
    pred = np.asarray(pred, dtype=np.float64).reshape(-1)
    target = np.asarray(target, dtype=np.float64).reshape(-1)
    return float(np.mean(np.abs(pred - target)))


def dataset_pmae(pred: np.ndarray, target: np.ndarray) -> float:
    pred = np.asarray(pred, dtype=np.float64).reshape(-1)
    target = np.asarray(target, dtype=np.float64).reshape(-1)
    denom = float(np.sum(np.abs(target)))
    if denom <= 0:
        return float("nan")
    return float(100.0 * np.sum(np.abs(pred - target)) / denom)


def plate_pmae(pred: np.ndarray, target: np.ndarray) -> float:
    """Mean per-plate percentage error. Plates with |y| == 0 are omitted."""
    pred = np.asarray(pred, dtype=np.float64).reshape(-1)
    target = np.asarray(target, dtype=np.float64).reshape(-1)
    mask = np.abs(target) > 0
    if not np.any(mask):
        return float("nan")
    return float(np.mean(100.0 * np.abs(pred[mask] - target[mask]) / np.abs(target[mask])))


METRIC_FNS = {
    "mae": mae,
    "dataset_pmae": dataset_pmae,
    "plate_pmae": plate_pmae,
}


def summarize(pred: np.ndarray, target: np.ndarray, columns: tuple[str, ...] = NUTRIENTS) -> dict:
    pred = np.asarray(pred, dtype=np.float64)
    target = np.asarray(target, dtype=np.float64)
    summary = {}
    for index, name in enumerate(columns):
        summary[name] = {
            metric: float(fn(pred[:, index], target[:, index]))
            for metric, fn in METRIC_FNS.items()
        }
    return summary
