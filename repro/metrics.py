"""Dataset PMAE, MAE, and plate PMAE."""

from __future__ import annotations

import numpy as np


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
    pred = np.asarray(pred, dtype=np.float64).reshape(-1)
    target = np.asarray(target, dtype=np.float64).reshape(-1)
    mask = np.abs(target) > 0
    if not np.any(mask):
        return float("nan")
    return float(np.mean(100.0 * np.abs(pred[mask] - target[mask]) / np.abs(target[mask])))


def summarize(pred: np.ndarray, target: np.ndarray, columns: tuple[str, ...]) -> dict:
    pred = np.asarray(pred, dtype=np.float64)
    target = np.asarray(target, dtype=np.float64)
    summary = {}
    dataset_values = []
    for index, name in enumerate(columns):
        item = {
            "mae": mae(pred[:, index], target[:, index]),
            "dataset_pmae": dataset_pmae(pred[:, index], target[:, index]),
            "plate_pmae": plate_pmae(pred[:, index], target[:, index]),
        }
        summary[name] = item
        dataset_values.append(item["dataset_pmae"])
    summary["mean5_dataset_pmae"] = float(np.mean(dataset_values))
    return summary
