"""Relative table-height maps from Nutrition5k depth_raw.png."""

from __future__ import annotations

import cv2
import numpy as np

from digi_nutri.protocol import BORDER_PIXELS, HEIGHT_MAX_MM, MORPH_CLOSE_K, RAW_UNIT_MM


def raw_to_millimeters(raw: np.ndarray) -> np.ndarray:
    """Convert 16-bit raw units to millimeters. One raw unit is 1e-4 m."""
    return np.asarray(raw, dtype=np.float32) * np.float32(RAW_UNIT_MM)


def border_values(millimeters: np.ndarray, width: int = BORDER_PIXELS) -> np.ndarray:
    """Return the millimeter samples on the outer frame of the image."""
    if millimeters.ndim != 2:
        raise ValueError(f"depth map must be HxW, got {millimeters.shape}")
    if width < 1:
        raise ValueError("border width must be positive")
    height, cols = millimeters.shape
    if height <= 2 * width or cols <= 2 * width:
        raise ValueError("image is smaller than the requested border")
    top = millimeters[:width, :].reshape(-1)
    bottom = millimeters[-width:, :].reshape(-1)
    left = millimeters[width:-width, :width].reshape(-1)
    right = millimeters[width:-width, -width:].reshape(-1)
    return np.concatenate([top, bottom, left, right])


def close_3x3(scaled: np.ndarray) -> np.ndarray:
    kernel = np.ones((MORPH_CLOSE_K, MORPH_CLOSE_K), np.uint8)
    closed = cv2.morphologyEx(np.asarray(scaled, dtype=np.float32), cv2.MORPH_CLOSE, kernel)
    return np.asarray(closed, dtype=np.float32)


def depth_raw_to_height01(raw: np.ndarray) -> np.ndarray:
    """Map a raw depth image to a fixed [0, 1] relative-height image.

    D_table is the median of the border millimeter values, including zeros.
    Height is clip(D_table - D_raw, 0, 150 mm), then divided by 150.
    A later image is not re-stretched to its own maximum. One 3x3 closing
    fills small holes after that fixed scale.
    """
    millimeters = raw_to_millimeters(raw)
    table = float(np.median(border_values(millimeters)))
    height_mm = np.clip(table - millimeters, 0.0, HEIGHT_MAX_MM)
    scaled = height_mm / np.float32(HEIGHT_MAX_MM)
    return close_3x3(scaled)
