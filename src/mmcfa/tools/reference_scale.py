"""Reference-scale tool for ECUSTFD-style coin/card calibration."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

COIN_MM = 25.0


@dataclass
class RefResult:
    success: bool
    w_hat: float
    vol_hat: float
    mmpp: float
    error: str | None = None


class ReferenceScaleTool:
    def __init__(self, coin_mm: float = COIN_MM):
        self.coin_mm = coin_mm

    def predict_instance(
        self,
        food_px: float,
        coin_px: float | None,
        k_area_per_vol: float,
        rho: float,
        global_mmpp: float | None = None,
        use_ref: bool = True,
    ) -> RefResult:
        if use_ref:
            if coin_px is None or coin_px <= 0:
                return RefResult(False, float("nan"), float("nan"), float("nan"), "coin_missing")
            mmpp = self.coin_mm / float(coin_px)
        else:
            if global_mmpp is None or global_mmpp <= 0:
                return RefResult(False, float("nan"), float("nan"), float("nan"), "no_global_mmpp")
            mmpp = float(global_mmpp)
        if food_px <= 0 or k_area_per_vol <= 0 or rho <= 0:
            return RefResult(False, float("nan"), float("nan"), mmpp, "invalid_geometry")
        food_mm2 = food_px * mmpp ** 2
        vol_hat = food_mm2 / k_area_per_vol
        w_hat = max(0.0, vol_hat * rho)
        return RefResult(True, float(w_hat), float(vol_hat), float(mmpp))


def instance_table(df: pd.DataFrame) -> pd.DataFrame:
    """Median T-view per instance, plus optional S-view."""
    t = df[df["view"] == "T"].groupby("inst", as_index=False).agg(
        cat=("cat", "first"),
        food_px=("food_px", "median"),
        coin_px=("coin_px", "median"),
        vol_ml=("vol_ml", "first"),
        wt_g=("wt_g", "first"),
    )
    s = df[df["view"] == "S"].groupby("inst", as_index=False).agg(
        food_px_s=("food_px", "median"),
        coin_px_s=("coin_px", "median"),
    )
    return t.merge(s, on="inst", how="left")
