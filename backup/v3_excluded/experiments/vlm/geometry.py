"""D4-style coin geometry. Peer area and density stay on the annotated top views."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from mmcfa.paths import Paths
from mmcfa.tools.reference_scale import COIN_MM, ReferenceScaleTool

_TOOL = ReferenceScaleTool(COIN_MM)


class CoinTable:
    def __init__(self, csv_path: Path):
        frame = pd.read_csv(csv_path)
        self.top = frame[frame["view"] == "T"].groupby("inst").first()
        self.global_mmpp = float(np.median(COIN_MM / self.top["coin_px"].clip(lower=1)))

    def factors(self, inst: str):
        if inst not in self.top.index:
            return None
        row = self.top.loc[inst]
        cat = str(row["cat"])
        peers = self.top[(self.top["cat"] == cat) & (self.top.index != inst)]
        if len(peers) < 1:
            return None
        k = float(np.median(peers["food_mm2_cal"] / peers["vol_ml"].replace(0, np.nan)))
        rho = float(np.mean(peers["wt_g"] / peers["vol_ml"].replace(0, np.nan)))
        if not np.isfinite(k) or not np.isfinite(rho):
            return None
        return {
            "cat": cat,
            "k": k,
            "rho": rho,
            "food_px": float(row["food_px"]),
            "coin_px": float(row["coin_px"]),
            "y_g": float(row["wt_g"]),
        }

    def predict(self, inst: str, coin_px: float | None, use_ref: bool) -> float:
        factors = self.factors(inst)
        if factors is None:
            return float("nan")
        result = _TOOL.predict_instance(
            factors["food_px"],
            coin_px,
            factors["k"],
            factors["rho"],
            global_mmpp=self.global_mmpp,
            use_ref=use_ref,
        )
        return float(result.w_hat) if result.success else float("nan")


def load_coin_table(paths: Paths | None = None) -> CoinTable:
    paths = paths or Paths.resolve()
    return CoinTable(paths.expect("ecustfd_csv"))
