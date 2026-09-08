"""Base MM-CFA inference API. Paths from config. No experiment on import."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch

from ..models import WeightEffNet4, WeightNet3ch, WeightResNet4, load_state, predict_np
from ..paths import Paths
from ..runtime import log


@dataclass
class BasePrediction:
    weight: np.ndarray
    weight_rn: np.ndarray
    weight_eff: np.ndarray
    disagree: np.ndarray
    channels: int
    version: str


class BaseMMCFATool:
    def __init__(self, paths: Paths | None = None, device: str | None = None):
        self.paths = paths or Paths.resolve()
        self.device = torch.device(device or ("cuda:0" if torch.cuda.is_available() else "cpu"))
        self._m4 = None
        self._m3 = None

    def _load4(self):
        if self._m4 is not None:
            return
        rn_p = self.paths.file("w_rn4")
        ef_p = self.paths.file("w_eff4")
        if rn_p is None or ef_p is None:
            raise FileNotFoundError("4ch weight checkpoints missing")
        rn = load_state(WeightResNet4(), rn_p, self.device)
        ef = load_state(WeightEffNet4(), ef_p, self.device)
        self._m4 = (rn, ef)

    def _load3(self):
        if self._m3 is not None:
            return
        rn_p = self.paths.file("w_rn3")
        ef_p = self.paths.file("w_eff3")
        if rn_p is None or ef_p is None:
            raise FileNotFoundError("3ch weight checkpoints missing")
        rn = load_state(WeightNet3ch("resnet18"), rn_p, self.device)
        ef = load_state(WeightNet3ch("efficientnet_b0"), ef_p, self.device)
        self._m3 = (rn, ef)

    def predict_4ch(self, X4: np.ndarray) -> BasePrediction:
        self._load4()
        rn, ef = self._load4() if False else self._m4
        pr = predict_np(rn, X4, self.device)
        pe = predict_np(ef, X4, self.device)
        ens = 0.5 * (pr + pe)
        return BasePrediction(ens, pr, pe, np.abs(pr - pe), 4, "4ch-final")

    def predict_3ch(self, X3: np.ndarray) -> BasePrediction:
        self._load3()
        rn, ef = self._m3
        pr = predict_np(rn, X3, self.device)
        pe = predict_np(ef, X3, self.device)
        ens = 0.5 * (pr + pe)
        return BasePrediction(ens, pr, pe, np.abs(pr - pe), 3, "3ch-clean")

    def predict_4ch_depth_ablate(self, X4: np.ndarray) -> BasePrediction:
        """Same 4ch weights, depth channel zeroed. Fair within-model depth ablation."""
        X = np.array(X4, copy=True)
        X[:, 3] = 0
        out = self.predict_4ch(X)
        out.version = "4ch-depth0"
        return out
