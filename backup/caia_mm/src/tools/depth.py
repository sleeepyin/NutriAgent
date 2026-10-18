"""Depth action: RGB vs RGB-D under a consistent training protocol."""
from __future__ import annotations

from .base_mm_cfa import BaseMMCFATool, BasePrediction


class DepthTool:
    def __init__(self, base: BaseMMCFATool | None = None):
        self.base = base or BaseMMCFATool()

    def rgb(self, X4) -> BasePrediction:
        return self.base.predict_3ch(X4[:, :3])

    def rgbd(self, X4) -> BasePrediction:
        return self.base.predict_4ch(X4)

    def rgbd_depth0(self, X4) -> BasePrediction:
        return self.base.predict_4ch_depth_ablate(X4)
