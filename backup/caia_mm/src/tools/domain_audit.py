"""Domain / quality audit. OOD is a gate, not an error magnitude claim."""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler


@dataclass
class DomainAudit:
    ood_score: float
    ood_pvalue: float
    quality_flags: list[str] = field(default_factory=list)
    recommended_action_mask: dict[str, bool] = field(default_factory=dict)


class DomainAuditTool:
    def __init__(self):
        self.nn: NearestNeighbors | None = None
        self.scaler: StandardScaler | None = None
        self.ref_dist: np.ndarray | None = None

    def fit(self, feat_ref: np.ndarray) -> None:
        feat_ref = np.asarray(feat_ref, dtype=float)
        self.scaler = StandardScaler()
        z = self.scaler.fit_transform(feat_ref)
        self.nn = NearestNeighbors(n_neighbors=min(15, len(z)), metric="euclidean")
        self.nn.fit(z)
        d, _ = self.nn.kneighbors(z)
        self.ref_dist = d.mean(axis=1)

    def score_features(self, feat: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        if self.nn is None or self.scaler is None or self.ref_dist is None:
            z = np.zeros(len(feat))
            return z, np.ones(len(feat))
        x = self.scaler.transform(np.asarray(feat, dtype=float))
        d, _ = self.nn.kneighbors(x)
        dist = d.mean(axis=1)
        ref = self.ref_dist
        p = np.array([float(np.mean(ref >= di)) for di in dist])
        ood = (dist - np.median(ref)) / (np.std(ref) + 1e-6)
        ood = 1.0 / (1.0 + np.exp(-ood))
        return ood, p

    def quality_from_crop(self, x: np.ndarray) -> list[str]:
        flags = []
        rgb = x[:3]
        if float(np.std(rgb)) < 0.08:
            flags.append("low_contrast")
        if float(np.mean(np.abs(rgb))) < 0.05:
            flags.append("too_dark_or_empty")
        if x.shape[0] >= 4:
            depth = x[3]
            if float(np.std(depth)) < 0.02:
                flags.append("depth_flat")
        return flags
