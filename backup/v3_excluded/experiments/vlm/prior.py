"""Name-prior shrinkage at tau=4. Policy medians come from the action tables."""
from __future__ import annotations

from collections import defaultdict

import numpy as np

from mmcfa.data_build import build_ecustfd
from mmcfa.paths import Paths

from .holdout import _attach_ecustfd, align_mf3d
from .names import match_name

TAU = 4.0


def _stats(rows: list[dict]) -> dict:
    buckets: dict[str, list[float]] = defaultdict(list)
    for row in rows:
        if row["split"] == "policy":
            buckets[row["cat"]].append(float(row["y"]))
    med = {cat: float(np.median(values)) for cat, values in buckets.items() if values}
    count = {cat: len(values) for cat, values in buckets.items()}
    flat = [y for values in buckets.values() for y in values]
    return {
        "median": med,
        "count": count,
        "global": float(np.median(flat)) if flat else float("nan"),
    }


def shrink(cat: str | None, stats: dict, cache: dict, tau: float = TAU) -> float:
    if not cat:
        return float(stats["global"])
    n = int(stats["count"].get(cat, 0))
    med = float(stats["median"].get(cat, stats["global"]))
    raw = cache.get(cat, {}).get("typical_mass_g")
    try:
        llm = float(raw)
    except (TypeError, ValueError):
        llm = float("nan")
    if not np.isfinite(llm):
        llm = float("nan")
    if n <= 0 or not np.isfinite(llm):
        return llm if np.isfinite(llm) else float(stats["global"])
    return float((n * med + tau * llm) / (n + tau))


class NamePrior:
    def __init__(self, paths: Paths | None = None, cache: dict | None = None):
        paths = paths or Paths.resolve()
        ec = build_ecustfd(paths)
        _attach_ecustfd(ec, paths)
        mf = align_mf3d(paths)
        self.by_dataset = {
            "ecustfd": {"stats": _stats(ec), "labels": sorted({r["cat"] for r in ec})},
            "mf3d": {"stats": _stats(mf), "labels": sorted({r["cat"] for r in mf})},
        }
        self.cache = cache or {}

    def predict_annotated(self, dataset: str, cat: str, tau: float = TAU) -> float:
        block = self.by_dataset.get(dataset)
        if block is None:
            return float("nan")
        return shrink(cat, block["stats"], self.cache, tau)

    def predict_vlm_name(self, dataset: str, food_name: str, tau: float = TAU):
        block = self.by_dataset.get(dataset)
        if block is None:
            return float("nan"), None, "none"
        matched, kind = match_name(food_name, block["labels"])
        return shrink(matched, block["stats"], self.cache, tau), matched, kind
