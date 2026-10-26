"""Post-perception cleanup for component hypotheses.

VDR image runs have emitted the same food name several times (for example
five copies of "Sweet Potato Cubes" at 100 g). Summing those copies inflates
mass and energy. Dedupe keeps one entry per normalized name.
"""

from __future__ import annotations

import re
import statistics
from typing import Any, Sequence

_WS_RE = re.compile(r"\s+")


def normalize_hyp_name(name: str) -> str:
    """Casefold, strip, and collapse internal whitespace. Punctuation stays."""
    return _WS_RE.sub(" ", (name or "").casefold().strip())


def dedupe_hypotheses(
    hypotheses: Sequence[dict[str, Any]] | None,
) -> list[dict[str, Any]]:
    """Merge hypotheses that share a normalized name.

    Portion rule: the median of the duplicate portion estimates (a single
    robust guess). Duplicates are not summed. Missing portions are ignored
    in the median; if every copy omitted a portion, the merged entry has
    ``portion_g=None`` so the caller can apply its default.
    """
    order: list[str] = []
    groups: dict[str, list[dict[str, Any]]] = {}
    for hyp in hypotheses or []:
        raw_name = hyp.get("name") or hyp.get("hyp") or hyp.get("query") or ""
        key = normalize_hyp_name(str(raw_name))
        if not key:
            continue
        if key not in groups:
            order.append(key)
            groups[key] = []
        groups[key].append(hyp)

    merged: list[dict[str, Any]] = []
    for key in order:
        group = groups[key]
        portions: list[float] = []
        for hyp in group:
            portion = hyp.get("portion_g")
            if portion is None or portion == "":
                continue
            try:
                portions.append(float(portion))
            except (TypeError, ValueError):
                continue
        display = str(group[0].get("name") or group[0].get("hyp") or group[0].get("query")).strip()
        merged.append(
            {
                "name": display,
                "portion_g": statistics.median(portions) if portions else None,
            }
        )
    return merged
