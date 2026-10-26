"""FNDDS retrieval tool.

Default catalog: data/fndds/2019-2020/foods.json if present, else toy.
Override with VDR_FNDDS_PATH.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .fndds_catalog import iter_foods


@dataclass
class RetrieveResult:
    query: str
    cands: list[dict[str, Any]]
    selected_code: str | None


_TOKEN_RE = re.compile(r"[a-z0-9]+", re.I)


def _tokens(text: str) -> set[str]:
    return {t.lower() for t in _TOKEN_RE.findall(text or "")}


def _score_food(query: str, food: dict[str, Any]) -> float:
    q = (query or "").strip().lower()
    if not q:
        return 0.0
    name = str(food.get("name", "")).lower()
    aliases = [str(a).lower() for a in food.get("aliases", [])]
    head = name.split(",")[0].strip()
    qt = _tokens(q)
    if not qt:
        return 0.0

    # Exact full-string matches
    if q == name or q in aliases:
        return 1.0
    if q == head:
        return 0.98

    hay_fields = [name, head, *aliases]
    ht_name = _tokens(name)
    ht_head = _tokens(head)
    ht_all = set().union(*(_tokens(h) for h in hay_fields if h))

    scores: list[float] = []

    # Contiguous phrase: prefer denser (shorter) names
    for h in hay_fields:
        if h and q in h:
            density = len(q) / max(len(h), 1)
            scores.append(0.70 + 0.28 * density)

    # All query tokens present in name — density + start-with boost
    # (handles FNDDS order like "Rice, white, cooked")
    if qt.issubset(ht_name):
        density = len(qt) / max(len(ht_name), 1)
        start_boost = 0.08 if any(name.startswith(t) for t in qt) else 0.0
        # comma-style: first tokens of name overlap query strongly
        head_overlap = len(qt & ht_head) / len(qt)
        scores.append(0.72 + 0.22 * density + start_boost + 0.05 * head_overlap)

    # Head-focused soft match
    if ht_head:
        prec = len(qt & ht_head) / len(qt)
        rec = len(qt & ht_head) / len(ht_head)
        if prec + rec:
            f1 = 2 * prec * rec / (prec + rec)
            scores.append(0.55 + 0.35 * f1)

    # Weak Jaccard fallback
    if ht_all:
        inter = len(qt & ht_all)
        if inter:
            jacc = inter / len(qt | ht_all)
            scores.append(0.40 * jacc + 0.15 * (inter / len(qt)))

    return max(scores) if scores else 0.0


def retrieve(
    query: str,
    top_k: int = 5,
    catalog_path: str | Path | None = None,
    **kwargs: Any,
) -> RetrieveResult:
    """Retrieve Top-k FNDDS food codes for a text query."""
    scored: list[tuple[float, dict[str, Any]]] = []
    for food in iter_foods(catalog_path):
        s = _score_food(query, food)
        if s > 0:
            scored.append((s, food))
    scored.sort(key=lambda x: (-x[0], len(str(x[1].get("name", ""))), x[1]["code"]))
    cands = [
        {
            "code": f["code"],
            "score": round(s, 4),
            "name": f.get("name"),
        }
        for s, f in scored[: max(1, int(top_k))]
    ]
    selected = cands[0]["code"] if cands else None
    return RetrieveResult(query=query, cands=cands, selected_code=selected)
