"""Post-perception cleanup for component hypotheses.

VDR image runs have emitted the same food name several times (for example
five copies of "Sweet Potato Cubes" at 100 g). Summing those copies inflates
mass and energy. Dedupe keeps one entry per normalized name.

A second cleanup merges one structural fragment with one short filling
before retrieve. "Cheese" + "Crust" was binding a cheese block and a
stuffed-crust pizza, which counts the cheese twice. The merged query is
retrieved once. Its portion is the larger hint, not the sum.
"""

from __future__ import annotations

import re
import statistics
from typing import Any, Sequence

_WS_RE = re.compile(r"\s+")
_TOKEN_RE = re.compile(r"[a-z0-9]+", re.I)

# Structural pieces that are not a whole dish on their own. Kept small on
# purpose: crust/dough plus one filling is the double-count we have seen.
STRUCTURAL_FRAGMENT_TOKENS = frozenset({"crust", "dough"})
_MAX_FRAGMENT_TOKENS = 2


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


def _name_tokens(name: str) -> list[str]:
    tokens: list[str] = []
    for tok in _TOKEN_RE.findall(name or ""):
        t = tok.lower()
        if len(t) > 3 and t.endswith("s") and not t.endswith("ss"):
            t = t[:-1]
        tokens.append(t)
    return tokens


def _portion_value(hyp: dict[str, Any]) -> float | None:
    portion = hyp.get("portion_g")
    if portion is None or portion == "":
        return None
    try:
        return float(portion)
    except (TypeError, ValueError):
        return None


def merge_complementary_fragments(
    hypotheses: Sequence[dict[str, Any]] | None,
) -> list[dict[str, Any]]:
    """Collapse one structural fragment plus one short filling into one query.

    Only a two-hypothesis list is eligible. "Cheese" + "Broccoli" has no
    structural token and stays split. Three or more hypotheses stay split so
    a side next to a crust is not swallowed into the crust query.
    """
    hyps = [dict(h) for h in (hypotheses or [])]
    if len(hyps) != 2:
        return hyps
    token_lists = [_name_tokens(str(h.get("name") or "")) for h in hyps]
    part_indexes = [
        i
        for i, toks in enumerate(token_lists)
        if toks
        and (set(toks) & STRUCTURAL_FRAGMENT_TOKENS)
        and len(toks) <= _MAX_FRAGMENT_TOKENS
    ]
    if len(part_indexes) != 1:
        return hyps
    part_i = part_indexes[0]
    fill_i = 1 - part_i
    fill_toks = token_lists[fill_i]
    if not fill_toks or len(fill_toks) > _MAX_FRAGMENT_TOKENS:
        return hyps
    if set(fill_toks) & STRUCTURAL_FRAGMENT_TOKENS:
        return hyps
    filling = hyps[fill_i]
    part = hyps[part_i]
    fill_name = str(filling.get("name") or "").strip()
    part_name = str(part.get("name") or "").strip()
    if not fill_name or not part_name:
        return hyps
    portions = [p for p in (_portion_value(filling), _portion_value(part)) if p is not None]
    return [
        {
            "name": f"{fill_name} {part_name}",
            "portion_g": max(portions) if portions else None,
        }
    ]
