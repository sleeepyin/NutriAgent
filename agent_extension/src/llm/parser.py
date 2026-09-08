"""Constrained parser: natural language -> ontology. No numeric target leakage."""
from __future__ import annotations

from ..tools.semantic_density import ONTOLOGY


def parse_answers(text: str) -> dict[str, str]:
    s = (text or "").lower()
    out = {}
    for key, opts in ONTOLOGY.items():
        for o in opts:
            if o.replace("-", " ") in s or o in s:
                out[key] = o
                break
        out.setdefault(key, "unknown" if key == "added_fat" else opts[0])
    return out
