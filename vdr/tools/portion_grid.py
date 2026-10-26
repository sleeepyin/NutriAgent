"""Portion grid search tool (stub).

Scans a discrete gram grid for a fixed food_code to reduce nutrition error proxy.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Sequence


@dataclass
class PortionResult:
    code: str
    portion_g: float
    grid: list[float]
    scores: list[float] | None = None


def search_portion(
    code: str,
    grid_g: Sequence[float] | None = None,
    **kwargs: Any,
) -> PortionResult:
    """Pick portion_g on a gram grid for the given FNDDS code.

    Raises:
        NotImplementedError: until nutrition lookup + scoring are wired.
    """
    raise NotImplementedError(
        "portion_grid.search_portion: stub. Wire after nutrition_lookup + rule scores."
    )
