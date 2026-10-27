"""Portion grid search for a fixed FNDDS code.

B1 used to trust the raw VLM ``portion_g``. This tool replaces that number
with the best-scoring point on a discrete gram grid.

Default grid (``DEFAULT_PORTION_GRID_G``), edible grams:

    5, 10, 15, 20, 25, 30, 40, 50, 60, 75, 80,
    100, 110, 125, 150, 175, 200, 250, 300, 400, 500

The points cover a garnish (5–20 g), a typical plated fruit or vegetable
(25–150 g), and a full side (200–500 g). 500 g is the top of the grid, so a
runaway or duplicated VLM mass cannot be selected. Every default point sits
inside ``rule_verify`` portion bounds (1 g–1500 g).

Score (higher is better), for hint ``h`` and grid point ``g``:

- closeness = 1 - min(|g - h| / max(h, 1), 1.5)
- if ``energy_kcal_per_100g * g / 100`` exceeds
  ``COMPONENT_ENERGY_SOFT_CAP_KCAL`` (800 kcal for one component), subtract
  a penalty that grows with the overflow, capped at 1.5
- if the hint is at most ``DENSE_PENALTY_HINT_MAX_G`` (500 g, the top of the
  default grid), the food is at least ``DENSE_KCAL_PER_100G`` (300 kcal/100 g),
  and ``g`` is above ``DENSE_PORTION_FLOOR_G`` (100 g), subtract a second
  penalty that grows with both the extra grams and the density. A 100 g hint
  is unpenalized: a plate of almonds is often about that size, and crushing
  it repeats the old mass error in the other direction. Hints above the grid
  stay on the 800 kcal rule, which is what rejects a 5000 g runaway.

The raw hint is not a candidate unless it already lies on the grid. Ties
prefer the point closer to the hint, then the smaller gram value.

This is a fixed lexical/numeric prior. It is not a learned regression weight,
and it does not call the volume tool.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

from vdr.tools.nutrition_lookup import lookup_per_100g
from vdr.tools.rule_verify import PORTION_MAX_G, PORTION_MIN_G

# See module docstring. Keep this tuple and the docstring in sync.
DEFAULT_PORTION_GRID_G: tuple[float, ...] = (
    5,
    10,
    15,
    20,
    25,
    30,
    40,
    50,
    60,
    75,
    80,
    100,
    110,
    125,
    150,
    175,
    200,
    250,
    300,
    400,
    500,
)

# One component above this is treated as a runaway hint, not a plated side.
COMPONENT_ENERGY_SOFT_CAP_KCAL = 800.0

# Density-aware gram penalty. See the module docstring.
# 300 kcal/100 g is above pizza, rice, eggs, and feta, and below hard cheese
# (~368), seeds, and nuts. The floor is 100 g so a correct handful stays put
# while a 150 g nut or hard-cheese hint is pulled down.
DENSE_KCAL_PER_100G = 300.0
DENSE_PORTION_FLOOR_G = 100.0
DENSE_PORTION_SCALE = 3.2
DENSE_PENALTY_HINT_MAX_G = 500.0


@dataclass
class PortionResult:
    code: str
    portion_g: float
    grid: list[float]
    scores: list[float] | None = None
    hint_g: float | None = None


def score_portion_candidate(
    portion_g: float,
    hint_g: float,
    energy_per_100g: float,
    *,
    energy_cap_kcal: float = COMPONENT_ENERGY_SOFT_CAP_KCAL,
) -> float:
    """Score one grid point. Higher is better. See module docstring."""
    hint = float(hint_g)
    denom = max(hint, 1.0)
    rel = abs(float(portion_g) - hint) / denom
    closeness = 1.0 - min(rel, 1.5)
    kcal = float(energy_per_100g) * float(portion_g) / 100.0
    penalty = 0.0
    if kcal > energy_cap_kcal:
        penalty = min(1.5, (kcal - energy_cap_kcal) / energy_cap_kcal)
    penalty += _dense_gram_penalty(float(portion_g), hint, float(energy_per_100g))
    return closeness - penalty


def _dense_gram_penalty(portion_g: float, hint_g: float, energy_per_100g: float) -> float:
    """Pull a large hint down when the code is energy-dense.

    Zero at or below 100 g, and zero when the hint itself is past the default
    grid (the 800 kcal cap owns that runaway). See the module docstring.
    """
    if hint_g > DENSE_PENALTY_HINT_MAX_G:
        return 0.0
    if energy_per_100g < DENSE_KCAL_PER_100G or portion_g <= DENSE_PORTION_FLOOR_G:
        return 0.0
    excess = (portion_g - DENSE_PORTION_FLOOR_G) / DENSE_PORTION_FLOOR_G
    dens_factor = min(2.0, (energy_per_100g - DENSE_KCAL_PER_100G) / DENSE_KCAL_PER_100G)
    return min(1.5, excess * dens_factor * DENSE_PORTION_SCALE)


def search_portion(
    code: str,
    grid_g: Sequence[float] | None = None,
    hint_g: float | None = None,
    catalog_path: str | Path | None = None,
    energy_per_100g: float | None = None,
    **kwargs: Any,
) -> PortionResult:
    """Pick ``portion_g`` on the gram grid for one FNDDS code.

    ``hint_g`` is the VLM (or caller) estimate. When it is omitted the hint
    is 100 g, which is also on the default grid.
    """
    if hint_g is None:
        hint_g = kwargs.get("hint_g", 100.0)
    hint = float(hint_g)
    raw_grid = list(grid_g) if grid_g is not None else list(DEFAULT_PORTION_GRID_G)
    grid = [float(g) for g in raw_grid if PORTION_MIN_G <= float(g) <= PORTION_MAX_G]
    if not grid:
        raise ValueError("portion grid has no points inside portion bounds")

    if energy_per_100g is None:
        per = lookup_per_100g(code, catalog_path=catalog_path)
        energy_per_100g = float(per.get("energy_kcal", 0.0))
    energy = float(energy_per_100g)

    scores = [score_portion_candidate(g, hint, energy) for g in grid]
    best_i = max(
        range(len(grid)),
        key=lambda i: (scores[i], -abs(grid[i] - hint), -grid[i]),
    )
    return PortionResult(
        code=str(code),
        portion_g=float(grid[best_i]),
        grid=grid,
        scores=[round(s, 6) for s in scores],
        hint_g=hint,
    )
