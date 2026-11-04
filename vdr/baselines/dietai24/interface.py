"""DietAI24 stage interfaces.

Published pipeline (Yan et al., Communications Medicine 2025):

1. Index FNDDS 2019–2020 descriptions once with ``text-embedding-3-large``.
2. GPT Vision describes the image; a GPT model expands that text into queries.
3. Embed the queries, search the vector store, drop duplicate food codes.
4. GPT Vision keeps the codes that are actually in the image.
5. For each code, choose an FNDDS portion descriptor (or an adjusted amount
   in FNDDS units) and map it to grams with the portions-and-weights table.
6. Scale each code's per-100 g nutrient vector by grams/100 and sum.

This module does not call those models. A later live client would implement
the protocols below. The gram-scaling step is pure and is owned by the runner
so a stub cannot skip it.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, Protocol, Sequence, runtime_checkable

from vdr.baselines.dietai24.config import (
    MACRO_KEYS,
    PUBLIC_N_KEYS,
    DietAI24Config,
    DietAI24ScaffoldError,
)

DIETAI24_STAGES = (
    "index",
    "describe_and_expand",
    "retrieve_candidates",
    "infer_food_codes",
    "estimate_portions",
    "aggregate_nutrients",
)


@dataclass(frozen=True)
class ImageRef:
    """Image handle. The dry-run uses a fixture URI and does not read pixels."""

    dish_id: str
    uri: str


@dataclass(frozen=True)
class IndexManifest:
    release: str
    expected_food_codes: int
    embedding_model: str
    executed: bool
    reason: str


@dataclass(frozen=True)
class QueryExpansion:
    description: str
    queries: tuple[str, ...]
    live: bool


@dataclass(frozen=True)
class CandidateCode:
    code: str
    description: str
    score: float


@dataclass(frozen=True)
class PortionSpec:
    """Qualitative FNDDS descriptor plus the gram equivalent from the weights table.

    ``from_portion_grid`` must stay false. The hardened in-repo baseline uses
    a gram grid; DietAI24 does not.
    """

    code: str
    descriptor: str
    grams: float
    from_portion_grid: bool = False


@dataclass(frozen=True)
class NutrientSpec:
    """Per-100 g vector for one code. The dry-run supplies a fixture, not a catalog lookup."""

    code: str
    per_100g: tuple[tuple[str, float], ...]


@dataclass(frozen=True)
class DietAI24DryCase:
    """Stage inputs for a dry-run. Nothing here is a model output."""

    description: str
    queries: tuple[str, ...]
    candidates: tuple[CandidateCode, ...]
    selected_codes: tuple[str, ...]
    portions: tuple[PortionSpec, ...]
    nutrients: tuple[NutrientSpec, ...]


def nutrient_map(spec: NutrientSpec) -> dict[str, float]:
    return {key: float(value) for key, value in spec.per_100g}


def scale_per_100g(per_100g: Mapping[str, float], grams: float) -> dict[str, float]:
    """Published contribution ``v * grams / 100`` for the public macro fields.

    The paper scales all 65 FNDDS components. This scaffold scales the five
    fields the VDR schema stores and records that the other 60 were not run.
    """

    missing = [key for key in MACRO_KEYS if key not in per_100g]
    if missing:
        raise DietAI24ScaffoldError(f"per-100 g vector missing {missing}")
    if grams < 0:
        raise DietAI24ScaffoldError("grams must be >= 0")
    factor = float(grams) / 100.0
    scaled = {key: float(per_100g[key]) * factor for key in MACRO_KEYS}
    scaled["mass_g"] = float(grams)
    return scaled


def sum_contributions(rows: Sequence[Mapping[str, float]]) -> dict[str, float]:
    total = {key: 0.0 for key in PUBLIC_N_KEYS}
    for row in rows:
        for key in PUBLIC_N_KEYS:
            total[key] += float(row[key])
    return total


def dedupe_candidates(candidates: Sequence[CandidateCode]) -> tuple[CandidateCode, ...]:
    """Drop duplicate codes, keeping the first. The paper removes duplicates after search."""

    seen: set[str] = set()
    kept: list[CandidateCode] = []
    for candidate in candidates:
        if candidate.code in seen:
            continue
        seen.add(candidate.code)
        kept.append(candidate)
    return tuple(kept)


@runtime_checkable
class DietAI24Indexer(Protocol):
    def index(self, config: DietAI24Config) -> IndexManifest:
        """Build the vector store. A faithful client embeds all 5624 descriptions."""


@runtime_checkable
class DietAI24Retriever(Protocol):
    def expand_queries(self, image: ImageRef, config: DietAI24Config) -> QueryExpansion:
        """Image description plus the expanded query list."""

    def retrieve(
        self,
        expansion: QueryExpansion,
        index: IndexManifest,
        config: DietAI24Config,
    ) -> tuple[CandidateCode, ...]:
        """Similarity search. The runner dedupes the returned codes."""


@runtime_checkable
class DietAI24Estimator(Protocol):
    def infer_codes(
        self,
        image: ImageRef,
        candidates: Sequence[CandidateCode],
        config: DietAI24Config,
    ) -> tuple[str, ...]:
        """Codes judged present. Each code must be one of ``candidates``."""

    def choose_portions(
        self,
        image: ImageRef,
        codes: Sequence[str],
        config: DietAI24Config,
    ) -> tuple[PortionSpec, ...]:
        """Descriptor and gram equivalent for every inferred code."""

    def per_100g_rows(
        self,
        codes: Sequence[str],
        config: DietAI24Config,
    ) -> tuple[NutrientSpec, ...]:
        """Per-100 g nutrient rows. A faithful client reads FNDDS, including 65 components."""
