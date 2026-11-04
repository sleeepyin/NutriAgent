"""DietAI24 protocol stub. Not the hardened RAG-once baseline.

The historical in-repo path remains ``vdr.baselines.b1_rag_once`` and is not
a faithful DietAI24 run. Stage protocols live in ``interface``. The CPU
dry-run entrypoint is ``python -m vdr.baselines.dietai24.dry_run``.
"""

from vdr.baselines.dietai24.config import (
    HARDENED_B1_ENERGY_MAE,
    HARDENED_B1_IS_FAITHFUL_DIETAI24,
    DietAI24Config,
    DietAI24ScaffoldError,
    FaithfulDietAI24NotImplemented,
)
from vdr.baselines.dietai24.interface import (
    DIETAI24_STAGES,
    DietAI24Estimator,
    DietAI24Indexer,
    DietAI24Retriever,
    scale_per_100g,
    sum_contributions,
)

__all__ = [
    "DIETAI24_STAGES",
    "DietAI24Config",
    "DietAI24Estimator",
    "DietAI24Indexer",
    "DietAI24Retriever",
    "DietAI24ScaffoldError",
    "FaithfulDietAI24NotImplemented",
    "HARDENED_B1_ENERGY_MAE",
    "HARDENED_B1_IS_FAITHFUL_DIETAI24",
    "scale_per_100g",
    "sum_contributions",
]
