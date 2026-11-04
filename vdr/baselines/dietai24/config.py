"""Pinned DietAI24 protocol constants.

The numbers here describe the published protocol. They are not results
measured by this repository. ``HARDENED_B1_IS_FAITHFUL_DIETAI24`` stays
false: the frozen overhead-pool energy MAE 110.72 comes from
``vdr.baselines.b1_rag_once``, which is a different system.
"""

from __future__ import annotations

from dataclasses import dataclass


PAPER = "Yan et al., Communications Medicine 2025, s43856-025-01159-0"
OFFICIAL_CODE = "https://github.com/Runz96/DietAI24"
OFFICIAL_ENTRYPOINTS = (
    "rag_food_code.py",
    "rag_portion_size.py",
    "nutrient_estimatte.py",
    "nutrient_estimatte_mx.py",
)

FNDDS_RELEASE = "2019-2020"
EXPECTED_FOOD_CODES = 5624
NUTRIENT_COUNT = 65
EMBEDDING_MODEL = "text-embedding-3-large"
MLLM_FAMILY = "gpt-4-vision"
TEMPERATURE = 0.0
PORTION_SOURCE = "fndds_portions_and_weights"
VECTOR_STORE = "langchain"

# Frozen historical run of the in-repo hardened RAG-once path. Not this protocol.
HARDENED_B1_MODULE = "vdr.baselines.b1_rag_once"
HARDENED_B1_ENERGY_MAE = 110.72
HARDENED_B1_IS_FAITHFUL_DIETAI24 = False

PUBLIC_N_KEYS = ("energy_kcal", "protein_g", "carb_g", "fat_g", "mass_g")
MACRO_KEYS = ("energy_kcal", "protein_g", "carb_g", "fat_g")


class DietAI24ScaffoldError(RuntimeError):
    """The dry-run fixture or config broke the published stage contract."""


class FaithfulDietAI24NotImplemented(DietAI24ScaffoldError):
    """A live DietAI24 eval (GPT Vision, embeddings, portions file) is not in this landing."""


@dataclass(frozen=True)
class DietAI24Config:
    """Call shape for the official protocol. Live calls stay off.

    ``fidelity`` is ``scaffold`` only. That means the stage order and the
    gram-scaling formula are the published ones, and no model is called.
    It is not a faithful eval and it is not an approx table row.
    """

    baseline_id: str = "b1_dietai24"
    fidelity: str = "scaffold"
    fndds_release: str = FNDDS_RELEASE
    expected_food_codes: int = EXPECTED_FOOD_CODES
    nutrient_count: int = NUTRIENT_COUNT
    embedding_model: str = EMBEDDING_MODEL
    mllm_family: str = MLLM_FAMILY
    temperature: float = TEMPERATURE
    portion_source: str = PORTION_SOURCE
    vector_store: str = VECTOR_STORE
    allow_live: bool = False
    main_table_eligible: bool = False

    def __post_init__(self) -> None:
        if self.fidelity != "scaffold":
            raise FaithfulDietAI24NotImplemented(
                "DietAI24 fidelity must stay 'scaffold' until a live "
                f"GPT Vision and embedding run exists. Refusing {self.fidelity!r}."
            )
        if self.allow_live or self.main_table_eligible:
            raise FaithfulDietAI24NotImplemented(
                "live DietAI24 evaluation is not implemented; "
                "main_table_eligible and allow_live stay false"
            )
        if self.baseline_id != "b1_dietai24":
            raise DietAI24ScaffoldError(f"unexpected baseline id {self.baseline_id!r}")
        if self.fndds_release != FNDDS_RELEASE or self.expected_food_codes != EXPECTED_FOOD_CODES:
            raise DietAI24ScaffoldError("protocol pins FNDDS 2019-2020 with 5624 codes")
        if self.nutrient_count != NUTRIENT_COUNT:
            raise DietAI24ScaffoldError("protocol estimates 65 nutrients")
        if self.temperature != TEMPERATURE:
            raise DietAI24ScaffoldError("published DietAI24 calls use temperature 0")
        if self.portion_source != PORTION_SOURCE:
            raise DietAI24ScaffoldError(
                "portions come from FNDDS portion descriptors, not portion_grid"
            )
        if self.embedding_model != EMBEDDING_MODEL or self.vector_store != VECTOR_STORE:
            raise DietAI24ScaffoldError("published retrieval is text-embedding-3-large via LangChain")
        if self.mllm_family != MLLM_FAMILY:
            raise DietAI24ScaffoldError("published MLLM family is GPT Vision, not the in-repo Qwen path")
