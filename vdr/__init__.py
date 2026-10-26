"""Verified Dietary Reasoning (VDR) package.

Hard constraints (do not violate in this repo):
- Do not load legacy nutrition-regression weights.
- Volume tools default OFF.
- Compute assumption: 2x 4090D 24GB.
- Public data only: Nutrition5K + FNDDS; human labels 200-500 dishes.
"""

__version__ = "0.1.0"
