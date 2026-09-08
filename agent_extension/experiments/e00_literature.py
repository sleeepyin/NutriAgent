#!/usr/bin/env python3
"""Phase 0: literature matrix from the audited neighbor list. No first-claim."""
from __future__ import annotations

import argparse
import csv
from pathlib import Path

from .common import add_common_args, out_dir, save_status

ROWS = [
    ("Open-KNEAD", "2607.12911", "nutrition Agent + retrieval", "semantic tool / strong baseline", "not first nutrition Agent"),
    ("OmniFood-Bench", "2607.08423", "VLM nutrient benchmark", "related-work + gap", "visual-semantic vs physical mass gap already discussed"),
    ("FoodAgent", "IEEE BSN 2025", "LLM expert routing", "matched LLM routing baseline", "routing itself is not novel"),
    ("PIDA", "KSEM", "food active clarification", "matched question baseline", "Q&A-only is insufficient"),
    ("FoodSage", "CHI EA 2025", "dialogue under recognition uncertainty", "related-work", "identification not continuous mass"),
    ("CVPRW 2026 interactive nutrition", "CVPRW 2026", "same-image self-QA risk", "must distinguish new evidence vs extra tokens", "no new physical evidence"),
    ("Conf-Gen", "2605.28920", "conformal generative / questions", "protocol neighbor", "not continuous food regression"),
    ("C-IP", "NeurIPS 2025", "conformal question selection", "matched question policy", "wide-interval => ask is not novel"),
    ("ALMA", "Machine Learning 2026", "conformal + active multimodal", "matched multimodal acquire baseline", "general cross-point already covered"),
    ("MiCP", "2604.01413", "adaptive stopping", "stopping itself not a contribution", "compare terminal vs all-state"),
    ("BCEA", "2606.16667", "post-policy recalibration", "required validity protocol", "not a theory first"),
    ("RouteCert", "2608.15520", "path/modality-group certification", "matched path baseline", "regression adaptation"),
    ("DCP-MT", "PMLR 266 2025", "multi-target joint validity", "joint coverage baseline", "five-output joint must be reported"),
    ("Conformal Policy Learning", "2311.01457", "robot policy under shift", "related-work", "not food measurement"),
    ("Co-GLANCE", "2606.09919", "active perception teaming", "related-work", "robot allocation"),
    ("AFA / ACO / BRiG-AFA / NM-PPG / AFABench", "multiple", "generic AFA", "high-order baseline if n allows", "tabular feature acquire != heterogeneous tools"),
    ("Nutrition5K", "CVPR 2021", "RGB-D food dataset", "D_base + depth pairing", ""),
    ("NutritionVerse-Real", "2401.08598", "real multi-view dishes", "view + OOD", ""),
    ("MetaFood3D", "2409.01966", "3D + density + nutrients", "morphology / density ceiling", "GT volume is oracle"),
    ("ECUSTFD", "public", "coin-reference food images", "reference tool", "GT box is annotation oracle"),
]


def main(argv=None) -> int:
    ap = add_common_args(argparse.ArgumentParser())
    args = ap.parse_args(argv)
    d = out_dir("e00_literature", args)
    csv_path = d / "literature_matrix.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["work", "ref", "overlap", "how_we_use", "what_we_must_not_claim"])
        w.writerows(ROWS)
    # also copy to prereg folder-adjacent results
    note = {
        "first_claim": False,
        "to_the_best_of_knowledge_allowed": False,
        "reason": "Systematic Scopus/WoS/GS search is not executed in this run. Matrix is the 2026-09-04 audited neighbor list. Do not add first-claim language.",
        "required_before_submission": [
            "Scopus",
            "Web of Science",
            "Google Scholar",
            "IEEE Xplore",
            "ACM DL",
            "PubMed",
        ],
    }
    save_status(d, True, n=len(ROWS), note=note, csv=str(csv_path))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
