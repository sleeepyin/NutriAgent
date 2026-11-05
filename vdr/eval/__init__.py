"""Evaluation metrics.

Toy MAE / evidence helpers stay in ``metrics``. P0 offline reports (no
training) live beside them: action confusion, provenance validity, and
selective coverage / MAE / AURC. Closed-loop FixRate / WorsenRate
lives in ``fixrate``. Hold-out energy MAE / Coverage / Selective MAE / AURC
lives in ``energy_selective``.
"""

from vdr.eval.action_metrics import (
    ACTIONS,
    balanced_accuracy,
    confusion_matrix,
    format_report,
    macro_f1,
    overall_accuracy,
    per_class_prf,
)
from vdr.eval.metrics import (
    evidence_accuracy,
    fix_rate,
    has_evidence_path,
    mae,
    mape,
    nutrition_field_mae_mape,
    worsen_rate,
)
from vdr.eval.provenance import (
    FNDDS_PINNED_N_CODES,
    FNDDS_PINNED_RELEASE,
    CatalogPinError,
    check_record,
    evaluate as evaluate_provenance,
    load_catalog,
    scale_expected,
)
from vdr.eval.fixrate import (
    BUDGET_K,
    FIXRATE_EPSILON_KCAL,
    evaluate_fixrate,
    format_fixrate_report,
)
from vdr.eval.energy_selective import (
    DEFINITION_ID as ENERGY_DEFINITION_ID,
    evaluate_energy_table,
    format_energy_report,
)
from vdr.eval.selective import aurc, coverage_and_selective_mae, risk_coverage_curve

__all__ = [
    "ACTIONS",
    "CatalogPinError",
    "FNDDS_PINNED_N_CODES",
    "FNDDS_PINNED_RELEASE",
    "BUDGET_K",
    "ENERGY_DEFINITION_ID",
    "FIXRATE_EPSILON_KCAL",
    "aurc",
    "balanced_accuracy",
    "check_record",
    "confusion_matrix",
    "coverage_and_selective_mae",
    "evaluate_energy_table",
    "evaluate_fixrate",
    "evaluate_provenance",
    "evidence_accuracy",
    "fix_rate",
    "format_energy_report",
    "format_fixrate_report",
    "format_report",
    "has_evidence_path",
    "load_catalog",
    "macro_f1",
    "mae",
    "mape",
    "nutrition_field_mae_mape",
    "overall_accuracy",
    "per_class_prf",
    "risk_coverage_curve",
    "scale_expected",
    "worsen_rate",
]
